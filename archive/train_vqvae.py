import optax
import equinox as eqx
import jax.numpy as jnp
from tensorboardX import SummaryWriter
from layers.VQVAE import VQVAE
from datasets import load_dataset
import datetime
import jax


def update_codebook_ema(model, updates: tuple, codebook_indices, key=None):
    avg_updates = jax.tree.map(lambda x: jax.numpy.mean(x, axis=0), updates)

    # Calculate which codes are too often used and yeet them. Prior is uniform.
    h = jnp.histogram(
        codebook_indices, bins=model.quantizer.K, range=(0, model.quantizer.K)
    )[0] / len(codebook_indices)
    part_that_should_be = 1 / model.quantizer.K
    mask = (h > 2 * part_that_should_be) | (h < 0.5 * part_that_should_be)
    rand_embed = (
        jax.random.normal(key, (model.quantizer.K, model.quantizer.D)) * mask[:, None]
    )
    avg_updates = (
        avg_updates[0],
        avg_updates[1],
        jnp.where(mask[:, None], rand_embed, avg_updates[2]),
    )

    where = lambda q: (
        q.quantizer.cluster_size,
        q.quantizer.codebook_avg,
        q.quantizer.codebook,
    )

    # Update the codebook and other trackers.
    model = eqx.tree_at(where, model, avg_updates)
    return model


@eqx.filter_jit
@eqx.filter_value_and_grad(has_aux=True)
def calculate_losses(model, x):
    z_e, z_q, codebook_updates, y = jax.vmap(model)(x)

    # Are the inputs and outputs close?
    reconstruct_loss = jnp.mean(jnp.linalg.norm((x - y), ord=2, axis=(1, 2)))

    # Are the output vectors z_e close to the codes z_q ?
    commit_loss = jnp.mean(
        jnp.linalg.norm(z_e - jax.lax.stop_gradient(z_q), ord=2, axis=(1, 2))
    )

    total_loss = reconstruct_loss + commit_loss

    return total_loss, (reconstruct_loss, commit_loss, codebook_updates, y)


@eqx.filter_jit
def make_step(model, optimizer, opt_state, x):
    (total_loss, (reconstruct_loss, commit_loss, codebook_updates, y)), grads = (
        calculate_losses(model, x)
    )
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    # model = update_codebook_ema(model, codebook_updates[0], codebook_updates[1], key)

    return (
        model,
        opt_state,
        total_loss,
        reconstruct_loss,
        commit_loss,
        codebook_updates,
        y,
    )
    

key1, key2 = jax.random.split(jax.random.key(1), 2)

model = VQVAE(key=key1)

optimizer = optax.adam(1e-4)
opt_state = optimizer.init(model)

writer = SummaryWriter(log_dir='./runs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

epochs = 50
batch_size = 32
step = 0

dataset = load_dataset("blabble-io/libritts_r", "clean", streaming=True)
dataloader= dataset["train.clean.360"].batch(batch_size=batch_size)

for epoch in range(epochs):
    eqx.tree_serialise_leaves(f"checkpoints/{epoch}.eqx", model)

    for batch in enumerate(dataloader):

        model, opt_state, total_loss, reconstruct_loss, commit_loss, codebook_updates, y = make_step(model, optimizer, opt_state, batch)

        # Log codebook updates to TensorBoard
        writer.add_scalar('Loss/Total', total_loss, step)
        writer.add_scalar('Loss/Reconstruct', reconstruct_loss, step)
        writer.add_scalar('Loss/Commit', commit_loss, step)
        step += 1
        # writer.add_histogram('Codebook Updates/Code ids used', jnp.reshape(codebook_updates[1], -1), step)
        # writer.add_histogram('Codebook Updates/Code means', jnp.mean(codebook_updates[0][2], axis=(0,2)), step)
        # writer.add_histogram('Codebook Updates/Code stds', jnp.std(codebook_updates[0][2], axis=(0,2)), step)
        # if (i // batch_size) % 20 == 0:
        #     print(batch.shape)
        #     print(y.shape)
        #     ax1.clear()
        #     ax2.clear()
        #     ax1.imshow(batch[0], aspect='auto', origin='lower')
        #     ax2.imshow(y[0], aspect='auto', origin='lower')
        #     display(fig)
        #     clear_output(wait=True)
    # plt.imshow(y[0])
