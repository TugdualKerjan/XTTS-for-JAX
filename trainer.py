from datasets import load_dataset
import numpy
from mytokenizer import Tokenizer
from fixtts import FiXTTS

import optax
import equinox as eqx
import jax


class Trainer:

    tokenizer: Tokenizer
    fixtts: FiXTTS

    def __init__(self):
        self.tokenizer = Tokenizer(checkpoint_dir="checkpoints", sample_rate=22050)
        print("✅ Tokenizer initialized")

        self.fixtts = FiXTTS()
        print("✅ FiXTTS initialized")

    def train_gpt(self):
        learning_rate = 1e-5

        optimizer = optax.inject_hyperparams(optax.adamw)(learning_rate=learning_rate)
        optimizer_state = optimizer.init(eqx.filter(self.fixtts.gpt, eqx.is_array))

        # @eqx.filter_jit
        def loss(model, x, y):
            # text_logits, audio_logits = jax.vmap(
            #     model, in_axes=(0, 0), out_axes=(0, 0)
            # )(x, y)
            # print(f"😁 Text and Audio shapes:\n{x.shape}\n{y.shape}")

            text_logits, audio_logits = model(x, y[:-1])

            # audio_logits = jax.numpy.clip(audio_logits, 1e-3)
            # y = jax.numpy.array(y, dtype=jax.numpy.int32)
            # jax.debug.print(
            #     f"👀 Text and Audio output shape:\n{text_logits.shape}\n{audio_logits.shape}"
            # )
            # print(jax.numpy.sum(audio_logits[1, -1]))
            # print(y[1, -1])

            audio_loss = optax.softmax_cross_entropy_with_integer_labels(
                logits=audio_logits, labels=y
            )
            text_loss = optax.softmax_cross_entropy_with_integer_labels(
                logits=text_logits, labels=x[1:]
            )

            # print(f"Text loss: {text_loss.mean()}")
            # print(f"Audio loss: {audio_loss.mean()}")

            return jax.numpy.mean(text_loss) + jax.numpy.mean(audio_loss)

        def make_step(model, optimizer_state, x, y):
            losses, grads = eqx.filter_value_and_grad(loss)(model, x, y)
            updates, optimizer_state = optimizer.update(grads, optimizer_state, model)
            model = eqx.apply_updates(model, updates)
            return model, optimizer_state, losses

        dataloader = self.get_dataloader(self.tokenizer, batch_size=1)

        print("🦀🦀🦀 Training")
        for batch in dataloader:
            for i in range(len(batch["text"])):
                x, y = batch["text"][i], batch["audio"][i]
                self.fixtts.gpt, optimizer_state, losses = make_step(
                    self.fixtts.gpt, optimizer_state, x, y
                )
                print(f"Loss: {losses}")

    def get_dataloader(self, tokenizer, batch_size):
        dataset = load_dataset("blabble-io/libritts_r", "clean", streaming=True)

        def encode(sample):
            # print(sample)
            text, audio = tokenizer.encode(
                sample["text_normalized"],
                sample["audio"]["array"],
                sample["audio"]["sampling_rate"],
            )
            print(text)

            return {"text": text, "audio": audio}

        def remove_too_long(sample):
            return len(sample["text"]) + len(sample["audio"]) <= 200

        def pad(sample):
            return {
                "audio": tokenizer.pad(
                    sample["audio"], len(sample["text"]) + len(sample["audio"])
                )
            }

        dataset = (
            dataset.map(
                encode,
                remove_columns=[
                    "text_original",
                    "speaker_id",
                    "chapter_id",
                    "id",
                    "path",
                    "text_normalized",
                ],
            )
            .filter(remove_too_long)
            .map(pad)
        )

        return dataset["train.clean.360"].batch(batch_size=batch_size)


trainer = Trainer()
trainer.train_gpt()
