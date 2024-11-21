from dataclasses import dataclass
from layers.gpt import GPT
import jax


@dataclass
class GPTConfig:
    block_size: int = 200
    vocab_size: int = (
        50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    )
    n_layer: int = 3
    n_head: int = 3
    n_embd: int = 512
    dropout: float = 0.0
    bias: bool = False
    embd_pdrop = 0.1
    layer_norm_epsilon = 0.01


mymodel = GPT(GPTConfig(), key=jax.random.key(1))
jax.vmap(mymodel)(
    jax.numpy.ones((15, 32), dtype=jax.numpy.int32),
    jax.numpy.ones((15, 168), dtype=jax.numpy.int32),
)
