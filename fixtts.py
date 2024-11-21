from dataclasses import dataclass
import os
from layers.gpt import GPT
from layers.hifigan import Generator
from layers.resnet import ResNet
from layers.perciever import PerceiverResampler

import jax
import equinox as eqx


@dataclass
class GPTConfig:
    block_size: int = 200
    # vocab_size: int = (
    #     50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    # )
    n_layer: int = 16
    n_head: int = 16
    n_embd: int = 512
    dropout: float = 0.05
    bias: bool = True
    embd_pdrop = 0.05
    layer_norm_epsilon = 0.01


class FiXTTS:

    gpt: GPT
    hifigan: Generator
    # perciever: PerceiverResampler
    # resnet: ResNet

    def __init__(self):
        # self.perciever = PerceiverResampler(
        #     dim=self.model_dim,
        #     depth=2,
        #     dim_context=self.model_dim,
        #     num_latents=32,
        #     dim_head=64,
        #     heads=8,
        #     ff_mult=4,
        #     use_flash_attn=False,
        #     key=jax.random.key(1),
        # )

        self.gpt = GPT(GPTConfig(), key=jax.random.key(1))

        self.hifigan = Generator(
            1024, 1, h_u=512, cond_channels=512, key=jax.random.key(1)
        )

        # self.resnet = ResNet(
        #     input_dims=64, proj_dim=512, log_input=True, key=jax.random.key(1)
        # )

        # self.dvae =

    # def inference(self, input_ids):
    #     current_inputs = input_ids

    #     while

    def load_checkpoints(self, checkpoint_dir: str):
        self.perciever = eqx.tree_deserialise_leaves(
            os.path.join(checkpoint_dir, "xttsperciever.eqx"), self.perciever
        )

        self.gpt = eqx.tree_deserialise_leaves(
            os.path.join(checkpoint_dir, "xttsgpt.eqx"),
            self.gpt,
        )

        self.hifigan = eqx.tree_deserialise_leaves(
            os.path.join(checkpoint_dir, "xttshifigan.eqx"),
            self.hifigan,
        )

        self.resnet = eqx.tree_deserialise_leaves(
            os.path.join(checkpoint_dir, "xttsresnet.eqx"),
            self.resnet,
        )
