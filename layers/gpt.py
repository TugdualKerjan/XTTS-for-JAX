#!/usr/bin/env python
# coding: utf-8


import jax
import equinox as eqx
import equinox.nn as nn
import jax.numpy as jnp
import typing as tp


class SwiGLU(eqx.Module):
    W: nn.Linear
    V: nn.Linear
    b: jax.Array
    c: jax.Array

    def __init__(self, input_dim, output_dim, key):
        key1, key2, key3, key4 = jax.random.split(key, 4)
        self.W = nn.Linear(input_dim, output_dim, key=key1)
        self.V = nn.Linear(input_dim, output_dim, key=key2)
        self.b = jax.random.normal(key3, (output_dim))
        self.c = jax.random.normal(key4, (output_dim))

    # @eqx.filter_jit
    def __call__(self, x):
        return jax.nn.swish((self.W(x) + self.b) * (self.V(x) + self.c))


class MLP(eqx.Module):
    c_fc: nn.Linear
    c_proj: nn.Linear
    drop: nn.Dropout
    act: SwiGLU

    def __init__(self, config, key):

        key1, key2, key3 = jax.random.split(key, 3)

        self.c_fc = nn.Linear(
            config.n_embd, 4 * config.n_embd, use_bias=config.bias, key=key1
        )
        self.act = SwiGLU(4 * config.n_embd, 4 * config.n_embd, key=key2)
        self.c_proj = nn.Linear(
            4 * config.n_embd, config.n_embd, use_bias=config.bias, key=key3
        )
        self.drop = nn.Dropout(config.dropout)

    # TODO: Interesting take on the fact that vmap should be applied here ?
    # @eqx.filter_jit
    def __call__(self, x):
        y = self.c_fc(x)
        y = self.act(y)
        y = self.c_proj(y)
        y = self.drop(y)

        return y


import math


class CausalSelfAttention(eqx.Module):
    attnk: nn.Linear
    attnq: nn.Linear
    attnv: nn.Linear
    proj: nn.Linear

    resid_dropout: nn.Dropout
    attn_dropout: nn.Dropout

    mask: jax.Array = eqx.field(static=True)

    def __init__(self, config, key):
        key1, key2, key3, key4 = jax.random.split(key, 4)

        self.attnk = nn.Linear(
            config.n_embd, config.n_embd, use_bias=config.bias, key=key1
        )
        self.attnv = nn.Linear(
            config.n_embd, config.n_embd, use_bias=config.bias, key=key2
        )
        self.attnq = nn.Linear(
            config.n_embd, config.n_embd, use_bias=config.bias, key=key3
        )
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.proj = nn.Linear(
            config.n_embd, config.n_embd, use_bias=config.bias, key=key4
        )

        self.mask = jnp.tril(jnp.ones((config.block_size, config.block_size)))

    # Could play arround with the different attention score calculations (Baidhu ?)
    # X is an embedding, it should self attend.

    # @eqx.filter_jit
    def __call__(self, x):
        # x = jnp.swapaxes(x, -1, -2)
        T, C = x.shape  # Seq length and embedding dim.

        q = jax.vmap(self.attnq)(x)
        k = jax.vmap(self.attnk)(x)
        v = jax.vmap(self.attnv)(x)

        att = jnp.matmul(q, jnp.transpose(k)) / math.sqrt(jnp.shape(k)[-1])
        att = jnp.where(
            jax.numpy.equal(jax.lax.stop_gradient(self.mask[:T, :T]), 0),
            float("-inf"),
            att,
        )
        att = jax.nn.softmax(att, axis=-1)
        att = self.attn_dropout(att)

        y = jnp.matmul(att, v)

        y = jax.vmap(self.proj)(y)
        y = self.resid_dropout(y)
        return y


class Block(eqx.Module):
    norm: nn.LayerNorm
    attn: CausalSelfAttention
    mlp: MLP

    def __init__(self, config, key):
        key1, key2 = jax.random.split(key, 2)

        self.norm = nn.LayerNorm(config.n_embd, use_bias=config.bias)
        self.attn = CausalSelfAttention(config, key=key1)
        self.mlp = MLP(config, key=key2)

    # @eqx.filter_jit
    def __call__(self, x):
        y = jax.vmap(self.norm)(x)
        y = self.attn(
            y
        )  # Can't vmap as the whole point is exchange info between tokens.
        x = y + x

        y = jax.vmap(self.norm)(y)
        y = jax.vmap(self.mlp)(y)
        x = y + x

        return x


class TransformerLayer(eqx.Module):
    text_wte: nn.Embedding  # Token embeddings
    text_wpe: nn.Embedding  # Positional embeddings
    audio_wpe: nn.Embedding  # Positional embeddings
    audio_wte: nn.Embedding  # Positional embeddings

    drop: nn.Dropout

    layers: list
    norm: nn.LayerNorm

    def __init__(self, config, key):
        key1, key2, key3, key4 = jax.random.split(key, 4)

        self.drop = nn.Dropout(config.embd_pdrop, deterministic=True)

        self.text_wte = nn.Embedding(50257, config.n_embd, key=key1)
        self.text_wpe = nn.Embedding(200, config.n_embd, key=key2)
        self.audio_wte = nn.Embedding(1025, config.n_embd, key=key3)
        self.audio_wpe = nn.Embedding(200, config.n_embd, key=key4)

        self.layers = [Block(config, key) for _ in range(config.n_layer)]
        self.norm = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    # @eqx.filter_jit
    def __call__(
        self,
        text_token_ids,  # One ID inputted ?
        audio_token_ids,  # One ID inputted ?
        attention_mask: tp.Optional[jax.Array] = None,  # Used !
        output_attentions: tp.Optional[bool] = None,  # Isn't used
        use_cache: tp.Optional[bool] = False,  # Set to true.
    ):

        text_input_shape = text_token_ids.shape
        audio_input_shape = audio_token_ids.shape

        # Should use better positional embeddings with cos and sin.
        # if past_key_values is None:
        #     past_length = 0
        #     past_key_values = tuple([None] * len(self.h))
        # else:
        #     past_length = past_key_values[0].shape[-2]

        text_position_ids = jax.numpy.arange(0, text_input_shape[-1])

        audio_position_ids = jax.numpy.arange(0, audio_input_shape[-1])

        text_input_embeds = jax.vmap(self.text_wte)(text_token_ids)
        text_pos_input_embeds = jax.vmap(self.text_wpe)(text_position_ids)

        audio_input_embeds = jax.vmap(self.audio_wte)(audio_token_ids)
        audio_pos_input_embeds = jax.vmap(self.audio_wpe)(audio_position_ids)

        # Dropout at the first layer ? Seems a bit aggressive...
        x = self.drop(
            jax.numpy.concat(
                [
                    text_input_embeds + text_pos_input_embeds,
                    audio_input_embeds + audio_pos_input_embeds,
                ],
                axis=-2,
            )
        )

        for block in self.layers:
            x = block(x)
        x = jax.vmap(self.norm)(x)

        return x


class GPT(eqx.Module):
    transformer: TransformerLayer
    text_lm_head: nn.Linear
    audio_lm_head: nn.Linear

    def __init__(self, config, key):
        key1, key2, key3 = jax.random.split(key, 3)

        self.transformer = TransformerLayer(config, key1)
        self.text_lm_head = nn.Linear(config.n_embd, 50257, use_bias=False, key=key2)
        self.audio_lm_head = nn.Linear(config.n_embd, 1025, use_bias=False, key=key3)

    def __call__(self, text_token_ids, audio_token_ids):

        output = self.transformer(text_token_ids, audio_token_ids)
        text_y = output[: text_token_ids.shape[-1] - 1, :]
        audio_y = output[text_token_ids.shape[-1] - 1 :, :]

        text_logits = jax.vmap(self.text_lm_head)(text_y)
        text_logits = jax.nn.softmax(text_logits, axis=-1)
        audio_logits = jax.vmap(self.audio_lm_head)(audio_y)
        audio_logits = jax.nn.softmax(audio_logits, axis=-1)
        return text_logits, audio_logits
