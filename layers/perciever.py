#!/usr/bin/env python
# coding: utf-8



import jax
import jax.numpy as jnp
import equinox as eqx
import equinox.nn as nn
import typing as tp
from jax import config

# config.update("jax_enable_x64", True)




class GEGLU(eqx.Module):
    def __call__(self, x):
        x, gate = jnp.split(x, 2, axis=-1)
        return jax.nn.gelu(gate, approximate=False) * x




from typing import Optional


class CausalConv1d(eqx.nn.Conv1d):
    causal_padding: int

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        (kernel_size,) = self.kernel_size
        (dilation,) = self.dilation
        (stride,) = self.stride

        assert stride == 1
        self.causal_padding = dilation * (kernel_size - 1)

    def __call__(self, x: jax.Array, *, key: Optional[jax.Array] = None) -> jax.Array:
        causal_padded_x = jax.numpy.pad(
            x, ((0, 0), (self.causal_padding, 0)), mode="constant", constant_values=0.0
        )
        # print(causal_padded_x.shape)
        return super().__call__(causal_padded_x, key=key)




class RMSNorm(eqx.Module):
    scale: float

    def __init__(self, dim, scale=None):
        self.scale = dim**0.5

    def __call__(self, x):
        return (x / jax.numpy.linalg.norm(x, axis=-1)) * self.scale




class Attend(eqx.Module):
    dropout: float
    causal: bool
    attn_dropout: nn.Dropout

    def __init__(self, dropout=0.0, causal=False, use_flash=False):
        self.dropout = dropout
        self.attn_dropout = eqx.nn.Dropout(dropout, inference=True)

        self.causal = causal

    def get_mask(self, n):
        return jnp.triu(jnp.ones((n, n), dtype=bool), k=1)

    def __call__(self, q, k, v, mask=None):
        n = q.shape[-2]
        scale = q.shape[-1] ** -0.5
        kq = jnp.matmul(q, jnp.transpose(k, (0, 2, 1))) * scale
        # Key mask
        if mask is not None:
            mask = jnp.expand_dims(mask, 0)
            kq = jnp.where(mask, kq, jnp.zeros_like(mask))

        if self.causal:
            kq = jax.numpy.where(self.get_mask(n), kq, -jnp.finfo(kq.dtype).max)

        attn = jax.nn.softmax(kq, axis=-1)
        # attn = self.attn_dropout(attn)

        out = jnp.matmul(attn, v)

        return out




from functools import partial
from einops import rearrange


class Attention(eqx.Module):

    cross_attn_include_queries: bool
    scale: float
    heads: int

    attend: Attend
    to_q: nn.Linear
    to_kv: nn.Linear
    to_out: nn.Linear

    dim_inner: int

    def __init__(
        self,
        dim,
        *,
        dim_context=None,
        causal=False,
        dim_head=64,
        heads=8,
        dropout=0.0,
        use_flash=False,
        cross_attn_include_queries=False,
        key=None,
    ):
        key1, key2, key3 = jax.random.split(key, 3)

        self.scale = dim_head**-0.5
        self.heads = heads
        self.cross_attn_include_queries = cross_attn_include_queries
        self.dim_inner = dim_head * heads

        self.attend = Attend(dropout, causal)

        self.to_q = nn.Linear(dim, self.dim_inner, use_bias=False, key=key1)
        self.to_kv = nn.Linear(dim, self.dim_inner * 2, use_bias=False, key=key2)
        self.to_out = nn.Linear(self.dim_inner, dim, use_bias=False, key=key3)

    # @partial(jax.jit, static_argnums=2)
    def __call__(self, x, context, mask=None):

        # Should the kv, cross attention, include the query values ?
        context = jnp.concat([x, context], axis=-2)
        q, k, v = (
            jax.vmap(jax.vmap(self.to_q))(x),
            *jnp.split(jax.vmap(jax.vmap(self.to_kv))(context), 2, axis=-1),
        )
        # q = jnp.reshape(q, shape=(q.shape[0], self.heads, q.shape[-2], -1))

        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v)
        )

        # k = jnp.reshape(k, shape=(k.shape[0], self.heads, k.shape[-2], -1))
        # v = jnp.reshape(v, shape=(v.shape[0], self.heads, v.shape[-2], -1))
        out = jax.vmap(self.attend)(q, k, v, mask)
        out = rearrange(out, "b h n d -> b n (h d)")

        return jax.vmap(jax.vmap(self.to_out))(out)




class FeedForward(eqx.Module):
    causal_conv: bool
    ff1: nn.Linear
    ff2: nn.Linear
    act: GEGLU
    conv: CausalConv1d

    def __init__(self, dim, mult=4, causal_conv=False, key=None):
        key1, key2, key3 = jax.random.split(key, 3)

        self.causal_conv = causal_conv
        dim_inner = int(dim * mult * 2 / 3)
        self.conv = CausalConv1d(dim_inner, dim_inner, 3, key=key3)
        self.act = GEGLU()
        self.ff1 = nn.Linear(dim, dim_inner * 2, key=key1)
        self.ff2 = nn.Linear(dim_inner, dim, key=key2)

    def __call__(self, x):
        y = jax.vmap(self.ff1)(x)
        y = self.act(y)
        if self.causal_conv:
            y = jnp.permute_dims(y, (1, 0))
            y = self.conv(y)
            y = jnp.permute_dims(y, (1, 0))
        y = jax.vmap(self.ff2)(y)

        return y




from einops import repeat


class PerceiverResampler(eqx.Module):

    proj_context: jax.Array
    latents: jax.Array
    layers: list
    norm: RMSNorm

    def __init__(
        self,
        *,
        dim,
        depth=2,
        dim_context=None,
        num_latents=32,
        dim_head=64,
        heads=8,
        ff_mult=4,
        use_flash_attn=False,
        key=None,
    ):

        key1, key2, key3 = jax.random.split(key, 3)
        if dim_context is None:
            dim_context = dim

        self.proj_context = (
            nn.Linear(dim_context, dim, key=key1)
            if dim != dim_context
            else nn.Identity()
        )

        self.latents = jax.random.normal(key3, (num_latents, dim))

        self.layers = [
            (
                Attention(
                    dim=dim,
                    dim_head=dim_head,
                    heads=heads,
                    use_flash=use_flash_attn,
                    cross_attn_include_queries=True,
                    key=y1,
                ),
                FeedForward(dim=dim, mult=ff_mult, key=y1),
            )
            for y1 in jax.random.split(key2, depth)
        ]

        self.norm = RMSNorm(dim)

    def __call__(self, x, mask=None):
        # print(f"Shape of x: {x.shape}")
        y = jax.vmap(self.proj_context)(x)
        # print(f"Shape of y: {y.shape}")
        latents = repeat(self.latents, "n d -> b n d", b=x.shape[0])
        # print(f"Shape of latent: {self.latents.shape}")
        # latents = j

        for attn, ff in self.layers:
            print(latents[0, 0])

            latents = attn(latents, y, mask) + latents
            print(latents[0, 0])
            latents = jax.vmap(ff)(latents) + latents
        return jax.vmap(jax.vmap(self.norm))(latents)
