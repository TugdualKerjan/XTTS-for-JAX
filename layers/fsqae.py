import jax
import equinox as eqx
import equinox.nn as nn
import jax.numpy as jnp

class Encoder(eqx.Module):
    conv1: nn.Conv1d
    conv2: nn.Conv1d
    conv3: nn.Conv1d
    conv4: nn.Conv1d

    def __init__(self, in_channels, hidden_channels, latent_channels, key):
        key1, key2, key3, key4, key5 = jax.random.split(key, 5)

        self.conv1 = nn.Conv1d(
            in_channels,
            hidden_channels,
            kernel_size=4,
            stride=2,
            padding="SAME",
            key=key1,
        )
        self.conv2 = nn.Conv1d(
            hidden_channels,
            hidden_channels,
            kernel_size=4,
            stride=2,
            padding="SAME",
            key=key2,
        )
        self.conv3 = nn.Conv1d(
            hidden_channels,
            hidden_channels,
            kernel_size=4,
            stride=2,
            padding="SAME",
            key=key3,
        )
        self.conv4 = nn.Conv1d(
            hidden_channels, latent_channels, kernel_size=1, key=key4
        )

    def __call__(self, x):
        y = self.conv1(x)
        y = jax.nn.relu(y)
        y = self.conv2(y)
        y = jax.nn.relu(y)
        y = self.conv3(y)
        y = jax.nn.relu(y)
        y = self.conv4(y)

        return y


class Decoder(eqx.Module):
    conv1: nn.ConvTranspose1d
    conv2: nn.ConvTranspose1d
    conv3: nn.ConvTranspose1d
    conv4: nn.ConvTranspose1d

    def __init__(self, in_channels, hidden_channels, latent_channels, key):
        key1, key2, key3, key4, key5 = jax.random.split(key, 5)

        self.conv1 = nn.ConvTranspose1d(
            latent_channels,
            hidden_channels,
            kernel_size=4,
            stride=2,
            padding="SAME",
            key=key1,
        )
        self.conv2 = nn.ConvTranspose1d(
            hidden_channels,
            hidden_channels,
            kernel_size=4,
            stride=2,
            padding="SAME",
            key=key2,
        )
        self.conv3 = nn.ConvTranspose1d(
            hidden_channels,
            hidden_channels,
            kernel_size=4,
            stride=2,
            padding="SAME",
            key=key3,
        )
        self.conv4 = nn.ConvTranspose1d(
            hidden_channels, in_channels, kernel_size=1, key=key4
        )

    def __call__(self, x):
        y = self.conv1(x)
        y = jax.nn.relu(y)
        y = self.conv2(y)
        y = jax.nn.relu(y)
        y = self.conv3(y)
        y = jax.nn.relu(y)
        y = self.conv4(y)

        return y


Codeword = jax.Array
Indices = jax.Array


class FSQ(eqx.Module):
    """Quantizer, taken from https://github.com/google-research/google-research/blob/master/fsq/fsq.ipynb"""

    _levels: list[int]
    _levels_np: jax.Array
    _eps: float
    _basis: jax.Array
    _implicit_codebook: jax.Array

    def __init__(self, levels: list[int], eps: float = 1e-3):
        self._levels = levels
        self._eps = eps
        self._levels_np = jnp.asarray(levels)
        self._basis = jnp.concatenate(
            (jnp.array([1]), jnp.cumprod(self._levels_np[:-1]))
        )

        self._implicit_codebook = self.indexes_to_codes(jnp.arange(self.codebook_size))

    @property
    def num_dimensions(self) -> int:
        """Number of dimensions expected from inputs."""
        return len(self._levels)

    @property
    def codebook_size(self):
        """Size of the codebook."""
        return jnp.prod(jnp.array(self._levels))

    @property
    def codebook(self):
        """Returns the implicit codebook. Shape (prod(levels), num_dimensions)."""
        return self._implicit_codebook

    @eqx.filter_jit
    def round_ste(self, z):
        """Round with straight through gradients."""
        zhat = jnp.round(z)
        return z + jax.lax.stop_gradient(zhat - z)

    @eqx.filter_jit
    def bound(self, z: jax.Array) -> jax.Array:
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels_np - 1) * (1 - self._eps) / 2
        offset = jnp.where(self._levels_np % 2 == 1, 0.0, 0.5)
        shift = jnp.tan(offset / half_l)
        return jnp.tanh(z + shift) * half_l - offset

    @eqx.filter_jit
    def __call__(self, z: jax.Array) -> Codeword:
        """Quanitzes z, returns quantized zhat, same shape as z."""
        quantized = self.round_ste(self.bound(z))

        # Renormalize to [-1, 1].
        half_width = self._levels_np // 2
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized):
        # Scale and shift to range [0, ..., L-1]
        half_width = self._levels_np // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat):
        half_width = self._levels_np // 2
        return (zhat - half_width) / half_width

    def codes_to_indexes(self, zhat: Codeword) -> Indices:
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.num_dimensions
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(axis=-1).astype(jnp.uint32)

    def indexes_to_codes(self, indices: Indices) -> Codeword:
        """Inverse of `indexes_to_codes`."""
        indices = indices[..., jnp.newaxis]
        codes_non_centered = jnp.mod(
            jnp.floor_divide(indices, self._basis), self._levels_np
        )
        return self._scale_and_shift_inverse(codes_non_centered)


class VQVAE(eqx.Module):
    encoder: Encoder
    decoder: Decoder
    quantizer: FSQ

    def __init__(self, in_channels, hidden_channels, latent_channels, levels, key=None):
        key1, key2 = jax.random.split(key)

        self.encoder = Encoder(in_channels, hidden_channels, latent_channels, key1)
        self.decoder = Decoder(in_channels, hidden_channels, latent_channels, key2)
        self.quantizer = FSQ(levels=levels)
        print("✅ Model initialized")

    def __call__(self, x):
        x = jnp.expand_dims(x, 0)
        
        z_e = self.encoder(x)
        z_q = self.quantizer(z_e)
        y = self.decoder(z_q)

        y = jnp.squeeze(y)
        return y