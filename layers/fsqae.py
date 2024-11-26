import jax
import equinox as eqx
import equinox.nn as nn
import jax.numpy as jnp

class ResUnit(eqx.Module):
    conv1: jax.Array
    conv2: jax.Array
    
    def __init__(self, channel_in, channel_out, kernel_size, dilation, bias=True, key=None):
        key1, key2 = jax.random.split(key)
        
        self.conv1 = nn.WeightNorm(nn.Conv1d(channel_in, channel_out, kernel_size, stride=1, dilation=dilation, padding="SAME", use_bias=bias, key=key1))
        self.conv2 = nn.WeightNorm(nn.Conv1d(channel_out, channel_out, kernel_size=1, stride=1,use_bias=bias, padding="SAME",key=key2))

    @eqx.filter_jit
    def __call__(self, x):
        y = self.conv1(jax.nn.elu(x))
        y = self.conv2(jax.nn.elu(y))
        return y + x
    
class Upsample(eqx.Module):
    scale_factor:int
    mode:str
    
    def __init__(self, scale_factor, mode):
        self.scale_factor = scale_factor
        self.mode = mode
    
    @eqx.filter_jit
    def __call__(self, x):
        new_height = x.shape[1] * self.scale_factor
        return jax.image.resize(x, (x.shape[0], new_height), method=self.mode)
    
class ResBlock(eqx.Module):
    suite: nn.Sequential

    def __init__(self, channel_in, channel_out, kernel_size:int,stride:int, mode:str, dilations=(1, 3, 9), bias=True, key=None):
        key1, key2 = jax.random.split(key)

        res_channels = channel_in if mode == 'encoder' else channel_out
        
        res_units = [
                    ResUnit(res_channels, res_channels, kernel_size=kernel_size, bias=False, dilation=dilation, key=k) for dilation, k in zip(dilations, jax.random.split(key1, len(dilations)))
                ]

        if mode == "encoder":
            if channel_in == channel_out:
                self.suite = nn.Sequential(res_units + [
                    nn.AvgPool1d(kernel_size=stride, stride=stride),
                ])
            else:
                self.suite = nn.Sequential(res_units + [
                    nn.WeightNorm(nn.Conv1d(channel_in, channel_out, kernel_size=(2*stride), stride=stride, use_bias=bias, padding="SAME", key=key2))
                ])
        elif mode == "decoder":
            if channel_in == channel_out:
                self.suite = nn.Sequential([
                    Upsample(scale_factor=stride, mode="nearest"),
                ]+ res_units)
            else:
                self.suite = nn.Sequential([
                    nn.WeightNorm(nn.ConvTranspose1d(channel_in, channel_out, kernel_size=(2*stride), stride=stride, use_bias=bias, padding="SAME", key=key2))
                ] + res_units)
                
    @eqx.filter_jit
    def __call__(self, x):
        out = x
        for unit in self.suite:
            out = unit(out)
        return out

class Encoder(eqx.Module):
    suite: list
    
    def __init__(self, in_channels, hidden_channels, latent_channels, kernel_size=5, channel_ratios: tuple =(1,4,8,8,16,16),
                 strides:tuple=(2,2,2,5,5),key=None):
        key0, key1, key2 = jax.random.split(key, 3)
        self.suite = [nn.WeightNorm(nn.Conv1d(
            in_channels,
            hidden_channels * channel_ratios[0],
            kernel_size=kernel_size,
            stride=1,
            padding="SAME",
            use_bias=False,
            key=key0,            
        ))] +[
            ResBlock(
                hidden_channels * channel_ratios[idx],
                hidden_channels * channel_ratios[idx + 1],
                kernel_size=kernel_size,
                stride= strides[idx],
                mode="encoder",
                bias=True,
                key=k,
            ) for idx, k in enumerate(jax.random.split(key1, len(strides)))
        ] +[nn.WeightNorm(nn.Conv1d(
            hidden_channels * channel_ratios[-1], latent_channels, kernel_size=1, padding="SAME", key=key2
        ))]
        
    @eqx.filter_jit
    def __call__(self, x):
        out = x 
        
        for unit in self.suite:
            out = unit(out)
            # jax.debug.print("{x}", x=out[0][:10])
            out = jax.nn.elu(out)

        return out



class Decoder(eqx.Module):
    suite: list
    
    def __init__(self, in_channels, hidden_channels, latent_channels, kernel_size=5, channel_ratios: tuple =(16,16,8,8,4,1),
                 strides:tuple=(5,5,2,2,2),key=None):
        key0, key1, key2 = jax.random.split(key, 3)
        self.suite = [nn.WeightNorm(nn.ConvTranspose1d(
            latent_channels,
            hidden_channels * channel_ratios[0],
            kernel_size=1,
            padding="SAME",
            use_bias=True,
            key=key0,            
        ))] +[
            ResBlock(
                hidden_channels * channel_ratios[idx],
                hidden_channels * channel_ratios[idx + 1],
                kernel_size=kernel_size,
                stride= strides[idx],
                mode="decoder",
                bias=True,
                key=k,
            ) for idx, k in enumerate(jax.random.split(key1, len(strides)))
        ] +[nn.WeightNorm(nn.ConvTranspose1d(
            hidden_channels * channel_ratios[-1], in_channels, kernel_size=kernel_size, stride=1, padding="SAME", key=key2
        ))]
        
    @eqx.filter_jit
    def __call__(self, x):
        out = x 
        for unit in self.suite:
            out = unit(out)
            # jax.debug.print("{x}", x=out[0][:10])
            out = jax.nn.elu(out)

        return out

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

        self.encoder = Encoder(in_channels, hidden_channels, latent_channels, key=key1)
        self.decoder = Decoder(in_channels, hidden_channels, latent_channels, key=key2)
        self.quantizer = FSQ(levels=levels)
        print("✅ Model initialized")

    @eqx.filter_jit
    def __call__(self, x):
        
        z_e = self.encoder(x)
        reshaped_z_e = jnp.reshape(z_e, (-1, 5)) #16000 Hz -> 16Hz = 1000 points per code downsampled from 1000 to 5. Map each set of 5 to their respective code and map back. There are 4 levels thus 5 * (2 ** 4) = 80bits per codeword
        reshaped_z_q = self.quantizer(reshaped_z_e)
        z_q = jnp.reshape(reshaped_z_q, z_e.shape)
        y = self.decoder(z_q)
        return y