import os
import jax
import librosa
import numpy
import tiktoken
import jax.numpy as jnp
import equinox as eqx
from layers.fsqae import VQVAE


class Tokenizer:

    encoding: tiktoken.Encoding
    vqvae: VQVAE
    sample_rate: int
    eoa = 1024
    max_length: int

    def __init__(
        self, checkpoint_dir: str, sample_rate: int = 22050, max_length: int = 200
    ):
        base_encoding = tiktoken.get_encoding("r50k_base")

        self.max_length = max_length

        my_special_tokens = base_encoding._special_tokens
        my_special_tokens.update({"<|endofaudio|>": 50256, "<|pad|>": 50257})

        self.encoding = tiktoken.Encoding(
            name="myEncoder",
            pat_str=base_encoding._pat_str,
            mergeable_ranks=base_encoding._mergeable_ranks,
            special_tokens=base_encoding._special_tokens,
        )

        vqvae = VQVAE(1, 32, 4, [6], key=jax.random.key(1))
        self.vqvae = eqx.tree_deserialise_leaves(
            os.path.join(checkpoint_dir, "fsq.eqx"), vqvae
        )

        self.sample_rate = sample_rate

    def pad(self, input, actual_length):
        padding_to_add = 200 - actual_length
        result = numpy.pad(
            input, pad_width=((0, padding_to_add)), constant_values=self.eoa
        )
        return result

    def encode(self, text, audio_array, sample_rate):

        tokenized_text = self.encoding.encode(text, allowed_special={"<|endoftext|>"})
        tokenized_text.append(self.encoding.encode_single_token("<|endoftext|>"))
        tokenized_audio = self.tokenize_audio(audio_array, sample_rate)

        tokenized_audio = jax.numpy.concatenate(
            [tokenized_audio, jax.numpy.array([self.eoa])], axis=-1
        )
        return jax.numpy.array(numpy.array(tokenized_text)), tokenized_audio

    def tokenize_audio(self, audio_array, sample_rate):
        if sample_rate != self.sample_rate:
            audio_array = librosa.resample(
                audio_array, orig_sr=sample_rate, target_sr=self.sample_rate
            )

        freq=16
        stride = int(16000 / freq)
        
        input = []
        for i in range(0, (int(len(audio_array)//stride) -1)):
            input.append(audio_array[i*stride:i*stride+stride])

        input = jnp.array(input)
        input = jnp.expand_dims(input, 1)
        print(f"Shape of the input : {input.shape}")
        
        y = jax.vmap(self.vqvae.encoder)(input)
        print(f"Shape of the output of encoder : {y.shape}")
        
        y = jax.vmap(self.vqvae.quantizer)(y)
        print(f"Shape of the output of quantizer : {y.shape}")
        y = jax.vmap(self.vqvae.quantizer.codes_to_indexes)(jnp.expand_dims(y, -1))
        print(y)

        

        return y

    def decode(self, tokens):
        return self.encoding.decode(tokens, errors="[ERR_CAN'T FIND TOKEN]")
