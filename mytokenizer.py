import os
import jax
import librosa
import numpy
import tiktoken
import torch
import torchaudio
import equinox as eqx
from layers.VQVAE import VQVAE


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

        vqvae = VQVAE(jax.random.key(1))
        self.vqvae = eqx.tree_deserialise_leaves(
            os.path.join(checkpoint_dir, "xttsvqvae.eqx"), vqvae
        )

        self.sample_rate = 22050

    def dvae_wav_to_mel(
        self,
        wav,
        mel_norms_file="./mel_stats.pth",
        mel_norms=None,
        device=torch.device("cpu"),
    ):
        mel_stft = torchaudio.transforms.MelSpectrogram(
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            power=2,
            normalized=False,
            sample_rate=22050,
            f_min=0,
            f_max=8000,
            n_mels=80,
            norm="slaney",
        ).to(device)
        wav = wav.to(device)
        mel = mel_stft(wav)
        mel = torch.log(torch.clamp(mel, min=1e-5))
        if mel_norms is None:
            mel_norms = torch.load(
                mel_norms_file, weights_only=True, map_location=device
            )
        mel = mel / mel_norms.unsqueeze(0).unsqueeze(-1)
        return mel

    def mel_to_dvae_wav(
        self,
        mel,
        mel_norms_file="./mel_stats.pth",
        mel_norms=None,
        device=torch.device("cpu"),
    ):
        if mel_norms is None:
            mel_norms = torch.load(
                mel_norms_file, weights_only=True, map_location=device
            )
        mel = mel * mel_norms.unsqueeze(0).unsqueeze(-1)
        mel = torch.exp(mel)

        mel_stft = torchaudio.transforms.MelSpectrogram(
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            power=2,
            normalized=False,
            sample_rate=22050,
            f_min=0,
            f_max=8000,
            n_mels=80,
            norm="slaney",
        ).to(device)

        # Create the inverse mel filter bank
        inv_mel_basis = torch.pinverse(mel_stft.mel_scale.fb).to(device)

        # Convert mel spectrogram to linear spectrogram
        spec = torch.matmul(inv_mel_basis, mel)

        # Use Griffin-Lim algorithm to convert spectrogram to waveform
        griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            power=2,
            n_iter=32,
        ).to(device)

        wav = griffin_lim(spec)
        return wav

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

        mels = self.dvae_wav_to_mel(
            torch.from_numpy(numpy.array(audio_array, dtype=numpy.float32))
        )

        mels = jax.numpy.array(numpy.array(mels))

        y = jax.vmap(self.vqvae.encoder)(mels)
        _, (_, y) = jax.vmap(self.vqvae.quantizer)(y)

        return y[0]

    def decode(self, tokens):
        return self.encoding.decode(tokens, errors="[ERR_CAN'T FIND TOKEN]")
