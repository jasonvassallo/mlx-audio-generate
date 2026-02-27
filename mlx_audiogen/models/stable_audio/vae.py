"""Oobleck VAE (autoencoder) for Stable Audio Open.

Encodes audio waveforms to continuous latent representations and decodes them back.
Uses Snake activation (learned sinusoidal modulation) and strided convolutions.

Ported from sandst1/stable-audio-mlx.
"""

import mlx.core as mx
import mlx.nn as nn

from .config import OobleckConfig


class Snake1d(nn.Module):
    """Snake activation with log-scale alpha/beta parameters.

    Parameters are stored in log-space and exponentiated at runtime.
    Forward: x + (1 / (exp(beta) + eps)) * sin(exp(alpha) * x)^2
    """

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = mx.zeros((channels,))
        self.beta = mx.zeros((channels,))

    def __call__(self, x: mx.array) -> mx.array:
        alpha = mx.exp(self.alpha)
        beta = mx.exp(self.beta)
        return x + (1.0 / (beta + 1e-9)) * mx.square(mx.sin(alpha * x))


class ResidualUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__()
        self.layers = nn.Sequential(
            Snake1d(in_channels),
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=7,
                dilation=dilation,
                padding=((7 - 1) * dilation) // 2,
            ),
            Snake1d(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
        )

    def __call__(self, x: mx.array) -> mx.array:
        return x + self.layers(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.layers = nn.Sequential(
            ResidualUnit(in_channels, in_channels, dilation=1),
            ResidualUnit(in_channels, in_channels, dilation=3),
            ResidualUnit(in_channels, in_channels, dilation=9),
            Snake1d(in_channels),
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=2 * stride,
                stride=stride,
                padding=(2 * stride - stride) // 2,
            ),
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.layers(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.layers = nn.Sequential(
            Snake1d(in_channels),
            nn.ConvTranspose1d(
                in_channels,
                out_channels,
                kernel_size=2 * stride,
                stride=stride,
                padding=(2 * stride - stride) // 2,
            ),
            ResidualUnit(out_channels, out_channels, dilation=1),
            ResidualUnit(out_channels, out_channels, dilation=3),
            ResidualUnit(out_channels, out_channels, dilation=9),
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.layers(x)


class OobleckEncoder(nn.Module):
    def __init__(self, config: OobleckConfig):
        super().__init__()
        layers: list[nn.Module] = []
        layers.append(
            nn.Conv1d(
                config.in_channels,
                config.channels,
                kernel_size=7,
                padding=3,
            )
        )

        c_in = config.channels
        for i, stride in enumerate(config.strides):
            c_out = config.channels * config.c_mults[i]
            layers.append(EncoderBlock(c_in, c_out, stride))
            c_in = c_out

        layers.append(Snake1d(c_in))
        layers.append(nn.Conv1d(c_in, config.latent_dim * 2, kernel_size=3, padding=1))

        self.layers = nn.Sequential(*layers)

    def __call__(self, x: mx.array) -> mx.array:
        return self.layers(x)


class OobleckDecoder(nn.Module):
    def __init__(self, config: OobleckConfig):
        super().__init__()
        layers: list[nn.Module] = []
        c_hidden = config.channels * config.c_mults[-1]

        layers.append(nn.Conv1d(config.latent_dim, c_hidden, kernel_size=7, padding=3))

        strides = list(reversed(config.strides))
        c_mults = list(reversed(config.c_mults))
        c_in = c_hidden

        for i, stride in enumerate(strides):
            if i < len(c_mults) - 1:
                c_out = config.channels * c_mults[i + 1]
            else:
                c_out = config.channels
            layers.append(DecoderBlock(c_in, c_out, stride))
            c_in = c_out

        layers.append(Snake1d(c_in))
        layers.append(
            nn.Conv1d(c_in, config.in_channels, kernel_size=7, padding=3, bias=False)
        )
        if config.final_tanh:
            layers.append(nn.Tanh())

        self.layers = nn.Sequential(*layers)

    def __call__(self, x: mx.array) -> mx.array:
        return self.layers(x)


class AutoencoderOobleck(nn.Module):
    def __init__(self, config: OobleckConfig = OobleckConfig()):
        super().__init__()
        self.encoder = OobleckEncoder(config)
        self.decoder = OobleckDecoder(config)

    def encode(self, x: mx.array) -> mx.array:
        return self.encoder(x)

    def decode(self, x: mx.array) -> mx.array:
        return self.decoder(x)

    def __call__(self, x: mx.array) -> mx.array:
        return self.decode(self.encode(x))
