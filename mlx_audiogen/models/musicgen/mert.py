"""MERT feature extractor for MusicGen style conditioning.

MERT (Music undERstanding Transformer) is a HuBERT-like model trained on music.
Used as a frozen feature extractor in MusicGen-Style to extract audio
representations that the style conditioner processes.

Architecture:
    - 7 CNN layers (conv + optional GroupNorm + GELU) — downsamples 320x
    - Feature projection: Linear(512, 768) + LayerNorm
    - Convolutional positional encoding (groups=16, kernel=128)
    - 12 post-norm transformer layers (768 dim, 12 heads, 3072 ffn)

Input: raw audio at 24kHz → Output: features at 75Hz, shape (B, T', 768).

Reference: m-a-p/MERT-v1-95M
"""

from dataclasses import dataclass, field
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


@dataclass
class MERTConfig:
    """Configuration for the MERT feature extractor."""

    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    conv_dim: list[int] = field(
        default_factory=lambda: [512, 512, 512, 512, 512, 512, 512]
    )
    conv_kernel: list[int] = field(default_factory=lambda: [10, 3, 3, 3, 3, 2, 2])
    conv_stride: list[int] = field(default_factory=lambda: [5, 2, 2, 2, 2, 2, 2])
    conv_bias: bool = False
    num_conv_pos_embeddings: int = 128
    num_conv_pos_embedding_groups: int = 16
    sample_rate: int = 24000
    layer_norm_eps: float = 1e-5

    @classmethod
    def from_dict(cls, d: dict) -> "MERTConfig":
        """Create config from a dictionary (e.g. HF config.json)."""
        import inspect

        valid = inspect.signature(cls).parameters
        return cls(**{k: v for k, v in d.items() if k in valid})


class MERTConvLayer(nn.Module):
    """Single convolutional feature extraction layer.

    The first layer uses GroupNorm; subsequent layers have no normalization.
    All layers use GELU activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        bias: bool = False,
        has_group_norm: bool = False,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, bias=bias
        )
        self.has_group_norm = has_group_norm
        if has_group_norm:
            self.layer_norm = nn.GroupNorm(out_channels, out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv(x)
        if self.has_group_norm:
            x = self.layer_norm(x)
        x = nn.gelu(x)
        return x


class MERTFeatureExtractor(nn.Module):
    """7-layer CNN feature extractor from raw audio to feature vectors.

    Total downsampling: 5 * 2^6 = 320 → 24000/320 = 75 Hz feature rate.
    """

    def __init__(self, config: MERTConfig):
        super().__init__()
        self.conv_layers = []
        for i in range(len(config.conv_dim)):
            in_c = 1 if i == 0 else config.conv_dim[i - 1]
            out_c = config.conv_dim[i]
            layer = MERTConvLayer(
                in_c,
                out_c,
                config.conv_kernel[i],
                config.conv_stride[i],
                bias=config.conv_bias,
                has_group_norm=(i == 0),
            )
            self.conv_layers.append(layer)

    def __call__(self, x: mx.array) -> mx.array:
        """Process raw audio. Input: (B, T, 1), Output: (B, T', 512)."""
        for layer in self.conv_layers:
            x = layer(x)
        return x


class MERTFeatureProjection(nn.Module):
    """Projects extracted features (512 dim) to hidden dimension (768)."""

    def __init__(self, config: MERTConfig):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.conv_dim[-1], eps=config.layer_norm_eps)
        self.projection = nn.Linear(config.conv_dim[-1], config.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.layer_norm(x)
        x = self.projection(x)
        return x


class MERTPositionalConvEmbedding(nn.Module):
    """Convolutional positional encoding (groups=16, kernel=128)."""

    def __init__(self, config: MERTConfig):
        super().__init__()
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.num_conv_pos_embeddings,
            padding=config.num_conv_pos_embeddings // 2,
            groups=config.num_conv_pos_embedding_groups,
        )

    def __call__(self, x: mx.array) -> mx.array:
        pos = self.conv(x)
        # Trim to match input length (conv may produce one extra frame)
        pos = pos[:, : x.shape[1], :]
        return nn.gelu(pos)


class MERTAttention(nn.Module):
    """Multi-head self-attention with bias (HuBERT-style).

    Unlike MusicGen's attention, MERT/HuBERT attention has bias on all
    Q/K/V/out projections.
    """

    def __init__(self, config: MERTConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim**-0.5
        # HuBERT attention uses bias=True (unlike MusicGen decoder)
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        B, T, _ = x.shape
        q = (
            self.q_proj(x)
            .reshape(B, T, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        k = (
            self.k_proj(x)
            .reshape(B, T, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.v_proj(x)
            .reshape(B, T, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.out_proj(out)


class MERTFeedForward(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, config: MERTConfig):
        super().__init__()
        self.intermediate_dense = nn.Linear(
            config.hidden_size, config.intermediate_size
        )
        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.gelu(self.intermediate_dense(x))
        return self.output_dense(x)


class MERTEncoderLayer(nn.Module):
    """Post-norm transformer encoder layer (HuBERT-style).

    Layout: x = norm(x + attn(x)), x = norm(x + ffn(x))
    This differs from MusicGen's decoder which uses pre-norm.
    """

    def __init__(self, config: MERTConfig):
        super().__init__()
        self.attention = MERTAttention(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = MERTFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.layer_norm(x + self.attention(x))
        x = self.final_layer_norm(x + self.feed_forward(x))
        return x


class MERTEncoder(nn.Module):
    """MERT transformer encoder with positional conv embedding."""

    def __init__(self, config: MERTConfig):
        super().__init__()
        self.pos_conv_embed = MERTPositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layers = [
            MERTEncoderLayer(config) for _ in range(config.num_hidden_layers)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        pos = self.pos_conv_embed(x)
        x = x + pos
        x = self.layer_norm(x)
        for layer in self.layers:
            x = layer(x)
        return x


class MERTModel(nn.Module):
    """MERT feature extraction model (inference-only).

    Processes raw audio at 24kHz into feature representations at 75Hz.
    Output shape: (batch, time_steps, 768).
    """

    def __init__(self, config: Optional[MERTConfig] = None):
        super().__init__()
        if config is None:
            config = MERTConfig()
        self.config = config
        self.feature_extractor = MERTFeatureExtractor(config)
        self.feature_projection = MERTFeatureProjection(config)
        self.encoder = MERTEncoder(config)

    def __call__(self, audio: mx.array) -> mx.array:
        """Extract features from audio.

        Args:
            audio: Raw audio waveform at 24kHz.
                Shape (B, T) or (B, T, 1).

        Returns:
            Feature representations, shape (B, T', 768) at ~75Hz.
        """
        if audio.ndim == 2:
            audio = audio[..., mx.newaxis]
        features = self.feature_extractor(audio)
        features = self.feature_projection(features)
        features = self.encoder(features)
        return features
