"""Style conditioner for MusicGen-Style.

Processes audio through MERT → projection → transformer → BatchNorm →
RVQ → downsampling to produce style conditioning tokens for cross-attention.

Pipeline:
    1. MERT extracts audio features at 75Hz → (B, T, 768)
    2. Linear projection: 768 → style_dim (512 for default scale)
    3. 8-layer pre-norm transformer encoder
    4. BatchNorm1d (affine=False, inference-mode with running stats)
    5. RVQ quantization (3 codebooks at eval, 1024 bins each)
    6. Downsampling by factor 15

Reference: facebookresearch/audiocraft — StyleConditioner in conditioners.py
"""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .mert import MERTModel


@dataclass
class StyleConfig:
    """Configuration for the style conditioner."""

    dim: int = 512
    num_heads: int = 8
    num_layers: int = 8
    ffn_dim: int = 2048
    ds_factor: int = 15
    n_q: int = 3  # Number of RVQ codebooks at eval
    bins: int = 1024  # Codebook size
    excerpt_length: float = 3.0  # Seconds of audio to extract
    mert_sample_rate: int = 24000  # MERT input sample rate
    mert_hidden_size: int = 768


class StyleTransformerBlock(nn.Module):
    """Pre-norm self-attention transformer block for style encoding.

    The audiocraft style transformer uses pre-norm (norm_first=True),
    GELU activation, and no bias on attention/FF projections.

    Attribute names align with audiocraft key structure after prefix
    stripping: norm1, self_attn, norm2, linear1, linear2.
    """

    def __init__(self, dim: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.norm1 = nn.LayerNorm(dim)
        # Audiocraft uses fused in_proj_weight; we use separate Q/K/V
        # (conversion script splits the fused weight)
        self.self_attn_q_proj = nn.Linear(dim, dim, bias=False)
        self.self_attn_k_proj = nn.Linear(dim, dim, bias=False)
        self.self_attn_v_proj = nn.Linear(dim, dim, bias=False)
        self.self_attn_out_proj = nn.Linear(dim, dim, bias=False)

        self.norm2 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, ffn_dim, bias=False)
        self.linear2 = nn.Linear(ffn_dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        B, T, _ = x.shape
        head_dim = x.shape[-1] // self.num_heads

        # Self-attention with pre-norm
        xn = self.norm1(x)
        q = (
            self.self_attn_q_proj(xn)
            .reshape(B, T, self.num_heads, head_dim)
            .transpose(0, 2, 1, 3)
        )
        k = (
            self.self_attn_k_proj(xn)
            .reshape(B, T, self.num_heads, head_dim)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.self_attn_v_proj(xn)
            .reshape(B, T, self.num_heads, head_dim)
            .transpose(0, 2, 1, 3)
        )
        attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        attn = attn.transpose(0, 2, 1, 3).reshape(B, T, -1)
        x = x + self.self_attn_out_proj(attn)

        # FFN with pre-norm + GELU
        xn = self.norm2(x)
        x = x + self.linear2(nn.gelu(self.linear1(xn)))

        return x


class VectorQuantizer(nn.Module):
    """Single codebook vector quantizer (nearest-neighbor lookup).

    At inference time: finds the nearest codebook vector for each input,
    returns the codebook vector (straight-through for backprop not needed).
    """

    def __init__(self, dim: int, bins: int = 1024):
        super().__init__()
        self.codebook = nn.Embedding(bins, dim)
        self.bins = bins

    def __call__(self, x: mx.array) -> mx.array:
        """Quantize input vectors via nearest-neighbor codebook lookup.

        Args:
            x: Input vectors, shape (B, T, dim).

        Returns:
            Quantized vectors, shape (B, T, dim).
        """
        B, T, D = x.shape
        flat = x.reshape(-1, D)  # (B*T, D)
        cb = self.codebook.weight  # (bins, D)

        # L2 distance: ||x-c||² = ||x||² + ||c||² - 2·x·cᵀ
        dist = (flat**2).sum(-1, keepdims=True) + (cb**2).sum(-1) - 2 * flat @ cb.T
        indices = mx.argmin(dist, axis=-1)
        quantized = self.codebook(indices)
        return quantized.reshape(B, T, D)


class ResidualVectorQuantizer(nn.Module):
    """Residual Vector Quantizer (RVQ) with multiple codebooks.

    Each codebook quantizes the residual from the previous stage:
        q₀ = VQ(x)
        q₁ = VQ(x - q₀)
        q₂ = VQ(x - q₀ - q₁)
        output = q₀ + q₁ + q₂

    Uses ``setattr`` for dynamic codebook count so MLX parameter discovery
    finds all codebook weights.
    """

    def __init__(self, dim: int, n_q: int = 3, bins: int = 1024):
        super().__init__()
        self.n_q = n_q
        for i in range(n_q):
            setattr(self, f"layers_{i}", VectorQuantizer(dim, bins))

    def __call__(self, x: mx.array) -> mx.array:
        """Apply residual quantization.

        Args:
            x: Input, shape (B, T, dim).

        Returns:
            Sum of quantized residuals, shape (B, T, dim).
        """
        quantized = mx.zeros_like(x)
        residual = x
        for i in range(self.n_q):
            layer: VectorQuantizer = getattr(self, f"layers_{i}")
            q = layer(residual)
            quantized = quantized + q
            residual = residual - q
        return quantized


class StyleConditioner(nn.Module):
    """Full style conditioning pipeline for MusicGen-Style.

    Extracts style tokens from a reference audio excerpt for use as
    cross-attention conditioning in the MusicGen decoder alongside
    (or instead of) text tokens.

    Components loaded from converted weights:
        - MERT: separate mert.safetensors (frozen, ~95M params)
        - embed, transformer, batch_norm, rvq: style.safetensors
    """

    def __init__(self, config: StyleConfig):
        super().__init__()
        self.config = config

        # MERT feature extractor (loaded separately)
        self.mert = MERTModel()

        # Linear projection from MERT dim to style dim
        self.embed = nn.Linear(config.mert_hidden_size, config.dim)

        # Style transformer encoder
        self.transformer_layers = [
            StyleTransformerBlock(config.dim, config.num_heads, config.ffn_dim)
            for _ in range(config.num_layers)
        ]

        # Batch normalization running statistics (affine=False)
        # Stored as plain arrays, applied as: (x - mean) / sqrt(var + eps)
        self.batch_norm_running_mean = mx.zeros(config.dim)
        self.batch_norm_running_var = mx.ones(config.dim)

        # Residual vector quantizer
        self.rvq = ResidualVectorQuantizer(config.dim, n_q=config.n_q, bins=config.bins)

        self.ds_factor = config.ds_factor

    def __call__(self, audio: mx.array, sample_rate: int = 32000) -> mx.array:
        """Extract style conditioning tokens from audio.

        Args:
            audio: Audio waveform, shape (B, T) or (1, T).
            sample_rate: Sample rate of input audio.

        Returns:
            Style tokens, shape (B, T', style_dim) for cross-attention.
        """
        # Resample to MERT's 24kHz if needed
        mert_sr = self.config.mert_sample_rate
        if sample_rate != mert_sr:
            ratio = mert_sr / sample_rate
            new_len = int(audio.shape[-1] * ratio)
            old_indices = mx.arange(audio.shape[-1]).astype(mx.float32)
            new_indices = mx.linspace(0, audio.shape[-1] - 1, new_len)
            if audio.ndim == 1:
                audio = mx.interp(new_indices, old_indices, audio)
            else:
                # Per-batch resampling
                resampled = []
                for i in range(audio.shape[0]):
                    resampled.append(mx.interp(new_indices, old_indices, audio[i]))
                audio = mx.stack(resampled)

        # Extract center excerpt
        excerpt_samples = int(self.config.excerpt_length * mert_sr)
        if audio.shape[-1] > excerpt_samples:
            start = (audio.shape[-1] - excerpt_samples) // 2
            audio = audio[..., start : start + excerpt_samples]

        # Ensure batch dimension
        if audio.ndim == 1:
            audio = audio[mx.newaxis]

        # Run MERT → (B, T, 768)
        features = self.mert(audio)

        # Project to style dim → (B, T, 512)
        features = self.embed(features)

        # Style transformer
        for layer in self.transformer_layers:
            features = layer(features)

        # Batch normalization (inference mode: use running stats)
        features = (features - self.batch_norm_running_mean) / mx.sqrt(
            self.batch_norm_running_var + 1e-5
        )

        # RVQ quantization → discrete bottleneck
        features = self.rvq(features)

        # Downsample by fixed factor
        features = features[:, :: self.ds_factor]

        return features
