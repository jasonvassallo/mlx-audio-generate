"""Diffusion Transformer (DiT) for Stable Audio Open.

Uses rectified flow with a prepend-style global conditioning approach
(not adaptive layer norm). Blocks are stored as explicit attributes
for MLX parameter discovery.

Ported from sandst1/stable-audio-mlx.
"""

import math
import re
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .config import DiTConfig


class FourierFeatures(nn.Module):
    """Learnable Fourier features for timestep embedding."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = mx.random.normal((out_features // 2, in_features))

    def __call__(self, x: mx.array) -> mx.array:
        if x.ndim == 1:
            x = x[:, None]
        f = 2 * math.pi * (x @ self.weight.T)
        return mx.concatenate([mx.cos(f), mx.sin(f)], axis=-1)


class RotaryEmbedding(nn.Module):
    """Rotary position embeddings."""

    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
        self.inv_freq = inv_freq

    def __call__(self, seq_len: int) -> mx.array:
        t = mx.arange(seq_len, dtype=mx.float32)
        freqs = t[:, None] * self.inv_freq[None, :]
        return mx.concatenate([freqs, freqs], axis=-1)


def rotate_half(x: mx.array) -> mx.array:
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(x: mx.array, freqs: mx.array) -> mx.array:
    """Apply partial rotary embeddings — only to the first rot_dim dims."""
    rot_dim = freqs.shape[-1]

    x_rot = x[..., :rot_dim]
    x_pass = x[..., rot_dim:]

    cos = mx.cos(freqs)[None, None, :, :]
    sin = mx.sin(freqs)[None, None, :, :]

    x_rot = (x_rot * cos) + (rotate_half(x_rot) * sin)

    if x_pass.shape[-1] > 0:
        return mx.concatenate([x_rot, x_pass], axis=-1)
    return x_rot


class GLU(nn.Module):
    """Gated Linear Unit with SiLU activation."""

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.proj(x)
        x, gate = mx.split(x, 2, axis=-1)
        return x * nn.silu(gate)


class FeedForward(nn.Module):
    """FFN with GLU. Uses nn.Identity placeholder at index 1 to align
    weight keys with PyTorch: ff.ff.layers.0.proj and ff.ff.layers.2."""

    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        inner_dim = int(dim * mult)
        self.ff = nn.Sequential(
            GLU(dim, inner_dim),
            nn.Identity(),
            nn.Linear(inner_dim, dim),
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.ff(x)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

        self.q_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)

    def __call__(
        self, x: mx.array, rotary_freqs: Optional[mx.array] = None
    ) -> mx.array:
        B, L, D = x.shape
        qkv = self.to_qkv(x)
        q, k, v = mx.split(qkv, 3, axis=-1)

        q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if rotary_freqs is not None:
            q = apply_rotary_pos_emb(q, rotary_freqs)
            k = apply_rotary_pos_emb(k, rotary_freqs)

        dots = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn = mx.softmax(dots, axis=-1)
        out = attn @ v

        out = out.transpose(0, 2, 1, 3).reshape(B, L, D)
        return self.to_out(out)


class CrossAttention(nn.Module):
    """Cross-attention with grouped query attention (GQA) support."""

    def __init__(self, dim: int, cond_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.kv_heads = cond_dim // self.head_dim
        self.scale = self.head_dim**-0.5

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(cond_dim, cond_dim * 2, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

        self.q_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)

    def __call__(self, x: mx.array, cond: mx.array) -> mx.array:
        B, L, D = x.shape
        _, S, _ = cond.shape

        q = self.to_q(x)
        kv = self.to_kv(cond)
        k, v = mx.split(kv, 2, axis=-1)

        q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, S, self.kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, S, self.kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.num_heads != self.kv_heads:
            repeats = self.num_heads // self.kv_heads
            k = mx.repeat(k, repeats, axis=1)
            v = mx.repeat(v, repeats, axis=1)

        dots = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn = mx.softmax(dots, axis=-1)
        out = attn @ v

        out = out.transpose(0, 2, 1, 3).reshape(B, L, D)
        return self.to_out(out)


class DiTBlock(nn.Module):
    """Standard pre-norm transformer block (no adaLN)."""

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.pre_norm = nn.LayerNorm(config.embed_dim, eps=1e-6)
        self.self_attn = SelfAttention(config.embed_dim, config.num_heads)

        self.cross_attend = config.cond_token_dim > 0
        if self.cross_attend:
            self.cross_attend_norm = nn.LayerNorm(config.embed_dim, eps=1e-6)
            cond_dim = (
                config.embed_dim if config.project_cond_tokens else config.cond_token_dim
            )
            self.cross_attn = CrossAttention(config.embed_dim, cond_dim, config.num_heads)

        self.ff_norm = nn.LayerNorm(config.embed_dim, eps=1e-6)
        self.ff = FeedForward(config.embed_dim)

    def __call__(
        self,
        x: mx.array,
        cond: Optional[mx.array] = None,
        rotary_freqs: Optional[mx.array] = None,
    ) -> mx.array:
        x = x + self.self_attn(self.pre_norm(x), rotary_freqs)

        if self.cross_attend and cond is not None:
            x = x + self.cross_attn(self.cross_attend_norm(x), cond)

        x = x + self.ff(self.ff_norm(x))
        return x


class TransformerCore(nn.Module):
    """Input/output projections and rotary embeddings."""

    def __init__(self, dim_in: int, dim: int, dim_out: int, num_heads: int):
        super().__init__()
        self.project_in = nn.Linear(dim_in, dim, bias=False)
        self.project_out = nn.Linear(dim, dim_out, bias=False)
        head_dim = dim // num_heads
        rotary_dim = max(head_dim // 2, 32)
        self.rotary_pos_emb = RotaryEmbedding(rotary_dim)


class StableAudioDiT(nn.Module):
    """Stable Audio Diffusion Transformer."""

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.config = config

        self.timestep_features = FourierFeatures(1, config.timestep_features_dim)
        self.to_timestep_embed = nn.Sequential(
            nn.Linear(config.timestep_features_dim, config.embed_dim),
            nn.SiLU(),
            nn.Linear(config.embed_dim, config.embed_dim),
        )

        if config.global_cond_dim > 0:
            self.to_global_embed = nn.Sequential(
                nn.Linear(config.global_cond_dim, config.embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(config.embed_dim, config.embed_dim, bias=False),
            )

        if config.cond_token_dim > 0:
            cond_embed_dim = (
                config.embed_dim if config.project_cond_tokens else config.cond_token_dim
            )
            self.to_cond_embed = nn.Sequential(
                nn.Linear(config.cond_token_dim, cond_embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(cond_embed_dim, cond_embed_dim, bias=False),
            )

        self.preprocess_conv = nn.Conv1d(
            config.io_channels, config.io_channels, kernel_size=1, bias=False
        )
        self.postprocess_conv = nn.Conv1d(
            config.io_channels, config.io_channels, kernel_size=1, bias=False
        )

        dim_in = config.io_channels * config.patch_size
        dim_out = config.io_channels * config.patch_size
        self.transformer = TransformerCore(
            dim_in, config.embed_dim, dim_out, config.num_heads
        )

        # Create blocks as explicit attributes for MLX parameter discovery.
        # Dynamic count based on config.depth.
        self.num_blocks = config.depth
        for i in range(config.depth):
            setattr(self, f"block_{i}", DiTBlock(config))

    def load_weights(self, weights):
        """Load weights with key remapping."""
        if isinstance(weights, dict):
            weights = list(weights.items())

        new_weights = {}
        for k, v in weights:
            nk = k

            if k.startswith("blocks."):
                nk = re.sub(r"blocks\.(\d+)\.", r"block_\1.", k)
                if nk.endswith(".gamma"):
                    nk = nk[:-6] + ".weight"
                elif nk.endswith(".beta"):
                    nk = nk[:-5] + ".bias"
                elif ".ff.ff." in nk:
                    nk = nk.replace(".ff.ff.", ".ff.ff.layers.")

            elif k.startswith("to_timestep_embed."):
                parts = k.split(".")
                if len(parts) >= 2 and parts[1].isdigit():
                    nk = f"to_timestep_embed.layers.{'.'.join(parts[1:])}"

            elif k.startswith("to_global_embed."):
                parts = k.split(".")
                if len(parts) >= 2 and parts[1].isdigit():
                    nk = f"to_global_embed.layers.{'.'.join(parts[1:])}"

            elif k.startswith("to_cond_embed."):
                parts = k.split(".")
                if len(parts) >= 2 and parts[1].isdigit():
                    nk = f"to_cond_embed.layers.{'.'.join(parts[1:])}"

            new_weights[nk] = v

        super().load_weights(list(new_weights.items()))

    def __call__(
        self,
        x: mx.array,
        t: mx.array,
        cross_attn_cond: Optional[mx.array] = None,
        global_embed: Optional[mx.array] = None,
        **kwargs,
    ) -> mx.array:
        """Forward pass.

        Args:
            x: Input latents (B, C, T)
            t: Timesteps (B,)
            cross_attn_cond: Cross-attention conditioning (B, S, cond_dim)
            global_embed: Global conditioning (B, global_cond_dim)
        """
        B, C, T = x.shape

        # Preprocess conv (residual), MLX Conv1d expects (B, T, C)
        x_t = x.transpose(0, 2, 1)
        x_t = x_t + self.preprocess_conv(x_t)

        # Timestep embedding
        t_embed = self.to_timestep_embed(self.timestep_features(t))

        # Global conditioning
        if global_embed is not None and hasattr(self, "to_global_embed"):
            global_embed = self.to_global_embed(global_embed)
            global_cond = global_embed + t_embed
        else:
            global_cond = t_embed

        # Cross-attention conditioning projection
        if cross_attn_cond is not None and hasattr(self, "to_cond_embed"):
            cross_attn_cond = self.to_cond_embed(cross_attn_cond)

        # Project input
        x_t = self.transformer.project_in(x_t)

        # Prepend global conditioning as a token
        prepend_embed = global_cond[:, None, :]
        x_t = mx.concatenate([prepend_embed, x_t], axis=1)

        # Rotary embeddings for full sequence (prepend + original)
        rotary_freqs = self.transformer.rotary_pos_emb(T + 1)

        # Transformer blocks
        for i in range(self.num_blocks):
            block = getattr(self, f"block_{i}")
            x_t = block(x_t, cond=cross_attn_cond, rotary_freqs=rotary_freqs)

        # Remove prepended token
        x_t = x_t[:, 1:, :]

        # Project output
        x_t = self.transformer.project_out(x_t)

        # Postprocess conv (residual)
        x_t = x_t + self.postprocess_conv(x_t)

        # Transpose back to (B, C, T)
        return x_t.transpose(0, 2, 1)
