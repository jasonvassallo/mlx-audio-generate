"""T5 Encoder Model for MLX.

Shared text encoder used by both MusicGen and Stable Audio Open for
conditioning audio generation on text prompts. Both models use T5-base
(768 hidden dim, 12 heads, 12 layers).

Ported from sandst1/stable-audio-mlx with type hints added.
Reference: https://github.com/sandst1/stable-audio-mlx
"""

import math
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


@dataclass
class T5Config:
    d_model: int = 768
    num_heads: int = 12
    d_kv: int = 64
    d_ff: int = 3072
    num_layers: int = 12
    vocab_size: int = 32128
    dropout_rate: float = 0.0  # Disabled for inference
    layer_norm_epsilon: float = 1e-6
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128

    @classmethod
    def from_dict(cls, d: dict) -> "T5Config":
        import inspect

        valid = inspect.signature(cls).parameters
        return cls(**{k: v for k, v in d.items() if k in valid})


class T5LayerNorm(nn.Module):
    """T5-specific RMSNorm: x * weight * rsqrt(mean(x^2) + eps).

    Unlike standard LayerNorm, T5 does not subtract the mean and has no bias.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        variance = mx.mean(mx.square(x), axis=-1, keepdims=True)
        return x * mx.rsqrt(variance + self.eps) * self.weight


class T5DenseActDense(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.wo(nn.relu(self.wi(x)))


class T5Attention(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias: bool = False):
        super().__init__()
        self.is_decoder = False
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance

        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.d_kv = config.d_kv
        self.inner_dim = self.d_kv * self.num_heads

        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(
                self.relative_attention_num_buckets, self.num_heads
            )

    def compute_bias(self, query_length: int, key_length: int) -> mx.array:
        context_position = mx.arange(query_length, dtype=mx.int32)[:, None]
        memory_position = mx.arange(key_length, dtype=mx.int32)[None, :]
        relative_position = memory_position - context_position

        rp_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(rp_bucket)  # (q, k, heads)
        values = mx.transpose(values, (2, 0, 1))  # (heads, q, k)
        return values.astype(mx.float32)

    def _relative_position_bucket(
        self,
        relative_position: mx.array,
        bidirectional: bool = True,
        num_buckets: int = 32,
        max_distance: int = 128,
    ) -> mx.array:
        ret = 0
        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            ret += (n < 0).astype(mx.int32) * num_buckets
            n = mx.abs(n)
        else:
            n = mx.maximum(n, 0)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            mx.log(n.astype(mx.float32) / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).astype(mx.int32)
        val_if_large = mx.minimum(val_if_large, num_buckets - 1)

        return mx.where(is_small, n, val_if_large) + ret

    def __call__(
        self,
        hidden_states: mx.array,
        mask: Optional[mx.array] = None,
        position_bias: Optional[mx.array] = None,
    ) -> tuple[mx.array, Optional[mx.array]]:
        B, L, _ = hidden_states.shape

        q = self.q(hidden_states)
        k = self.k(hidden_states)
        v = self.v(hidden_states)

        q = q.reshape(B, L, self.num_heads, self.d_kv).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.num_heads, self.d_kv).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.num_heads, self.d_kv).transpose(0, 2, 1, 3)

        scores = q @ k.transpose(0, 1, 3, 2)

        if self.has_relative_attention_bias:
            if position_bias is None:
                position_bias = self.compute_bias(L, L)
                position_bias = mx.expand_dims(position_bias, 0)
            scores += position_bias
        elif position_bias is not None:
            scores += position_bias

        if mask is not None:
            scores = scores + mask

        attn_weights = mx.softmax(scores, axis=-1)
        out = attn_weights @ v

        out = out.transpose(0, 2, 1, 3).reshape(B, L, self.inner_dim)
        return self.o(out), position_bias


class T5Block(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias: bool = False):
        super().__init__()
        self.self_attn_norm = T5LayerNorm(config.d_model, config.layer_norm_epsilon)
        self.self_attn = T5Attention(
            config, has_relative_attention_bias=has_relative_attention_bias
        )
        self.ff_norm = T5LayerNorm(config.d_model, config.layer_norm_epsilon)
        self.ff = T5DenseActDense(config)

    def __call__(
        self,
        hidden_states: mx.array,
        mask: Optional[mx.array] = None,
        position_bias: Optional[mx.array] = None,
    ) -> tuple[mx.array, Optional[mx.array]]:
        normed = self.self_attn_norm(hidden_states)
        attn_output, position_bias = self.self_attn(normed, mask, position_bias)
        hidden_states = hidden_states + attn_output

        normed = self.ff_norm(hidden_states)
        ff_output = self.ff(normed)
        hidden_states = hidden_states + ff_output

        return hidden_states, position_bias


class T5Stack(nn.Module):
    def __init__(self, config: T5Config, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__()
        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)

        self.block = [
            T5Block(config, has_relative_attention_bias=(i == 0))
            for i in range(config.num_layers)
        ]

        self.final_layer_norm = T5LayerNorm(config.d_model, config.layer_norm_epsilon)

    def __call__(
        self, input_ids: mx.array, attention_mask: Optional[mx.array] = None
    ) -> mx.array:
        hidden_states = self.embed_tokens(input_ids)

        mask = None
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                expanded = attention_mask[:, None, None, :]
                mask = (1.0 - expanded.astype(mx.float32)) * -1e9
            else:
                mask = attention_mask

        position_bias = None
        for layer in self.block:
            hidden_states, position_bias = layer(hidden_states, mask, position_bias)

        hidden_states = self.final_layer_norm(hidden_states)

        if attention_mask is not None:
            mask_expanded = attention_mask[:, :, None].astype(mx.float32)
            hidden_states = hidden_states * mask_expanded

        return hidden_states


class T5EncoderModel(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = T5Stack(config, embed_tokens=self.shared)

    def __call__(
        self, input_ids: mx.array, attention_mask: Optional[mx.array] = None
    ) -> mx.array:
        return self.encoder(input_ids, attention_mask)
