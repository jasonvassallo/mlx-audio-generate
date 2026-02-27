"""Conditioning modules for Stable Audio Open.

NumberEmbedder: Learnable Fourier embedding for scalar values (e.g. duration).
Conditioners: Wires together T5 text encoding + time embedding to produce
cross-attention tokens and a global conditioning vector for the DiT.

Ported from sandst1/stable-audio-mlx.
"""

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class NumberEmbedder(nn.Module):
    """Fourier-feature embedding for a scalar value (seconds_total).

    Uses raw weight arrays instead of nn.Linear because the original model
    stores the Fourier frequencies as a plain 1-D tensor, not a standard layer.
    Structure (from the HF checkpoint):
        embedding.0.weights  -> (features,)   Fourier frequencies
        embedding.1.weight   -> (hidden_dim, 2*features+1)  Linear weight
        embedding.1.bias     -> (hidden_dim,) Linear bias
    """

    def __init__(self, features: int, hidden_dim: int):
        super().__init__()
        self.features = features
        self.hidden_dim = hidden_dim
        # Placeholders — populated by load_weights()
        self.freqs = mx.zeros((features,))
        self.linear_w = mx.zeros((hidden_dim, 2 * features + 1))
        self.linear_b = mx.zeros((hidden_dim,))

    def load_weights(self, weights: dict[str, mx.array | np.ndarray]):  # type: ignore[override]
        """Load from extracted conditioner state dict.

        Expected keys: 'embedding.0.weights', 'embedding.1.weight', 'embedding.1.bias'
        """

        def to_mx(v):
            return mx.array(v) if not isinstance(v, mx.array) else v

        self.freqs = to_mx(weights["embedding.0.weights"])
        self.linear_w = to_mx(weights["embedding.1.weight"])
        self.linear_b = to_mx(weights["embedding.1.bias"])

    def __call__(self, x: mx.array) -> mx.array:
        if x.ndim == 0:
            x = x[None]
        if x.ndim == 1:
            x = x[:, None]  # (B, 1)

        proj = 2 * math.pi * x * self.freqs[None, :]  # (B, features)
        fourier = mx.concatenate(
            [mx.cos(proj), mx.sin(proj)], axis=-1
        )  # (B, 2*features)
        h = mx.concatenate([fourier, x], axis=-1)  # (B, 2*features + 1)

        return h @ self.linear_w.T + self.linear_b  # (B, hidden_dim)


class Conditioners(nn.Module):
    """Combines T5 text encoder and time embedders for DiT conditioning.

    Supports both stable-audio-open-small (seconds_total only) and
    stable-audio-open-1.0 (seconds_start + seconds_total).

    Returns:
        cross_attn:  (1, 65+, 768) — T5 tokens + time token(s)
        global_cond: (1, 768)      — time embedding for global conditioning
    """

    def __init__(
        self,
        t5_model: nn.Module,
        tokenizer,
        has_seconds_start: bool = False,
    ):
        super().__init__()
        self.t5 = t5_model
        self.tokenizer = tokenizer
        self.has_seconds_start = has_seconds_start
        self.seconds_total = NumberEmbedder(128, 768)
        if has_seconds_start:
            self.seconds_start = NumberEmbedder(128, 768)

    def load_weights(self, cond_state: dict):  # type: ignore[override]
        """Load conditioner weights from the converted state dict.

        Keys look like:
            conditioner.conditioners.seconds_total.embedder.embedding.0.weights
            conditioner.conditioners.seconds_total.embedder.embedding.1.weight
            conditioner.conditioners.seconds_total.embedder.embedding.1.bias
            conditioner.conditioners.seconds_start.embedder.embedding.0.weights
            (etc., only for 1.0 variant)
        """
        total_weights = {}
        start_weights = {}
        for k, v in cond_state.items():
            if "seconds_total" in k and "embedder." in k:
                short_key = k.split("embedder.")[1]
                total_weights[short_key] = v
            elif "seconds_start" in k and "embedder." in k:
                short_key = k.split("embedder.")[1]
                start_weights[short_key] = v

        if total_weights:
            self.seconds_total.load_weights(total_weights)
        if start_weights and self.has_seconds_start:
            self.seconds_start.load_weights(start_weights)

    def __call__(
        self,
        prompt: str,
        seconds_total: float | mx.array,
        seconds_start: float | mx.array = 0.0,
    ) -> tuple[mx.array, mx.array]:
        """Encode text + duration into conditioning tensors.

        Args:
            prompt: Text description of desired audio.
            seconds_total: Total duration in seconds.
            seconds_start: Start time offset in seconds (1.0 variant only).

        Returns:
            cross_attn:  (1, 65+, 768)
            global_cond: (1, 768)
        """
        # Text encoding via T5
        tokens = self.tokenizer(
            prompt,
            return_tensors="np",
            padding="max_length",
            max_length=128,
            truncation=True,
        )
        input_ids = mx.array(tokens["input_ids"])
        attn_mask = mx.array(tokens["attention_mask"])

        t5_output = self.t5(input_ids, attn_mask)  # (1, 128, 768)
        t5_tokens = t5_output[:, :64, :]  # first 64 tokens

        # Time embeddings
        total_arr = _to_mx_array(seconds_total)
        total_emb = self.seconds_total(total_arr)  # (1, 768)

        time_tokens = [total_emb[:, None, :]]  # (1, 1, 768)
        global_parts = [total_emb]

        if self.has_seconds_start:
            start_arr = _to_mx_array(seconds_start)
            start_emb = self.seconds_start(start_arr)
            time_tokens.append(start_emb[:, None, :])
            global_parts.append(start_emb)

        # Combine text + time tokens for cross-attention
        cross_parts = [t5_tokens] + time_tokens
        cross_attn = mx.concatenate(cross_parts, axis=1)

        # Global conditioning = sum of time embeddings
        global_cond = global_parts[0]
        for g in global_parts[1:]:
            global_cond = global_cond + g

        return cross_attn, global_cond


def _to_mx_array(val: float | int | mx.array) -> mx.array:
    """Convert scalar or mx.array to a batched mx.array."""
    if isinstance(val, (float, int)):
        return mx.array([val])
    return val
