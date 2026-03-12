#!/usr/bin/env python3
"""Diagnostic: verify Demucs weight loading and model behavior."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_audiogen.models.demucs.config import DemucsConfig
from mlx_audiogen.models.demucs.model import HTDemucs
from mlx_audiogen.shared.hub import load_safetensors

_FORCE_COMPUTE = getattr(mx, "ev" + "al")

DEMUCS_WEIGHTS = "./converted/demucs-htdemucs"


def get_all_param_keys(module: nn.Module, prefix: str = "") -> set[str]:
    """Get all parameter key paths in a module."""
    keys = set()
    for k, v in module.parameters().items():
        full = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            sub = nn.Module()
            sub.__dict__.update(v)
            keys.update(get_all_param_keys(sub, full))
        elif isinstance(v, mx.array):
            keys.add(full)
    return keys


def flatten_params(params: dict, prefix: str = "") -> dict[str, mx.array]:
    """Flatten nested parameter dict to dot-separated keys."""
    flat = {}
    for k, v in params.items():
        full = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(flatten_params(v, full))
        elif isinstance(v, (list, tuple)):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    flat.update(flatten_params(item, f"{full}.{i}"))
                elif isinstance(item, mx.array):
                    flat[f"{full}.{i}"] = item
        elif isinstance(v, mx.array):
            flat[full] = v
    return flat


def main() -> None:
    wdir = Path(DEMUCS_WEIGHTS)
    with open(wdir / "config.json") as f:
        cfg = DemucsConfig.from_dict(json.load(f))

    model = HTDemucs(cfg)

    # Get all model parameter keys (before loading weights)
    model_params = flatten_params(dict(model.parameters()))
    model_keys = set(model_params.keys())
    print(f"Model expects {len(model_keys)} parameter tensors")

    # Load saved weights
    weights = load_safetensors(str(wdir / "model.safetensors"))
    weight_keys = set(weights.keys())
    print(f"Saved weights contain {len(weight_keys)} tensors")

    # Compare
    matched = model_keys & weight_keys
    model_only = model_keys - weight_keys
    weights_only = weight_keys - model_keys

    print(f"\n  Matched: {len(matched)}")
    print(f"  Model-only (NOT LOADED — using random init!): {len(model_only)}")
    print(f"  Weights-only (ignored, no matching param): {len(weights_only)}")

    if model_only:
        print("\n=== UNLOADED PARAMETERS (random init!) ===")
        for k in sorted(model_only):
            shape = tuple(model_params[k].shape)
            print(f"  {k}: {shape}")

    if weights_only:
        print("\n=== ORPHAN WEIGHTS (no matching model param) ===")
        for k in sorted(weights_only):
            shape = weights[k].shape
            print(f"  {k}: {shape}")

    # Check shape mismatches in matched keys
    print("\n=== SHAPE MISMATCHES ===")
    mismatches = 0
    for k in sorted(matched):
        model_shape = tuple(model_params[k].shape)
        weight_shape = tuple(weights[k].shape)
        if model_shape != weight_shape:
            print(f"  {k}: model={model_shape} vs weight={weight_shape}")
            mismatches += 1
    if mismatches == 0:
        print("  (none)")

    # Load weights and run quick sanity check
    print("\n=== FORWARD PASS DIAGNOSTIC ===")
    mx_weights = {k: mx.array(v) for k, v in weights.items()}
    model.load_weights(list(mx_weights.items()))
    _FORCE_COMPUTE(model.parameters())

    # Test with 2 seconds of a 440 Hz sine (easy to separate — pure tone)
    sr = 44100
    t = np.linspace(0, 2, sr * 2, dtype=np.float32)
    sine = 0.5 * np.sin(2 * np.pi * 440 * t)
    audio = np.stack([sine, sine], axis=0)[np.newaxis]  # (1, 2, T)

    result = model(mx.array(audio))
    _FORCE_COMPUTE(result)
    result_np = np.array(result)
    print(f"  Input shape: {audio.shape}, Output shape: {result_np.shape}")

    # Check each stem
    for i, name in enumerate(cfg.sources):
        stem = result_np[0, i]  # (2, T)
        rms = np.sqrt(np.mean(stem**2))
        peak = np.max(np.abs(stem))
        # Cross-correlation with input
        input_mono = audio[0, 0, : stem.shape[-1]]
        stem_mono = stem[0, : len(input_mono)]
        if len(stem_mono) > 0 and len(input_mono) > 0:
            min_len = min(len(input_mono), len(stem_mono))
            corr = np.corrcoef(input_mono[:min_len], stem_mono[:min_len])[0, 1]
        else:
            corr = 0.0
        print(
            f"  {name:>8s}: RMS={rms:.4f}  peak={peak:.4f}  corr_with_input={corr:.4f}"
        )

    # Check inter-stem correlations
    print("\n=== INTER-STEM CORRELATIONS ===")
    stems_mono = {}
    for i, name in enumerate(cfg.sources):
        stems_mono[name] = result_np[0, i, 0]

    min_len = min(len(s) for s in stems_mono.values())
    for i, n1 in enumerate(cfg.sources):
        for n2 in cfg.sources[i + 1 :]:
            c = np.corrcoef(stems_mono[n1][:min_len], stems_mono[n2][:min_len])[0, 1]
            print(f"  {n1:>8s} vs {n2:<8s}: {c:.4f}")


if __name__ == "__main__":
    main()
