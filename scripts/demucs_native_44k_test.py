#!/usr/bin/env python3
"""Test Demucs with native 44.1kHz audio (no resampling needed).

This isolates model-inherent artifacts from resampling artifacts.
If distortion appears here too, it's a model quality issue, not resampling.
If distortion is gone, the resampler is the culprit.

Usage::

    uv run python scripts/demucs_native_44k_test.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import soundfile as sf

DEMUCS_WEIGHTS = "./converted/demucs-htdemucs"
OUTPUT_DIR = Path("output/demucs_native_44k_test")


def generate_test_audio(sr: int = 44100, duration: float = 5.0) -> np.ndarray:
    """Generate a multi-instrument test signal at native 44.1kHz.

    Creates a mix of:
    - Kick drum pattern (low sine bursts at ~60 Hz)
    - Bass line (sine sweep 80-200 Hz)
    - Mid-range tone (sine at 440 Hz with tremolo)
    - High harmonics (sine at 2000 Hz, quieter)
    """
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)

    # Kick drum: 60Hz sine with exponential decay, every 0.5s
    kick = np.zeros_like(t)
    beat_period = 0.5
    for beat_start in np.arange(0, duration, beat_period):
        mask = (t >= beat_start) & (t < beat_start + 0.15)
        local_t = t[mask] - beat_start
        kick[mask] = 0.8 * np.sin(2 * np.pi * 60 * local_t) * np.exp(-20 * local_t)

    # Bass: gentle sine at 100 Hz
    bass = 0.3 * np.sin(2 * np.pi * 100 * t)

    # Mid tone: 440 Hz with amplitude modulation (tremolo)
    tremolo = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)
    mid = 0.2 * np.sin(2 * np.pi * 440 * t) * tremolo

    # High harmonics: quiet 2 kHz tone
    high = 0.1 * np.sin(2 * np.pi * 2000 * t)

    mix = kick + bass + mid + high
    # Normalize to prevent clipping
    peak = np.max(np.abs(mix))
    if peak > 0.95:
        mix *= 0.95 / peak

    return mix


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sr = 44100  # Native Demucs rate — NO RESAMPLING

    # Generate test audio
    print("=== Generating 44.1kHz test audio (no resampling needed) ===")
    audio = generate_test_audio(sr=sr, duration=5.0)
    print(f"  Shape: {audio.shape}, SR: {sr}, Duration: {len(audio) / sr:.1f}s")

    original_path = OUTPUT_DIR / "original_44k.wav"
    sf.write(str(original_path), audio, sr)
    print(f"  Saved: {original_path}")

    # Separate with Demucs
    print("\n=== Separating with MLX Demucs (native 44.1kHz — no resampling) ===")
    from mlx_audiogen.models.demucs.pipeline import DemucsPipeline

    t0 = time.perf_counter()
    pipeline = DemucsPipeline.from_pretrained(DEMUCS_WEIGHTS)
    print(f"  Model loaded in {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    stems = pipeline.separate(audio, sample_rate=sr)
    sep_time = time.perf_counter() - t0
    print(f"  Separated in {sep_time:.1f}s")

    # Save stems
    print(f"\n=== Saving stems (stereo, {sr} Hz) ===")
    for name, stem in stems.items():
        stem_path = OUTPUT_DIR / f"stem_{name}_44k.wav"
        sf.write(str(stem_path), stem.T, sr)
        rms = np.sqrt(np.mean(stem**2))
        peak = np.max(np.abs(stem))
        print(f"  {name:>8s}: RMS={rms:.4f}  peak={peak:.4f}  → {stem_path}")

    print(f"\n  Output directory: {OUTPUT_DIR.resolve()}")
    print("  Compare these stems (no resampling) with the 32kHz demo stems.")
    print("  If distortion is gone here, it was a resampling artifact.")
    print("  If distortion is still present, it's inherent model quality.")


if __name__ == "__main__":
    main()
