#!/usr/bin/env python3
"""End-to-end demo: MusicGen → Demucs → stem WAV files.

Usage::

    uv run python scripts/demucs_e2e_demo.py

Generates a short clip with MusicGen, separates with Demucs, and saves
each stem + the original as WAV files in ``output/demucs_e2e_demo/``.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import soundfile as sf

MUSICGEN_WEIGHTS = "./converted/musicgen-small"
DEMUCS_WEIGHTS = "./converted/demucs-htdemucs"
OUTPUT_DIR = Path("output/demucs_e2e_demo")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Generate audio with MusicGen ---
    print("=== Step 1: Generating audio with MusicGen ===")
    from mlx_audiogen.models.musicgen.pipeline import MusicGenPipeline

    t0 = time.perf_counter()
    mg_pipeline = MusicGenPipeline.from_pretrained(MUSICGEN_WEIGHTS)
    print(f"  Model loaded in {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    audio = mg_pipeline.generate(
        prompt="upbeat rock drums and funky bass with electric guitar",
        seconds=10.0,
        seed=42,
    )
    gen_time = time.perf_counter() - t0
    duration = len(audio) / 32000
    print(f"  Generated {duration:.1f}s audio in {gen_time:.1f}s")

    # Save original
    original_path = OUTPUT_DIR / "original.wav"
    sf.write(str(original_path), audio, 32000)
    print(f"  Saved: {original_path}")

    # --- Step 2: Separate with Demucs ---
    print("\n=== Step 2: Separating with MLX Demucs ===")
    from mlx_audiogen.models.demucs.pipeline import DemucsPipeline

    t0 = time.perf_counter()
    dm_pipeline = DemucsPipeline.from_pretrained(DEMUCS_WEIGHTS)
    print(f"  Model loaded in {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    stems = dm_pipeline.separate(
        audio,
        sample_rate=32000,
        progress_callback=lambda p: print(f"  Progress: {p * 100:.0f}%", end="\r"),
    )
    sep_time = time.perf_counter() - t0
    print(f"\n  Separated in {sep_time:.1f}s")

    # --- Step 3: Save stems ---
    # Stems are returned at original sample rate (32 kHz for MusicGen)
    stem_sr = 32000
    print(f"\n=== Step 3: Saving stems (stereo, {stem_sr} Hz) ===")
    for name, stem in stems.items():
        stem_path = OUTPUT_DIR / f"stem_{name}.wav"
        sf.write(str(stem_path), stem.T, stem_sr)
        rms = np.sqrt(np.mean(stem**2))
        print(f"  {name:>8s}: RMS={rms:.4f}  → {stem_path}")

    # --- Step 4: Quality summary ---
    print("\n=== Quality Summary ===")
    stem_sum = sum(stems.values())
    for name, stem in stems.items():
        rms = np.sqrt(np.mean(stem**2))
        peak = np.max(np.abs(stem))
        print(f"  {name:>8s}: RMS={rms:.4f}  peak={peak:.4f}")

    sum_rms = np.sqrt(np.mean(stem_sum**2))
    print(f"  {'sum':>8s}: RMS={sum_rms:.4f}")
    print(f"\n  Output directory: {OUTPUT_DIR.resolve()}")
    print("  Listen to each stem to verify separation quality!")


if __name__ == "__main__":
    main()
