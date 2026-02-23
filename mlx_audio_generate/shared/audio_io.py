"""Audio file I/O utilities."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import mlx.core as mx


def save_wav(
    path: str | Path,
    audio: np.ndarray | mx.array,
    sample_rate: int,
    channels: int = 1,
) -> None:
    """Save audio array to a WAV file.

    Args:
        path: Output file path.
        audio: Audio data. Can be MLX array or numpy array.
            Expected shapes: (samples,), (1, samples), (channels, samples),
            or (batch, channels, samples) where batch=1.
        sample_rate: Sample rate in Hz (e.g. 32000, 44100).
        channels: Number of audio channels (1=mono, 2=stereo).
    """
    import soundfile as sf

    # Convert MLX array to numpy if needed
    try:
        import mlx.core as mx

        if isinstance(audio, mx.array):
            audio = np.array(audio)
    except ImportError:
        pass

    # Squeeze batch dimension if present
    while audio.ndim > 2:
        audio = audio[0]

    # Transpose from (channels, samples) to (samples, channels) for soundfile
    if audio.ndim == 2 and audio.shape[0] <= audio.shape[1]:
        audio = audio.T

    # Clip to valid range
    audio = np.clip(audio, -1.0, 1.0)

    sf.write(str(path), audio, sample_rate)


def load_wav(
    path: str | Path, target_sample_rate: int | None = None
) -> tuple[np.ndarray, int]:
    """Load a WAV file. Returns (samples, sample_rate)."""
    import soundfile as sf

    audio, sr = sf.read(str(path))
    if target_sample_rate is not None and sr != target_sample_rate:
        # Basic resampling via linear interpolation
        ratio = target_sample_rate / sr
        n_samples = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, n_samples)
        audio = np.interp(indices, np.arange(len(audio)), audio)
        sr = target_sample_rate
    return audio, sr


def play_audio(path: str | Path) -> None:
    """Play audio using macOS afplay. Fails silently on non-macOS."""
    try:
        subprocess.run(["afplay", str(path)], check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
