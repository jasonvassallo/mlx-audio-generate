"""Chromagram feature extraction for MusicGen melody conditioning.

Extracts 12-bin chromagram features from audio waveforms using STFT,
normalizes, and converts to one-hot encoding. This matches the behaviour
of ``MusicgenMelodyFeatureExtractor`` from HuggingFace ``transformers``.

The chromagram maps spectral energy to 12 pitch classes (C, C#, D, ...),
capturing the harmonic content of music without specific octave information.

Processing pipeline:
    1. Mono conversion + optional resampling
    2. STFT (n_fft=16384, hop_length=4096, Hann window)
    3. Chroma filter bank maps FFT bins -> 12 chroma bins
    4. Normalize by max value per frame
    5. Argmax -> one-hot encoding (most salient pitch class per frame)
"""

import numpy as np


def _chroma_filter_bank(
    sr: int,
    n_fft: int,
    n_chroma: int = 12,
    tuning: float = 0.0,
) -> np.ndarray:
    """Build a chroma filter bank mapping FFT bins to 12 pitch classes.

    Each row of the output maps one FFT frequency bin to the 12 chroma bins
    based on the pitch class it falls closest to.

    Args:
        sr: Sample rate in Hz.
        n_fft: FFT size.
        n_chroma: Number of chroma bins (always 12).
        tuning: Tuning deviation in fractions of a chroma bin.

    Returns:
        Filter bank of shape (n_chroma, 1 + n_fft // 2).
    """
    n_bins = 1 + n_fft // 2
    freqs = np.linspace(0, sr / 2, n_bins)

    # Avoid log(0) — set DC bin to a very low frequency
    freqs[0] = 1e-5

    # Convert Hz to MIDI note numbers, then to chroma bin index
    # MIDI: 69 = A4 = 440 Hz, so note = 12 * log2(f/440) + 69
    midi = 12.0 * np.log2(freqs / 440.0) + 69.0 + tuning
    chroma_idx = np.mod(midi, 12)

    # Create filter bank: each chroma bin gets energy from nearby FFT bins
    # Using cosine window with width 2 semitones
    fb = np.zeros((n_chroma, n_bins), dtype=np.float32)
    for c in range(n_chroma):
        # Distance from this chroma bin (circular)
        diff = chroma_idx - c
        # Wrap to [-6, 6]
        diff = np.mod(diff + 6, 12) - 6
        # Cosine weighting within 1.5 semitones
        mask = np.abs(diff) < 1.5
        fb[c, mask] = np.cos(diff[mask] * np.pi / 3.0)

    # Normalize each chroma bin
    norms = fb.sum(axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    fb /= norms

    return fb


def extract_chroma(
    audio: np.ndarray,
    sr: int = 32000,
    n_fft: int = 16384,
    hop_length: int = 4096,
    n_chroma: int = 12,
    chroma_length: int = 235,
) -> np.ndarray:
    """Extract one-hot chromagram features from audio.

    Args:
        audio: Audio waveform, shape (samples,) or (channels, samples).
            Will be converted to mono if stereo.
        sr: Sample rate of the input audio.
        n_fft: FFT window size.
        hop_length: STFT hop length.
        n_chroma: Number of chroma bins (12 = chromatic scale).
        chroma_length: Output sequence length (pad or truncate).

    Returns:
        One-hot chroma features, shape (1, chroma_length, n_chroma).
        Each frame has exactly one 1 (most salient pitch class).
    """
    # Convert to mono if stereo
    if audio.ndim > 1:
        audio = audio.mean(axis=0)

    # Ensure float32
    audio = audio.astype(np.float32)

    # STFT with Hann window
    window = np.hanning(n_fft).astype(np.float32)
    n_frames = 1 + (len(audio) - n_fft) // hop_length

    if n_frames <= 0:
        # Audio too short — pad with zeros
        pad_len = n_fft - len(audio)
        audio = np.pad(audio, (0, max(0, pad_len)))
        n_frames = 1

    # Compute magnitude spectrogram
    mag = np.zeros((1 + n_fft // 2, n_frames), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_length
        frame = audio[start : start + n_fft] * window
        spectrum = np.fft.rfft(frame)
        mag[:, i] = np.abs(spectrum)

    # Apply chroma filter bank
    fb = _chroma_filter_bank(sr, n_fft, n_chroma)
    chroma = fb @ mag  # (n_chroma, n_frames)

    # Normalize per frame
    frame_max = chroma.max(axis=0, keepdims=True)
    frame_max = np.where(frame_max > 0, frame_max, 1.0)
    chroma = (chroma / frame_max).astype(np.float32)

    # Transpose to (n_frames, n_chroma)
    chroma = chroma.T

    # Argmax -> one-hot
    best_bin = chroma.argmax(axis=-1)
    one_hot = np.zeros_like(chroma)
    one_hot[np.arange(len(best_bin)), best_bin] = 1.0

    # Pad or truncate to chroma_length
    if one_hot.shape[0] < chroma_length:
        pad = np.zeros(
            (chroma_length - one_hot.shape[0], n_chroma),
            dtype=np.float32,
        )
        # Default padding: first chroma bin = 1 (C note)
        pad[:, 0] = 1.0
        one_hot = np.concatenate([one_hot, pad], axis=0)
    else:
        one_hot = one_hot[:chroma_length]

    # Add batch dimension: (1, chroma_length, n_chroma)
    return one_hot[np.newaxis]
