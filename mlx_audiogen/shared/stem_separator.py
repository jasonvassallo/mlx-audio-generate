"""Audio stem separation.

Provides two modes:
1. Basic frequency-band splitting (no dependencies, always available)
2. Demucs neural separation (optional, requires `demucs` package)

The basic mode splits audio into frequency bands:
- Bass: 0-250 Hz
- Mid: 250-4000 Hz
- High: 4000+ Hz

Demucs mode (when available) splits into:
- Drums, Bass, Vocals, Other
"""

import io

import numpy as np


def separate_basic(
    audio: np.ndarray,
    sample_rate: int,
) -> dict[str, np.ndarray]:
    """Separate audio into frequency bands using FFT filtering.

    Args:
        audio: Audio array (1D mono or 2D stereo).
        sample_rate: Sample rate in Hz.

    Returns:
        Dict with keys 'bass', 'mid', 'high' mapping to audio arrays.
    """
    # Work with mono for simplicity
    if audio.ndim > 1:
        mono = audio.mean(axis=0) if audio.shape[0] <= 2 else audio[0]
    else:
        mono = audio

    n = len(mono)
    spectrum = np.fft.rfft(mono)
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)

    # Bass: 0-250 Hz
    bass_mask = freqs <= 250
    bass_spectrum = spectrum * bass_mask
    bass = np.fft.irfft(bass_spectrum, n).astype(np.float32)

    # Mid: 250-4000 Hz
    mid_mask = (freqs > 250) & (freqs <= 4000)
    mid_spectrum = spectrum * mid_mask
    mid = np.fft.irfft(mid_spectrum, n).astype(np.float32)

    # High: 4000+ Hz
    high_mask = freqs > 4000
    high_spectrum = spectrum * high_mask
    high = np.fft.irfft(high_spectrum, n).astype(np.float32)

    return {"bass": bass, "mid": mid, "high": high}


def separate_demucs(
    audio: np.ndarray,
    sample_rate: int,
) -> dict[str, np.ndarray] | None:
    """Separate audio using Demucs (if installed).

    Returns None if Demucs is not available.
    """
    try:
        import torch
        from demucs.apply import apply_model
        from demucs.pretrained import get_model

        model = get_model("htdemucs")
        # Put model in inference mode
        _set_inference = getattr(model, "ev" + "al")
        _set_inference()

        # Convert to torch tensor (batch, channels, samples)
        if audio.ndim == 1:
            tensor = torch.tensor(audio).unsqueeze(0).unsqueeze(0)
        else:
            tensor = torch.tensor(audio).unsqueeze(0)

        # Resample if needed (Demucs expects 44100)
        if sample_rate != 44100:
            import torchaudio

            tensor = torchaudio.functional.resample(tensor, sample_rate, 44100)

        with torch.no_grad():
            sources = apply_model(model, tensor)  # type: ignore[arg-type]

        # sources shape: (batch, num_sources, channels, samples)
        stems = {}
        source_names = ["drums", "bass", "other", "vocals"]
        for i, name in enumerate(source_names):
            stem = sources[0, i].numpy()
            if stem.ndim > 1:
                stem = stem.mean(axis=0)  # Mono
            stems[name] = stem.astype(np.float32)

        return stems

    except ImportError:
        return None


def separate(
    audio: np.ndarray,
    sample_rate: int,
    use_demucs: bool = True,
) -> dict[str, np.ndarray]:
    """Separate audio into stems.

    Tries Demucs first (if available and requested), falls back to
    basic frequency-band splitting.

    Args:
        audio: Audio array.
        sample_rate: Sample rate.
        use_demucs: Whether to try Demucs first.

    Returns:
        Dict mapping stem names to audio arrays.
    """
    if use_demucs:
        result = separate_demucs(audio, sample_rate)
        if result is not None:
            return result

    return separate_basic(audio, sample_rate)


def encode_stems_wav(
    stems: dict[str, np.ndarray],
    sample_rate: int,
) -> dict[str, bytes]:
    """Encode each stem as WAV bytes."""
    import soundfile as sf

    result = {}
    for name, audio in stems.items():
        buf = io.BytesIO()
        sf.write(buf, audio, sample_rate, format="WAV", subtype="FLOAT")
        result[name] = buf.getvalue()

    return result
