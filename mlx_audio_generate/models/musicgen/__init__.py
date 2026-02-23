"""MusicGen model — autoregressive text-to-music generation."""

from .config import AudioEncoderConfig, DecoderConfig, MusicGenConfig, TextEncoderConfig
from .convert import convert_musicgen
from .model import MusicGenModel
from .pipeline import MusicGenPipeline

__all__ = [
    "MusicGenPipeline",
    "MusicGenModel",
    "MusicGenConfig",
    "DecoderConfig",
    "AudioEncoderConfig",
    "TextEncoderConfig",
    "convert_musicgen",
]
