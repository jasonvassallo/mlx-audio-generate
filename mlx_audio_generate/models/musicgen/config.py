"""Configuration dataclasses for MusicGen.

Reads the nested HuggingFace config.json format with three sub-configs:
  - text_encoder (T5): d_model, num_heads, num_layers, d_ff, d_kv, vocab_size
  - decoder: hidden_size, num_hidden_layers, num_attention_heads, ffn_dim, num_codebooks
  - audio_encoder (EnCodec): codebook_size, sampling_rate, num_codebooks, upsampling_ratios
"""

import inspect
from dataclasses import dataclass, field


@dataclass
class DecoderConfig:
    """Autoregressive transformer decoder configuration."""

    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    ffn_dim: int = 4096
    num_codebooks: int = 4
    vocab_size: int = 2048
    max_position_embeddings: int = 2048
    activation_function: str = "gelu"
    dropout: float = 0.1
    attention_dropout: float = 0.0
    activation_dropout: float = 0.0
    bos_token_id: int = 2048
    pad_token_id: int = 2048

    @classmethod
    def from_dict(cls, d: dict) -> "DecoderConfig":
        return cls(**{k: v for k, v in d.items() if k in inspect.signature(cls).parameters})


@dataclass
class AudioEncoderConfig:
    """EnCodec audio encoder/decoder configuration."""

    codebook_size: int = 2048
    codebook_dim: int = 128
    sampling_rate: int = 32000
    audio_channels: int = 1
    hidden_size: int = 128
    num_filters: int = 64
    num_residual_layers: int = 1
    num_lstm_layers: int = 2
    kernel_size: int = 7
    last_kernel_size: int = 7
    residual_kernel_size: int = 3
    dilation_growth_rate: int = 2
    compress: int = 2
    upsampling_ratios: list[int] = field(default_factory=lambda: [8, 5, 4, 4])
    norm_type: str = "weight_norm"
    pad_mode: str = "reflect"
    use_causal_conv: bool = False
    use_conv_shortcut: bool = False
    trim_right_ratio: float = 1.0

    @property
    def frame_rate(self) -> float:
        """Audio frames per second = sampling_rate / product(upsampling_ratios)."""
        stride = 1
        for r in self.upsampling_ratios:
            stride *= r
        return self.sampling_rate / stride

    @classmethod
    def from_dict(cls, d: dict) -> "AudioEncoderConfig":
        return cls(**{k: v for k, v in d.items() if k in inspect.signature(cls).parameters})


@dataclass
class TextEncoderConfig:
    """T5 text encoder configuration (subset needed for model construction)."""

    d_model: int = 768
    d_ff: int = 3072
    d_kv: int = 64
    num_heads: int = 12
    num_layers: int = 12
    vocab_size: int = 32128
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    model_name_or_path: str = "t5-base"

    @classmethod
    def from_dict(cls, d: dict) -> "TextEncoderConfig":
        # HF uses _name_or_path; we map it to model_name_or_path
        mapped = dict(d)
        if "_name_or_path" in mapped:
            mapped["model_name_or_path"] = mapped.pop("_name_or_path")
        return cls(**{k: v for k, v in mapped.items() if k in inspect.signature(cls).parameters})


@dataclass
class MusicGenConfig:
    """Top-level MusicGen config combining all sub-configs.

    Can be loaded from HF's config.json via ``MusicGenConfig.from_dict(json_data)``
    or constructed with defaults for musicgen-small.
    """

    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    audio_encoder: AudioEncoderConfig = field(default_factory=AudioEncoderConfig)
    text_encoder: TextEncoderConfig = field(default_factory=TextEncoderConfig)

    @classmethod
    def from_dict(cls, d: dict) -> "MusicGenConfig":
        decoder = DecoderConfig.from_dict(d.get("decoder", {}))
        audio_encoder = AudioEncoderConfig.from_dict(d.get("audio_encoder", {}))
        text_encoder = TextEncoderConfig.from_dict(d.get("text_encoder", {}))
        return cls(decoder=decoder, audio_encoder=audio_encoder, text_encoder=text_encoder)
