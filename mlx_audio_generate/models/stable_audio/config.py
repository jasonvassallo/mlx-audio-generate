"""Configuration dataclasses for Stable Audio Open."""

import inspect
from dataclasses import dataclass, field


@dataclass
class DiTConfig:
    io_channels: int = 64
    embed_dim: int = 1024
    depth: int = 16
    num_heads: int = 8
    cond_token_dim: int = 768
    global_cond_dim: int = 768
    project_cond_tokens: bool = True
    transformer_type: str = "continuous_transformer"
    global_cond_type: str = "prepend"
    patch_size: int = 1
    timestep_features_dim: int = 256

    @classmethod
    def from_dict(cls, d: dict) -> "DiTConfig":
        valid = inspect.signature(cls).parameters
        return cls(**{k: v for k, v in d.items() if k in valid})


@dataclass
class OobleckConfig:
    in_channels: int = 2
    channels: int = 128
    c_mults: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    strides: list[int] = field(default_factory=lambda: [2, 4, 4, 8, 8])
    latent_dim: int = 64
    use_snake: bool = True
    final_tanh: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "OobleckConfig":
        valid = inspect.signature(cls).parameters
        return cls(**{k: v for k, v in d.items() if k in valid})


@dataclass
class StableAudioConfig:
    dit: DiTConfig = field(default_factory=DiTConfig)
    vae: OobleckConfig = field(default_factory=OobleckConfig)
    sample_rate: int = 44100
    sample_size: int = 524288

    @classmethod
    def from_dict(cls, d: dict) -> "StableAudioConfig":
        dit_cfg = DiTConfig.from_dict(d.get("dit", {}))
        vae_cfg = OobleckConfig.from_dict(d.get("vae", {}))
        return cls(
            dit=dit_cfg,
            vae=vae_cfg,
            sample_rate=d.get("sample_rate", 44100),
            sample_size=d.get("sample_size", 524288),
        )
