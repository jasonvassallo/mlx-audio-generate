"""Stable Audio Open generation pipeline.

Orchestrates the full text-to-audio workflow:
    1. Text + duration conditioning via T5 and NumberEmbedder
    2. Rectified-flow diffusion sampling in latent space
    3. VAE decoding back to waveform

Weights are stored as separate safetensors files produced by convert.py.
"""

import json
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from transformers import AutoTokenizer

from mlx_audio_generate.shared.hub import load_safetensors
from mlx_audio_generate.shared.t5 import T5Config, T5EncoderModel

from .conditioners import Conditioners
from .config import StableAudioConfig
from .dit import StableAudioDiT
from .sampling import get_rf_schedule, sample_euler, sample_rk4
from .vae import AutoencoderOobleck


def _materialize(x: mx.array) -> None:
    """Force MLX lazy graph materialization on GPU (mlx.core function)."""
    mx.eval(x)  # noqa: S307


class StableAudioPipeline:
    """End-to-end Stable Audio Open generation pipeline."""

    def __init__(
        self,
        vae: AutoencoderOobleck,
        dit: StableAudioDiT,
        conditioners: Conditioners,
        config: StableAudioConfig,
    ):
        self.vae = vae
        self.dit = dit
        self.conditioners = conditioners
        self.config = config

    @classmethod
    def from_pretrained(
        cls,
        weights_dir: Optional[str] = None,
        repo_id: str = "stabilityai/stable-audio-open-small",
    ) -> "StableAudioPipeline":
        """Load a converted pipeline from a local directory.

        Args:
            weights_dir: Path to directory with converted safetensors files.
                If None, downloads and converts automatically.
            repo_id: HuggingFace repo ID (used if weights_dir is None or
                to download the tokenizer).
        """
        if weights_dir is None:
            raise ValueError(
                "weights_dir is required. Run `mlx-audio-convert "
                "--model stabilityai/stable-audio-open-small` first."
            )

        weights_path = Path(weights_dir)

        # Load config
        config = _load_config(weights_path)

        # Load tokenizer
        tokenizer = _load_tokenizer(weights_path, repo_id)

        # Build and load VAE
        print("Loading VAE...")
        vae = AutoencoderOobleck(config.vae)
        vae_weights = load_safetensors(weights_path / "vae.safetensors")
        vae.load_weights(list((k, mx.array(v)) for k, v in vae_weights.items()))
        nn.eval(vae)  # type: ignore[attr-defined]

        # Build and load DiT
        print("Loading DiT...")
        dit = StableAudioDiT(config.dit)
        dit_weights = load_safetensors(weights_path / "dit.safetensors")
        dit.load_weights(list((k, mx.array(v)) for k, v in dit_weights.items()))
        nn.eval(dit)  # type: ignore[attr-defined]

        # Build and load T5
        print("Loading T5...")
        t5_config = _load_t5_config(weights_path)
        t5 = T5EncoderModel(t5_config)
        t5_weights = load_safetensors(weights_path / "t5.safetensors")
        t5.load_weights(list((k, mx.array(v)) for k, v in t5_weights.items()))
        nn.eval(t5)  # type: ignore[attr-defined]

        # Build conditioners and load embedder weights
        print("Loading conditioners...")
        conditioners = Conditioners(t5, tokenizer)
        cond_weights = load_safetensors(weights_path / "conditioners.safetensors")
        conditioners.load_weights({k: mx.array(v) for k, v in cond_weights.items()})

        print("Pipeline ready.")
        return cls(vae, dit, conditioners, config)

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        seconds_total: float = 30.0,
        steps: int = 100,
        cfg_scale: float = 7.0,
        sigma_max: float = 1.0,
        seed: Optional[int] = None,
        sampler: str = "euler",
    ) -> mx.array:
        """Generate audio from a text prompt.

        Args:
            prompt: Text description of desired audio.
            negative_prompt: Negative prompt for CFG.
            seconds_total: Duration in seconds.
            steps: Number of diffusion steps.
            cfg_scale: Classifier-free guidance scale.
            sigma_max: Maximum sigma for rectified flow schedule.
            seed: Random seed for reproducibility.
            sampler: 'euler' (fast) or 'rk4' (accurate).

        Returns:
            Audio tensor of shape (1, channels, samples).
        """
        # Validate prompt
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string.")
        _MAX_PROMPT_CHARS = 2000
        if len(prompt) > _MAX_PROMPT_CHARS:
            print(
                f"Warning: Prompt is {len(prompt)} chars (max recommended: "
                f"{_MAX_PROMPT_CHARS}). It will be truncated by the tokenizer."
            )

        if seed is not None:
            mx.random.seed(seed)
        else:
            # Use OS entropy for non-reproducible generation
            import os

            mx.random.seed(int.from_bytes(os.urandom(4)))

        # Conditioning
        print("Encoding conditioning...")
        cond_tokens, global_cond = self.conditioners(prompt, seconds_total)

        uncond_tokens = None
        if cfg_scale > 1.0:
            uncond_tokens, _ = self.conditioners(negative_prompt, seconds_total)

        # Initialize noise in latent space
        latent_rate = self.config.sample_rate / 2048  # ~21.5 Hz for 44100
        latent_length = int(seconds_total * latent_rate)
        latents = mx.random.normal((1, 64, latent_length))

        # Timestep schedule
        timesteps = get_rf_schedule(steps, sigma_max)

        # Sample
        print(f"Sampling ({sampler.upper()}, {steps} steps, CFG {cfg_scale})...")
        sampler_fn = {"euler": sample_euler, "rk4": sample_rk4}.get(sampler)
        if sampler_fn is None:
            raise ValueError(f"Unknown sampler '{sampler}'. Use 'euler' or 'rk4'.")

        latents = sampler_fn(
            self.dit,
            latents,
            timesteps,
            cond_tokens,
            uncond_tokens,
            global_cond,
            cfg_scale,
            steps,
        )

        # Decode latents to audio
        print("Decoding latents to audio...")
        # DiT outputs (B, C, T); VAE decoder expects (B, T, C) for MLX Conv1d
        latents = latents.transpose(0, 2, 1)
        _materialize(latents)

        audio = self.vae.decode(latents)
        _materialize(audio)

        # VAE outputs (B, T, C) in MLX layout; transpose to (B, C, T)
        audio = audio.transpose(0, 2, 1)
        _materialize(audio)

        return audio


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_config(weights_path: Path) -> StableAudioConfig:
    """Load model config from JSON with basic validation."""
    config_file = weights_path / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(
            f"config.json not found in {weights_path}. Run mlx-audio-convert first."
        )
    with open(config_file) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"config.json must be a JSON object, got {type(data)}")
    return StableAudioConfig.from_dict(data)


def _load_t5_config(weights_path: Path) -> T5Config:
    """Load T5 config from JSON or fall back to defaults."""
    t5_config_file = weights_path / "t5_config.json"
    if t5_config_file.exists():
        with open(t5_config_file) as f:
            data = json.load(f)
        return T5Config(
            **{k: v for k, v in data.items() if k in T5Config.__dataclass_fields__}
        )
    return T5Config()


def _load_tokenizer(weights_path: Path, repo_id: str):
    """Load tokenizer from local dir or download from HuggingFace."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(  # nosec B615 — local path
            str(weights_path)
        )
        print(f"Loaded tokenizer from {weights_path}")
        return tokenizer
    except (OSError, ValueError, KeyError):
        pass

    # Fall back to downloading from HF
    print("Warning: Tokenizer not found locally. Downloading from HuggingFace...")
    # Both stable-audio-open-small and 1.0 use the same T5 tokenizer
    for source in [repo_id, "stabilityai/stable-audio-open-1.0"]:
        try:
            tokenizer = AutoTokenizer.from_pretrained(  # nosec B615 — known HF repo
                source, subfolder="tokenizer"
            )
            tokenizer.save_pretrained(str(weights_path))
            print(f"Saved tokenizer to {weights_path}")
            return tokenizer
        except (OSError, ValueError, KeyError):
            continue

    # Last resort: try loading as a plain T5 tokenizer
    print("Warning: Falling back to generic T5-base tokenizer.")
    tokenizer = AutoTokenizer.from_pretrained(  # nosec B615 — known HF repo
        "google-t5/t5-base"
    )
    tokenizer.save_pretrained(str(weights_path))
    return tokenizer
