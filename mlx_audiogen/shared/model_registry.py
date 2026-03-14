"""Model registry mapping model names to HuggingFace repo IDs.

Pre-converted MLX weights are published to jasonvassallo/* repos on HuggingFace.
This registry enables auto-download: if a weights directory doesn't exist,
the pipeline can download pre-converted weights directly from HF.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default cache directory for auto-downloaded models
DEFAULT_MODELS_DIR = Path.home() / ".mlx-audiogen" / "models"

# Map of model short names → HuggingFace repo IDs with pre-converted MLX weights
MODEL_REGISTRY: dict[str, str] = {
    # MusicGen variants
    "musicgen-small": "jasonvassallo/mlx-musicgen-small",
    "musicgen-medium": "jasonvassallo/mlx-musicgen-medium",
    "musicgen-large": "jasonvassallo/mlx-musicgen-large",
    "musicgen-stereo-small": "jasonvassallo/mlx-musicgen-stereo-small",
    "musicgen-stereo-medium": "jasonvassallo/mlx-musicgen-stereo-medium",
    "musicgen-stereo-large": "jasonvassallo/mlx-musicgen-stereo-large",
    "musicgen-melody": "jasonvassallo/mlx-musicgen-melody",
    "musicgen-melody-large": "jasonvassallo/mlx-musicgen-melody-large",
    "musicgen-stereo-melody": "jasonvassallo/mlx-musicgen-stereo-melody",
    "musicgen-stereo-melody-large": "jasonvassallo/mlx-musicgen-stereo-melody-large",
    "musicgen-style": "jasonvassallo/mlx-musicgen-style",
    # Stable Audio variants
    "stable-audio": "jasonvassallo/mlx-stable-audio-open-small",
    "stable-audio-1.0": "jasonvassallo/mlx-stable-audio-open-1.0",
    # Demucs (source separation)
    "demucs-htdemucs": "jasonvassallo/demucs-htdemucs-mlx",
}


def resolve_weights_dir(
    weights_dir: Optional[str],
    model_name: Optional[str] = None,
    required_files: Optional[list[str]] = None,
) -> Path:
    """Resolve a weights directory, auto-downloading from HF if needed.

    Priority order:
      1. Use weights_dir if it exists as a directory with required files
      2. Check DEFAULT_MODELS_DIR / model_name
      3. Download from MODEL_REGISTRY to DEFAULT_MODELS_DIR / model_name

    Args:
        weights_dir: Explicit path, or a model name from the registry.
        model_name: Short name (e.g. "musicgen-small") for registry lookup.
            If weights_dir looks like a model name (no slashes, in registry),
            it's used as model_name automatically.
        required_files: Files that must exist in the weights directory
            (e.g. ["config.json", "decoder.safetensors"]).

    Returns:
        Path to the resolved weights directory.

    Raises:
        FileNotFoundError: If weights can't be found or downloaded.
    """
    # If both are None, caller must provide at least one
    if weights_dir is None and model_name is None:
        raise ValueError(
            "weights_dir is required. Run `mlx-audiogen-convert` first, "
            "or pass a model name from the registry: "
            + ", ".join(sorted(MODEL_REGISTRY.keys()))
        )

    # If weights_dir is a model name from the registry, treat it as such
    if weights_dir and "/" not in weights_dir and weights_dir in MODEL_REGISTRY:
        model_name = weights_dir
        weights_dir = None

    # 1. Explicit path — use directly if it exists
    if weights_dir is not None:
        path = Path(weights_dir)
        if path.is_dir():
            if required_files and not _has_required_files(path, required_files):
                logger.warning(
                    "Directory %s exists but missing required files: %s",
                    path,
                    [f for f in required_files if not (path / f).exists()],
                )
            return path
        # Path doesn't exist — fall through to auto-download
        logger.info("Weights directory %s not found, checking auto-download...", path)

    # 2. Check default models dir
    if model_name and (DEFAULT_MODELS_DIR / model_name).is_dir():
        cached = DEFAULT_MODELS_DIR / model_name
        if not required_files or _has_required_files(cached, required_files):
            logger.info("Using cached model: %s", cached)
            return cached

    # 3. Auto-download from HuggingFace
    if model_name and model_name in MODEL_REGISTRY:
        return _auto_download(model_name)

    # Can't resolve
    hint = ""
    if model_name:
        hint = f" Model '{model_name}' not found in registry."
    raise FileNotFoundError(
        f"Weights directory not found: {weights_dir or model_name}.{hint} "
        "Run `mlx-audiogen-convert` to convert weights, or use a registered "
        f"model name: {', '.join(sorted(MODEL_REGISTRY.keys()))}"
    )


def _auto_download(model_name: str) -> Path:
    """Download pre-converted weights from HuggingFace to the models cache."""
    from mlx_audiogen.shared.hub import download_model

    repo_id = MODEL_REGISTRY[model_name]
    dest = DEFAULT_MODELS_DIR / model_name
    dest.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Auto-downloading %s from %s...", model_name, repo_id)
    print(f"Downloading {model_name} from {repo_id}...")

    # Download to HF cache, then symlink to our models dir
    hf_path = download_model(
        repo_id,
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"],
    )

    # Create symlink from our models dir to HF cache
    if not dest.exists():
        dest.symlink_to(hf_path)
        logger.info("Linked %s → %s", dest, hf_path)

    print(f"Model ready: {dest}")
    return dest


def _has_required_files(path: Path, required: list[str]) -> bool:
    """Check if all required files exist in a directory."""
    return all((path / f).exists() for f in required)


def list_available_models() -> list[str]:
    """List model names that are available (downloaded) in the models cache."""
    if not DEFAULT_MODELS_DIR.is_dir():
        return []
    return sorted(
        d.name
        for d in DEFAULT_MODELS_DIR.iterdir()
        if d.is_dir() and (d / "config.json").exists()
    )


def list_registry_models() -> list[str]:
    """List all model names in the registry (downloadable)."""
    return sorted(MODEL_REGISTRY.keys())
