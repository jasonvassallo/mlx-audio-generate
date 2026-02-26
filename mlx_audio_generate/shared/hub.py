"""HuggingFace Hub integration for downloading and loading model weights."""

import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Retry configuration for transient network failures
_MAX_RETRIES = 3
_RETRY_DELAY_SECONDS = 2.0


def download_model(
    repo_id: str,
    allow_patterns: list[str] | None = None,
    revision: str | None = None,
    force: bool = False,
) -> Path:
    """Download a model from HuggingFace Hub with retry logic.

    Retries up to 3 times with exponential backoff on transient network errors.

    Returns the path to the cached snapshot directory.
    """
    from huggingface_hub import snapshot_download

    if allow_patterns is None:
        allow_patterns = ["*.json", "*.safetensors", "*.model", "*.txt"]

    last_error: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            logger.info(
                "Downloading %s (attempt %d/%d)",
                repo_id,
                attempt,
                _MAX_RETRIES,
            )
            path = snapshot_download(  # nosec B615 — revision passed by caller
                repo_id,
                allow_patterns=allow_patterns,
                revision=revision,
                force_download=force,
            )
            logger.info("Downloaded %s to %s", repo_id, path)
            return Path(path)
        except (OSError, ConnectionError, TimeoutError) as e:
            last_error = e
            if attempt < _MAX_RETRIES:
                delay = _RETRY_DELAY_SECONDS * (2 ** (attempt - 1))
                logger.warning(
                    "Download attempt %d failed: %s. Retrying in %.1fs...",
                    attempt,
                    e,
                    delay,
                )
                time.sleep(delay)
            else:
                logger.error("Download failed after %d attempts: %s", _MAX_RETRIES, e)

    raise last_error  # type: ignore[misc]


def load_safetensors(path: str | Path) -> dict[str, np.ndarray]:
    """Load a single safetensors file as numpy arrays."""
    from safetensors import safe_open

    weights = {}
    with safe_open(str(path), framework="numpy", device="cpu") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)
    return weights


def load_all_safetensors(directory: str | Path) -> dict[str, np.ndarray]:
    """Load all safetensors files from a directory, merged into one dict."""
    directory = Path(directory)
    weights = {}
    for sf_file in sorted(directory.glob("*.safetensors")):
        if sf_file.name == "tokenizer.safetensors":
            continue
        weights.update(load_safetensors(sf_file))
    return weights


def load_pytorch_bin(path: str | Path) -> dict[str, np.ndarray]:
    """Load a PyTorch .bin file and convert tensors to numpy arrays.

    Requires torch to be installed (``pip install mlx-audio-generate[convert]``).
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required to load .bin weight files. "
            "Install it with: pip install mlx-audio-generate[convert]"
        ) from exc

    state_dict = torch.load(str(path), map_location="cpu", weights_only=True)
    weights: dict[str, np.ndarray] = {}
    for key, tensor in state_dict.items():
        weights[key] = tensor.numpy()
    return weights


def save_safetensors(weights: dict[str, np.ndarray], path: str | Path) -> None:
    """Save a dict of numpy arrays as a safetensors file."""
    from safetensors.numpy import save_file

    save_file(weights, str(path))
