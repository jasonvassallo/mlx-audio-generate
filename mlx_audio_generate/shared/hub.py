"""HuggingFace Hub integration for downloading and loading model weights."""

from pathlib import Path

import numpy as np


def download_model(
    repo_id: str,
    allow_patterns: list[str] | None = None,
    revision: str | None = None,
    force: bool = False,
) -> Path:
    """Download a model from HuggingFace Hub.

    Returns the path to the cached snapshot directory.
    """
    from huggingface_hub import snapshot_download

    if allow_patterns is None:
        allow_patterns = ["*.json", "*.safetensors", "*.model", "*.txt"]

    path = snapshot_download(
        repo_id,
        allow_patterns=allow_patterns,
        revision=revision,
        force_download=force,
    )
    return Path(path)


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


def save_safetensors(weights: dict[str, np.ndarray], path: str | Path) -> None:
    """Save a dict of numpy arrays as a safetensors file."""
    from safetensors.numpy import save_file

    save_file(weights, str(path))
