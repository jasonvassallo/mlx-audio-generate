"""MLX-specific utility functions for weight conversion and manipulation."""

import numpy as np


def transpose_conv1d_weight(weight: np.ndarray) -> np.ndarray:
    """Transpose Conv1d weight from PyTorch to MLX format.

    PyTorch: (out_channels, in_channels, kernel_size)
    MLX:     (out_channels, kernel_size, in_channels)
    """
    return np.transpose(weight, (0, 2, 1))


def transpose_conv_transpose1d_weight(weight: np.ndarray) -> np.ndarray:
    """Transpose ConvTranspose1d weight from PyTorch to MLX format.

    PyTorch: (in_channels, out_channels, kernel_size)
    MLX:     (out_channels, kernel_size, in_channels)
    """
    return np.transpose(weight, (1, 2, 0))


def fuse_weight_norm(weight_g: np.ndarray, weight_v: np.ndarray) -> np.ndarray:
    """Fuse weight normalization parameters into a single weight tensor.

    weight_g: magnitude scalar (out_channels, 1, 1)
    weight_v: direction tensor (out_channels, in_channels, kernel_size)
    Returns: g * (v / ||v||) with norm over axes (1, 2)
    """
    norm = np.linalg.norm(weight_v, axis=(1, 2), keepdims=True)
    return weight_g * (weight_v / (norm + 1e-12))


def remap_key(key: str, mappings: list[tuple[str, str]]) -> str:
    """Apply a sequence of (pattern, replacement) pairs to a weight key."""
    for pattern, replacement in mappings:
        key = key.replace(pattern, replacement)
    return key
