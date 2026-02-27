"""HTTP server for MLX audio generation.

Provides a FastAPI-based REST API for generating audio from text prompts.
Designed for integration with Max for Live and other HTTP clients.

Usage:
    mlx-audiogen-server --weights-dir ./converted/musicgen-small --port 8420

    # Or with uvicorn directly:
    uvicorn mlx_audiogen.server:app --port 8420

Install server dependencies:
    uv sync --extra server
"""

from .app import app, main

__all__ = ["app", "main"]
