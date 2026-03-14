"""Shared httpx async client factory for enrichment APIs."""

from __future__ import annotations

import httpx

_USER_AGENT = "mlx-audiogen/0.1.0 (https://github.com/jasonvassallo/mlx-audiogen)"


def create_client(timeout: float = 10.0) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        headers={"User-Agent": _USER_AGENT, "Accept": "application/json"},
        timeout=timeout,
        follow_redirects=True,
    )
