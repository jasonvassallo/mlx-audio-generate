"""Async token-bucket rate limiter for external API calls.

Each API endpoint gets its own :class:`ApiRateLimiter` instance to enforce
per-service rate limits independently.

Usage::

    mb_limiter = ApiRateLimiter(max_per_second=1.0)   # MusicBrainz: 1 req/s
    lfm_limiter = ApiRateLimiter(max_per_second=5.0)  # Last.fm: 5 req/s
    dc_limiter = ApiRateLimiter(max_per_second=1.0)   # Discogs: 1 req/s

    await mb_limiter.acquire()
    # ... make API call ...
"""

from __future__ import annotations

import asyncio
import time


class ApiRateLimiter:
    """Simple async rate limiter that enforces a minimum interval between calls.

    Args:
        max_per_second: Maximum number of requests allowed per second.
    """

    def __init__(self, max_per_second: float) -> None:
        self._interval: float = 1.0 / max_per_second
        self._last_call: float = 0.0
        self._lock: asyncio.Lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until the next request slot is available, then mark it used."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            if elapsed < self._interval:
                await asyncio.sleep(self._interval - elapsed)
            self._last_call = time.monotonic()
