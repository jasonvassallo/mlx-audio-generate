"""Discogs database search client."""
from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

from .clients import create_client
from .rate_limiter import ApiRateLimiter

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.discogs.com/database/search"


def _parse_discogs_search_response(data: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Extract structured metadata from a Discogs search response.

    Returns a dict with labels, styles, genres, year, and country — or
    ``None`` if the response contains no results.
    """
    results = data.get("results", [])
    if not results:
        return None

    item = results[0]

    # Year may be a string or missing
    raw_year = item.get("year")
    year: Optional[int] = None
    if raw_year is not None:
        try:
            year = int(raw_year)
        except (ValueError, TypeError):
            pass

    return {
        "discogs_id": item.get("id"),
        "title": item.get("title"),
        "genres": item.get("genre", []),
        "styles": item.get("style", []),
        "labels": item.get("label", []),
        "year": year,
        "country": item.get("country"),
        "catno": item.get("catno"),
    }


async def search_discogs(
    artist: str,
    title: str,
    token: str,
    rate_limiter: ApiRateLimiter,
    client: Optional[httpx.AsyncClient] = None,
) -> Optional[dict[str, Any]]:
    """Search Discogs for a release matching artist and title.

    Returns parsed metadata or ``None`` on errors / empty results.
    """
    await rate_limiter.acquire()

    params = {
        "q": f"{artist} {title}",
        "type": "master",
        "per_page": "1",
    }
    headers = {"Authorization": f"Discogs token={token}"}

    owns_client = client is None
    if owns_client:
        client = create_client()

    try:
        resp = await client.get(_BASE_URL, params=params, headers=headers)
        if resp.status_code == 429:
            logger.warning("Discogs rate limited (429)")
            return None
        resp.raise_for_status()
        return _parse_discogs_search_response(resp.json())
    except httpx.HTTPError as exc:
        logger.warning("Discogs request failed: %s", exc)
        return None
    finally:
        if owns_client:
            await client.aclose()
