"""MusicBrainz recording search client."""

from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

from .clients import create_client
from .rate_limiter import ApiRateLimiter

logger = logging.getLogger(__name__)

_BASE_URL = "https://musicbrainz.org/ws/2/recording"


def _parse_musicbrainz_response(data: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Extract structured metadata from a MusicBrainz recording search response.

    Returns a dict with tags, genres, release_group, artist_mbid, and
    similar_artists — or ``None`` if the response contains no recordings.
    """
    recordings = data.get("recordings", [])
    if not recordings:
        return None

    rec = recordings[0]

    # Artist info
    artist_mbid: Optional[str] = None
    artist_credit = rec.get("artist-credit", [])
    if artist_credit:
        artist = artist_credit[0].get("artist", {})
        artist_mbid = artist.get("id")

    # Tags
    tags = rec.get("tags", [])

    # Release group from first release
    release_group: Optional[str] = None
    releases = rec.get("releases", [])
    if releases:
        rg = releases[0].get("release-group", {})
        release_group = rg.get("id")

    return {
        "recording_mbid": rec.get("id"),
        "title": rec.get("title"),
        "artist_mbid": artist_mbid,
        "tags": tags,
        "genres": [t["name"] for t in tags if t.get("name")],
        "release_group": release_group,
        "similar_artists": [],  # requires a separate lookup
    }


async def search_musicbrainz(
    artist: str,
    title: str,
    rate_limiter: ApiRateLimiter,
    client: Optional[httpx.AsyncClient] = None,
) -> Optional[dict[str, Any]]:
    """Search MusicBrainz for a recording by artist and title.

    Returns parsed metadata or ``None`` on errors / empty results.
    """
    await rate_limiter.acquire()

    query = f'recording:"{title}" AND artist:"{artist}"'
    params = {"query": query, "fmt": "json", "limit": "1"}

    owns_client = client is None
    if owns_client:
        client = create_client()
    assert client is not None

    try:
        resp = await client.get(_BASE_URL, params=params)
        if resp.status_code == 429:
            logger.warning("MusicBrainz rate limited (429)")
            return None
        resp.raise_for_status()
        return _parse_musicbrainz_response(resp.json())
    except httpx.HTTPError as exc:
        logger.warning("MusicBrainz request failed: %s", exc)
        return None
    finally:
        if owns_client:
            await client.aclose()
