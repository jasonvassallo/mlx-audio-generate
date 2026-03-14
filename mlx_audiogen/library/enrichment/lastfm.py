"""Last.fm track info client."""
from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

from .clients import create_client
from .rate_limiter import ApiRateLimiter

logger = logging.getLogger(__name__)

_BASE_URL = "https://ws.audioscrobbler.com/2.0/"


def _parse_lastfm_track_response(data: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Extract structured metadata from a Last.fm track.getInfo response.

    Returns a dict with tags, similar_tracks, similar_artists, play_count,
    and listeners — or ``None`` if the response contains an error key.
    """
    if "error" in data:
        return None

    track = data.get("track", {})
    if not track:
        return None

    # Tags
    top_tags = track.get("toptags", {}).get("tag", [])
    tags = [{"name": t["name"], "count": int(t.get("count", 0))} for t in top_tags]

    # Similar tracks
    similar_raw = track.get("similar", {}).get("track", [])
    similar_tracks = [
        {
            "name": s.get("name"),
            "artist": s.get("artist", {}).get("name"),
            "match": float(s.get("match", 0)),
        }
        for s in similar_raw
    ]

    # Similar artists (derived from similar tracks)
    similar_artists = list(
        {s["artist"] for s in similar_tracks if s.get("artist")}
    )

    return {
        "tags": tags,
        "similar_tracks": similar_tracks,
        "similar_artists": similar_artists,
        "play_count": int(track.get("playcount", 0)),
        "listeners": int(track.get("listeners", 0)),
    }


async def search_lastfm(
    artist: str,
    title: str,
    api_key: str,
    rate_limiter: ApiRateLimiter,
    client: Optional[httpx.AsyncClient] = None,
) -> Optional[dict[str, Any]]:
    """Fetch track info from Last.fm.

    Returns parsed metadata or ``None`` on errors / API errors.
    """
    await rate_limiter.acquire()

    params = {
        "method": "track.getInfo",
        "artist": artist,
        "track": title,
        "api_key": api_key,
        "format": "json",
    }

    owns_client = client is None
    if owns_client:
        client = create_client()

    try:
        resp = await client.get(_BASE_URL, params=params)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            logger.warning("Last.fm API error %s: %s", data["error"], data.get("message"))
            return None
        return _parse_lastfm_track_response(data)
    except httpx.HTTPError as exc:
        logger.warning("Last.fm request failed: %s", exc)
        return None
    finally:
        if owns_client:
            await client.aclose()
