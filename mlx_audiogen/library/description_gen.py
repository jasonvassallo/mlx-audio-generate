"""Template-based description generation from TrackInfo metadata.

Builds human-readable text prompts from whatever metadata fields are available,
suitable for use as LoRA training labels or as generation prompts.
"""

import statistics
from collections import Counter
from typing import Optional

from .models import TrackInfo


def generate_description(
    track: TrackInfo,
    enrichment: dict | None = None,
) -> str:
    """Build a comma-separated description phrase from available track metadata.

    Priority order:
      1. genre (lowercased)
      2. BPM (e.g. "122 BPM")
      3. key (e.g. "4A")
      4. artist style (e.g. "Jimpster style")
      5. enrichment genres / styles (from MusicBrainz and Discogs)

    If all of those fields are empty / None, falls back to the track title
    (lowercased) or — if that is also empty — the string "instrumental track".

    Args:
        track: A :class:`TrackInfo` instance.
        enrichment: Optional dict with ``musicbrainz``, ``lastfm``, and/or
            ``discogs`` keys, each containing enrichment data dicts (as
            returned by :meth:`EnrichmentDB.get_all_enrichment`).  When
            provided, additional genre/style tags are appended.

    Returns:
        A non-empty description string.
    """
    parts: list[str] = []

    if track.genre:
        parts.append(track.genre.lower())

    if track.bpm is not None:
        # Format as integer BPM when the value is whole, otherwise 1 decimal place
        bpm_str = (
            f"{int(track.bpm)} BPM"
            if track.bpm == int(track.bpm)
            else f"{track.bpm:.1f} BPM"
        )
        parts.append(bpm_str)

    if track.key:
        parts.append(track.key)

    if track.artist:
        parts.append(f"{track.artist} style")

    # Append enrichment tags (genres from MusicBrainz, styles from Discogs)
    if enrichment is not None:
        existing_lower = {p.lower() for p in parts}

        mb = enrichment.get("musicbrainz")
        if mb is not None:
            mb_data = mb.get("data", mb) if isinstance(mb, dict) else {}
            for tag in mb_data.get("tags", []):
                tag_name = tag if isinstance(tag, str) else tag.get("name", "")
                if tag_name and tag_name.lower() not in existing_lower:
                    parts.append(tag_name.lower())
                    existing_lower.add(tag_name.lower())

        dc = enrichment.get("discogs")
        if dc is not None:
            dc_data = dc.get("data", dc) if isinstance(dc, dict) else {}
            for style in dc_data.get("styles", []):
                style_name = style if isinstance(style, str) else str(style)
                if style_name and style_name.lower() not in existing_lower:
                    parts.append(style_name.lower())
                    existing_lower.add(style_name.lower())

    if parts:
        return ", ".join(parts)

    # Fallback: title or generic string
    if track.title:
        return track.title.lower()
    return "instrumental track"


def generate_playlist_prompt(tracks: list[TrackInfo]) -> dict:
    """Analyse a set of tracks and return playlist-level statistics + a prompt.

    Args:
        tracks: List of :class:`TrackInfo` instances (may be empty).

    Returns:
        A dict with the following keys:

        - ``bpm_median`` (float | None): Median BPM across tracks with BPM data.
        - ``bpm_range`` ([min, max] | None): Min/max BPM, or None if no BPM data.
        - ``top_keys`` (list[str]): Up to 3 most-common Camelot keys.
        - ``top_genres`` (list[str]): Up to 3 most-common genres.
        - ``top_artists`` (list[str]): Up to 5 most-common artists.
        - ``year_range`` ([min, max] | None): Earliest/latest release years.
        - ``track_count`` (int): Total number of tracks.
        - ``available_count`` (int): Tracks with ``file_available=True``.
        - ``prompt`` (str): A ready-to-use generation prompt string.
    """
    bpms: list[float] = [t.bpm for t in tracks if t.bpm is not None]
    keys: list[str] = [t.key for t in tracks if t.key]
    genres: list[str] = [t.genre for t in tracks if t.genre]
    artists: list[str] = [t.artist for t in tracks if t.artist]
    years: list[int] = [t.year for t in tracks if t.year is not None]

    bpm_median: Optional[float] = statistics.median(bpms) if bpms else None
    bpm_range: Optional[list] = [min(bpms), max(bpms)] if bpms else None
    year_range: Optional[list] = [min(years), max(years)] if years else None

    top_keys = [k for k, _ in Counter(keys).most_common(3)]
    top_genres = [g for g, _ in Counter(genres).most_common(3)]
    top_artists = [a for a, _ in Counter(artists).most_common(5)]

    available_count = sum(1 for t in tracks if t.file_available)

    # Build a natural-language prompt from the collected stats
    prompt_parts: list[str] = []
    if top_genres:
        prompt_parts.append(top_genres[0].lower())
    if bpm_median is not None:
        bpm_display = (
            f"{int(bpm_median)} BPM"
            if bpm_median == int(bpm_median)
            else f"{bpm_median:.1f} BPM"
        )
        prompt_parts.append(bpm_display)
    if top_keys:
        prompt_parts.append(top_keys[0])
    if top_artists:
        if len(top_artists) == 1:
            prompt_parts.append(f"influenced by {top_artists[0]}")
        else:
            artists_str = " and ".join(top_artists[:2])
            prompt_parts.append(f"influenced by {artists_str}")

    prompt = ", ".join(prompt_parts) if prompt_parts else "instrumental track"

    return {
        "bpm_median": bpm_median,
        "bpm_range": bpm_range,
        "top_keys": top_keys,
        "top_genres": top_genres,
        "top_artists": top_artists,
        "year_range": year_range,
        "track_count": len(tracks),
        "available_count": available_count,
        "prompt": prompt,
    }
