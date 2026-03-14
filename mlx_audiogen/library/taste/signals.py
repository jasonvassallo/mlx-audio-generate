"""Signal collectors for taste profile — library and generation analysis.

These functions aggregate raw track metadata and generation history into
weighted signal dicts that can be applied to a :class:`TasteProfile`.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Optional

from ..models import TrackInfo


def _weighted_tags_from_counter(
    counter: Counter, max_tags: int = 20
) -> list[dict[str, Any]]:
    """Convert a Counter to a sorted list of weighted tag dicts.

    The most common item gets weight 1.0; others are scaled proportionally.
    """
    if not counter:
        return []
    most_common = counter.most_common(max_tags)
    max_count = most_common[0][1]
    if max_count == 0:
        return []
    return [
        {"name": name, "weight": round(count / max_count, 4)}
        for name, count in most_common
    ]


def collect_library_signals(
    tracks: list[TrackInfo],
    enrichment_tags: Optional[dict[str, list[str]]] = None,
) -> dict[str, Any]:
    """Aggregate library tracks into taste signals.

    Genres and artists are weighted by ``play_count``.  BPM range uses
    the 10th and 90th percentiles.  Era distribution buckets years into
    decades.

    Args:
        tracks: List of :class:`TrackInfo` objects.
        enrichment_tags: Optional mapping of track_id -> extra tag strings
            from the enrichment database.

    Returns:
        Dict with keys matching TasteProfile library fields:
        ``top_genres``, ``top_artists``, ``bpm_range``, ``key_preferences``,
        ``era_distribution``, ``mood_profile``, ``style_tags``.
    """
    if not tracks:
        return {
            "top_genres": [],
            "top_artists": [],
            "bpm_range": (0.0, 0.0),
            "key_preferences": [],
            "era_distribution": {},
            "mood_profile": [],
            "style_tags": [],
        }

    genre_counter: Counter = Counter()
    artist_counter: Counter = Counter()
    key_counter: Counter = Counter()
    era_counter: Counter = Counter()
    bpms: list[float] = []

    for t in tracks:
        weight = max(t.play_count, 1)  # minimum weight of 1

        if t.genre:
            genre_counter[t.genre.strip()] += weight

        if t.artist:
            artist_counter[t.artist.strip()] += weight

        if t.key:
            key_counter[t.key.strip()] += weight

        if t.bpm is not None and t.bpm > 0:
            bpms.append(t.bpm)

        if t.year is not None and t.year > 0:
            decade = f"{(t.year // 10) * 10}s"
            era_counter[decade] += weight

        # Include enrichment tags if available
        if enrichment_tags and t.track_id in enrichment_tags:
            for tag in enrichment_tags[t.track_id]:
                genre_counter[tag] += weight

    # BPM range: 10th-90th percentile
    bpm_range = (0.0, 0.0)
    if bpms:
        bpms.sort()
        n = len(bpms)
        idx_lo = max(0, int(n * 0.1))
        idx_hi = min(n - 1, int(n * 0.9))
        bpm_range = (bpms[idx_lo], bpms[idx_hi])

    # Era distribution as proportions
    era_total = sum(era_counter.values())
    era_distribution = {}
    if era_total > 0:
        era_distribution = {
            k: round(v / era_total, 4) for k, v in era_counter.most_common()
        }

    return {
        "top_genres": _weighted_tags_from_counter(genre_counter),
        "top_artists": _weighted_tags_from_counter(artist_counter),
        "bpm_range": bpm_range,
        "key_preferences": _weighted_tags_from_counter(key_counter),
        "era_distribution": era_distribution,
        "mood_profile": [],  # requires enrichment / NLP
        "style_tags": [],  # requires enrichment / NLP
    }


def collect_generation_signals(
    style_profile: dict[str, Any],
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    """Convert prompt memory style profile and history into taste signals.

    Style profile lists are converted to :class:`WeightedTag` dicts with
    exponentially decaying weights (1.0, 0.85, 0.70, ...).

    Args:
        style_profile: Dict with ``genres``, ``moods``, ``instruments`` lists
            (as stored in prompt memory).
        history: List of generation history entries, each with at least
            ``duration_seconds`` and ``model`` keys.

    Returns:
        Dict with keys matching TasteProfile generation fields:
        ``gen_genres``, ``gen_moods``, ``gen_instruments``, ``kept_ratio``,
        ``avg_duration``, ``preferred_models``.
    """
    decay_base = 0.85

    def _to_weighted(items: list[str]) -> list[dict[str, Any]]:
        result = []
        for i, name in enumerate(items):
            weight = round(decay_base ** i, 4)
            result.append({"name": name, "weight": weight})
        return result

    gen_genres = _to_weighted(style_profile.get("genres", []))
    gen_moods = _to_weighted(style_profile.get("moods", []))
    gen_instruments = _to_weighted(style_profile.get("instruments", []))

    # Average duration
    durations = [h.get("duration_seconds", 0.0) for h in history if h.get("duration_seconds")]
    avg_duration = sum(durations) / len(durations) if durations else 0.0

    # Model usage counts
    model_counter: Counter = Counter()
    for h in history:
        model = h.get("model")
        if model:
            model_counter[model] += 1

    preferred_models = _weighted_tags_from_counter(model_counter)

    return {
        "gen_genres": gen_genres,
        "gen_moods": gen_moods,
        "gen_instruments": gen_instruments,
        "kept_ratio": 0.0,  # requires tracking kept/discarded
        "avg_duration": avg_duration,
        "preferred_models": preferred_models,
    }
