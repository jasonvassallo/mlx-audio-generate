"""TasteProfile and WeightedTag dataclasses for personal preference modeling.

A :class:`TasteProfile` captures a user's musical preferences derived from
their library listening history and generation activity.  It persists as a
JSON file at ``~/.mlx-audiogen/taste_profile.json`` by default.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


_DEFAULT_PATH = str(Path.home() / ".mlx-audiogen" / "taste_profile.json")


@dataclass
class WeightedTag:
    """A tag name with an associated weight (0.0 - 1.0)."""

    name: str
    weight: float

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "weight": self.weight}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WeightedTag:
        return cls(name=d["name"], weight=d["weight"])


@dataclass
class TasteProfile:
    """Aggregated user taste profile combining library and generation signals.

    Fields are grouped into four categories:

    - **Library**: derived from the user's music library (genres, artists, BPM, etc.)
    - **Generation**: derived from prompt memory and generation history
    - **Metadata**: bookkeeping (counts, timestamps, version)
    - **User**: manual overrides
    """

    # -- Library signals --
    top_genres: list[WeightedTag] = field(default_factory=list)
    top_artists: list[WeightedTag] = field(default_factory=list)
    bpm_range: tuple[float, float] = (0.0, 0.0)
    key_preferences: list[WeightedTag] = field(default_factory=list)
    era_distribution: dict[str, float] = field(default_factory=dict)
    mood_profile: list[WeightedTag] = field(default_factory=list)
    style_tags: list[WeightedTag] = field(default_factory=list)

    # -- Generation signals --
    gen_genres: list[WeightedTag] = field(default_factory=list)
    gen_moods: list[WeightedTag] = field(default_factory=list)
    gen_instruments: list[WeightedTag] = field(default_factory=list)
    kept_ratio: float = 0.0
    avg_duration: float = 0.0
    preferred_models: list[WeightedTag] = field(default_factory=list)

    # -- Metadata --
    library_track_count: int = 0
    generation_count: int = 0
    last_updated: str = ""
    version: int = 1

    # -- User --
    overrides: str = ""

    @classmethod
    def empty(cls) -> TasteProfile:
        """Create an empty profile with all defaults."""
        return cls()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the profile to a JSON-compatible dict."""
        return {
            "top_genres": [t.to_dict() for t in self.top_genres],
            "top_artists": [t.to_dict() for t in self.top_artists],
            "bpm_range": list(self.bpm_range),
            "key_preferences": [t.to_dict() for t in self.key_preferences],
            "era_distribution": self.era_distribution,
            "mood_profile": [t.to_dict() for t in self.mood_profile],
            "style_tags": [t.to_dict() for t in self.style_tags],
            "gen_genres": [t.to_dict() for t in self.gen_genres],
            "gen_moods": [t.to_dict() for t in self.gen_moods],
            "gen_instruments": [t.to_dict() for t in self.gen_instruments],
            "kept_ratio": self.kept_ratio,
            "avg_duration": self.avg_duration,
            "preferred_models": [t.to_dict() for t in self.preferred_models],
            "library_track_count": self.library_track_count,
            "generation_count": self.generation_count,
            "last_updated": self.last_updated,
            "version": self.version,
            "overrides": self.overrides,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TasteProfile:
        """Deserialize a profile from a dict."""

        def _tags(key: str) -> list[WeightedTag]:
            return [WeightedTag.from_dict(t) for t in d.get(key, [])]

        bpm = d.get("bpm_range", [0.0, 0.0])

        return cls(
            top_genres=_tags("top_genres"),
            top_artists=_tags("top_artists"),
            bpm_range=(float(bpm[0]), float(bpm[1])),
            key_preferences=_tags("key_preferences"),
            era_distribution=d.get("era_distribution", {}),
            mood_profile=_tags("mood_profile"),
            style_tags=_tags("style_tags"),
            gen_genres=_tags("gen_genres"),
            gen_moods=_tags("gen_moods"),
            gen_instruments=_tags("gen_instruments"),
            kept_ratio=d.get("kept_ratio", 0.0),
            avg_duration=d.get("avg_duration", 0.0),
            preferred_models=_tags("preferred_models"),
            library_track_count=d.get("library_track_count", 0),
            generation_count=d.get("generation_count", 0),
            last_updated=d.get("last_updated", ""),
            version=d.get("version", 1),
            overrides=d.get("overrides", ""),
        )

    def save(self, path: Optional[str] = None) -> None:
        """Save the profile to a JSON file.

        Args:
            path: File path.  Defaults to ``~/.mlx-audiogen/taste_profile.json``.
        """
        if path is None:
            path = _DEFAULT_PATH
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.last_updated = datetime.now(timezone.utc).isoformat()
        p.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Optional[str] = None) -> TasteProfile:
        """Load a profile from a JSON file.

        Returns an empty profile if the file does not exist.

        Args:
            path: File path.  Defaults to ``~/.mlx-audiogen/taste_profile.json``.
        """
        if path is None:
            path = _DEFAULT_PATH
        p = Path(path)
        if not p.exists():
            return cls.empty()
        data = json.loads(p.read_text())
        return cls.from_dict(data)
