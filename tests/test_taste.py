"""Tests for mlx_audiogen.library.taste — profile, signals, and engine."""

import json
import os
import tempfile

import pytest

from mlx_audiogen.library.models import TrackInfo
from mlx_audiogen.library.taste.profile import TasteProfile, WeightedTag
from mlx_audiogen.library.taste.signals import (
    collect_generation_signals,
    collect_library_signals,
)
from mlx_audiogen.library.taste.engine import TasteEngine


def _make_track(**kwargs) -> TrackInfo:
    defaults = {
        "track_id": "1",
        "title": "Test",
        "artist": "Artist",
        "album": "",
        "genre": "",
        "bpm": None,
        "key": None,
        "year": None,
        "rating": 0,
        "play_count": 0,
        "duration_seconds": 0,
        "comments": "",
        "file_path": None,
        "file_available": False,
        "source": "apple_music",
        "loved": False,
        "description": "",
        "description_edited": False,
    }
    defaults.update(kwargs)
    return TrackInfo(**defaults)


# ===========================================================================
# TasteProfile tests (4 tests)
# ===========================================================================


class TestTasteProfile:
    def test_create_empty(self):
        """TasteProfile.empty() returns a profile with zero counts."""
        p = TasteProfile.empty()
        assert p.library_track_count == 0
        assert p.generation_count == 0
        assert p.top_genres == []
        assert p.gen_genres == []
        assert p.version == 1

    def test_to_dict_roundtrip(self):
        """to_dict -> from_dict preserves all fields."""
        p = TasteProfile.empty()
        p.top_genres = [WeightedTag("house", 0.9), WeightedTag("techno", 0.7)]
        p.bpm_range = (120.0, 130.0)
        p.overrides = "prefer deep house"

        d = p.to_dict()
        p2 = TasteProfile.from_dict(d)
        assert len(p2.top_genres) == 2
        assert p2.top_genres[0].name == "house"
        assert p2.top_genres[0].weight == 0.9
        assert p2.bpm_range == (120.0, 130.0)
        assert p2.overrides == "prefer deep house"

    def test_save_and_load(self, tmp_path):
        """Profile saves to JSON and loads back correctly."""
        path = str(tmp_path / "taste.json")
        p = TasteProfile.empty()
        p.top_genres = [WeightedTag("ambient", 1.0)]
        p.library_track_count = 42
        p.save(path)

        p2 = TasteProfile.load(path)
        assert p2.library_track_count == 42
        assert len(p2.top_genres) == 1
        assert p2.top_genres[0].name == "ambient"

    def test_load_missing_returns_empty(self, tmp_path):
        """Loading from a non-existent path returns an empty profile."""
        path = str(tmp_path / "nonexistent.json")
        p = TasteProfile.load(path)
        assert p.library_track_count == 0
        assert p.top_genres == []


# ===========================================================================
# Library signals tests (3 tests)
# ===========================================================================


class TestLibrarySignals:
    def test_genre_weighting(self):
        """Genres are weighted by play_count."""
        tracks = [
            _make_track(track_id="1", genre="House", play_count=10),
            _make_track(track_id="2", genre="House", play_count=5),
            _make_track(track_id="3", genre="Techno", play_count=3),
        ]
        signals = collect_library_signals(tracks)
        # House should be top genre (higher total play_count)
        assert len(signals["top_genres"]) > 0
        top_genre = signals["top_genres"][0]
        assert top_genre["name"].lower() == "house"

    def test_bpm_range(self):
        """BPM range uses 10th-90th percentile."""
        tracks = [
            _make_track(track_id=str(i), bpm=float(bpm), play_count=1)
            for i, bpm in enumerate(range(100, 200, 5))  # 100,105,...195 = 20 tracks
        ]
        signals = collect_library_signals(tracks)
        lo, hi = signals["bpm_range"]
        # 10th percentile of 100-195 ~ 110, 90th ~ 185
        assert lo >= 100.0
        assert hi <= 200.0
        assert lo < hi

    def test_empty_tracks(self):
        """Empty track list returns empty signals."""
        signals = collect_library_signals([])
        assert signals["top_genres"] == []
        assert signals["bpm_range"] == (0.0, 0.0)
        assert signals["top_artists"] == []


# ===========================================================================
# Generation signals tests (1 test)
# ===========================================================================


class TestGenerationSignals:
    def test_from_prompt_memory(self):
        """Converts prompt memory style_profile to weighted tags with decay."""
        style_profile = {
            "genres": ["house", "techno", "ambient"],
            "moods": ["energetic", "dark"],
            "instruments": ["synthesizer"],
        }
        history = [
            {"duration_seconds": 10.0, "model": "musicgen"},
            {"duration_seconds": 20.0, "model": "musicgen"},
            {"duration_seconds": 15.0, "model": "stable_audio"},
        ]
        signals = collect_generation_signals(style_profile, history)
        # Genres should have decaying weights: 1.0, 0.85, 0.70...
        assert len(signals["gen_genres"]) == 3
        assert signals["gen_genres"][0]["weight"] == 1.0
        assert abs(signals["gen_genres"][1]["weight"] - 0.85) < 0.01
        # avg_duration
        assert abs(signals["avg_duration"] - 15.0) < 0.01
        # preferred_models
        assert "musicgen" in [m["name"] for m in signals["preferred_models"]]


# ===========================================================================
# TasteEngine tests (4 tests)
# ===========================================================================


class TestTasteEngine:
    def test_compute_from_library(self, tmp_path):
        """update_library_signals populates profile with library data."""
        path = str(tmp_path / "taste.json")
        engine = TasteEngine(profile_path=path)
        tracks = [
            _make_track(track_id="1", genre="House", play_count=10, bpm=128.0),
            _make_track(track_id="2", genre="Techno", play_count=5, bpm=135.0),
        ]
        engine.update_library_signals(tracks)
        p = engine.get_profile()
        assert p.library_track_count == 2
        assert len(p.top_genres) > 0

    def test_compute_from_generation(self, tmp_path):
        """update_generation_signals populates profile with gen data."""
        path = str(tmp_path / "taste.json")
        engine = TasteEngine(profile_path=path)
        style_profile = {"genres": ["house"], "moods": ["chill"], "instruments": []}
        history = [{"duration_seconds": 10.0, "model": "musicgen"}]
        engine.update_generation_signals(style_profile, history)
        p = engine.get_profile()
        assert p.generation_count == 1
        assert len(p.gen_genres) > 0

    def test_set_overrides(self, tmp_path):
        """set_overrides stores user override text in profile."""
        path = str(tmp_path / "taste.json")
        engine = TasteEngine(profile_path=path)
        engine.set_overrides("I prefer deep house over tech house")
        p = engine.get_profile()
        assert p.overrides == "I prefer deep house over tech house"

    def test_persists_to_disk(self, tmp_path):
        """Profile is saved to disk after each update."""
        path = str(tmp_path / "taste.json")
        engine = TasteEngine(profile_path=path)
        tracks = [_make_track(track_id="1", genre="House", play_count=5)]
        engine.update_library_signals(tracks)

        # Load from disk directly
        with open(path) as f:
            data = json.load(f)
        assert data["library_track_count"] == 1
