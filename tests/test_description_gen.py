"""Tests for mlx_audiogen.library.description_gen."""

from mlx_audiogen.library.description_gen import (
    generate_description,
    generate_playlist_prompt,
)
from mlx_audiogen.library.models import TrackInfo

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_track(**kwargs) -> TrackInfo:
    defaults = dict(
        track_id="1",
        title="Test Track",
        artist="Jimpster",
        album="Test Album",
        genre="Deep House",
        bpm=122.0,
        key="4A",
        year=2020,
        rating=80,
        play_count=5,
        duration_seconds=360.0,
        comments="",
        file_path="/tmp/test.wav",
        file_available=True,
        source="apple_music",
        loved=False,
        description="",
        description_edited=False,
    )
    defaults.update(kwargs)
    return TrackInfo(**defaults)


# ---------------------------------------------------------------------------
# generate_description
# ---------------------------------------------------------------------------


class TestGenerateDescription:
    def test_full_metadata(self):
        """All fields present → description contains genre, BPM, key."""
        track = _make_track(genre="Deep House", bpm=122.0, key="4A", artist="Jimpster")
        desc = generate_description(track)
        assert "deep house" in desc
        assert "122 BPM" in desc
        assert "4A" in desc

    def test_missing_bpm(self):
        """No BPM → 'BPM' not in output but other fields still appear."""
        track = _make_track(bpm=None)
        desc = generate_description(track)
        assert "BPM" not in desc
        # genre and key should still be present
        assert "deep house" in desc
        assert "4A" in desc

    def test_missing_key(self):
        """No key → description has genre + BPM but no key."""
        track = _make_track(key=None)
        desc = generate_description(track)
        assert "deep house" in desc
        assert "122 BPM" in desc
        # Key should not appear
        assert "4A" not in desc

    def test_missing_genre(self):
        """Empty genre → description is still non-empty (BPM + key)."""
        track = _make_track(genre="")
        desc = generate_description(track)
        assert desc  # non-empty
        assert "122 BPM" in desc

    def test_all_missing(self):
        """All metadata empty → still returns a non-empty string."""
        track = _make_track(genre="", bpm=None, key=None, artist="")
        desc = generate_description(track)
        assert desc  # non-empty

    def test_all_missing_uses_title(self):
        """No genre/BPM/key/artist → falls back to lowercased title."""
        track = _make_track(genre="", bpm=None, key=None, artist="", title="My Track")
        desc = generate_description(track)
        assert desc == "my track"

    def test_all_missing_no_title(self):
        """No metadata AND no title → returns 'instrumental track'."""
        track = _make_track(genre="", bpm=None, key=None, artist="", title="")
        desc = generate_description(track)
        assert desc == "instrumental track"

    def test_integer_bpm_display(self):
        """Whole BPM values are displayed without decimals."""
        track = _make_track(bpm=128.0)
        desc = generate_description(track)
        assert "128 BPM" in desc
        # Should NOT have '128.0 BPM'
        assert "128.0 BPM" not in desc

    def test_fractional_bpm_display(self):
        """Fractional BPM values keep one decimal place."""
        track = _make_track(bpm=124.5)
        desc = generate_description(track)
        assert "124.5 BPM" in desc

    def test_artist_style_suffix(self):
        """Artist name gets ' style' appended."""
        track = _make_track(artist="Shur-i-kan")
        desc = generate_description(track)
        assert "Shur-i-kan style" in desc


# ---------------------------------------------------------------------------
# generate_playlist_prompt
# ---------------------------------------------------------------------------


class TestGeneratePlaylistPrompt:
    def _make_tracks(self) -> list[TrackInfo]:
        """Return 3 tracks covering the standard fixture data."""
        return [
            _make_track(
                track_id="1",
                genre="Deep House",
                bpm=122.0,
                key="4A",
                artist="Jimpster",
                year=2020,
                file_available=True,
            ),
            _make_track(
                track_id="2",
                genre="Deep House",
                bpm=126.0,
                key="4A",
                artist="Shur-i-kan",
                year=2021,
                file_available=True,
            ),
            _make_track(
                track_id="3",
                genre="Acid Techno",
                bpm=135.0,
                key="8A",
                artist="Surgeon",
                year=2019,
                file_available=False,
            ),
        ]

    def test_generate_playlist_prompt_returns_dict(self):
        """generate_playlist_prompt returns a dict with all expected keys."""
        tracks = self._make_tracks()
        result = generate_playlist_prompt(tracks)
        assert isinstance(result, dict)
        expected_keys = {
            "bpm_median",
            "bpm_range",
            "top_keys",
            "top_genres",
            "top_artists",
            "year_range",
            "track_count",
            "available_count",
            "prompt",
        }
        assert set(result.keys()) == expected_keys

    def test_track_count_and_available_count(self):
        tracks = self._make_tracks()
        result = generate_playlist_prompt(tracks)
        assert result["track_count"] == 3
        assert result["available_count"] == 2  # one track has file_available=False

    def test_bpm_median(self):
        tracks = self._make_tracks()
        result = generate_playlist_prompt(tracks)
        # sorted bpms: [122, 126, 135] → median = 126
        assert result["bpm_median"] == 126.0

    def test_bpm_range(self):
        tracks = self._make_tracks()
        result = generate_playlist_prompt(tracks)
        assert result["bpm_range"] == [122.0, 135.0]

    def test_top_keys(self):
        tracks = self._make_tracks()
        result = generate_playlist_prompt(tracks)
        # "4A" appears twice, "8A" once
        assert result["top_keys"][0] == "4A"
        assert "8A" in result["top_keys"]

    def test_top_genres(self):
        tracks = self._make_tracks()
        result = generate_playlist_prompt(tracks)
        assert result["top_genres"][0] == "Deep House"
        assert "Acid Techno" in result["top_genres"]

    def test_top_artists(self):
        tracks = self._make_tracks()
        result = generate_playlist_prompt(tracks)
        # All 3 appear once; ordering is arbitrary but all should be present
        assert set(result["top_artists"]) == {"Jimpster", "Shur-i-kan", "Surgeon"}

    def test_year_range(self):
        tracks = self._make_tracks()
        result = generate_playlist_prompt(tracks)
        assert result["year_range"] == [2019, 2021]

    def test_prompt_non_empty(self):
        tracks = self._make_tracks()
        result = generate_playlist_prompt(tracks)
        assert result["prompt"]  # non-empty string

    def test_prompt_contains_genre(self):
        tracks = self._make_tracks()
        result = generate_playlist_prompt(tracks)
        # Top genre is "Deep House" → should appear lowercased in prompt
        assert "deep house" in result["prompt"].lower()

    def test_empty_track_list(self):
        """Empty list → defaults with None values and 'instrumental track' prompt."""
        result = generate_playlist_prompt([])
        assert result["bpm_median"] is None
        assert result["bpm_range"] is None
        assert result["year_range"] is None
        assert result["top_keys"] == []
        assert result["top_genres"] == []
        assert result["top_artists"] == []
        assert result["track_count"] == 0
        assert result["available_count"] == 0
        assert result["prompt"] == "instrumental track"

    def test_no_bpm_data(self):
        """Tracks without BPM → bpm_median and bpm_range are None."""
        tracks = [_make_track(track_id=str(i), bpm=None) for i in range(3)]
        result = generate_playlist_prompt(tracks)
        assert result["bpm_median"] is None
        assert result["bpm_range"] is None

    def test_single_artist_prompt(self):
        """Single artist uses 'influenced by <artist>' (no 'and')."""
        tracks = [_make_track(track_id=str(i), artist="Jimpster") for i in range(2)]
        result = generate_playlist_prompt(tracks)
        assert "influenced by Jimpster" in result["prompt"]
        assert " and " not in result["prompt"]

    def test_two_artists_prompt(self):
        """Two distinct artists uses 'influenced by X and Y'."""
        tracks = [
            _make_track(track_id="1", artist="Jimpster"),
            _make_track(track_id="2", artist="Shur-i-kan"),
        ]
        result = generate_playlist_prompt(tracks)
        assert "influenced by" in result["prompt"]
        assert " and " in result["prompt"]
