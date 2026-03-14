"""Tests for mlx_audiogen.library.cache.LibraryCache."""

from pathlib import Path

import pytest

from mlx_audiogen.library.cache import LibraryCache

# Path to the shared test fixtures
FIXTURES = Path(__file__).parent / "fixtures"
APPLE_MUSIC_XML = str(FIXTURES / "apple_music_sample.xml")
REKORDBOX_XML = str(FIXTURES / "rekordbox_sample.xml")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cache(tmp_path: Path) -> LibraryCache:
    """Create a fresh LibraryCache backed by a temp config dir."""
    return LibraryCache(config_dir=tmp_path)


# ---------------------------------------------------------------------------
# Source management
# ---------------------------------------------------------------------------


class TestAddSource:
    def test_add_source_returns_library_source(self, tmp_path):
        cache = _cache(tmp_path)
        src = cache.add_source("apple_music", APPLE_MUSIC_XML, "My Library")
        assert src.type == "apple_music"
        assert src.path == APPLE_MUSIC_XML
        assert src.label == "My Library"
        assert src.track_count == 0  # not yet scanned
        assert src.last_loaded is None

    def test_add_source_generates_unique_id(self, tmp_path):
        cache = _cache(tmp_path)
        s1 = cache.add_source("apple_music", APPLE_MUSIC_XML, "A")
        s2 = cache.add_source("rekordbox", REKORDBOX_XML, "B")
        assert s1.id != s2.id

    def test_add_source_persisted(self, tmp_path):
        """Source is written to library_sources.json."""
        cache = _cache(tmp_path)
        src = cache.add_source("apple_music", APPLE_MUSIC_XML, "Lib")
        assert (tmp_path / "library_sources.json").exists()
        # Re-loading should find the source
        cache2 = LibraryCache(config_dir=tmp_path)
        sources = cache2.list_sources()
        assert any(s.id == src.id for s in sources)


class TestListSources:
    def test_list_empty(self, tmp_path):
        assert _cache(tmp_path).list_sources() == []

    def test_list_sources(self, tmp_path):
        cache = _cache(tmp_path)
        cache.add_source("apple_music", APPLE_MUSIC_XML, "A")
        cache.add_source("rekordbox", REKORDBOX_XML, "B")
        sources = cache.list_sources()
        assert len(sources) == 2
        labels = {s.label for s in sources}
        assert labels == {"A", "B"}


class TestRemoveSource:
    def test_remove_source(self, tmp_path):
        cache = _cache(tmp_path)
        src = cache.add_source("apple_music", APPLE_MUSIC_XML, "X")
        cache.remove_source(src.id)
        assert not any(s.id == src.id for s in cache.list_sources())

    def test_remove_unknown_raises(self, tmp_path):
        with pytest.raises(KeyError):
            _cache(tmp_path).remove_source("deadbeef")

    def test_remove_evicts_cached_data(self, tmp_path):
        cache = _cache(tmp_path)
        src = cache.add_source("apple_music", APPLE_MUSIC_XML, "X")
        cache.scan(src.id)
        assert cache.get_track_count(src.id) > 0
        cache.remove_source(src.id)
        assert cache.get_track_count(src.id) == 0


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------


class TestScanAppleMusic:
    def test_scan_apple_music(self, tmp_path):
        """3 tracks in fixture → track_count == 3 after scan."""
        cache = _cache(tmp_path)
        src = cache.add_source("apple_music", APPLE_MUSIC_XML, "AM")
        updated = cache.scan(src.id)
        assert updated.track_count == 3
        assert updated.last_loaded is not None

    def test_scan_updates_source_stats(self, tmp_path):
        cache = _cache(tmp_path)
        src = cache.add_source("apple_music", APPLE_MUSIC_XML, "AM")
        assert src.track_count == 0
        cache.scan(src.id)
        assert cache.get_track_count(src.id) == 3

    def test_scan_apple_music_playlists(self, tmp_path):
        cache = _cache(tmp_path)
        src = cache.add_source("apple_music", APPLE_MUSIC_XML, "AM")
        cache.scan(src.id)
        playlists = cache.get_playlists(src.id)
        # Fixture has 3 playlists: Library, DJ Vassallo, Empty Playlist
        assert len(playlists) == 3

    def test_scan_playlist_tracks(self, tmp_path):
        cache = _cache(tmp_path)
        src = cache.add_source("apple_music", APPLE_MUSIC_XML, "AM")
        cache.scan(src.id)
        playlists = cache.get_playlists(src.id)
        # "DJ Vassallo" playlist has 2 tracks
        dj_pl = next(p for p in playlists if "vassallo" in p.id)
        tracks = cache.get_playlist_tracks(src.id, dj_pl.id)
        assert len(tracks) == 2


class TestScanRekordbox:
    def test_scan_rekordbox(self, tmp_path):
        """3 tracks in rekordbox fixture → track_count == 3."""
        cache = _cache(tmp_path)
        src = cache.add_source("rekordbox", REKORDBOX_XML, "RB")
        updated = cache.scan(src.id)
        assert updated.track_count == 3

    def test_scan_rekordbox_playlists(self, tmp_path):
        cache = _cache(tmp_path)
        src = cache.add_source("rekordbox", REKORDBOX_XML, "RB")
        cache.scan(src.id)
        playlists = cache.get_playlists(src.id)
        # Fixture has 2 leaf playlists: My Playlist + Empty
        assert len(playlists) == 2


# ---------------------------------------------------------------------------
# Search tracks
# ---------------------------------------------------------------------------


class TestSearchTracks:
    def _setup(self, tmp_path: Path) -> tuple[LibraryCache, str]:
        cache = _cache(tmp_path)
        src = cache.add_source("rekordbox", REKORDBOX_XML, "RB")
        cache.scan(src.id)
        return cache, src.id

    def test_search_all(self, tmp_path):
        cache, sid = self._setup(tmp_path)
        results = cache.search_tracks(sid, limit=100)
        assert len(results) == 3

    def test_search_by_q(self, tmp_path):
        """q='Deep' should find 'Deep House Anthem'."""
        cache, sid = self._setup(tmp_path)
        results = cache.search_tracks(sid, q="Deep")
        assert len(results) == 1
        assert results[0].title == "Deep House Anthem"

    def test_search_q_case_insensitive(self, tmp_path):
        cache, sid = self._setup(tmp_path)
        results = cache.search_tracks(sid, q="deep")
        assert len(results) == 1

    def test_search_by_genre(self, tmp_path):
        cache, sid = self._setup(tmp_path)
        results = cache.search_tracks(sid, genre="House")
        # "House" matches "House" genre exactly
        assert any(t.title == "Deep House Anthem" for t in results)

    def test_search_by_artist(self, tmp_path):
        cache, sid = self._setup(tmp_path)
        results = cache.search_tracks(sid, artist="Studio")
        assert len(results) == 1
        assert results[0].artist == "Studio Artist"

    def test_search_by_bpm_range(self, tmp_path):
        """BPM range 132-140 should find only 'Tech Minimal' (138 BPM)."""
        cache, sid = self._setup(tmp_path)
        results = cache.search_tracks(sid, bpm_min=132, bpm_max=140)
        assert len(results) == 1
        assert results[0].title == "Tech Minimal"

    def test_search_by_key_exact(self, tmp_path):
        cache, sid = self._setup(tmp_path)
        results = cache.search_tracks(sid, key="8A")
        assert len(results) == 1
        assert results[0].key == "8A"

    def test_search_key_no_partial(self, tmp_path):
        """Key filter is exact — '8' should not match '8A'."""
        cache, sid = self._setup(tmp_path)
        results = cache.search_tracks(sid, key="8")
        assert len(results) == 0

    def test_search_no_match(self, tmp_path):
        cache, sid = self._setup(tmp_path)
        results = cache.search_tracks(sid, q="ZZZ_NONEXISTENT")
        assert results == []

    def test_search_bpm_skips_none(self, tmp_path):
        """Tracks with no BPM are excluded when bpm_min/max is applied."""
        cache = _cache(tmp_path)
        src = cache.add_source("apple_music", APPLE_MUSIC_XML, "AM")
        cache.scan(src.id)
        # Sparse Track in apple_music fixture has no BPM
        results = cache.search_tracks(src.id, bpm_min=50)
        for t in results:
            assert t.bpm is not None

    def test_search_by_available(self, tmp_path):
        cache, sid = self._setup(tmp_path)
        # Rekordbox fixture tracks have non-existent paths → file_available=False
        results = cache.search_tracks(sid, available=False)
        assert len(results) == 3
        results_avail = cache.search_tracks(sid, available=True)
        assert len(results_avail) == 0

    def test_search_pagination_offset(self, tmp_path):
        cache, sid = self._setup(tmp_path)
        all_tracks = cache.search_tracks(sid, limit=100)
        page1 = cache.search_tracks(sid, offset=0, limit=2)
        page2 = cache.search_tracks(sid, offset=2, limit=2)
        combined = page1 + page2
        assert len(combined) == len(all_tracks)
        assert {t.track_id for t in combined} == {t.track_id for t in all_tracks}

    def test_search_limit(self, tmp_path):
        cache, sid = self._setup(tmp_path)
        results = cache.search_tracks(sid, limit=1)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Sort tracks
# ---------------------------------------------------------------------------


class TestSortTracks:
    def _setup(self, tmp_path: Path) -> tuple[LibraryCache, str]:
        cache = _cache(tmp_path)
        src = cache.add_source("rekordbox", REKORDBOX_XML, "RB")
        cache.scan(src.id)
        return cache, src.id

    def test_sort_by_bpm_asc(self, tmp_path):
        cache, sid = self._setup(tmp_path)
        results = cache.search_tracks(sid, sort="bpm", order="asc", limit=100)
        bpms = [t.bpm for t in results if t.bpm is not None]
        assert bpms == sorted(bpms)

    def test_sort_by_bpm_desc(self, tmp_path):
        """Sort by BPM descending; tracks with no BPM sort to end."""
        cache, sid = self._setup(tmp_path)
        results = cache.search_tracks(sid, sort="bpm", order="desc", limit=100)
        bpms_no_none = [t.bpm for t in results if t.bpm is not None]
        assert bpms_no_none == sorted(bpms_no_none, reverse=True)
        # None BPM tracks appear after non-None tracks
        for i, t in enumerate(results):
            if t.bpm is None:
                remaining = results[i:]
                assert all(r.bpm is None for r in remaining)

    def test_sort_by_title_asc(self, tmp_path):
        cache, sid = self._setup(tmp_path)
        results = cache.search_tracks(sid, sort="title", order="asc", limit=100)
        titles = [t.title.lower() for t in results]
        assert titles == sorted(titles)

    def test_sort_by_title_desc(self, tmp_path):
        cache, sid = self._setup(tmp_path)
        results = cache.search_tracks(sid, sort="title", order="desc", limit=100)
        titles = [t.title.lower() for t in results]
        assert titles == sorted(titles, reverse=True)

    def test_sort_none_values_last(self, tmp_path):
        """None BPM tracks always sort last regardless of order."""
        cache = _cache(tmp_path)
        src = cache.add_source("apple_music", APPLE_MUSIC_XML, "AM")
        cache.scan(src.id)
        for order in ("asc", "desc"):
            results = cache.search_tracks(src.id, sort="bpm", order=order, limit=100)
            seen_none = False
            for t in results:
                if t.bpm is None:
                    seen_none = True
                elif seen_none:
                    pytest.fail(f"Non-None BPM after None-BPM track (order={order})")


# ---------------------------------------------------------------------------
# Persistence across instances
# ---------------------------------------------------------------------------


class TestPersistSources:
    def test_persist_sources(self, tmp_path):
        """Sources added to one instance are visible in a new instance."""
        cache1 = LibraryCache(config_dir=tmp_path)
        src = cache1.add_source("apple_music", APPLE_MUSIC_XML, "Persistent")

        cache2 = LibraryCache(config_dir=tmp_path)
        sources = cache2.list_sources()
        assert any(s.id == src.id and s.label == "Persistent" for s in sources)

    def test_scan_stats_persisted(self, tmp_path):
        """After scanning, track_count is visible in a freshly loaded instance."""
        cache1 = LibraryCache(config_dir=tmp_path)
        src = cache1.add_source("apple_music", APPLE_MUSIC_XML, "Lib")
        cache1.scan(src.id)

        cache2 = LibraryCache(config_dir=tmp_path)
        sources = cache2.list_sources()
        loaded = next(s for s in sources if s.id == src.id)
        assert loaded.track_count == 3
        assert loaded.last_loaded is not None
