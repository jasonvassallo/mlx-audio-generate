"""Tests for mlx_audiogen.library.enrichment — SQLite cache + rate limiter."""

import asyncio
import time

import pytest

from mlx_audiogen.library.enrichment.enrichment_db import EnrichmentDB
from mlx_audiogen.library.enrichment.rate_limiter import ApiRateLimiter

# ===========================================================================
# EnrichmentDB tests (15 tests, in-memory SQLite)
# ===========================================================================


class TestEnrichmentDB:
    @pytest.fixture()
    def db(self):
        """Fresh in-memory enrichment database for each test."""
        return EnrichmentDB(":memory:")

    # -- get_or_create_track -------------------------------------------------

    def test_get_or_create_track(self, db):
        """Creates a new track and returns its row id."""
        track_id = db.get_or_create_track("Daft Punk", "Around the World")
        assert isinstance(track_id, int)
        assert track_id > 0

    def test_dedup_normalization(self, db):
        """Same artist+title with different casing/whitespace returns same id."""
        id1 = db.get_or_create_track("Daft Punk", "Around the World")
        id2 = db.get_or_create_track("  daft punk ", " around the world  ")
        assert id1 == id2

    def test_library_id_mapping(self, db):
        """get_or_create_track stores library source and track id."""
        track_id = db.get_or_create_track(
            "Daft Punk", "Da Funk",
            library_source="apple_music",
            library_track_id="AM-1234",
        )
        found = db.find_by_library_id("apple_music", "AM-1234")
        assert found == track_id

    # -- store/get all 3 sources --------------------------------------------

    def test_store_get_musicbrainz(self, db):
        """Store and retrieve MusicBrainz data."""
        tid = db.get_or_create_track("Artist", "Song")
        data = {"mbid": "abc-123", "tags": ["electronic"]}
        db.store_musicbrainz(tid, data)
        result = db.get_musicbrainz(tid)
        assert result is not None
        assert result["data"]["mbid"] == "abc-123"

    def test_store_get_lastfm(self, db):
        """Store and retrieve Last.fm data."""
        tid = db.get_or_create_track("Artist", "Song")
        data = {"listeners": 50000, "top_tags": ["dance"]}
        db.store_lastfm(tid, data)
        result = db.get_lastfm(tid)
        assert result is not None
        assert result["data"]["listeners"] == 50000

    def test_store_get_discogs(self, db):
        """Store and retrieve Discogs data."""
        tid = db.get_or_create_track("Artist", "Song")
        data = {"release_id": 999, "year": 1997}
        db.store_discogs(tid, data)
        result = db.get_discogs(tid)
        assert result is not None
        assert result["data"]["year"] == 1997

    # -- missing returns None -----------------------------------------------

    def test_missing_returns_none(self, db):
        """get_* returns None for tracks with no enrichment data."""
        tid = db.get_or_create_track("New Artist", "New Song")
        assert db.get_musicbrainz(tid) is None
        assert db.get_lastfm(tid) is None
        assert db.get_discogs(tid) is None

    # -- enrichment status --------------------------------------------------

    def test_enrichment_status_none(self, db):
        """Status is 'none' when no sources are stored."""
        tid = db.get_or_create_track("A", "B")
        assert db.get_enrichment_status(tid) == "none"

    def test_enrichment_status_partial(self, db):
        """Status is 'partial' when some but not all sources are stored."""
        tid = db.get_or_create_track("A", "B")
        db.store_musicbrainz(tid, {"mbid": "x"})
        assert db.get_enrichment_status(tid) == "partial"

    def test_enrichment_status_complete(self, db):
        """Status is 'complete' when all 3 sources are stored."""
        tid = db.get_or_create_track("A", "B")
        db.store_musicbrainz(tid, {"mbid": "x"})
        db.store_lastfm(tid, {"listeners": 1})
        db.store_discogs(tid, {"year": 2020})
        assert db.get_enrichment_status(tid) == "complete"

    # -- is_stale -----------------------------------------------------------

    def test_is_stale(self, db):
        """is_stale returns True when data is older than ttl_days."""
        tid = db.get_or_create_track("A", "B")
        db.store_musicbrainz(tid, {"mbid": "x"})
        # Not stale with default 90 days
        assert db.is_stale(tid, "musicbrainz", ttl_days=90) is False
        # Stale with 0 days (anything stored is immediately stale)
        assert db.is_stale(tid, "musicbrainz", ttl_days=0) is True
        # Missing source is always stale
        assert db.is_stale(tid, "lastfm", ttl_days=90) is True

    # -- update preserves other sources -------------------------------------

    def test_update_preserves_other_sources(self, db):
        """Updating one source does not affect other sources."""
        tid = db.get_or_create_track("A", "B")
        db.store_musicbrainz(tid, {"mbid": "x"})
        db.store_lastfm(tid, {"listeners": 100})

        # Update musicbrainz
        db.store_musicbrainz(tid, {"mbid": "y"})

        # lastfm should be untouched
        lfm = db.get_lastfm(tid)
        assert lfm is not None
        assert lfm["data"]["listeners"] == 100

        # musicbrainz should be updated
        mb = db.get_musicbrainz(tid)
        assert mb is not None
        assert mb["data"]["mbid"] == "y"

    # -- get_stats ----------------------------------------------------------

    def test_get_stats(self, db):
        """get_stats returns counts of tracks and enriched sources."""
        tid1 = db.get_or_create_track("A", "Song1")
        tid2 = db.get_or_create_track("B", "Song2")
        db.store_musicbrainz(tid1, {"mbid": "x"})
        db.store_lastfm(tid1, {"listeners": 1})
        db.store_discogs(tid1, {"year": 2020})
        db.store_musicbrainz(tid2, {"mbid": "y"})

        stats = db.get_stats()
        assert stats["total_tracks"] == 2
        assert stats["musicbrainz"] == 2
        assert stats["lastfm"] == 1
        assert stats["discogs"] == 1

    # -- find_by_library_id missing -----------------------------------------

    def test_find_by_library_id_missing(self, db):
        """find_by_library_id returns None for unknown library ids."""
        assert db.find_by_library_id("apple_music", "nonexistent") is None

    # -- get_all_enrichment -------------------------------------------------

    def test_get_all_enrichment(self, db):
        """get_all_enrichment returns all stored sources for a track."""
        tid = db.get_or_create_track("A", "B")
        db.store_musicbrainz(tid, {"mbid": "x"})
        db.store_lastfm(tid, {"listeners": 1})

        result = db.get_all_enrichment(tid)
        assert result["musicbrainz"] is not None
        assert result["lastfm"] is not None
        assert result["discogs"] is None


# ===========================================================================
# Rate limiter tests (2 tests)
# ===========================================================================


class TestApiRateLimiter:
    def test_allows_within_limit(self):
        """A single acquire should complete without delay."""
        limiter = ApiRateLimiter(max_per_second=10.0)

        async def _run():
            t0 = time.monotonic()
            await limiter.acquire()
            elapsed = time.monotonic() - t0
            # Should be nearly instant
            assert elapsed < 0.05

        asyncio.run(_run())

    def test_tracks_separate_apis(self):
        """Separate limiter instances do not interfere with each other."""
        mb_limiter = ApiRateLimiter(max_per_second=1.0)
        lfm_limiter = ApiRateLimiter(max_per_second=5.0)

        async def _run():
            # First acquire on each should be instant
            await mb_limiter.acquire()
            await lfm_limiter.acquire()

            # Second acquire on mb_limiter should be delayed (~1s)
            # Second acquire on lfm_limiter should be much faster (~0.2s)
            t0 = time.monotonic()
            await lfm_limiter.acquire()
            lfm_elapsed = time.monotonic() - t0

            t1 = time.monotonic()
            await mb_limiter.acquire()
            mb_elapsed = time.monotonic() - t1

            # lfm should be faster than mb
            assert lfm_elapsed < mb_elapsed
            # mb should take roughly 1 second (use 0.7 for timing tolerance)
            assert mb_elapsed >= 0.7

        asyncio.run(_run())
