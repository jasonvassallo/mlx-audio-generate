"""Integration tests for library scanner with real XML files.

These require the user's actual library exports on disk.
Run with: pytest -m integration -v
"""

from pathlib import Path

import pytest

from mlx_audiogen.library.cache import LibraryCache

APPLE_MUSIC_XML = Path.home() / "Music" / "Media" / "Library.xml"
REKORDBOX_XML = Path.home() / "Documents" / "rekordbox" / "rekordbox.xml"


@pytest.mark.integration
@pytest.mark.skipif(not APPLE_MUSIC_XML.exists(), reason="Apple Music XML not found")
def test_parse_real_apple_music(tmp_path: Path):
    cache = LibraryCache(config_dir=tmp_path)
    src = cache.add_source("apple_music", str(APPLE_MUSIC_XML), "Apple Music")
    cache.scan(src.id)
    assert cache.get_track_count(src.id) > 100
    playlists = cache.get_playlists(src.id)
    assert len(playlists) > 5


@pytest.mark.integration
@pytest.mark.skipif(not REKORDBOX_XML.exists(), reason="rekordbox XML not found")
def test_parse_real_rekordbox(tmp_path: Path):
    cache = LibraryCache(config_dir=tmp_path)
    src = cache.add_source("rekordbox", str(REKORDBOX_XML), "rekordbox")
    cache.scan(src.id)
    assert cache.get_track_count(src.id) > 100
    playlists = cache.get_playlists(src.id)
    assert len(playlists) > 5
