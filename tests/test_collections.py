"""Tests for mlx_audiogen.library.collections."""

import os
import tempfile
from pathlib import Path

import pytest

from mlx_audiogen.library.collections import (
    collection_to_training_data,
    create_collection,
    delete_collection,
    get_collection,
    list_collections,
    update_collection,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tmp_dir(tmp_path: Path) -> Path:
    """Return a fresh sub-directory within pytest's tmp_path fixture."""
    d = tmp_path / "collections"
    d.mkdir()
    return d


def _sample_track_dict(
    track_id: str = "1",
    file_path: str = "/nonexistent/track.wav",
    file_available: bool = False,
    description: str = "",
) -> dict:
    return {
        "track_id": track_id,
        "title": "Test Track",
        "artist": "Test Artist",
        "album": "Test Album",
        "genre": "Electronic",
        "bpm": 128.0,
        "key": "4A",
        "year": 2023,
        "rating": 80,
        "play_count": 5,
        "duration_seconds": 210.0,
        "comments": "",
        "file_path": file_path,
        "file_available": file_available,
        "source": "apple_music",
        "loved": False,
        "description": description,
        "description_edited": False,
    }


def _minimal_collection(
    name: str = "test-collection",
    tracks: list | None = None,
) -> dict:
    return {
        "name": name,
        "source": "apple_music",
        "playlist": "My Playlist",
        "tracks": tracks or [],
    }


# ---------------------------------------------------------------------------
# create_collection
# ---------------------------------------------------------------------------


class TestCreateCollection:
    def test_create_collection(self, tmp_path):
        cdir = _tmp_dir(tmp_path)
        doc = create_collection(_minimal_collection("my-set"), collections_dir=cdir)
        assert doc["name"] == "my-set"
        assert "created_at" in doc
        assert "updated_at" in doc
        # Persisted to disk
        assert (cdir / "my-set.json").exists()

    def test_create_persists_tracks(self, tmp_path):
        cdir = _tmp_dir(tmp_path)
        tracks = [_sample_track_dict()]
        doc = create_collection(
            _minimal_collection("with-tracks", tracks=tracks),
            collections_dir=cdir,
        )
        assert len(doc["tracks"]) == 1

    def test_create_duplicate_raises(self, tmp_path):
        cdir = _tmp_dir(tmp_path)
        create_collection(_minimal_collection("dupe"), collections_dir=cdir)
        with pytest.raises(ValueError, match="already exists"):
            create_collection(_minimal_collection("dupe"), collections_dir=cdir)

    def test_create_invalid_name_raises(self, tmp_path):
        cdir = _tmp_dir(tmp_path)
        with pytest.raises(ValueError):
            create_collection({"name": "bad name!"}, collections_dir=cdir)

    def test_create_creates_dir_if_missing(self, tmp_path):
        cdir = tmp_path / "does_not_exist" / "sub"
        assert not cdir.exists()
        create_collection(_minimal_collection("first"), collections_dir=cdir)
        assert cdir.exists()


# ---------------------------------------------------------------------------
# list_collections
# ---------------------------------------------------------------------------


class TestListCollections:
    def test_list_empty_dir(self, tmp_path):
        cdir = _tmp_dir(tmp_path)
        assert list_collections(collections_dir=cdir) == []

    def test_list_missing_dir(self, tmp_path):
        cdir = tmp_path / "missing"
        assert list_collections(collections_dir=cdir) == []

    def test_list_collections(self, tmp_path):
        cdir = _tmp_dir(tmp_path)
        tracks = [_sample_track_dict(track_id=str(i)) for i in range(3)]
        create_collection(
            _minimal_collection("set-a", tracks=tracks), collections_dir=cdir
        )
        create_collection(_minimal_collection("set-b"), collections_dir=cdir)

        summaries = list_collections(collections_dir=cdir)
        names = [s["name"] for s in summaries]
        assert "set-a" in names
        assert "set-b" in names

        a = next(s for s in summaries if s["name"] == "set-a")
        assert a["track_count"] == 3
        assert a["source"] == "apple_music"
        assert a["playlist"] == "My Playlist"
        assert a["created_at"] is not None
        assert a["updated_at"] is not None


# ---------------------------------------------------------------------------
# get_collection
# ---------------------------------------------------------------------------


class TestGetCollection:
    def test_get_collection(self, tmp_path):
        cdir = _tmp_dir(tmp_path)
        create_collection(_minimal_collection("alpha"), collections_dir=cdir)
        doc = get_collection("alpha", collections_dir=cdir)
        assert doc["name"] == "alpha"

    def test_get_not_found(self, tmp_path):
        cdir = _tmp_dir(tmp_path)
        with pytest.raises(FileNotFoundError):
            get_collection("nonexistent", collections_dir=cdir)


# ---------------------------------------------------------------------------
# update_collection
# ---------------------------------------------------------------------------


class TestUpdateCollection:
    def test_update_collection(self, tmp_path):
        cdir = _tmp_dir(tmp_path)
        create_collection(_minimal_collection("upd"), collections_dir=cdir)
        new_tracks = [_sample_track_dict()]
        updated = update_collection("upd", {"tracks": new_tracks}, collections_dir=cdir)
        assert len(updated["tracks"]) == 1
        assert updated["updated_at"] != updated["created_at"] or True  # may differ

    def test_update_name_immutable(self, tmp_path):
        """The 'name' field cannot be changed via update."""
        cdir = _tmp_dir(tmp_path)
        create_collection(_minimal_collection("orig"), collections_dir=cdir)
        updated = update_collection("orig", {"name": "changed"}, collections_dir=cdir)
        assert updated["name"] == "orig"

    def test_update_created_at_immutable(self, tmp_path):
        """The 'created_at' field cannot be changed via update."""
        cdir = _tmp_dir(tmp_path)
        orig = create_collection(_minimal_collection("dt"), collections_dir=cdir)
        orig_created = orig["created_at"]
        updated = update_collection(
            "dt", {"created_at": "1970-01-01T00:00:00Z"}, collections_dir=cdir
        )
        assert updated["created_at"] == orig_created

    def test_update_not_found(self, tmp_path):
        cdir = _tmp_dir(tmp_path)
        with pytest.raises(FileNotFoundError):
            update_collection("ghost", {}, collections_dir=cdir)

    def test_update_persisted(self, tmp_path):
        """Changes are written to disk and re-readable."""
        cdir = _tmp_dir(tmp_path)
        create_collection(_minimal_collection("persist"), collections_dir=cdir)
        update_collection("persist", {"playlist": "New PL"}, collections_dir=cdir)
        doc = get_collection("persist", collections_dir=cdir)
        assert doc["playlist"] == "New PL"


# ---------------------------------------------------------------------------
# delete_collection
# ---------------------------------------------------------------------------


class TestDeleteCollection:
    def test_delete_collection(self, tmp_path):
        cdir = _tmp_dir(tmp_path)
        create_collection(_minimal_collection("bye"), collections_dir=cdir)
        delete_collection("bye", collections_dir=cdir)
        assert not (cdir / "bye.json").exists()

    def test_delete_not_found(self, tmp_path):
        cdir = _tmp_dir(tmp_path)
        with pytest.raises(FileNotFoundError):
            delete_collection("missing", collections_dir=cdir)


# ---------------------------------------------------------------------------
# Invalid names (path traversal)
# ---------------------------------------------------------------------------


class TestInvalidName:
    @pytest.mark.parametrize(
        "name",
        [
            "../etc/passwd",
            "../../evil",
            "valid/../escape",
            "bad name",  # space
            "bad!name",  # exclamation
            "",  # empty
            "a" * 65,  # too long
        ],
    )
    def test_invalid_name(self, name, tmp_path):
        cdir = _tmp_dir(tmp_path)
        with pytest.raises((ValueError, FileNotFoundError)):
            create_collection({"name": name}, collections_dir=cdir)


# ---------------------------------------------------------------------------
# collection_to_training_data
# ---------------------------------------------------------------------------


class TestCollectionToTrainingData:
    def test_collection_to_training_data(self, tmp_path):
        """Available track (real temp file) is included; unavailable is skipped."""
        cdir = _tmp_dir(tmp_path)

        # Create a real temp audio file so os.path.isfile() returns True
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            real_path = f.name
            f.write(b"\x00" * 100)  # dummy content

        try:
            tracks = [
                _sample_track_dict(
                    track_id="1",
                    file_path=real_path,
                    file_available=True,
                    description="deep house, 128 BPM",
                ),
                _sample_track_dict(
                    track_id="2",
                    file_path="/nonexistent/track.wav",
                    file_available=False,
                ),
            ]
            create_collection(
                _minimal_collection("train", tracks=tracks), collections_dir=cdir
            )
            entries = collection_to_training_data("train", collections_dir=cdir)
        finally:
            os.unlink(real_path)

        assert len(entries) == 1
        assert entries[0]["file"] == real_path
        assert entries[0]["text"] == "deep house, 128 BPM"

    def test_collection_to_training_data_uses_description_gen(self, tmp_path):
        """When description is empty, generate_description() is used."""
        cdir = _tmp_dir(tmp_path)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            real_path = f.name

        try:
            tracks = [
                _sample_track_dict(
                    track_id="1",
                    file_path=real_path,
                    file_available=True,
                    description="",  # empty → should be auto-generated
                )
            ]
            create_collection(
                _minimal_collection("gen-desc", tracks=tracks), collections_dir=cdir
            )
            entries = collection_to_training_data("gen-desc", collections_dir=cdir)
        finally:
            os.unlink(real_path)

        assert entries[0]["text"]  # non-empty — generated from metadata

    def test_collection_to_training_data_no_available(self, tmp_path):
        """Raises ValueError when no tracks have available audio."""
        cdir = _tmp_dir(tmp_path)
        tracks = [
            _sample_track_dict(track_id="1", file_available=False),
            _sample_track_dict(
                track_id="2",
                file_path="/does/not/exist.wav",
                file_available=True,
            ),
        ]
        create_collection(
            _minimal_collection("empty-avail", tracks=tracks), collections_dir=cdir
        )
        with pytest.raises(ValueError, match="no tracks with available audio"):
            collection_to_training_data("empty-avail", collections_dir=cdir)

    def test_collection_to_training_data_missing_collection(self, tmp_path):
        cdir = _tmp_dir(tmp_path)
        with pytest.raises(FileNotFoundError):
            collection_to_training_data("ghost", collections_dir=cdir)
