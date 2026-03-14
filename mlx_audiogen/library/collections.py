"""Collection CRUD operations + training pipeline bridge.

A *collection* is a named, persisted set of :class:`~.models.TrackInfo`
objects (stored as JSON) that can be fed directly into the LoRA training
pipeline.  Collections are saved under ``~/.mlx-audiogen/collections/``
as ``<name>.json`` files.

Typical workflow:

1. Parse a library with :mod:`.parsers`.
2. Filter to the desired tracks.
3. Call :func:`create_collection` to persist the selection.
4. Call :func:`collection_to_training_data` to get a list ready for
   :func:`mlx_audiogen.lora.dataset.scan_dataset`.
"""

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .description_gen import generate_description
from .models import TrackInfo

DEFAULT_COLLECTIONS_DIR: Path = Path.home() / ".mlx-audiogen" / "collections"

# Collection names must be 1-64 characters: letters, digits, hyphens, underscores.
_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def _validate_name(name: str) -> None:
    """Raise ValueError for invalid or path-traversal collection names."""
    if ".." in name:
        raise ValueError(f"Collection name must not contain '..': {name!r}")
    if not _NAME_RE.match(name):
        raise ValueError(
            f"Collection name {name!r} is invalid. "
            "Use only letters, digits, hyphens, and underscores (1-64 chars)."
        )


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string (with 'Z' suffix)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _collection_path(name: str, collections_dir: Path) -> Path:
    return collections_dir / f"{name}.json"


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


def create_collection(
    data: dict,
    collections_dir: Path = DEFAULT_COLLECTIONS_DIR,
) -> dict:
    """Create a new collection and persist it to disk.

    Args:
        data: Dict that must contain at least ``"name"``.  Other fields
            (``"tracks"``, ``"source"``, ``"playlist"``, …) are stored
            verbatim.  ``created_at`` and ``updated_at`` are always set by
            this function.
        collections_dir: Directory where collection JSON files are stored.

    Returns:
        The stored collection dict (including timestamps).

    Raises:
        ValueError: If the name is invalid or the collection already exists.
    """
    name = data.get("name", "")
    _validate_name(name)

    collections_dir.mkdir(parents=True, exist_ok=True)
    path = _collection_path(name, collections_dir)
    if path.exists():
        raise ValueError(f"Collection {name!r} already exists.")

    now = _now_iso()
    doc = dict(data)
    doc["created_at"] = now
    doc["updated_at"] = now

    path.write_text(json.dumps(doc, indent=2))
    return doc


def list_collections(
    collections_dir: Path = DEFAULT_COLLECTIONS_DIR,
) -> list[dict]:
    """Return summary info for every collection in *collections_dir*.

    Each summary dict contains:
    ``name``, ``track_count``, ``source``, ``playlist``,
    ``created_at``, ``updated_at``.
    Missing fields default to sensible values (0 / "" / None).

    Returns an empty list if the directory does not exist.
    """
    if not collections_dir.is_dir():
        return []

    results = []
    for path in sorted(collections_dir.glob("*.json")):
        try:
            doc = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        tracks = doc.get("tracks", [])
        results.append(
            {
                "name": doc.get("name", path.stem),
                "track_count": len(tracks) if isinstance(tracks, list) else 0,
                "source": doc.get("source", ""),
                "playlist": doc.get("playlist", None),
                "created_at": doc.get("created_at", None),
                "updated_at": doc.get("updated_at", None),
            }
        )
    return results


def get_collection(
    name: str,
    collections_dir: Path = DEFAULT_COLLECTIONS_DIR,
) -> dict:
    """Load a collection by name.

    Raises:
        ValueError: If the name is invalid.
        FileNotFoundError: If no collection with that name exists.
    """
    _validate_name(name)
    path = _collection_path(name, collections_dir)
    if not path.exists():
        raise FileNotFoundError(f"Collection {name!r} not found.")
    return json.loads(path.read_text())


def update_collection(
    name: str,
    updates: dict,
    collections_dir: Path = DEFAULT_COLLECTIONS_DIR,
) -> dict:
    """Merge *updates* into an existing collection and persist.

    The ``updated_at`` field is always refreshed.  ``name``, ``created_at``
    cannot be changed via updates (those keys in *updates* are ignored).

    Raises:
        ValueError: If the name is invalid.
        FileNotFoundError: If no collection with that name exists.
    """
    doc = get_collection(name, collections_dir)

    for key, value in updates.items():
        if key in ("name", "created_at"):
            continue  # immutable
        doc[key] = value
    doc["updated_at"] = _now_iso()

    _collection_path(name, collections_dir).write_text(json.dumps(doc, indent=2))
    return doc


def delete_collection(
    name: str,
    collections_dir: Path = DEFAULT_COLLECTIONS_DIR,
) -> None:
    """Delete a collection file.

    Raises:
        ValueError: If the name is invalid.
        FileNotFoundError: If no collection with that name exists.
    """
    _validate_name(name)
    path = _collection_path(name, collections_dir)
    if not path.exists():
        raise FileNotFoundError(f"Collection {name!r} not found.")
    path.unlink()


# ---------------------------------------------------------------------------
# Training bridge
# ---------------------------------------------------------------------------


def collection_to_training_data(
    name: str,
    collections_dir: Path = DEFAULT_COLLECTIONS_DIR,
) -> list[dict[str, str]]:
    """Convert a collection to training data entries for the LoRA pipeline.

    Each entry has the shape ``{"file": "<absolute path>", "text": "<desc>"}``,
    matching the format expected by
    :func:`mlx_audiogen.lora.dataset.scan_dataset`.

    Only tracks that satisfy **all** of the following are included:

    - ``file_available`` is ``True``
    - The ``file_path`` actually exists on disk (``os.path.isfile``)

    The text description comes from ``description`` if it is non-empty (and
    ``description_edited`` is respected by the human editing flow), otherwise
    :func:`~.description_gen.generate_description` is called to produce one
    from metadata.

    Raises:
        ValueError: If no tracks with available audio exist in the collection.
        FileNotFoundError: If the collection does not exist.
    """
    doc = get_collection(name, collections_dir)
    raw_tracks: list[dict] = doc.get("tracks", [])

    entries: list[dict[str, str]] = []
    for raw in raw_tracks:
        if not raw.get("file_available", False):
            continue
        file_path: Optional[str] = raw.get("file_path")
        if not file_path or not os.path.isfile(file_path):
            continue

        # Use stored description if available; otherwise generate from metadata
        description: str = raw.get("description", "").strip()
        if not description:
            track = TrackInfo.from_dict(raw)
            description = generate_description(track)

        entries.append({"file": file_path, "text": description})

    if not entries:
        raise ValueError(
            f"Collection {name!r} has no tracks with available audio files. "
            "Ensure file_available=True and the files exist on disk."
        )

    return entries
