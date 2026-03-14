"""In-memory library cache with search, sort, filter, and pagination.

:class:`LibraryCache` wraps one or more parsed music library XML files
(Apple Music / rekordbox) and provides fast in-process querying without
repeated disk I/O.  Source metadata (path, label, stats) is persisted to
``~/.mlx-audiogen/library_sources.json`` so it survives across process
restarts; the actual track data lives only in memory and is rebuilt by
calling :meth:`~LibraryCache.scan`.

Typical usage::

    cache = LibraryCache()
    src = cache.add_source("apple_music", "~/Music/Library.xml", "My Library")
    cache.scan(src.id)
    tracks = cache.search_tracks(src.id, q="deep", bpm_min=120, limit=20)
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .models import LibrarySource, PlaylistInfo, TrackInfo
from .parsers import parse_apple_music_xml, parse_rekordbox_xml

DEFAULT_CONFIG_DIR: Path = Path.home() / ".mlx-audiogen"
_SOURCES_FILE = "library_sources.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class LibraryCache:
    """Manages multiple music library sources with in-memory track caching.

    Args:
        config_dir: Directory where ``library_sources.json`` is persisted.
            Defaults to ``~/.mlx-audiogen``.
    """

    def __init__(self, config_dir: Path = DEFAULT_CONFIG_DIR) -> None:
        self._config_dir = config_dir
        self._sources_path = config_dir / _SOURCES_FILE

        # In-memory data store: source_id → {tracks, playlists}
        self._data: dict[str, dict] = {}

        # Persisted source list
        self._sources: dict[str, LibrarySource] = {}
        self._load_sources()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _load_sources(self) -> None:
        """Load source list from disk (silently if missing or corrupt)."""
        if not self._sources_path.exists():
            return
        try:
            raw: list[dict] = json.loads(self._sources_path.read_text())
            self._sources = {
                d["id"]: LibrarySource.from_dict(d)
                for d in raw
                if isinstance(d, dict) and "id" in d
            }
        except (OSError, json.JSONDecodeError, KeyError):
            self._sources = {}

    def _save_sources(self) -> None:
        """Persist source list to disk, creating the directory if needed."""
        self._config_dir.mkdir(parents=True, exist_ok=True)
        data = [s.to_dict() for s in self._sources.values()]
        self._sources_path.write_text(json.dumps(data, indent=2))

    # ------------------------------------------------------------------
    # Source management
    # ------------------------------------------------------------------

    def add_source(self, type: str, path: str, label: str) -> LibrarySource:
        """Register a new library source.

        Generates a short UUID (first 8 hex chars) as the source ID and
        persists the updated source list to disk.

        Args:
            type: ``"apple_music"`` or ``"rekordbox"``.
            path: Filesystem path to the XML export file.
            label: Human-readable display name.

        Returns:
            The newly created :class:`~.models.LibrarySource`.
        """
        source_id = str(uuid.uuid4())[:8]
        source = LibrarySource(
            id=source_id,
            type=type,
            path=path,
            label=label,
            track_count=0,
            playlist_count=0,
            last_loaded=None,
        )
        self._sources[source_id] = source
        self._save_sources()
        return source

    def update_source(
        self,
        source_id: str,
        path: Optional[str] = None,
        label: Optional[str] = None,
    ) -> LibrarySource:
        """Update an existing source's path and/or label.

        Raises:
            KeyError: If *source_id* is not found.
        """
        source = self._get_source(source_id)
        if path is not None:
            source.path = path
        if label is not None:
            source.label = label
        self._save_sources()
        return source

    def remove_source(self, source_id: str) -> None:
        """Remove a source from the registry and evict its cached data.

        Raises:
            KeyError: If *source_id* is not found.
        """
        self._get_source(source_id)  # validates existence
        self._sources.pop(source_id, None)
        self._data.pop(source_id, None)
        self._save_sources()

    def list_sources(self) -> list[LibrarySource]:
        """Return all registered sources."""
        return list(self._sources.values())

    def _get_source(self, source_id: str) -> LibrarySource:
        if source_id not in self._sources:
            raise KeyError(f"Unknown source ID: {source_id!r}")
        return self._sources[source_id]

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------

    def scan(self, source_id: str) -> LibrarySource:
        """Parse the XML file for *source_id* and cache the results.

        Dispatches to :func:`~.parsers.parse_apple_music_xml` or
        :func:`~.parsers.parse_rekordbox_xml` based on the source type.

        Args:
            source_id: ID of a registered source.

        Returns:
            The updated :class:`~.models.LibrarySource` with current stats.

        Raises:
            KeyError: If *source_id* is not found.
            FileNotFoundError: If the XML file does not exist.
            ValueError: If the XML cannot be parsed.
        """
        source = self._get_source(source_id)

        if source.type == "apple_music":
            tracks, playlists = parse_apple_music_xml(source.path)
        elif source.type == "rekordbox":
            tracks, playlists = parse_rekordbox_xml(source.path)
        else:
            raise ValueError(f"Unknown source type: {source.type!r}")

        self._data[source_id] = {
            "tracks": tracks,  # dict[str, TrackInfo]
            "playlists": playlists,  # list[PlaylistInfo]
        }

        source.track_count = len(tracks)
        source.playlist_count = len(playlists)
        source.last_loaded = _now_iso()
        self._save_sources()
        return source

    # ------------------------------------------------------------------
    # Browsing
    # ------------------------------------------------------------------

    def get_track_count(self, source_id: str) -> int:
        """Return the number of cached tracks for *source_id*."""
        data = self._data.get(source_id, {})
        return len(data.get("tracks", {}))

    def get_playlists(self, source_id: str) -> list[PlaylistInfo]:
        """Return all playlists for *source_id*."""
        data = self._data.get(source_id, {})
        return list(data.get("playlists", []))

    def get_playlist_tracks(self, source_id: str, playlist_id: str) -> list[TrackInfo]:
        """Return all tracks belonging to a playlist.

        Args:
            source_id: Library source ID.
            playlist_id: Playlist slug (``PlaylistInfo.id``).

        Returns:
            Ordered list of :class:`~.models.TrackInfo` objects.
            Returns an empty list if the playlist or source is not found.
        """
        data = self._data.get(source_id, {})
        tracks: dict[str, TrackInfo] = data.get("tracks", {})
        playlists: list[PlaylistInfo] = data.get("playlists", [])

        playlist = next((p for p in playlists if p.id == playlist_id), None)
        if playlist is None:
            return []
        return [tracks[tid] for tid in playlist.track_ids if tid in tracks]

    # ------------------------------------------------------------------
    # Search / filter / sort / paginate
    # ------------------------------------------------------------------

    def search_tracks(  # noqa: PLR0913 — many filter params by design
        self,
        source_id: str,
        *,
        q: Optional[str] = None,
        artist: Optional[str] = None,
        album: Optional[str] = None,
        genre: Optional[str] = None,
        key: Optional[str] = None,
        bpm_min: Optional[float] = None,
        bpm_max: Optional[float] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        rating_min: Optional[int] = None,
        loved: Optional[bool] = None,
        available: Optional[bool] = None,
        sort: Optional[str] = None,
        order: str = "asc",
        offset: int = 0,
        limit: int = 50,
    ) -> list[TrackInfo]:
        """Search and filter tracks with optional sorting and pagination.

        All string filters are case-insensitive substring matches except
        *key* which requires an exact match.

        Args:
            source_id: Library source to query.
            q: Free-text search across title, artist, album, comments.
            artist: Substring match on artist.
            album: Substring match on album.
            genre: Substring match on genre.
            key: Exact Camelot key match (e.g. ``"4A"``).
            bpm_min: Minimum BPM (inclusive). Tracks with no BPM are excluded.
            bpm_max: Maximum BPM (inclusive). Tracks with no BPM are excluded.
            year_min: Minimum release year (inclusive).
            year_max: Maximum release year (inclusive).
            rating_min: Minimum normalized rating 0-100 (inclusive).
            loved: If given, filters to loved/unloved tracks.
            available: If given, filters to tracks where ``file_available``
                matches.
            sort: Any :class:`~.models.TrackInfo` field name.  ``None`` values
                sort last for both asc and desc.
            order: ``"asc"`` (default) or ``"desc"``.
            offset: Number of results to skip (for pagination).
            limit: Maximum results to return (default 50).

        Returns:
            Matching tracks after all filters, sorting, and pagination.
        """
        data = self._data.get(source_id, {})
        tracks: dict[str, TrackInfo] = data.get("tracks", {})
        results: list[TrackInfo] = list(tracks.values())

        # -- Free text --
        if q:
            q_lo = q.lower()
            results = [
                t
                for t in results
                if q_lo in t.title.lower()
                or q_lo in t.artist.lower()
                or q_lo in t.album.lower()
                or q_lo in t.comments.lower()
            ]

        # -- Substring filters --
        if artist:
            a_lo = artist.lower()
            results = [t for t in results if a_lo in t.artist.lower()]
        if album:
            al_lo = album.lower()
            results = [t for t in results if al_lo in t.album.lower()]
        if genre:
            g_lo = genre.lower()
            results = [t for t in results if g_lo in t.genre.lower()]

        # -- Exact key match --
        if key:
            results = [t for t in results if t.key == key]

        # -- Numeric ranges (skip tracks where the field is None) --
        if bpm_min is not None:
            results = [t for t in results if t.bpm is not None and t.bpm >= bpm_min]
        if bpm_max is not None:
            results = [t for t in results if t.bpm is not None and t.bpm <= bpm_max]
        if year_min is not None:
            results = [t for t in results if t.year is not None and t.year >= year_min]
        if year_max is not None:
            results = [t for t in results if t.year is not None and t.year <= year_max]
        if rating_min is not None:
            results = [t for t in results if t.rating >= rating_min]

        # -- Boolean filters --
        if loved is not None:
            results = [t for t in results if t.loved == loved]
        if available is not None:
            results = [t for t in results if t.file_available == available]

        # -- Sort --
        if sort:
            reverse = order.lower() == "desc"
            results = _sort_tracks(results, sort, reverse)

        # -- Pagination --
        return results[offset : offset + limit]


# ---------------------------------------------------------------------------
# Sorting helper
# ---------------------------------------------------------------------------


def _sort_key(track: TrackInfo, field: str) -> tuple[int, Any]:
    """Return a sort key that pushes None values to the end.

    Returns a 2-tuple ``(is_none: int, value)`` so that ``None`` values
    always sort after real values regardless of ascending/descending order.
    (When reversed, ``(1, None)`` > ``(0, value)`` for any value, which keeps
    Nones at the end.)
    """
    value = getattr(track, field, None)
    if value is None:
        return (1, "")  # None → sort last
    # Normalise strings for case-insensitive sorting
    if isinstance(value, str):
        return (0, value.lower())
    return (0, value)


def _sort_tracks(
    tracks: list[TrackInfo],
    field: str,
    reverse: bool,
) -> list[TrackInfo]:
    """Sort tracks by *field*, with None values always last.

    The two-tuple trick: sort ascending by (is_none, value).  For descending
    we negate the first element via a secondary sort to keep Nones at the end.
    """
    if not reverse:
        return sorted(tracks, key=lambda t: _sort_key(t, field))

    # Descending: sort non-None values descending, then Nones at end
    non_none = [t for t in tracks if getattr(t, field, None) is not None]
    none_tracks = [t for t in tracks if getattr(t, field, None) is None]
    non_none_sorted = sorted(
        non_none,
        key=lambda t: _sort_key(t, field),
        reverse=True,
    )
    return non_none_sorted + none_tracks
