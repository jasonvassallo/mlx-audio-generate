# Phase 9g-2: Music Library Scanner + Playlist-Driven Generation

**Date:** 2026-03-14
**Status:** Design approved, ready for implementation planning

## Summary

Add a Library tab to the web UI that connects to Apple Music and rekordbox XML exports, enabling users to browse their music libraries, select playlists/tracks, and either generate audio inspired by the selection's vibe or curate training collections for LoRA fine-tuning.

## Goals

1. Parse Apple Music and rekordbox XML library exports into a unified track model
2. Browse playlists and tracks with full search/sort/filter capabilities
3. Generate prompts from playlist metadata analysis ("Generate Like This")
4. Curate and save reusable training collections ("Train on These")
5. Support audio conditioning when local files are available (Option C)
6. Expand the color-coded tag system from 5 to 14 categories

## Non-Goals (deferred to later phases)

- Web enrichment via MusicBrainz/Last.fm/Discogs (Phase 9g-3)
- Spotify integration (deprecated Audio Features API — skip entirely)
- Taste profile learning engine (Phase 9g-3)
- Re-training loop / flywheel (Phase 9g-4)
- Spectral waveform coloring (Phase 10f-2)

## Architecture

### New Module: `mlx_audiogen/library/`

```
mlx_audiogen/library/
├── __init__.py
├── parsers.py        # Apple Music plist + rekordbox XML → TrackInfo
├── playlists.py      # Playlist extraction from both formats
├── cloud_paths.py    # Cloud storage path resolution
├── description_gen.py # Metadata → text description generation
└── collections.py    # Collection CRUD + persistence
```

### Data Model

```python
@dataclass
class TrackInfo:
    title: str
    artist: str
    album: str
    genre: str
    bpm: float | None
    key: str | None              # Camelot notation (e.g., "4A", "10B")
    year: int | None
    rating: int | None           # 0-100
    play_count: int
    duration_seconds: float
    comments: str
    file_path: str | None        # Resolved local path (None if streaming-only)
    file_available: bool         # Whether file_path exists on disk
    source: str                  # "apple_music" | "rekordbox"
    loved: bool
    description: str             # Auto-generated, user-editable

@dataclass
class PlaylistInfo:
    name: str
    track_count: int
    source: str

@dataclass
class LibrarySource:
    id: str
    type: str                    # "apple_music" | "rekordbox"
    path: str                    # Path to XML file
    label: str                   # Display name (e.g., "Apple Music", "rekordbox")
    track_count: int | None
    playlist_count: int | None
    last_loaded: str | None      # ISO timestamp
```

### Parsing Strategy

**Apple Music XML** (`~/Music/Media/Library.xml`):
- Format: Apple plist (XML property list)
- Parser: Python stdlib `plistlib` (safe by design, no entity expansion)
- Tracks: `<dict>` entries under `Tracks` key, keyed by Track ID
- Metadata: Name, Artist, Album, Genre, BPM, Year, Rating (0-100), Play Count, Total Time, Loved, Favorited
- Key extraction: Camelot key in Comments field (pattern: "4A - 7" → extract "4A")
- File paths: `Location` key contains `file:///` URLs (URL-encoded)
- Playlists: `Playlists` array, each with `Name` and `Playlist Items` (list of Track ID dicts)
- Size: ~650K lines, 12,104 tracks, 161 playlists

**rekordbox XML** (`~/Documents/rekordbox/rekordbox.xml`):
- Format: Custom XML (DJ_PLAYLISTS schema)
- Parser: `defusedxml.ElementTree` (prevents XXE, billion laughs attacks)
- Tracks: `<TRACK>` elements under `<COLLECTION>`
- Metadata: Name, Artist, Genre, AverageBpm, Tonality (Camelot key), Rating (0-255), PlayCount, TotalTime, DateAdded, Comments
- Additional: TEMPO elements (beat grid), POSITION_MARK elements (cue points)
- File paths: `Location` attribute contains `file://localhost/` URLs (URL-encoded)
- Playlists: `<NODE>` elements under `<PLAYLISTS>`, Type="0" = folder, Type="1" = playlist, child `<TRACK>` elements reference TrackID
- Size: ~193K lines, 4,211 tracks, 30 playlists

**Caching:**
- Parse once on library connect → in-memory dict/list cache
- Refresh via button or automatic on file mtime change
- No persistent cache — XML files are the source of truth

### Cloud Path Resolution

`cloud_paths.py` resolves `file://` URLs to local filesystem paths:

1. URL-decode the path (`urllib.parse.unquote`)
2. Strip scheme (`file://localhost/` → `/`, `file:///` → `/`)
3. Check if path exists on disk
4. For iCloud files: detect `.icloud` placeholder files → mark as unavailable
5. Auto-discover cloud providers at `~/Library/CloudStorage/`:
   - Dropbox: `Dropbox/`, `Dropbox-*/`
   - Google Drive: `GoogleDrive-*/`
   - iCloud: `~/Library/Mobile Documents/com~apple~CloudDocs/`
   - OneDrive: `OneDrive-*/`

### Description Generation

`description_gen.py` generates text descriptions from track metadata:

**Template mode** (instant):
```
"{genre}, {bpm} BPM, {key}, {artist} style"
```
With smart fallbacks: skip missing fields, infer mood from genre+BPM+key.

**LLM mode** (2-3 seconds):
Send metadata to local LLM with system prompt:
```
You are a music producer. Given this track's metadata, write a concise
audio generation prompt (1-2 sentences) that captures the sonic character.
Focus on genre, mood, instruments, and energy.
```

All descriptions are user-editable before training.

### Collections

Saved training selections at `~/.mlx-audiogen/collections/{name}.json`:

```json
{
  "name": "my-deep-house",
  "created_at": "2026-03-14T...",
  "updated_at": "2026-03-14T...",
  "source": "rekordbox",
  "playlist": "DJ Vassallo",
  "tracks": [
    {
      "title": "Box Jam",
      "artist": "Beaumont",
      "bpm": 122.0,
      "key": "4A",
      "genre": "Deep House",
      "comments": "4A - 7",
      "file_path": "/Users/.../Box Jam.mp3",
      "file_available": true,
      "description": "deep house groove, 122 BPM, 4A, warm bass",
      "description_edited": false
    }
  ]
}
```

Collections are independent of LoRA adapters — one collection can produce multiple adapters (different profiles), and collections can be edited and re-used.

## Server API

### Library Source Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/library/sources` | List configured library sources |
| `POST` | `/api/library/sources` | Add/update a library source |
| `DELETE` | `/api/library/sources/{id}` | Remove a library source |
| `POST` | `/api/library/scan/{id}` | Parse/refresh a library source |

### Library Browsing Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/library/playlists/{id}` | List playlists for a source |
| `GET` | `/api/library/tracks/{id}` | List/search/filter/sort tracks |
| `GET` | `/api/library/playlist-tracks/{id}/{playlist}` | Tracks in a playlist |

**Track query parameters:**
- `q` — free text search (title, artist, album, comments)
- `artist` — substring match
- `album` — substring match
- `genre` — substring match
- `key` — Camelot notation filter
- `bpm_min` / `bpm_max` — BPM range
- `year_min` / `year_max` — year range
- `rating_min` — minimum rating (0-100)
- `loved` — filter to loved/favorited
- `available` — filter to locally available files
- `sort` — field (`title`, `artist`, `album`, `year`, `bpm`, `key`, `genre`, `rating`, `play_count`)
- `order` — `asc` or `desc`
- `offset` / `limit` — pagination (default 50)

### AI Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/library/describe` | Generate descriptions for tracks |
| `POST` | `/api/library/suggest-name` | AI-suggest adapter name |
| `POST` | `/api/library/generate-prompt` | Analyze tracks → generate prompt |

### Collection Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/collections` | List saved collections |
| `GET` | `/api/collections/{name}` | Get a collection |
| `POST` | `/api/collections` | Create/save a collection |
| `PUT` | `/api/collections/{name}` | Update a collection |
| `DELETE` | `/api/collections/{name}` | Delete a collection |
| `GET` | `/api/collections/{name}/export` | Download as JSON |
| `POST` | `/api/collections/import` | Upload/restore a JSON collection |

### Library source config

Persisted at `~/.mlx-audiogen/library_sources.json`:
```json
[
  {"id": "am1", "type": "apple_music", "path": "~/Music/Media/Library.xml", "label": "Apple Music"},
  {"id": "rb1", "type": "rekordbox", "path": "~/Documents/rekordbox/rekordbox.xml", "label": "rekordbox"}
]
```

## Web UI

### Library Tab (new sidebar tab)

**Source Selector** (top):
- Dropdown of connected libraries
- "Add Library" button → form: type picker, path input, label
- Refresh button

**Playlist Browser** (below source selector):
- Scrollable list with track counts
- "All Tracks" option for full library
- Search bar with filter dropdowns (key, BPM range, genre)

**Track Table** (main content area, replaces History when Library tab active):
- Columns: checkbox, Title, Artist, Album, Year, BPM, Key, Genre, Comments, Rating (stars), Available (green/gray dot)
- All columns sortable (click header to toggle asc/desc)
- All columns searchable via top search bar
- Select-all checkbox in header row (select/deselect all visible)
- Click track → detail panel (full metadata, file path, play count)
- Multi-select via checkboxes

**Action Buttons** (bottom of track table):
- "Generate Like This" → metadata analysis → prompt preview → Generate tab
- "Train on These" → metadata editor → save collection → Train tab

### Metadata Editor (modal/inline)

Appears after "Train on These":
- Table of selected tracks with editable description column
- AI-suggested adapter name (editable text input)
- Profile picker (quick/balanced/deep)
- "Save Collection & Train" button
- "Save Collection" only button (for later)

### Enhanced Train Tab

New "Source" dropdown at top:
- "Folder" (existing behavior)
- "Collection" → picker from saved collections → shows tracks inline

### Color-Coded Tag Schema (14 categories)

| Category | Color | CSS Hex | Use |
|----------|-------|---------|-----|
| Genre | Amber | `#f59e0b` | Primary genre label |
| Sub-genre | Orange | `#f97316` | Genre refinement |
| Mood/Energy | Emerald | `#10b981` | Emotional character |
| Instrument/Timbre | Sky | `#0ea5e9` | Sonic texture |
| Vocal character | Violet | `#8b5cf6` | Voice description |
| Key/Harmony | Rose | `#f43f5e` | Musical key |
| BPM/Rhythm | Cyan | `#06b6d4` | Tempo, pulse |
| Era/Decade | Slate | `#64748b` | Time period |
| Production style | Fuchsia | `#d946ef` | Lo-fi, polished, etc. |
| Artist influence | Teal | `#14b8a6` | "Sounds like..." |
| Label/Scene | Indigo | `#6366f1` | Label, community |
| Structure | Lime | `#84cc16` | Build, drop, breakdown |
| Rating/Quality | Yellow | `#eab308` | Stars, favorites |
| Availability | Zinc | `#71717a` | File status |

Color relationships: warm cluster (identity), cool cluster (sonic character), accent cluster (feel), neutral (context).

## "Generate Like This" Flow

1. **Select tracks** in Library tab (playlist or manual selection)
2. **Click "Generate Like This"** → server analyzes:
   - BPM distribution (median, range)
   - Key clusters (most common + Camelot neighbors)
   - Genre profile (frequency-weighted)
   - Artist fingerprint (top artists)
   - Era (year range + median)
   - Mood inference (from genre + BPM + key)
3. **Prompt generation** (user picks mode):
   - Template mode (instant): structured prompt from analysis
   - LLM mode (2-3s): richer creative description via local LLM
4. **Preview card** (like existing EnhancePreview):
   - Use & Generate → sends to Generate tab
   - Edit → opens in prompt textarea
   - Regenerate → another LLM attempt
5. **Audio conditioning** (when files available):
   - Style conditioning (MusicGen style model): highest-rated track as reference
   - Reference audio (Stable Audio audio-to-audio): representative track
   - User can change reference track via dropdown
   - Grayed out with tooltip when no local files available

## Security

- **XML parsing**: `defusedxml` for rekordbox (prevents XXE, billion laughs). `plistlib` for Apple Music (stdlib, safe)
- **File size limit**: Reject XML files > 500MB
- **Path validation**: All `file://` URL resolution through `_validate_audio_path()` pattern — rejects `..` traversal, validates extensions
- **Cloud paths**: Validated against `~/Library/CloudStorage/` prefix
- **Collection paths**: Restricted to `~/.mlx-audiogen/collections/`
- **Collection names**: Validated with `^[a-zA-Z0-9_-]{1,64}$` regex
- **Input validation**: Pydantic models for all request bodies with field constraints

## Error Handling

- Missing/moved audio files: `file_available: false`, gray dot in UI, training skips gracefully
- Corrupt XML: clear error message ("Could not parse — try re-exporting from rekordbox/Apple Music")
- iCloud placeholder files (`.icloud`): detected, reported as unavailable
- Empty playlists: shown in list but disabled for actions
- Unicode track names: fully supported (UTF-8 throughout)

## Testing

- Unit tests with small fixture XMLs (5-10 tracks, 2-3 playlists) for both formats
- Parser edge cases: missing fields, URL-encoded paths, unicode, empty playlists, malformed XML
- Collection CRUD tests (create, read, update, delete, export, import)
- Description generation tests (template mode + edge cases)
- Cloud path resolution tests (various URL schemes, missing files, `.icloud` placeholders)
- API endpoint tests (search, sort, filter, pagination)
- Integration tests with real XML files (`@pytest.mark.integration`)

## Dependencies

- `defusedxml` — new dependency for safe XML parsing (add to pyproject.toml)
- All other dependencies already present (plistlib is stdlib, FastAPI/Pydantic for API)

## Files to Add/Modify

### New files:
- `mlx_audiogen/library/__init__.py`
- `mlx_audiogen/library/parsers.py`
- `mlx_audiogen/library/playlists.py`
- `mlx_audiogen/library/cloud_paths.py`
- `mlx_audiogen/library/description_gen.py`
- `mlx_audiogen/library/collections.py`
- `tests/test_library.py`
- `tests/fixtures/apple_music_sample.xml`
- `tests/fixtures/rekordbox_sample.xml`
- `web/src/components/LibraryPanel.tsx`
- `web/src/components/MetadataEditor.tsx`

### Modified files:
- `pyproject.toml` — add `defusedxml` dependency
- `.gitignore` — add `Library.xml`, `rekordbox.xml`
- `mlx_audiogen/server/app.py` — add 17 new endpoints
- `web/src/api/client.ts` — add library + collection API wrappers
- `web/src/store/useStore.ts` — add library state
- `web/src/types/api.ts` — add library/collection types
- `web/src/components/App.tsx` — add Library tab
- `web/src/components/TrainPanel.tsx` — add Collection source type
- `web/src/components/TabBar.tsx` — add Library tab
- `CLAUDE.md` — document new module and endpoints
