import { useCallback, useEffect, useState } from "react";
import { useStore } from "../store/useStore";
import { enrichTracks } from "../api/client";
import type { EnrichmentStatus } from "../types/api";

/** Format seconds as mm:ss. */
function fmtDuration(s: number): string {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

/** Render 1-5 stars from 0-100 rating. */
function Stars({ rating }: { rating: number | null }) {
  if (rating === null || rating === 0) return <span className="text-zinc-600">-</span>;
  const count = Math.round(rating / 20);
  return (
    <span className="text-amber-400">
      {"★".repeat(count)}
      <span className="text-zinc-700">{"★".repeat(5 - count)}</span>
    </span>
  );
}

/** Availability dot: green = available, gray = missing. */
function AvailDot({ available }: { available: boolean }) {
  return (
    <span
      className={`inline-block h-2 w-2 rounded-full ${
        available ? "bg-emerald-500" : "bg-zinc-600"
      }`}
      title={available ? "File available" : "File not found"}
    />
  );
}

/** Enrichment status dots: green = fetched, gray = not. */
function EnrichmentDots({ status }: { status?: EnrichmentStatus }) {
  const s = status ?? "none";
  const mb = s === "complete";
  const lf = s === "complete" || s === "partial";
  const dc = s === "complete";
  return (
    <span className="inline-flex items-center gap-1">
      <span
        className={`inline-block h-1.5 w-1.5 rounded-full ${mb ? "bg-emerald-500" : "bg-zinc-700"}`}
        title="MusicBrainz"
      />
      <span
        className={`inline-block h-1.5 w-1.5 rounded-full ${lf ? "bg-emerald-500" : "bg-zinc-700"}`}
        title="Last.fm"
      />
      <span
        className={`inline-block h-1.5 w-1.5 rounded-full ${dc ? "bg-emerald-500" : "bg-zinc-700"}`}
        title="Discogs"
      />
    </span>
  );
}

/** Sortable column header. */
function SortHeader({
  label,
  field,
  currentSort,
  currentOrder,
  onSort,
  className = "",
}: {
  label: string;
  field: string;
  currentSort: string | undefined;
  currentOrder: string;
  onSort: (field: string) => void;
  className?: string;
}) {
  const active = currentSort === field;
  return (
    <th
      className={`cursor-pointer select-none px-2 py-1.5 text-left text-[10px] font-medium uppercase tracking-wider hover:text-zinc-200 ${
        active ? "text-sky-400" : "text-zinc-500"
      } ${className}`}
      onClick={() => onSort(field)}
    >
      {label}
      {active && (
        <span className="ml-0.5">{currentOrder === "asc" ? "▲" : "▼"}</span>
      )}
    </th>
  );
}

/** Sidebar: source picker + playlist browser. */
export function LibrarySidebar() {
  const sources = useStore((s) => s.librarySources);
  const sourcesLoading = useStore((s) => s.librarySourcesLoading);
  const activeSourceId = useStore((s) => s.activeSourceId);
  const setActiveSourceId = useStore((s) => s.setActiveSourceId);
  const playlists = useStore((s) => s.playlists);
  const playlistsLoading = useStore((s) => s.playlistsLoading);
  const activePlaylistId = useStore((s) => s.activePlaylistId);
  const setActivePlaylistId = useStore((s) => s.setActivePlaylistId);
  const addSource = useStore((s) => s.addSource);
  const removeSource = useStore((s) => s.removeSource);
  const scanSource = useStore((s) => s.scanSource);
  const loadLibrarySources = useStore((s) => s.loadLibrarySources);
  const libraryTracksTotal = useStore((s) => s.libraryTracksTotal);

  // Add-source form
  const [showAdd, setShowAdd] = useState(false);
  const [addType, setAddType] = useState<"apple_music" | "rekordbox">("apple_music");
  const [addPath, setAddPath] = useState("");
  const [addLabel, setAddLabel] = useState("");
  const [addError, setAddError] = useState<string | null>(null);
  const [scanning, setScanning] = useState<string | null>(null);

  useEffect(() => {
    loadLibrarySources();
  }, [loadLibrarySources]);

  const handleAddSource = useCallback(async () => {
    if (!addPath.trim()) return;
    setAddError(null);
    try {
      await addSource(addType, addPath.trim(), addLabel.trim() || addType === "apple_music" ? "Apple Music" : "rekordbox");
      setShowAdd(false);
      setAddPath("");
      setAddLabel("");
    } catch (e) {
      setAddError(e instanceof Error ? e.message : "Failed to add source");
    }
  }, [addType, addPath, addLabel, addSource]);

  const handleScan = useCallback(async (id: string) => {
    setScanning(id);
    try {
      await scanSource(id);
    } catch {
      // Error handled in store
    }
    setScanning(null);
  }, [scanSource]);

  return (
    <div className="flex flex-1 flex-col overflow-hidden">
      <div className="space-y-3 p-5">
        <div className="flex items-center justify-between">
          <h3 className="text-xs font-medium uppercase tracking-wider text-zinc-400">
            Library
          </h3>
          <button
            onClick={() => setShowAdd(!showAdd)}
            className="text-xs text-sky-400 hover:text-sky-300"
          >
            {showAdd ? "Cancel" : "+ Add"}
          </button>
        </div>

        {/* Add source form */}
        {showAdd && (
          <div className="space-y-2 rounded border border-zinc-700 bg-zinc-800/50 p-3">
            <select
              value={addType}
              onChange={(e) => setAddType(e.target.value as "apple_music" | "rekordbox")}
              className="w-full rounded bg-zinc-800 px-2 py-1.5 text-sm text-zinc-200 border border-zinc-700 focus:border-sky-500 focus:outline-none"
            >
              <option value="apple_music">Apple Music</option>
              <option value="rekordbox">rekordbox</option>
            </select>
            <input
              type="text"
              value={addPath}
              onChange={(e) => setAddPath(e.target.value)}
              placeholder="Path to XML export..."
              className="w-full rounded bg-zinc-800 px-2 py-1.5 text-sm text-zinc-200 border border-zinc-700 focus:border-sky-500 focus:outline-none placeholder:text-zinc-600"
            />
            <input
              type="text"
              value={addLabel}
              onChange={(e) => setAddLabel(e.target.value)}
              placeholder="Display name (optional)"
              className="w-full rounded bg-zinc-800 px-2 py-1.5 text-sm text-zinc-200 border border-zinc-700 focus:border-sky-500 focus:outline-none placeholder:text-zinc-600"
            />
            {addError && (
              <p className="text-[10px] text-red-400">{addError}</p>
            )}
            <button
              onClick={handleAddSource}
              disabled={!addPath.trim()}
              className="w-full rounded bg-sky-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-sky-500 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Add Source
            </button>
          </div>
        )}

        {/* Source selector */}
        {sourcesLoading && (
          <p className="text-[10px] text-zinc-500">Loading sources...</p>
        )}
        {sources.length === 0 && !sourcesLoading && (
          <p className="text-[10px] text-zinc-500">
            No library sources configured. Click + Add to import your music library XML.
          </p>
        )}
        {sources.map((src) => (
          <div
            key={src.id}
            className={`group flex items-center gap-2 rounded border px-3 py-2 text-xs transition-colors cursor-pointer ${
              activeSourceId === src.id
                ? "border-sky-500/50 bg-sky-500/10 text-sky-300"
                : "border-zinc-700 bg-zinc-800/50 text-zinc-300 hover:border-zinc-600"
            }`}
            onClick={() => setActiveSourceId(src.id)}
          >
            <span className="flex-1 truncate font-medium">{src.label}</span>
            <span className="text-[10px] tabular-nums text-zinc-500">
              {src.track_count ?? 0} tracks
            </span>
            <button
              onClick={(e) => { e.stopPropagation(); handleScan(src.id); }}
              disabled={scanning === src.id}
              className="text-zinc-500 hover:text-sky-400 opacity-0 group-hover:opacity-100 transition-opacity"
              title="Rescan"
            >
              {scanning === src.id ? "..." : "↻"}
            </button>
            <button
              onClick={(e) => { e.stopPropagation(); removeSource(src.id); }}
              className="text-zinc-500 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-opacity"
              title="Remove"
            >
              ×
            </button>
          </div>
        ))}
      </div>

      {/* Playlist browser */}
      {activeSourceId && (
        <div className="flex-1 overflow-y-auto border-t border-zinc-700/50 px-5 py-3">
          <p className="mb-2 text-[10px] font-medium uppercase tracking-wider text-zinc-500">
            Playlists
          </p>
          {playlistsLoading && (
            <p className="text-[10px] text-zinc-500">Loading...</p>
          )}
          {/* All Tracks entry */}
          <button
            onClick={() => setActivePlaylistId(null)}
            className={`mb-1 flex w-full items-center justify-between rounded px-2 py-1.5 text-xs transition-colors ${
              activePlaylistId === null
                ? "bg-sky-500/10 text-sky-300"
                : "text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
            }`}
          >
            <span className="truncate">All Tracks</span>
            <span className="text-[10px] tabular-nums text-zinc-500">
              {libraryTracksTotal}
            </span>
          </button>
          {playlists.map((pl) => (
            <button
              key={pl.id}
              onClick={() => setActivePlaylistId(pl.id)}
              className={`mb-0.5 flex w-full items-center justify-between rounded px-2 py-1.5 text-xs transition-colors ${
                activePlaylistId === pl.id
                  ? "bg-sky-500/10 text-sky-300"
                  : "text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
              }`}
            >
              <span className="truncate">{pl.name}</span>
              <span className="text-[10px] tabular-nums text-zinc-500">
                {pl.track_count}
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

/** Main content: track table with search, sort, selection, and actions. */
export function LibraryTrackTable({ onTrainOnThese }: { onTrainOnThese?: () => void }) {
  const tracks = useStore((s) => s.libraryTracks);
  const loading = useStore((s) => s.libraryTracksLoading);
  const total = useStore((s) => s.libraryTracksTotal);
  const selectedIds = useStore((s) => s.selectedTrackIds);
  const toggleSelection = useStore((s) => s.toggleTrackSelection);
  const selectAll = useStore((s) => s.selectAllTracks);
  const deselectAll = useStore((s) => s.deselectAllTracks);
  const searchParams = useStore((s) => s.librarySearchParams);
  const setSearchParams = useStore((s) => s.setLibrarySearchParams);
  const activeSourceId = useStore((s) => s.activeSourceId);
  const activePlaylistId = useStore((s) => s.activePlaylistId);
  const generateLikeThis = useStore((s) => s.generateLikeThis);
  const generateLikeLoading = useStore((s) => s.generateLikeLoading);
  const generateLikeResult = useStore((s) => s.generateLikeResult);
  const clearGenerateLikeResult = useStore((s) => s.clearGenerateLikeResult);
  const setParam = useStore((s) => s.setParam);
  const setActiveTab = useStore((s) => s.setActiveTab);
  const generate = useStore((s) => s.generate);

  const [searchQuery, setSearchQuery] = useState("");
  const [enriching, setEnriching] = useState(false);

  // Debounced search
  useEffect(() => {
    const timer = setTimeout(() => {
      if (activeSourceId && !activePlaylistId) {
        setSearchParams({ ...searchParams, q: searchQuery || undefined, offset: 0 });
      }
    }, 300);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchQuery]);

  const handleSort = useCallback((field: string) => {
    const newOrder =
      searchParams.sort === field && searchParams.order === "asc" ? "desc" : "asc";
    setSearchParams({ ...searchParams, sort: field, order: newOrder, offset: 0 });
  }, [searchParams, setSearchParams]);

  const allSelected = tracks.length > 0 && selectedIds.size === tracks.length;
  const someSelected = selectedIds.size > 0;

  const handleGenerateLikeThis = useCallback(() => {
    if (!activeSourceId || selectedIds.size === 0) return;
    generateLikeThis(activeSourceId, Array.from(selectedIds));
  }, [activeSourceId, selectedIds, generateLikeThis]);

  // "Use & Generate" from the Generate Like This preview
  const handleUseAndGenerate = useCallback(() => {
    if (!generateLikeResult) return;
    setParam("prompt", generateLikeResult.prompt);
    clearGenerateLikeResult();
    setActiveTab("generate");
    // Delay generation slightly to let the tab switch render
    setTimeout(() => generate(), 100);
  }, [generateLikeResult, setParam, clearGenerateLikeResult, setActiveTab, generate]);

  const handleEditPrompt = useCallback(() => {
    if (!generateLikeResult) return;
    setParam("prompt", generateLikeResult.prompt);
    clearGenerateLikeResult();
    setActiveTab("generate");
  }, [generateLikeResult, setParam, clearGenerateLikeResult, setActiveTab]);

  const handleEnrichSelected = useCallback(async () => {
    if (!activeSourceId || selectedIds.size === 0) return;
    setEnriching(true);
    try {
      await enrichTracks({
        track_ids: Array.from(selectedIds),
        source_id: activeSourceId,
      });
    } catch {
      // Ignore — server may not support enrichment yet
    }
    setEnriching(false);
  }, [activeSourceId, selectedIds]);

  if (!activeSourceId) {
    return (
      <div className="flex flex-1 items-center justify-center text-zinc-500 text-sm">
        Select a library source to browse tracks
      </div>
    );
  }

  return (
    <div className="flex flex-1 flex-col overflow-hidden">
      {/* Search + actions bar */}
      <div className="flex items-center gap-3 border-b border-zinc-700/50 px-4 py-3">
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Search tracks..."
          disabled={!!activePlaylistId}
          className="flex-1 rounded bg-zinc-800 px-3 py-1.5 text-sm text-zinc-200 border border-zinc-700 focus:border-sky-500 focus:outline-none placeholder:text-zinc-600 disabled:opacity-50"
        />
        <span className="text-[10px] tabular-nums text-zinc-500">
          {someSelected && <>{selectedIds.size} selected / </>}
          {total} tracks
        </span>
      </div>

      {/* Generate Like This preview card */}
      {generateLikeResult && (
        <div className="mx-4 mt-3 rounded border border-sky-500/30 bg-sky-500/5 p-3 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-sky-300">
              Generated Prompt
            </span>
            <button
              onClick={clearGenerateLikeResult}
              className="text-zinc-500 hover:text-zinc-300 text-xs"
            >
              ×
            </button>
          </div>
          <p className="text-sm text-zinc-200">{generateLikeResult.prompt}</p>
          {/* Analysis tags */}
          <div className="flex flex-wrap gap-1">
            {generateLikeResult.top_genres.map((g) => (
              <span key={g} className="rounded bg-amber-500/10 px-1.5 py-0.5 text-[10px] text-amber-400">
                {g}
              </span>
            ))}
            {generateLikeResult.top_keys.map((k) => (
              <span key={k} className="rounded bg-purple-500/10 px-1.5 py-0.5 text-[10px] text-purple-400">
                {k}
              </span>
            ))}
            {generateLikeResult.bpm_median !== null && (
              <span className="rounded bg-emerald-500/10 px-1.5 py-0.5 text-[10px] text-emerald-400">
                {Math.round(generateLikeResult.bpm_median)} BPM
              </span>
            )}
          </div>
          <div className="flex gap-2">
            <button
              onClick={handleUseAndGenerate}
              className="rounded bg-sky-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-sky-500"
            >
              Use & Generate
            </button>
            <button
              onClick={handleEditPrompt}
              className="rounded border border-zinc-600 px-3 py-1.5 text-xs text-zinc-300 hover:border-zinc-500"
            >
              Edit
            </button>
          </div>
        </div>
      )}

      {/* Track table */}
      <div className="flex-1 overflow-auto">
        {loading ? (
          <div className="flex items-center justify-center py-12 text-sm text-zinc-500">
            Loading tracks...
          </div>
        ) : tracks.length === 0 ? (
          <div className="flex items-center justify-center py-12 text-sm text-zinc-500">
            {activePlaylistId ? "Empty playlist" : "No tracks found"}
          </div>
        ) : (
          <table className="w-full text-xs">
            <thead className="sticky top-0 bg-surface-0 z-10">
              <tr className="border-b border-zinc-700/50">
                <th className="w-8 px-2 py-1.5">
                  <input
                    type="checkbox"
                    checked={allSelected}
                    onChange={() => (allSelected ? deselectAll() : selectAll())}
                    className="accent-sky-500"
                  />
                </th>
                <th className="w-6 px-1 py-1.5" />
                <SortHeader label="Title" field="title" currentSort={searchParams.sort} currentOrder={searchParams.order ?? "asc"} onSort={handleSort} className="min-w-[120px]" />
                <SortHeader label="Artist" field="artist" currentSort={searchParams.sort} currentOrder={searchParams.order ?? "asc"} onSort={handleSort} className="min-w-[100px]" />
                <SortHeader label="Genre" field="genre" currentSort={searchParams.sort} currentOrder={searchParams.order ?? "asc"} onSort={handleSort} />
                <SortHeader label="BPM" field="bpm" currentSort={searchParams.sort} currentOrder={searchParams.order ?? "asc"} onSort={handleSort} className="w-16 text-right" />
                <SortHeader label="Key" field="key" currentSort={searchParams.sort} currentOrder={searchParams.order ?? "asc"} onSort={handleSort} className="w-12" />
                <SortHeader label="Dur" field="duration_seconds" currentSort={searchParams.sort} currentOrder={searchParams.order ?? "asc"} onSort={handleSort} className="w-14 text-right" />
                <SortHeader label="Rating" field="rating" currentSort={searchParams.sort} currentOrder={searchParams.order ?? "asc"} onSort={handleSort} className="w-20" />
                <th className="px-2 py-1.5 text-left text-[10px] font-medium uppercase tracking-wider text-zinc-500 w-16">
                  Enriched
                </th>
                <th className="px-2 py-1.5 text-left text-[10px] font-medium uppercase tracking-wider text-zinc-500 w-24">
                  Tags
                </th>
              </tr>
            </thead>
            <tbody>
              {tracks.map((track) => {
                const selected = selectedIds.has(track.track_id);
                return (
                  <tr
                    key={track.track_id}
                    className={`border-b border-zinc-800/50 transition-colors cursor-pointer ${
                      selected ? "bg-sky-500/5" : "hover:bg-zinc-800/30"
                    }`}
                    onClick={() => toggleSelection(track.track_id)}
                  >
                    <td className="px-2 py-1.5">
                      <input
                        type="checkbox"
                        checked={selected}
                        onChange={() => toggleSelection(track.track_id)}
                        onClick={(e) => e.stopPropagation()}
                        className="accent-sky-500"
                      />
                    </td>
                    <td className="px-1 py-1.5">
                      <AvailDot available={track.file_available} />
                    </td>
                    <td className="max-w-[200px] truncate px-2 py-1.5 text-zinc-200" title={track.title}>
                      {track.title}
                    </td>
                    <td className="max-w-[150px] truncate px-2 py-1.5 text-zinc-400" title={track.artist}>
                      {track.artist}
                    </td>
                    <td className="max-w-[100px] truncate px-2 py-1.5 text-zinc-400">
                      {track.genre}
                    </td>
                    <td className="px-2 py-1.5 text-right tabular-nums text-zinc-400">
                      {track.bpm !== null ? Math.round(track.bpm) : "-"}
                    </td>
                    <td className="px-2 py-1.5 tabular-nums text-zinc-400">
                      {track.key ?? "-"}
                    </td>
                    <td className="px-2 py-1.5 text-right tabular-nums text-zinc-400">
                      {fmtDuration(track.duration_seconds)}
                    </td>
                    <td className="px-2 py-1.5">
                      <Stars rating={track.rating} />
                    </td>
                    <td className="px-2 py-1.5">
                      <EnrichmentDots status={track.enrichment_status} />
                    </td>
                    <td className="px-2 py-1.5">
                      {(track.enrichment_status ?? "none") === "none" ? (
                        <span className="italic text-zinc-600 text-[10px]">Not enriched</span>
                      ) : (
                        <span className="rounded bg-emerald-500/10 px-1.5 py-0.5 text-[10px] text-emerald-400">
                          {track.enrichment_status}
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>

      {/* Bottom action bar */}
      {someSelected && (
        <div className="flex items-center gap-2 border-t border-zinc-700/50 px-4 py-3 bg-surface-1">
          <button
            onClick={handleGenerateLikeThis}
            disabled={generateLikeLoading}
            className="rounded bg-sky-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-sky-500 disabled:opacity-50"
          >
            {generateLikeLoading ? "Analyzing..." : "Generate Like This"}
          </button>
          <button
            onClick={onTrainOnThese}
            className="rounded border border-zinc-600 px-3 py-1.5 text-xs text-zinc-300 hover:border-zinc-500"
          >
            Train on These
          </button>
          <button
            onClick={handleEnrichSelected}
            disabled={enriching}
            className="rounded bg-purple-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-purple-500 disabled:opacity-50"
          >
            {enriching ? "Enriching..." : "Enrich Selected"}
          </button>
          <div className="flex-1" />
          <button
            onClick={deselectAll}
            className="text-[10px] text-zinc-500 hover:text-zinc-300"
          >
            Clear selection
          </button>
        </div>
      )}
    </div>
  );
}
