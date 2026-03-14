import { useCallback, useEffect, useState } from "react";
import { useStore } from "../store/useStore";
import {
  createCollection,
  describeLibraryTracks,
  suggestAdapterName,
  startTraining,
} from "../api/client";
import type { LibraryTrackInfo } from "../types/api";

type Profile = "quick" | "balanced" | "deep";

const PROFILES: Record<Profile, { label: string; desc: string }> = {
  quick: { label: "Quick & Light", desc: "rank 8, q+v only" },
  balanced: { label: "Balanced", desc: "rank 16, q+v+out" },
  deep: { label: "Deep", desc: "rank 32, all projections" },
};

interface Props {
  tracks: LibraryTrackInfo[];
  sourceId: string;
  onClose: () => void;
}

export default function MetadataEditor({ tracks, sourceId, onClose }: Props) {
  const models = useStore((s) => s.models);
  const fetchLoras = useStore((s) => s.fetchLoras);
  const loadCollections = useStore((s) => s.loadCollections);

  // Descriptions per track ID
  const [descriptions, setDescriptions] = useState<Record<string, string>>({});
  const [descriptionsLoading, setDescriptionsLoading] = useState(false);

  // Collection + training config
  const [adapterName, setAdapterName] = useState("");
  const [nameLoading, setNameLoading] = useState(false);
  const [baseModel, setBaseModel] = useState("musicgen-small");
  const [profile, setProfile] = useState<Profile>("balanced");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const musicgenModels = models.filter((m) => m.model_type === "musicgen");
  const nameValid = /^[a-zA-Z0-9_-]{1,64}$/.test(adapterName);

  // Auto-generate descriptions and suggest name on mount
  useEffect(() => {
    const trackIds = tracks.map((t) => t.track_id);

    // Generate descriptions
    setDescriptionsLoading(true);
    describeLibraryTracks(sourceId, trackIds)
      .then((res) => {
        setDescriptions(res.descriptions);
      })
      .catch((e) => console.error("Failed to generate descriptions:", e))
      .finally(() => setDescriptionsLoading(false));

    // Suggest adapter name
    setNameLoading(true);
    suggestAdapterName(sourceId, trackIds)
      .then((res) => {
        setAdapterName(res.name);
      })
      .catch((e) => console.error("Failed to suggest name:", e))
      .finally(() => setNameLoading(false));
  }, [tracks, sourceId]);

  const updateDescription = useCallback((trackId: string, desc: string) => {
    setDescriptions((prev) => ({ ...prev, [trackId]: desc }));
  }, []);

  const handleSaveCollection = useCallback(async (andTrain: boolean) => {
    if (!adapterName || !nameValid) return;
    setSaving(true);
    setError(null);
    setSuccess(null);

    try {
      // Build track data with descriptions
      const trackData = tracks.map((t) => ({
        ...t,
        description: descriptions[t.track_id] ?? "",
        description_edited: descriptions[t.track_id] !== undefined,
      }));

      // Create collection
      await createCollection({
        name: adapterName,
        source: sourceId,
        tracks: trackData,
      });

      await loadCollections();

      if (andTrain) {
        // Start training with collection
        await startTraining({
          collection: adapterName,
          base_model: baseModel,
          name: adapterName,
          profile,
        });
        await fetchLoras();
        setSuccess("Collection saved and training started");
      } else {
        setSuccess("Collection saved");
      }

      // Close after a brief delay so user sees success
      setTimeout(onClose, 1500);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to save");
    } finally {
      setSaving(false);
    }
  }, [adapterName, nameValid, tracks, descriptions, sourceId, baseModel, profile, loadCollections, fetchLoras, onClose]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="flex max-h-[85vh] w-[700px] flex-col rounded-lg border border-zinc-700 bg-zinc-900 shadow-xl">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-zinc-700 px-5 py-3">
          <h2 className="text-sm font-medium text-zinc-200">
            Curate Collection ({tracks.length} tracks)
          </h2>
          <button
            onClick={onClose}
            className="text-zinc-500 hover:text-zinc-300 text-lg"
          >
            ×
          </button>
        </div>

        {/* Config section */}
        <div className="space-y-3 border-b border-zinc-700/50 px-5 py-4">
          <div className="flex gap-3">
            {/* Adapter name */}
            <div className="flex-1 space-y-1">
              <label className="text-[10px] text-zinc-500">Adapter / Collection Name</label>
              <input
                type="text"
                value={adapterName}
                onChange={(e) => setAdapterName(e.target.value)}
                placeholder={nameLoading ? "Suggesting..." : "my-style"}
                className={`w-full rounded bg-zinc-800 px-2 py-1.5 text-sm text-zinc-200 border focus:outline-none placeholder:text-zinc-600 ${
                  adapterName && !nameValid
                    ? "border-red-500"
                    : "border-zinc-700 focus:border-sky-500"
                }`}
              />
            </div>
            {/* Base model */}
            <div className="w-40 space-y-1">
              <label className="text-[10px] text-zinc-500">Base Model</label>
              <select
                value={baseModel}
                onChange={(e) => setBaseModel(e.target.value)}
                className="w-full rounded bg-zinc-800 px-2 py-1.5 text-sm text-zinc-200 border border-zinc-700 focus:border-sky-500 focus:outline-none"
              >
                {musicgenModels.length > 0 ? (
                  musicgenModels.map((m) => (
                    <option key={m.name} value={m.name}>{m.name}</option>
                  ))
                ) : (
                  <option value="musicgen-small">musicgen-small</option>
                )}
              </select>
            </div>
          </div>

          {/* Profile cards */}
          <div className="grid grid-cols-3 gap-1.5">
            {(Object.entries(PROFILES) as [Profile, { label: string; desc: string }][]).map(
              ([key, { label, desc }]) => (
                <button
                  key={key}
                  onClick={() => setProfile(key)}
                  className={`rounded border px-2 py-2 text-center transition-colors ${
                    profile === key
                      ? "border-sky-500 bg-sky-500/10 text-sky-400"
                      : "border-zinc-700 bg-zinc-800 text-zinc-400 hover:border-zinc-600"
                  }`}
                >
                  <div className="text-xs font-medium">{label}</div>
                  <div className="text-[10px] text-zinc-500">{desc}</div>
                </button>
              ),
            )}
          </div>
        </div>

        {/* Track descriptions table */}
        <div className="flex-1 overflow-y-auto px-5 py-3">
          {descriptionsLoading ? (
            <p className="py-4 text-center text-xs text-zinc-500">
              Generating descriptions...
            </p>
          ) : (
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-zinc-700/50">
                  <th className="px-2 py-1.5 text-left text-[10px] font-medium uppercase tracking-wider text-zinc-500 w-1/3">
                    Track
                  </th>
                  <th className="px-2 py-1.5 text-left text-[10px] font-medium uppercase tracking-wider text-zinc-500">
                    Description
                  </th>
                </tr>
              </thead>
              <tbody>
                {tracks.map((track) => (
                  <tr key={track.track_id} className="border-b border-zinc-800/50">
                    <td className="px-2 py-2 align-top">
                      <div className="text-zinc-200 truncate" title={track.title}>
                        {track.title}
                      </div>
                      <div className="text-[10px] text-zinc-500 truncate">
                        {track.artist}
                        {track.bpm !== null && ` · ${Math.round(track.bpm)} BPM`}
                        {track.key && ` · ${track.key}`}
                      </div>
                    </td>
                    <td className="px-2 py-2">
                      <textarea
                        value={descriptions[track.track_id] ?? ""}
                        onChange={(e) => updateDescription(track.track_id, e.target.value)}
                        rows={2}
                        className="w-full rounded bg-zinc-800 px-2 py-1 text-xs text-zinc-200 border border-zinc-700 focus:border-sky-500 focus:outline-none resize-none placeholder:text-zinc-600"
                        placeholder="Describe this track..."
                      />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        {/* Footer actions */}
        <div className="flex items-center gap-2 border-t border-zinc-700 px-5 py-3">
          {error && <p className="text-xs text-red-400">{error}</p>}
          {success && <p className="text-xs text-emerald-400">{success}</p>}
          <div className="flex-1" />
          <button
            onClick={() => handleSaveCollection(false)}
            disabled={saving || !adapterName || !nameValid}
            className="rounded border border-zinc-600 px-3 py-1.5 text-xs text-zinc-300 hover:border-zinc-500 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Save Collection
          </button>
          <button
            onClick={() => handleSaveCollection(true)}
            disabled={saving || !adapterName || !nameValid}
            className="rounded bg-sky-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-sky-500 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {saving ? "Saving..." : "Save & Train"}
          </button>
        </div>
      </div>
    </div>
  );
}
