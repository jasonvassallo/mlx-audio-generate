import { useStore } from "../store/useStore";

const RETENTION_OPTIONS = [
  { label: "Keep forever", value: 0 },
  { label: "1 hour", value: 1 },
  { label: "6 hours", value: 6 },
  { label: "24 hours", value: 24 },
  { label: "7 days", value: 168 },
  { label: "30 days", value: 720 },
];

export default function SettingsPanel() {
  const settings = useStore((s) => s.settings);
  const updateSettings = useStore((s) => s.updateSettings);
  const history = useStore((s) => s.history);
  const favoriteCount = history.filter((h) => h.favorite).length;

  return (
    <div className="space-y-4">
      {/* Master Playback */}
      <div className="space-y-3">
        <h3 className="text-xs font-medium uppercase tracking-wider text-text-secondary">
          Master Playback
        </h3>

        {/* Master BPM */}
        <div className="space-y-1">
          <div className="flex items-center justify-between">
            <label className="text-xs font-medium text-text-secondary">
              Target BPM
            </label>
            <span className="text-xs tabular-nums text-accent">
              {settings.masterBpm}
            </span>
          </div>
          <input
            type="range"
            min={40}
            max={240}
            step={1}
            value={settings.masterBpm}
            onChange={(e) =>
              updateSettings({ masterBpm: parseInt(e.target.value) })
            }
            className="w-full"
          />
          <div className="flex justify-between text-xs text-text-muted">
            <span>40</span>
            <span>240</span>
          </div>
          <p className="text-xs text-text-muted">
            Set each clip's source BPM to enable tempo sync.
          </p>
        </div>

        {/* Pitch mode toggle */}
        <div className="flex items-center justify-between">
          <div>
            <label className="text-xs font-medium text-text-secondary">
              {settings.preservePitch ? "Time-Stretch" : "Vinyl / Turntable"}
            </label>
            <p className="text-xs text-text-muted mt-0.5">
              {settings.preservePitch
                ? "Tempo changes, pitch stays locked"
                : "Pitch follows tempo (like vinyl)"}
            </p>
          </div>
          <button
            onClick={() =>
              updateSettings({ preservePitch: !settings.preservePitch })
            }
            className={`
              relative h-6 w-11 rounded-full transition-colors
              ${settings.preservePitch ? "bg-accent" : "bg-border-light"}
            `}
          >
            <span
              className={`
                absolute top-0.5 h-5 w-5 rounded-full bg-white transition-transform
                ${settings.preservePitch ? "left-[22px]" : "left-0.5"}
              `}
            />
          </button>
        </div>
      </div>

      {/* History Settings */}
      <div className="space-y-3 pt-3 border-t border-border">
        <h3 className="text-xs font-medium uppercase tracking-wider text-text-secondary">
          History
        </h3>

        {/* Auto-delete retention */}
        <div className="space-y-1">
          <label className="text-xs font-medium text-text-secondary">
            Auto-delete after
          </label>
          <select
            value={settings.retentionHours}
            onChange={(e) =>
              updateSettings({ retentionHours: parseInt(e.target.value) })
            }
            className="
              w-full rounded border border-border bg-surface-2 px-2 py-1.5
              text-xs text-text-primary
              focus:border-accent focus:outline-none
            "
          >
            {RETENTION_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
          <p className="text-xs text-text-muted">
            Favorites are never auto-deleted.
            {favoriteCount > 0 && (
              <span className="text-accent"> {favoriteCount} favorited.</span>
            )}
          </p>
        </div>
      </div>
    </div>
  );
}
