import { useStore } from "../store/useStore";
import AudioDeviceSelector from "./AudioDeviceSelector";

interface TransportBarProps {
  connected: boolean;
}

export default function TransportBar({ connected }: TransportBarProps) {
  const settings = useStore((s) => s.settings);
  const updateSettings = useStore((s) => s.updateSettings);
  const isGenerating = useStore((s) => s.isGenerating);
  const activeJob = useStore((s) => s.activeJob);
  const serverUrl = useStore((s) => s.serverUrl);

  const progress = activeJob?.progress ?? 0;
  const progressPct = Math.round(progress * 100);

  return (
    <div className="flex h-10 shrink-0 items-center gap-6 border-t border-border bg-surface-1 px-4">
      {/* Connection status */}
      <div className="flex items-center gap-1.5 shrink-0">
        <span
          className={`h-2 w-2 rounded-full ${
            connected
              ? serverUrl
                ? "bg-info"        /* blue = remote */
                : "bg-success"     /* green = local */
              : "bg-text-muted"    /* gray = disconnected */
          }`}
        />
        <span className="text-xs text-text-muted">
          {connected
            ? serverUrl
              ? "Remote"
              : "Local"
            : "Offline"}
        </span>
      </div>

      {/* Divider */}
      <div className="h-4 w-px bg-border" />

      {/* Master BPM — compact inline control */}
      <div className="flex items-center gap-2 shrink-0">
        <label className="text-xs text-text-muted">BPM</label>
        <input
          type="number"
          min={40}
          max={240}
          value={settings.masterBpm}
          onChange={(e) =>
            updateSettings({
              masterBpm: Math.max(40, Math.min(240, parseInt(e.target.value) || 120)),
            })
          }
          className="
            w-14 rounded border border-border bg-surface-2 px-1.5 py-0.5
            text-xs tabular-nums text-accent text-center
            focus:border-accent focus:outline-none
          "
        />
        <input
          type="range"
          min={40}
          max={240}
          step={1}
          value={settings.masterBpm}
          onChange={(e) =>
            updateSettings({ masterBpm: parseInt(e.target.value) })
          }
          className="w-24"
        />
      </div>

      {/* Divider */}
      <div className="h-4 w-px bg-border" />

      {/* Pitch mode toggle — compact pill */}
      <button
        onClick={() =>
          updateSettings({ preservePitch: !settings.preservePitch })
        }
        className={`
          flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs transition-colors
          ${
            settings.preservePitch
              ? "bg-accent/15 text-accent border border-accent/30"
              : "bg-surface-2 text-text-secondary border border-border hover:border-border-light"
          }
        `}
        title={
          settings.preservePitch
            ? "Time-Stretch: tempo changes, pitch stays locked"
            : "Vinyl: pitch follows tempo (like a turntable)"
        }
      >
        {/* Musical note icon */}
        <svg
          width="10"
          height="10"
          viewBox="0 0 24 24"
          fill="currentColor"
        >
          <path d="M12 3v10.55c-.59-.34-1.27-.55-2-.55C7.79 13 6 14.79 6 17s1.79 4 4 4 4-1.79 4-4V7h4V3h-6z" />
        </svg>
        {settings.preservePitch ? "Stretch" : "Vinyl"}
      </button>

      {/* Spacer */}
      <div className="flex-1" />

      {/* Generation progress (when active) */}
      {isGenerating && (
        <div className="flex items-center gap-2 shrink-0">
          <div className="h-1 w-20 overflow-hidden rounded-full bg-surface-3">
            <div
              className="h-full rounded-full bg-accent transition-all duration-300"
              style={{ width: `${Math.max(2, progressPct)}%` }}
            />
          </div>
          <span className="text-xs tabular-nums text-accent">
            {progressPct}%
          </span>
        </div>
      )}

      {/* Audio device selector — inline */}
      <div className="shrink-0">
        <AudioDeviceSelector compact />
      </div>
    </div>
  );
}
