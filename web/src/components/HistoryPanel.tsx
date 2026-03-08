import { useStore } from "../store/useStore";
import AudioPlayer from "./AudioPlayer";

export default function HistoryPanel() {
  const history = useStore((s) => s.history);
  const clearHistory = useStore((s) => s.clearHistory);

  if (history.length === 0) {
    return (
      <div className="flex flex-1 items-center justify-center text-sm text-text-muted">
        Generated audio will appear here
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-medium uppercase tracking-wider text-text-secondary">
          History ({history.length})
        </h3>
        <button
          onClick={clearHistory}
          className="text-xs text-text-muted hover:text-error transition-colors"
        >
          Clear all
        </button>
      </div>

      <div className="space-y-3 overflow-y-auto">
        {history.map((entry, i) => (
          <div
            key={entry.job.id}
            className="rounded border border-border bg-surface-1 p-3 space-y-2"
          >
            {/* Job metadata */}
            <div className="flex items-start justify-between gap-2">
              <div className="flex-1 min-w-0">
                <p className="text-xs text-text-primary truncate">
                  {entry.job.prompt}
                </p>
                <div className="flex gap-2 mt-1 text-xs text-text-muted">
                  <span>
                    {entry.job.model === "musicgen"
                      ? "MusicGen"
                      : "Stable Audio"}
                  </span>
                  <span>{entry.job.seconds}s</span>
                  {entry.job.completed_at && entry.job.created_at && (
                    <span>
                      {(
                        entry.job.completed_at - entry.job.created_at
                      ).toFixed(1)}
                      s gen time
                    </span>
                  )}
                </div>
              </div>
              <span className="shrink-0 text-xs tabular-nums text-text-muted">
                #{history.length - i}
              </span>
            </div>

            {/* Audio player */}
            <AudioPlayer
              src={entry.audioUrl}
              title={`${entry.job.model}_${entry.job.id}`}
              autoPlay={i === 0}
            />
          </div>
        ))}
      </div>
    </div>
  );
}
