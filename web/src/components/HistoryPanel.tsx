import { useStore } from "../store/useStore";
import AudioPlayer from "./AudioPlayer";

export default function HistoryPanel() {
  const history = useStore((s) => s.history);
  const historyLoaded = useStore((s) => s.historyLoaded);
  const clearHistory = useStore((s) => s.clearHistory);
  const toggleFavorite = useStore((s) => s.toggleFavorite);
  const deleteHistoryEntry = useStore((s) => s.deleteHistoryEntry);

  if (!historyLoaded) {
    return (
      <div className="flex flex-1 items-center justify-center text-sm text-text-muted">
        Loading history...
      </div>
    );
  }

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
            key={entry.id}
            className={`
              rounded border bg-surface-1 p-3 space-y-2
              ${entry.favorite ? "border-accent/40" : "border-border"}
            `}
          >
            {/* Job metadata + actions */}
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
                      s gen
                    </span>
                  )}
                  <span>
                    {new Date(entry.createdAt).toLocaleTimeString()}
                  </span>
                </div>
              </div>

              {/* Action buttons */}
              <div className="flex items-center gap-1 shrink-0">
                {/* Favorite toggle */}
                <button
                  onClick={() => toggleFavorite(entry.id)}
                  className={`
                    p-1 rounded transition-colors
                    ${
                      entry.favorite
                        ? "text-accent hover:text-accent-hover"
                        : "text-text-muted hover:text-text-secondary"
                    }
                  `}
                  title={
                    entry.favorite
                      ? "Remove from favorites"
                      : "Add to favorites (protected from auto-delete)"
                  }
                >
                  {entry.favorite ? (
                    <svg
                      width="14"
                      height="14"
                      viewBox="0 0 24 24"
                      fill="currentColor"
                    >
                      <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z" />
                    </svg>
                  ) : (
                    <svg
                      width="14"
                      height="14"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                    >
                      <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z" />
                    </svg>
                  )}
                </button>

                {/* Delete */}
                <button
                  onClick={() => deleteHistoryEntry(entry.id)}
                  className="p-1 rounded text-text-muted hover:text-error transition-colors"
                  title="Delete this generation"
                >
                  <svg
                    width="14"
                    height="14"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                  >
                    <path d="M18 6L6 18M6 6l12 12" />
                  </svg>
                </button>

                {/* Entry number */}
                <span className="text-xs tabular-nums text-text-muted ml-1">
                  #{history.length - i}
                </span>
              </div>
            </div>

            {/* Audio player */}
            <AudioPlayer
              src={entry.audioUrl}
              title={`${entry.job.model}_${entry.id}`}
              autoPlay={i === 0 && !entry.favorite}
            />
          </div>
        ))}
      </div>
    </div>
  );
}
