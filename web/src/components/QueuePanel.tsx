import { useStore } from "../store/useStore";

export default function QueuePanel() {
  const queue = useStore((s) => s.queue);
  const queueRunning = useStore((s) => s.queueRunning);
  const queueProgress = useStore((s) => s.queueProgress);
  const addToQueue = useStore((s) => s.addToQueue);
  const removeFromQueue = useStore((s) => s.removeFromQueue);
  const clearQueue = useStore((s) => s.clearQueue);
  const runQueue = useStore((s) => s.runQueue);

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-medium uppercase tracking-wider text-text-secondary">
          Queue ({queue.length})
        </h3>
        <div className="flex gap-2">
          {queue.length > 0 && !queueRunning && (
            <button
              onClick={clearQueue}
              className="text-xs text-text-muted hover:text-error transition-colors"
            >
              Clear
            </button>
          )}
        </div>
      </div>

      {/* Add current params to queue */}
      <button
        onClick={() => addToQueue()}
        disabled={queueRunning}
        className="
          w-full rounded border border-dashed border-border py-2
          text-xs text-text-muted hover:text-accent hover:border-accent/40
          transition-colors disabled:opacity-50
        "
      >
        + Add current prompt to queue
      </button>

      {/* Queue items */}
      {queue.length > 0 && (
        <div className="space-y-1.5 max-h-48 overflow-y-auto">
          {queue.map((item, i) => (
            <div
              key={i}
              className={`
                flex items-center gap-2 rounded bg-surface-2 px-2 py-1.5
                ${queueProgress && i < queueProgress.current ? "opacity-40" : ""}
                ${queueProgress && i === queueProgress.current - 1 ? "border border-accent/40" : "border border-transparent"}
              `}
            >
              <span className="text-xs tabular-nums text-text-muted w-5">
                {i + 1}.
              </span>
              <span className="flex-1 text-xs text-text-primary truncate">
                {item.prompt}
              </span>
              <span className="text-[10px] text-text-muted shrink-0">
                {item.model === "stable_audio" ? "SA" : "MG"} {item.seconds}s
              </span>
              {!queueRunning && (
                <button
                  onClick={() => removeFromQueue(i)}
                  className="text-text-muted hover:text-error transition-colors shrink-0"
                >
                  <svg
                    width="10"
                    height="10"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                  >
                    <path d="M18 6L6 18M6 6l12 12" />
                  </svg>
                </button>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Run / Cancel queue */}
      {queue.length > 0 && (
        <div className="space-y-2">
          {queueRunning ? (
            <>
              <div className="h-1.5 w-full overflow-hidden rounded-full bg-surface-3">
                <div
                  className="h-full rounded-full bg-accent transition-all duration-300"
                  style={{
                    width: `${queueProgress ? (queueProgress.current / queueProgress.total) * 100 : 0}%`,
                  }}
                />
              </div>
              <div className="flex items-center justify-between">
                <span className="text-xs text-text-muted">
                  {queueProgress?.current}/{queueProgress?.total} complete
                </span>
                <button
                  onClick={() => useStore.setState({ queueRunning: false })}
                  className="text-xs text-error hover:text-error/80"
                >
                  Cancel
                </button>
              </div>
            </>
          ) : (
            <button
              onClick={runQueue}
              className="
                w-full rounded py-2 text-xs font-bold uppercase tracking-wider
                bg-accent/20 text-accent hover:bg-accent/30
                transition-colors
              "
            >
              Run Queue ({queue.length} items)
            </button>
          )}
        </div>
      )}
    </div>
  );
}
