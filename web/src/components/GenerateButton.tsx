import { useStore } from "../store/useStore";

export default function GenerateButton() {
  const generate = useStore((s) => s.generate);
  const isGenerating = useStore((s) => s.isGenerating);
  const activeJob = useStore((s) => s.activeJob);
  const generateError = useStore((s) => s.generateError);

  // Progress estimation: ~2x realtime (same as Max for Live client)
  const progress = (() => {
    if (!activeJob || !isGenerating) return 0;
    const elapsed = (Date.now() / 1000 - activeJob.created_at) * 1000;
    const estimated = activeJob.seconds * 1000 * 2;
    return Math.min(95, (elapsed / estimated) * 100);
  })();

  return (
    <div className="space-y-2">
      <button
        onClick={generate}
        disabled={isGenerating}
        className="
          w-full rounded py-3 text-sm font-bold uppercase tracking-wider
          transition-all duration-150
          bg-accent text-surface-0 hover:bg-accent-hover
          active:scale-[0.98]
          disabled:opacity-70 disabled:cursor-not-allowed disabled:active:scale-100
        "
      >
        {isGenerating ? "Generating..." : "Generate"}
      </button>

      {/* Progress bar */}
      {isGenerating && (
        <div className="space-y-1">
          <div className="h-1 w-full overflow-hidden rounded-full bg-surface-3">
            <div
              className="h-full rounded-full bg-accent transition-all duration-500"
              style={{ width: `${progress}%` }}
            />
          </div>
          <div className="text-center text-xs text-text-muted">
            {activeJob?.status === "queued" && "Queued..."}
            {activeJob?.status === "running" && "Generating audio..."}
          </div>
        </div>
      )}

      {/* Error display */}
      {generateError && !isGenerating && (
        <div className="rounded border border-error/30 bg-error/10 px-3 py-2 text-xs text-error">
          {generateError}
        </div>
      )}
    </div>
  );
}
