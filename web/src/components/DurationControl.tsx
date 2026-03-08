import { useState, useEffect } from "react";
import { useStore } from "../store/useStore";

type DurationMode = "seconds" | "bars";

/**
 * Calculate exact bar duration in seconds.
 * In 4/4 time: bar = 4 beats × (60 / BPM)
 */
function barDuration(bpm: number): number {
  return 4 * (60 / bpm);
}

export default function DurationControl() {
  const params = useStore((s) => s.params);
  const setParam = useStore((s) => s.setParam);
  const isGenerating = useStore((s) => s.isGenerating);

  const [mode, setMode] = useState<DurationMode>("seconds");
  const [bpm, setBpm] = useState(120);
  const [bars, setBars] = useState(4);

  // When in BPM mode, update the seconds param whenever BPM or bars change
  useEffect(() => {
    if (mode === "bars") {
      const exactSeconds = bars * barDuration(bpm);
      // Round to 6 decimal places to avoid floating point noise
      const rounded = Math.round(exactSeconds * 1000000) / 1000000;
      setParam("seconds", rounded);
    }
  }, [mode, bpm, bars, setParam]);

  return (
    <div className="space-y-2">
      {/* Mode toggle */}
      <div className="flex items-center justify-between">
        <label className="text-xs font-medium uppercase tracking-wider text-text-secondary">
          Duration
        </label>
        <div className="flex rounded border border-border overflow-hidden">
          <button
            onClick={() => setMode("seconds")}
            disabled={isGenerating}
            className={`
              px-2 py-0.5 text-xs transition-colors
              ${mode === "seconds" ? "bg-accent text-surface-0" : "bg-surface-2 text-text-secondary hover:text-text-primary"}
              disabled:opacity-50
            `}
          >
            Time
          </button>
          <button
            onClick={() => setMode("bars")}
            disabled={isGenerating}
            className={`
              px-2 py-0.5 text-xs transition-colors
              ${mode === "bars" ? "bg-accent text-surface-0" : "bg-surface-2 text-text-secondary hover:text-text-primary"}
              disabled:opacity-50
            `}
          >
            Bars
          </button>
        </div>
      </div>

      {mode === "seconds" ? (
        /* Seconds mode — direct slider */
        <div className="space-y-1">
          <div className="flex items-center justify-between">
            <span className="text-xs text-text-secondary">Length</span>
            <span className="text-xs tabular-nums text-text-primary">
              {params.seconds}s
            </span>
          </div>
          <input
            type="range"
            min={0.5}
            max={60}
            step={0.5}
            value={params.seconds}
            onChange={(e) => setParam("seconds", parseFloat(e.target.value))}
            disabled={isGenerating}
            className="w-full disabled:opacity-50"
          />
        </div>
      ) : (
        /* Bars mode — BPM + bar count → exact seconds */
        <div className="space-y-2">
          {/* BPM */}
          <div className="space-y-1">
            <div className="flex items-center justify-between">
              <span className="text-xs text-text-secondary">BPM</span>
              <input
                type="number"
                value={bpm}
                onChange={(e) =>
                  setBpm(Math.max(20, Math.min(300, parseInt(e.target.value) || 120)))
                }
                disabled={isGenerating}
                className="
                  w-16 rounded border border-border bg-surface-2 px-1.5 py-0.5
                  text-xs tabular-nums text-text-primary text-right
                  focus:border-accent focus:outline-none
                  disabled:opacity-50
                "
              />
            </div>
            <input
              type="range"
              min={40}
              max={240}
              step={1}
              value={bpm}
              onChange={(e) => setBpm(parseInt(e.target.value))}
              disabled={isGenerating}
              className="w-full disabled:opacity-50"
            />
          </div>

          {/* Bars */}
          <div className="space-y-1">
            <div className="flex items-center justify-between">
              <span className="text-xs text-text-secondary">Bars (4/4)</span>
              <span className="text-xs tabular-nums text-text-primary">
                {bars}
              </span>
            </div>
            <input
              type="range"
              min={1}
              max={32}
              step={1}
              value={bars}
              onChange={(e) => setBars(parseInt(e.target.value))}
              disabled={isGenerating}
              className="w-full disabled:opacity-50"
            />
          </div>

          {/* Calculated duration display */}
          <div className="rounded bg-surface-0 px-2 py-1.5 text-xs">
            <div className="flex justify-between text-text-secondary">
              <span>
                {bars} bar{bars > 1 ? "s" : ""} @ {bpm} BPM
              </span>
              <span className="tabular-nums text-accent font-medium">
                {(bars * barDuration(bpm)).toFixed(3)}s
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
