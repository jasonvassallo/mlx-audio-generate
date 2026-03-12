import { useState } from "react";
import { useStore } from "../store/useStore";

export default function EnhancePreview() {
  const enhanceResult = useStore((s) => s.enhanceResult);
  const clearEnhanceResult = useStore((s) => s.clearEnhanceResult);
  const setParam = useStore((s) => s.setParam);
  const generate = useStore((s) => s.generate);
  const isGenerating = useStore((s) => s.isGenerating);

  const [editing, setEditing] = useState(false);
  const [editText, setEditText] = useState("");

  if (!enhanceResult) return null;

  const { original, enhanced, analysis_tags, used_llm, warning } =
    enhanceResult;

  const handleAcceptAndGenerate = () => {
    setParam("prompt", enhanced);
    clearEnhanceResult();
    generate();
  };

  const handleEdit = () => {
    setEditing(true);
    setEditText(enhanced);
  };

  const handleSaveEdit = () => {
    setParam("prompt", editText);
    clearEnhanceResult();
    setEditing(false);
  };

  const handleUseOriginal = () => {
    clearEnhanceResult();
  };

  return (
    <div className="rounded border border-accent/30 bg-surface-2 p-3 space-y-3">
      <div className="flex items-center justify-between">
        <h4 className="text-xs font-medium uppercase tracking-wider text-accent">
          Enhanced Prompt
        </h4>
        {!used_llm && (
          <span className="text-xs text-warning">Template fallback</span>
        )}
      </div>

      {warning && (
        <div className="rounded bg-warning/10 px-2 py-1.5 text-xs text-warning">
          {warning}
        </div>
      )}

      {/* Analysis tags */}
      {analysis_tags && (
        <div className="flex flex-wrap gap-1">
          {analysis_tags.genres.map((g) => (
            <span
              key={`g-${g}`}
              className="rounded bg-warning/20 px-1.5 py-0.5 text-xs text-warning"
            >
              {g}
            </span>
          ))}
          {analysis_tags.moods.map((m) => (
            <span
              key={`m-${m}`}
              className="rounded bg-success/20 px-1.5 py-0.5 text-xs text-success"
            >
              {m}
            </span>
          ))}
          {analysis_tags.instruments.map((i) => (
            <span
              key={`i-${i}`}
              className="rounded bg-info/20 px-1.5 py-0.5 text-xs text-info"
            >
              {i}
            </span>
          ))}
          {analysis_tags.missing.map((m) => (
            <span
              key={`x-${m}`}
              className="rounded bg-surface-3 px-1.5 py-0.5 text-xs text-text-muted"
            >
              + {m}
            </span>
          ))}
        </div>
      )}

      {/* Enhanced text or edit field */}
      {editing ? (
        <textarea
          value={editText}
          onChange={(e) => setEditText(e.target.value)}
          rows={3}
          className="
            w-full resize-none rounded border border-border bg-surface-1 px-3 py-2
            text-sm text-text-primary
            focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent-dim
          "
        />
      ) : (
        <div className="space-y-1">
          <p className="text-xs text-text-primary leading-relaxed">
            {enhanced}
          </p>
          {enhanced !== original && (
            <p className="text-xs text-text-muted">
              Original: {original}
            </p>
          )}
        </div>
      )}

      {/* Action buttons */}
      <div className="flex gap-2">
        {editing ? (
          <>
            <button
              onClick={handleSaveEdit}
              disabled={!editText.trim()}
              className="rounded bg-accent px-3 py-1 text-xs font-medium text-surface-0 disabled:opacity-50"
            >
              Use Edited
            </button>
            <button
              onClick={() => setEditing(false)}
              className="rounded bg-surface-3 px-3 py-1 text-xs text-text-muted hover:text-text-secondary"
            >
              Cancel
            </button>
          </>
        ) : (
          <>
            <button
              onClick={handleAcceptAndGenerate}
              disabled={isGenerating}
              className="rounded bg-accent px-3 py-1 text-xs font-medium text-surface-0 disabled:opacity-50"
            >
              Accept & Generate
            </button>
            <button
              onClick={handleEdit}
              className="rounded bg-surface-3 px-3 py-1 text-xs text-text-muted hover:text-text-secondary"
            >
              Edit
            </button>
            <button
              onClick={handleUseOriginal}
              className="rounded bg-surface-3 px-3 py-1 text-xs text-text-muted hover:text-text-secondary"
            >
              Use Original
            </button>
          </>
        )}
      </div>
    </div>
  );
}
