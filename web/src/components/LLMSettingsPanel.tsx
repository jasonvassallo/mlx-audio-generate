import { useEffect, useRef } from "react";
import { useStore } from "../store/useStore";
import { getMemoryExportUrl } from "../api/client";

export default function LLMSettingsPanel() {
  // --- Server settings ---
  const serverSettings = useStore((s) => s.serverSettings);
  const serverSettingsLoaded = useStore((s) => s.serverSettingsLoaded);
  const loadServerSettings = useStore((s) => s.loadServerSettings);
  const updateServerSetting = useStore((s) => s.updateServerSetting);

  // --- LLM models ---
  const llmModels = useStore((s) => s.llmModels);
  const llmStatus = useStore((s) => s.llmStatus);
  const loadLLMModels = useStore((s) => s.loadLLMModels);
  const selectLLM = useStore((s) => s.selectLLM);
  const loadLLMStatus = useStore((s) => s.loadLLMStatus);

  // --- Prompt memory ---
  const promptMemory = useStore((s) => s.promptMemory);
  const loadPromptMemory = useStore((s) => s.loadPromptMemory);
  const clearPromptMemory = useStore((s) => s.clearPromptMemory);
  const importPromptMemory = useStore((s) => s.importPromptMemory);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Load data on mount
  useEffect(() => {
    loadServerSettings();
    loadLLMModels();
    loadLLMStatus();
    loadPromptMemory();
  }, [loadServerSettings, loadLLMModels, loadLLMStatus, loadPromptMemory]);

  const handleImport = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      importPromptMemory(file);
      // Reset the input so the same file can be re-imported
      e.target.value = "";
    }
  };

  return (
    <div className="space-y-5">
      {/* --- AI Enhancement --- */}
      <section className="space-y-3">
        <h3 className="text-xs font-medium uppercase tracking-wider text-text-secondary">
          AI Enhancement
        </h3>

        {/* AI Enhance toggle */}
        <div className="flex items-center justify-between">
          <div>
            <label className="text-xs font-medium text-text-secondary">
              AI Enhance
            </label>
            <p className="text-xs text-text-muted mt-0.5">
              Enhance prompts before generation
            </p>
          </div>
          <button
            onClick={() =>
              updateServerSetting({
                ai_enhance: !serverSettings.ai_enhance,
              })
            }
            className={`
              relative h-6 w-11 rounded-full transition-colors
              ${serverSettings.ai_enhance ? "bg-accent" : "bg-border-light"}
            `}
          >
            <span
              className={`
                absolute top-0.5 h-5 w-5 rounded-full bg-white transition-transform
                ${serverSettings.ai_enhance ? "left-[22px]" : "left-0.5"}
              `}
            />
          </button>
        </div>

        {/* LLM Model selector */}
        <div className="space-y-1">
          <label className="text-xs font-medium text-text-secondary">
            LLM Model
          </label>
          <select
            value={serverSettings.llm_model ?? ""}
            onChange={(e) => {
              const modelId = e.target.value || null;
              updateServerSetting({ llm_model: modelId });
              if (modelId) selectLLM(modelId);
            }}
            className="
              w-full rounded border border-border bg-surface-2 px-2 py-1.5
              text-xs text-text-primary
              focus:border-accent focus:outline-none
            "
          >
            <option value="">None (template fallback)</option>
            {llmModels.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name} ({m.size_gb.toFixed(1)} GB — {m.source})
              </option>
            ))}
          </select>
          {llmStatus?.loaded && (
            <p className="text-xs text-success">
              Loaded — {llmStatus.memory_mb.toFixed(0)} MB,{" "}
              idle {Math.round(llmStatus.idle_seconds)}s
            </p>
          )}
          {!serverSettingsLoaded && (
            <p className="text-xs text-text-muted">Loading settings...</p>
          )}
        </div>

        {/* History context slider */}
        <div className="space-y-1">
          <div className="flex items-center justify-between">
            <label className="text-xs font-medium text-text-secondary">
              History Context
            </label>
            <span className="text-xs tabular-nums text-accent">
              {serverSettings.history_context_count}
            </span>
          </div>
          <input
            type="range"
            min={0}
            max={200}
            step={5}
            value={serverSettings.history_context_count}
            onChange={(e) =>
              updateServerSetting({
                history_context_count: parseInt(e.target.value),
              })
            }
            className="w-full"
          />
          <div className="flex justify-between text-xs text-text-muted">
            <span>0 (off)</span>
            <span>200</span>
          </div>
          <p className="text-xs text-text-muted">
            Recent prompts sent as LLM context for personalization.
          </p>
        </div>
      </section>

      {/* --- Prompt Memory --- */}
      <section className="space-y-3 border-t border-border pt-4">
        <h3 className="text-xs font-medium uppercase tracking-wider text-text-secondary">
          Prompt Memory
        </h3>

        {/* Style profile summary */}
        {promptMemory && (
          <div className="rounded bg-surface-2 p-2.5 space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="text-xs text-text-muted">Generations</span>
              <span className="text-xs tabular-nums text-text-primary">
                {promptMemory.style_profile.generation_count}
              </span>
            </div>
            {promptMemory.style_profile.preferred_duration > 0 && (
              <div className="flex items-center justify-between">
                <span className="text-xs text-text-muted">
                  Preferred duration
                </span>
                <span className="text-xs tabular-nums text-text-primary">
                  {promptMemory.style_profile.preferred_duration}s
                </span>
              </div>
            )}
            {promptMemory.style_profile.top_genres.length > 0 && (
              <div className="flex flex-wrap gap-1 mt-1">
                {promptMemory.style_profile.top_genres.map((g) => (
                  <span
                    key={g}
                    className="rounded bg-warning/20 px-1.5 py-0.5 text-xs text-warning"
                  >
                    {g}
                  </span>
                ))}
                {promptMemory.style_profile.top_moods.map((m) => (
                  <span
                    key={m}
                    className="rounded bg-success/20 px-1.5 py-0.5 text-xs text-success"
                  >
                    {m}
                  </span>
                ))}
                {promptMemory.style_profile.top_instruments.map((i) => (
                  <span
                    key={i}
                    className="rounded bg-info/20 px-1.5 py-0.5 text-xs text-info"
                  >
                    {i}
                  </span>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Memory actions */}
        <div className="flex gap-2">
          <a
            href={getMemoryExportUrl()}
            download="prompt_memory.json"
            className="rounded bg-surface-3 px-3 py-1 text-xs text-text-muted hover:text-text-secondary"
          >
            Export
          </a>
          <button
            onClick={() => fileInputRef.current?.click()}
            className="rounded bg-surface-3 px-3 py-1 text-xs text-text-muted hover:text-text-secondary"
          >
            Import
          </button>
          <button
            onClick={clearPromptMemory}
            className="rounded bg-error/20 px-3 py-1 text-xs text-error hover:bg-error/30"
          >
            Clear
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".json"
            onChange={handleImport}
            className="hidden"
          />
        </div>
      </section>
    </div>
  );
}
