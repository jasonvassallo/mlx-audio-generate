import { useStore } from "../store/useStore";

export default function ModelSelector() {
  const models = useStore((s) => s.models);
  const currentModel = useStore((s) => s.params.model);
  const setParam = useStore((s) => s.setParam);
  const isGenerating = useStore((s) => s.isGenerating);

  // Group models by type
  const modelTypes = [...new Set(models.map((m) => m.model_type))];

  return (
    <div className="space-y-2">
      <label className="block text-xs font-medium uppercase tracking-wider text-text-secondary">
        Model
      </label>
      <div className="flex gap-2">
        {modelTypes.length === 0 ? (
          <div className="text-xs text-text-muted">No models available</div>
        ) : (
          modelTypes.map((type) => {
            const isActive = currentModel === type;
            const modelsOfType = models.filter((m) => m.model_type === type);
            const anyLoaded = modelsOfType.some((m) => m.is_loaded);

            return (
              <button
                key={type}
                onClick={() =>
                  setParam("model", type as "musicgen" | "stable_audio")
                }
                disabled={isGenerating}
                className={`
                  flex items-center gap-2 rounded px-4 py-2 text-xs font-medium
                  transition-all duration-150
                  ${
                    isActive
                      ? "bg-accent text-surface-0"
                      : "bg-surface-3 text-text-secondary hover:bg-border-light hover:text-text-primary"
                  }
                  disabled:opacity-50 disabled:cursor-not-allowed
                `}
              >
                <span>{type === "musicgen" ? "MusicGen" : "Stable Audio"}</span>
                {anyLoaded && (
                  <span
                    className={`h-1.5 w-1.5 rounded-full ${
                      isActive ? "bg-surface-0" : "bg-success"
                    }`}
                  />
                )}
              </button>
            );
          })
        )}
      </div>
      {/* Show specific model name */}
      {models.length > 0 && (
        <div className="text-xs text-text-muted">
          {models
            .filter((m) => m.model_type === currentModel)
            .map((m) => m.name)
            .join(", ")}
        </div>
      )}
    </div>
  );
}
