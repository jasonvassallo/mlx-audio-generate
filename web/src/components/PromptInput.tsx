import { useStore } from "../store/useStore";

export default function PromptInput() {
  const prompt = useStore((s) => s.params.prompt);
  const negativePrompt = useStore((s) => s.params.negative_prompt);
  const model = useStore((s) => s.params.model);
  const setParam = useStore((s) => s.setParam);
  const isGenerating = useStore((s) => s.isGenerating);
  const generate = useStore((s) => s.generate);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      generate();
    }
  };

  return (
    <div className="space-y-3">
      <div className="space-y-1.5">
        <label className="block text-xs font-medium uppercase tracking-wider text-text-secondary">
          Prompt
        </label>
        <textarea
          value={prompt}
          onChange={(e) => setParam("prompt", e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={isGenerating}
          placeholder="Describe the audio you want to generate..."
          rows={3}
          className="
            w-full resize-none rounded border border-border bg-surface-2 px-3 py-2
            text-sm text-text-primary placeholder:text-text-muted
            focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent-dim
            disabled:opacity-50
          "
        />
        <div className="flex items-center justify-between text-xs text-text-muted">
          <span>{prompt.length} / 5000</span>
          <span>Cmd+Enter to generate</span>
        </div>
      </div>

      {model === "stable_audio" && (
        <div className="space-y-1.5">
          <label className="block text-xs font-medium uppercase tracking-wider text-text-secondary">
            Negative Prompt
          </label>
          <textarea
            value={negativePrompt ?? ""}
            onChange={(e) => setParam("negative_prompt", e.target.value)}
            disabled={isGenerating}
            placeholder="What to avoid in the generated audio..."
            rows={2}
            className="
              w-full resize-none rounded border border-border bg-surface-2 px-3 py-2
              text-sm text-text-primary placeholder:text-text-muted
              focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent-dim
              disabled:opacity-50
            "
          />
        </div>
      )}
    </div>
  );
}
