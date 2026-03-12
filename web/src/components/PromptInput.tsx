import { useState } from "react";
import { useStore } from "../store/useStore";
import TagAutocomplete from "./TagAutocomplete";

export default function PromptInput() {
  const prompt = useStore((s) => s.params.prompt);
  const negativePrompt = useStore((s) => s.params.negative_prompt);
  const model = useStore((s) => s.params.model);
  const setParam = useStore((s) => s.setParam);
  const isGenerating = useStore((s) => s.isGenerating);
  const generate = useStore((s) => s.generate);
  const tagDatabase = useStore((s) => s.tagDatabase);
  const enhancePrompt = useStore((s) => s.enhancePrompt);
  const enhanceLoading = useStore((s) => s.enhanceLoading);
  const serverSettings = useStore((s) => s.serverSettings);

  const [showTags, setShowTags] = useState(false);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      generate();
    }
    if (e.key === "Escape") {
      setShowTags(false);
    }
  };

  const handleTagSelect = (tag: string) => {
    // Replace the last partial token with the selected tag
    const parts = prompt.split(/,/);
    parts[parts.length - 1] = ` ${tag}`;
    setParam("prompt", parts.join(",").replace(/^[\s,]+/, ""));
  };

  return (
    <div className="space-y-3">
      <div className="space-y-1.5">
        <label className="block text-xs font-medium uppercase tracking-wider text-text-secondary">
          Prompt
        </label>
        <div className="relative">
          <textarea
            value={prompt}
            onChange={(e) => {
              setParam("prompt", e.target.value);
              setShowTags(true);
            }}
            onKeyDown={handleKeyDown}
            onBlur={() => {
              // Delay so click on autocomplete registers first
              setTimeout(() => setShowTags(false), 200);
            }}
            onFocus={() => setShowTags(true)}
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
          <TagAutocomplete
            query={prompt}
            tagDatabase={tagDatabase}
            onSelect={handleTagSelect}
            onDismiss={() => setShowTags(false)}
            visible={showTags}
          />
        </div>
        <div className="flex items-center justify-between text-xs text-text-muted">
          <span>{prompt.length} / 5000</span>
          <div className="flex items-center gap-3">
            {serverSettings.ai_enhance && (
              <button
                onClick={enhancePrompt}
                disabled={enhanceLoading || !prompt.trim() || isGenerating}
                className="text-accent hover:text-accent-hover disabled:opacity-50"
              >
                {enhanceLoading ? "Enhancing..." : "Enhance"}
              </button>
            )}
            <span>Cmd+Enter to generate</span>
          </div>
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
