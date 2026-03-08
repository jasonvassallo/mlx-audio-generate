import { useStore } from "../store/useStore";

export default function Header() {
  const models = useStore((s) => s.models);
  const loadedCount = models.filter((m) => m.is_loaded).length;

  return (
    <header className="flex items-center justify-between border-b border-border px-6 py-3 bg-surface-1">
      <div className="flex items-center gap-3">
        <h1 className="text-base font-bold tracking-wide text-text-primary">
          MLX AudioGen
        </h1>
        <span className="text-xs text-text-muted">v0.1.0</span>
      </div>
      <div className="flex items-center gap-4 text-xs text-text-secondary">
        <span>
          {models.length} model{models.length !== 1 ? "s" : ""} available
        </span>
        {loadedCount > 0 && (
          <span className="text-success">
            {loadedCount} loaded
          </span>
        )}
      </div>
    </header>
  );
}
