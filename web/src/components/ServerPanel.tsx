import { useCallback, useEffect, useState } from "react";
import { useStore } from "../store/useStore";

/**
 * Server connection settings — lets users point the UI at a remote
 * mlx-audiogen server (e.g., a Mac Studio on the LAN).
 */
export default function ServerPanel() {
  const serverUrl = useStore((s) => s.serverUrl);
  const setServerUrl = useStore((s) => s.setServerUrl);

  // Local input state (not committed until user clicks Connect)
  const [input, setInput] = useState(serverUrl);
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<
    "connected" | "failed" | null
  >(null);

  // Keep input in sync when serverUrl changes externally
  useEffect(() => {
    setInput(serverUrl);
  }, [serverUrl]);

  const isRemote = serverUrl !== "";
  const isDirty = input !== serverUrl;

  const testConnection = useCallback(async (url: string) => {
    setTesting(true);
    setTestResult(null);
    const base = url ? url.replace(/\/+$/, "") + "/api" : "/api";
    try {
      const res = await fetch(`${base}/health`, {
        signal: AbortSignal.timeout(3000),
      });
      setTestResult(res.ok ? "connected" : "failed");
    } catch {
      setTestResult("failed");
    } finally {
      setTesting(false);
    }
  }, []);

  const handleConnect = useCallback(async () => {
    const trimmed = input.trim().replace(/\/+$/, "");

    // Validate URL format for non-empty input
    if (trimmed) {
      try {
        const parsed = new URL(trimmed);
        if (!["http:", "https:"].includes(parsed.protocol)) {
          setTestResult("failed");
          return;
        }
      } catch {
        setTestResult("failed");
        return;
      }
    }

    // Test the connection first
    setTesting(true);
    setTestResult(null);
    const base = trimmed ? trimmed + "/api" : "/api";
    try {
      const res = await fetch(`${base}/health`, {
        signal: AbortSignal.timeout(3000),
      });
      if (res.ok) {
        setTestResult("connected");
        setServerUrl(trimmed);
      } else {
        setTestResult("failed");
      }
    } catch {
      setTestResult("failed");
    } finally {
      setTesting(false);
    }
  }, [input, setServerUrl]);

  const handleReset = useCallback(() => {
    setInput("");
    setTestResult(null);
    setServerUrl("");
  }, [setServerUrl]);

  return (
    <section className="space-y-3">
      <h3 className="text-xs font-medium uppercase tracking-wider text-text-secondary">
        Server Connection
      </h3>

      {/* Connection status indicator */}
      <div className="flex items-center gap-2">
        <span
          className={`h-2 w-2 rounded-full ${
            isRemote
              ? "bg-info"
              : "bg-success"
          }`}
        />
        <span className="text-xs text-text-secondary">
          {isRemote
            ? `Remote: ${serverUrl}`
            : "Local server"}
        </span>
      </div>

      {/* URL input */}
      <div className="space-y-1">
        <label className="text-xs font-medium text-text-secondary">
          Server URL
        </label>
        <input
          type="url"
          value={input}
          onChange={(e) => {
            setInput(e.target.value);
            setTestResult(null);
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter") handleConnect();
          }}
          placeholder="http://192.168.1.100:8420"
          className="
            w-full rounded border border-border bg-surface-2 px-2 py-1.5
            text-xs text-text-primary placeholder:text-text-muted
            focus:border-accent focus:outline-none
          "
        />
        <p className="text-xs text-text-muted">
          Leave empty for local. Enter a remote server address to offload
          generation.
        </p>
      </div>

      {/* Action buttons */}
      <div className="flex gap-2">
        <button
          onClick={handleConnect}
          disabled={testing || !isDirty}
          className={`
            rounded px-3 py-1 text-xs font-medium transition-colors
            ${
              testing || !isDirty
                ? "bg-surface-3 text-text-muted cursor-not-allowed"
                : "bg-accent text-surface-0 hover:bg-accent/80"
            }
          `}
        >
          {testing ? "Testing..." : "Connect"}
        </button>
        {isRemote && (
          <button
            onClick={handleReset}
            className="rounded bg-surface-3 px-3 py-1 text-xs text-text-muted hover:text-text-secondary"
          >
            Reset to Local
          </button>
        )}
        {!isDirty && serverUrl && (
          <button
            onClick={() => testConnection(serverUrl)}
            disabled={testing}
            className="rounded bg-surface-3 px-3 py-1 text-xs text-text-muted hover:text-text-secondary"
          >
            Test
          </button>
        )}
      </div>

      {/* Test result feedback */}
      {testResult === "connected" && (
        <p className="text-xs text-success">Connected successfully.</p>
      )}
      {testResult === "failed" && (
        <p className="text-xs text-error">
          Connection failed. Check the URL and ensure the server is running
          {input.trim() && " with CORS enabled"}.
        </p>
      )}
    </section>
  );
}
