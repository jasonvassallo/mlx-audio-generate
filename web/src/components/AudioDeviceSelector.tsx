import { useState, useEffect } from "react";

/**
 * Audio output device selector.
 *
 * Uses the Web Audio API's `enumerateDevices()` to list available audio
 * output devices, and `setSinkId()` on audio elements to route playback
 * to the selected device.
 *
 * NOTE: `setSinkId` requires a user gesture (click) before it can be called.
 * The browser may also prompt for microphone permission when enumerating
 * devices (this is a browser security requirement, not a bug).
 */

/** Global sink ID shared across all AudioPlayer instances. */
let _globalSinkId = "";
const _listeners = new Set<(id: string) => void>();

export function getGlobalSinkId(): string {
  return _globalSinkId;
}

export function onSinkIdChange(fn: (id: string) => void): () => void {
  _listeners.add(fn);
  return () => _listeners.delete(fn);
}

function setGlobalSinkId(id: string) {
  _globalSinkId = id;
  _listeners.forEach((fn) => fn(id));
}

interface AudioDevice {
  deviceId: string;
  label: string;
}

interface AudioDeviceSelectorProps {
  compact?: boolean;
}

export default function AudioDeviceSelector({ compact = false }: AudioDeviceSelectorProps) {
  const [devices, setDevices] = useState<AudioDevice[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadDevices() {
      try {
        // Must request permission to get device labels
        if (navigator.mediaDevices?.enumerateDevices) {
          const allDevices = await navigator.mediaDevices.enumerateDevices();
          const audioOutputs = allDevices
            .filter((d) => d.kind === "audiooutput")
            .map((d) => ({
              deviceId: d.deviceId,
              label: d.label || `Output ${d.deviceId.slice(0, 8)}`,
            }));
          setDevices(audioOutputs);
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : "Cannot list audio devices");
      }
    }
    loadDevices();
  }, []);

  const handleChange = (deviceId: string) => {
    setSelectedId(deviceId);
    setGlobalSinkId(deviceId);
  };

  if (devices.length <= 1 && !error) {
    // Only one output device (or none) — no need to show selector
    return null;
  }

  if (compact) {
    return (
      <div className="flex items-center gap-1.5">
        {/* Speaker icon */}
        <svg
          width="12"
          height="12"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          className="text-text-muted shrink-0"
        >
          <path d="M11 5L6 9H2v6h4l5 4V5z" />
          <path d="M19.07 4.93a10 10 0 010 14.14M15.54 8.46a5 5 0 010 7.07" />
        </svg>
        {error ? (
          <span className="text-xs text-text-muted">{error}</span>
        ) : (
          <select
            value={selectedId}
            onChange={(e) => handleChange(e.target.value)}
            className="
              max-w-[140px] rounded border border-border bg-surface-2 px-1.5 py-0.5
              text-xs text-text-secondary
              focus:border-accent focus:outline-none
            "
          >
            <option value="">Default</option>
            {devices.map((d) => (
              <option key={d.deviceId} value={d.deviceId}>
                {d.label}
              </option>
            ))}
          </select>
        )}
      </div>
    );
  }

  return (
    <div className="space-y-1">
      <label className="text-xs font-medium uppercase tracking-wider text-text-secondary">
        Audio Output
      </label>
      {error ? (
        <div className="text-xs text-text-muted">{error}</div>
      ) : (
        <select
          value={selectedId}
          onChange={(e) => handleChange(e.target.value)}
          className="
            w-full rounded border border-border bg-surface-2 px-2 py-1.5
            text-xs text-text-primary
            focus:border-accent focus:outline-none
          "
        >
          <option value="">System Default</option>
          {devices.map((d) => (
            <option key={d.deviceId} value={d.deviceId}>
              {d.label}
            </option>
          ))}
        </select>
      )}
    </div>
  );
}
