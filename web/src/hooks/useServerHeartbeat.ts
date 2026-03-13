import { useEffect, useRef, useState } from "react";
import { useStore } from "../store/useStore";
import { getApiBase } from "../api/client";

const HEARTBEAT_INTERVAL = 3000; // 3 seconds
const MAX_FAILURES = 3; // Disconnect after 3 consecutive failures (9 seconds)

/**
 * Polls /api/health and tracks server connectivity.
 * Re-initializes when the server URL changes so remote servers
 * are checked immediately on configuration.
 */
export function useServerHeartbeat() {
  const [connected, setConnected] = useState(true);
  const failCount = useRef(0);
  const serverUrl = useStore((s) => s.serverUrl);

  useEffect(() => {
    // Reset on URL change — optimistically assume connected
    failCount.current = 0;
    setConnected(true);

    const check = async () => {
      try {
        const res = await fetch(`${getApiBase()}/health`, {
          signal: AbortSignal.timeout(2000),
        });
        if (res.ok) {
          failCount.current = 0;
          setConnected(true);
        } else {
          failCount.current++;
        }
      } catch {
        failCount.current++;
      }

      if (failCount.current >= MAX_FAILURES) {
        setConnected(false);
      }
    };

    // Check immediately on URL change
    check();
    const interval = setInterval(check, HEARTBEAT_INTERVAL);

    return () => clearInterval(interval);
  }, [serverUrl]);

  return connected;
}
