/**
 * IndexedDB persistence layer for generation history.
 *
 * Stores audio blobs and metadata in the browser's IndexedDB so history
 * survives page refreshes and browser restarts. WAV files are stored as
 * binary Blobs — IndexedDB handles large binary data efficiently, unlike
 * localStorage which has a ~5MB limit.
 *
 * Schema:
 *   Store "history" — keyed by job ID
 *     { id, job, audioBlob, favorite, createdAt }
 *
 *   Store "settings"  — keyed by setting name
 *     { key, value }
 */

import type { JobInfo } from "../types/api";

const DB_NAME = "mlx-audiogen";
const DB_VERSION = 1;
const HISTORY_STORE = "history";
const SETTINGS_STORE = "settings";

export interface PersistedEntry {
  id: string;
  job: JobInfo;
  audioBlob: Blob;
  favorite: boolean;
  createdAt: number; // Unix timestamp (ms)
  /** Estimated source BPM (user-editable per entry). 0 = unknown. */
  sourceBpm: number;
}

export interface HistorySettings {
  /** Auto-delete entries older than this many hours. 0 = never. */
  retentionHours: number;
  /** Master/target BPM for loop playback. 0 = no tempo adjustment. */
  masterBpm: number;
  /** true = time-stretch (pitch stays), false = vinyl (pitch follows tempo). */
  preservePitch: boolean;
}

const DEFAULT_SETTINGS: HistorySettings = {
  retentionHours: 0, // Keep forever by default
  masterBpm: 120,
  preservePitch: true,
};

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(HISTORY_STORE)) {
        db.createObjectStore(HISTORY_STORE, { keyPath: "id" });
      }
      if (!db.objectStoreNames.contains(SETTINGS_STORE)) {
        db.createObjectStore(SETTINGS_STORE, { keyPath: "key" });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

// ---------------------------------------------------------------------------
// History CRUD
// ---------------------------------------------------------------------------

export async function saveEntry(entry: PersistedEntry): Promise<void> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(HISTORY_STORE, "readwrite");
    tx.objectStore(HISTORY_STORE).put(entry);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

export async function loadAllEntries(): Promise<PersistedEntry[]> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(HISTORY_STORE, "readonly");
    const req = tx.objectStore(HISTORY_STORE).getAll();
    req.onsuccess = () => {
      // Sort newest first
      const entries = (req.result as PersistedEntry[]).sort(
        (a, b) => b.createdAt - a.createdAt,
      );
      resolve(entries);
    };
    req.onerror = () => reject(req.error);
  });
}

export async function deleteEntry(id: string): Promise<void> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(HISTORY_STORE, "readwrite");
    tx.objectStore(HISTORY_STORE).delete(id);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

export async function updateFavorite(
  id: string,
  favorite: boolean,
): Promise<void> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(HISTORY_STORE, "readwrite");
    const store = tx.objectStore(HISTORY_STORE);
    const getReq = store.get(id);
    getReq.onsuccess = () => {
      const entry = getReq.result as PersistedEntry | undefined;
      if (entry) {
        entry.favorite = favorite;
        store.put(entry);
      }
    };
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

export async function updateSourceBpm(
  id: string,
  sourceBpm: number,
): Promise<void> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(HISTORY_STORE, "readwrite");
    const store = tx.objectStore(HISTORY_STORE);
    const getReq = store.get(id);
    getReq.onsuccess = () => {
      const entry = getReq.result as PersistedEntry | undefined;
      if (entry) {
        entry.sourceBpm = sourceBpm;
        store.put(entry);
      }
    };
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

export async function clearAllEntries(): Promise<void> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(HISTORY_STORE, "readwrite");
    tx.objectStore(HISTORY_STORE).clear();
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

/**
 * Delete non-favorite entries older than the retention period.
 * Returns the number of entries deleted.
 */
export async function purgeExpiredEntries(
  retentionHours: number,
): Promise<number> {
  if (retentionHours <= 0) return 0;

  const cutoff = Date.now() - retentionHours * 60 * 60 * 1000;
  const entries = await loadAllEntries();
  let deleted = 0;

  for (const entry of entries) {
    if (!entry.favorite && entry.createdAt < cutoff) {
      await deleteEntry(entry.id);
      deleted++;
    }
  }

  return deleted;
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

export async function loadSettings(): Promise<HistorySettings> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(SETTINGS_STORE, "readonly");
    const req = tx.objectStore(SETTINGS_STORE).get("history");
    req.onsuccess = () => {
      const result = req.result as { key: string; value: HistorySettings } | undefined;
      resolve(result?.value ?? DEFAULT_SETTINGS);
    };
    req.onerror = () => reject(req.error);
  });
}

export async function saveSettings(settings: HistorySettings): Promise<void> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(SETTINGS_STORE, "readwrite");
    tx.objectStore(SETTINGS_STORE).put({ key: "history", value: settings });
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}
