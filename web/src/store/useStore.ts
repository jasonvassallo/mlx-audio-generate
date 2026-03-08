import { create } from "zustand";
import type { GenerateRequest, JobInfo, ModelInfo } from "../types/api";
import {
  fetchModels,
  submitGeneration,
  fetchJobStatus,
  getAudioUrl,
} from "../api/client";
import {
  saveEntry,
  loadAllEntries,
  deleteEntry,
  updateFavorite,
  clearAllEntries,
  purgeExpiredEntries,
  loadSettings,
  saveSettings,
  type PersistedEntry,
  type HistorySettings,
} from "./historyDb";

/** A generation entry with a local blob URL for playback. */
export interface HistoryEntry {
  id: string;
  job: JobInfo;
  audioUrl: string; // blob: URL for local playback
  favorite: boolean;
  createdAt: number;
}

interface AppState {
  // --- Models ---
  models: ModelInfo[];
  modelsLoading: boolean;
  modelsError: string | null;
  loadModels: () => Promise<void>;

  // --- Generation Parameters ---
  params: GenerateRequest;
  setParam: <K extends keyof GenerateRequest>(
    key: K,
    value: GenerateRequest[K],
  ) => void;

  // --- Active Job ---
  activeJob: JobInfo | null;
  isGenerating: boolean;
  generateError: string | null;
  generate: () => Promise<void>;

  // --- History (IndexedDB-backed) ---
  history: HistoryEntry[];
  historyLoaded: boolean;
  loadHistory: () => Promise<void>;
  toggleFavorite: (id: string) => Promise<void>;
  deleteHistoryEntry: (id: string) => Promise<void>;
  clearHistory: () => Promise<void>;

  // --- Settings ---
  settings: HistorySettings;
  settingsLoaded: boolean;
  loadSettings: () => Promise<void>;
  updateSettings: (settings: Partial<HistorySettings>) => Promise<void>;
}

const DEFAULT_PARAMS: GenerateRequest = {
  model: "musicgen",
  prompt: "",
  seconds: 5,
  temperature: 1.0,
  top_k: 250,
  guidance_coef: 3.0,
  steps: 8,
  cfg_scale: 6.0,
  sampler: "euler",
  seed: null,
  melody_path: null,
  style_audio_path: null,
  style_coef: 5.0,
};

const POLL_INTERVAL = 500;

/** Convert a PersistedEntry (with Blob) to a HistoryEntry (with blob URL). */
function toHistoryEntry(entry: PersistedEntry): HistoryEntry {
  return {
    id: entry.id,
    job: entry.job,
    audioUrl: URL.createObjectURL(entry.audioBlob),
    favorite: entry.favorite,
    createdAt: entry.createdAt,
  };
}

export const useStore = create<AppState>((set, get) => ({
  // --- Models ---
  models: [],
  modelsLoading: false,
  modelsError: null,
  loadModels: async () => {
    set({ modelsLoading: true, modelsError: null });
    try {
      const models = await fetchModels();
      set({ models, modelsLoading: false });
      if (models.length > 0 && !get().params.model) {
        set((s) => ({
          params: { ...s.params, model: models[0]!.model_type },
        }));
      }
    } catch (e) {
      set({
        modelsError: e instanceof Error ? e.message : String(e),
        modelsLoading: false,
      });
    }
  },

  // --- Generation Parameters ---
  params: DEFAULT_PARAMS,
  setParam: (key, value) =>
    set((s) => ({ params: { ...s.params, [key]: value } })),

  // --- Active Job ---
  activeJob: null,
  isGenerating: false,
  generateError: null,
  generate: async () => {
    const { params, isGenerating } = get();
    if (isGenerating) return;
    if (!params.prompt.trim()) {
      set({ generateError: "Prompt is required" });
      return;
    }

    set({ isGenerating: true, generateError: null, activeJob: null });

    try {
      const { id } = await submitGeneration(params);

      // Poll until done
      const poll = async (): Promise<JobInfo> => {
        const job = await fetchJobStatus(id);
        set({ activeJob: job });
        if (job.status === "done" || job.status === "error") {
          return job;
        }
        await new Promise((r) => setTimeout(r, POLL_INTERVAL));
        return poll();
      };

      const finalJob = await poll();

      if (finalJob.status === "done") {
        // Eagerly download the audio blob before server cleans it up (5 min)
        const audioRes = await fetch(getAudioUrl(finalJob.id));
        const audioBlob = await audioRes.blob();
        const now = Date.now();

        // Persist to IndexedDB
        const persisted: PersistedEntry = {
          id: finalJob.id,
          job: finalJob,
          audioBlob,
          favorite: false,
          createdAt: now,
        };
        await saveEntry(persisted);

        // Add to in-memory history
        const entry: HistoryEntry = {
          id: finalJob.id,
          job: finalJob,
          audioUrl: URL.createObjectURL(audioBlob),
          favorite: false,
          createdAt: now,
        };
        set((s) => ({
          history: [entry, ...s.history],
          isGenerating: false,
        }));

        // Run auto-purge if retention is configured
        const { settings } = get();
        if (settings.retentionHours > 0) {
          const deleted = await purgeExpiredEntries(settings.retentionHours);
          if (deleted > 0) {
            // Reload history to reflect purged entries
            await get().loadHistory();
          }
        }
      } else {
        set({
          generateError: finalJob.error ?? "Generation failed",
          isGenerating: false,
        });
      }
    } catch (e) {
      set({
        generateError: e instanceof Error ? e.message : String(e),
        isGenerating: false,
      });
    }
  },

  // --- History (IndexedDB-backed) ---
  history: [],
  historyLoaded: false,
  loadHistory: async () => {
    try {
      const entries = await loadAllEntries();
      const historyEntries = entries.map(toHistoryEntry);
      set({ history: historyEntries, historyLoaded: true });
    } catch (e) {
      console.error("Failed to load history:", e);
      set({ historyLoaded: true });
    }
  },

  toggleFavorite: async (id: string) => {
    const entry = get().history.find((h) => h.id === id);
    if (!entry) return;
    const newFav = !entry.favorite;
    await updateFavorite(id, newFav);
    set((s) => ({
      history: s.history.map((h) =>
        h.id === id ? { ...h, favorite: newFav } : h,
      ),
    }));
  },

  deleteHistoryEntry: async (id: string) => {
    // Revoke blob URL to free memory
    const entry = get().history.find((h) => h.id === id);
    if (entry) URL.revokeObjectURL(entry.audioUrl);

    await deleteEntry(id);
    set((s) => ({
      history: s.history.filter((h) => h.id !== id),
    }));
  },

  clearHistory: async () => {
    // Revoke all blob URLs
    get().history.forEach((h) => URL.revokeObjectURL(h.audioUrl));
    await clearAllEntries();
    set({ history: [] });
  },

  // --- Settings ---
  settings: { retentionHours: 0 },
  settingsLoaded: false,
  loadSettings: async () => {
    try {
      const settings = await loadSettings();
      set({ settings, settingsLoaded: true });
    } catch (e) {
      console.error("Failed to load settings:", e);
      set({ settingsLoaded: true });
    }
  },

  updateSettings: async (partial) => {
    const newSettings = { ...get().settings, ...partial };
    await saveSettings(newSettings);
    set({ settings: newSettings });

    // If retention was just enabled, run a purge now
    if (partial.retentionHours && partial.retentionHours > 0) {
      const deleted = await purgeExpiredEntries(partial.retentionHours);
      if (deleted > 0) {
        await get().loadHistory();
      }
    }
  },
}));
