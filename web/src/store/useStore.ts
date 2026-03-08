import { create } from "zustand";
import type { GenerateRequest, JobInfo, ModelInfo } from "../types/api";
import {
  fetchModels,
  submitGeneration,
  fetchJobStatus,
  getAudioUrl,
} from "../api/client";

/** A completed generation with its audio URL. */
export interface HistoryEntry {
  job: JobInfo;
  audioUrl: string;
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

  // --- History ---
  history: HistoryEntry[];
  clearHistory: () => void;
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
      // Auto-select first model if none selected
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
      // Submit generation request
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
        // Add to history
        set((s) => ({
          history: [
            { job: finalJob, audioUrl: getAudioUrl(finalJob.id) },
            ...s.history,
          ],
          isGenerating: false,
        }));
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

  // --- History ---
  history: [],
  clearHistory: () => set({ history: [] }),
}));
