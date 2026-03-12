import type {
  EnhanceResponse,
  GenerateRequest,
  GenerateResponse,
  JobInfo,
  LLMModelInfo,
  LLMStatus,
  ModelInfo,
  PresetInfo,
  PromptAnalysis,
  PromptMemoryData,
  ServerSettings,
  StemResult,
  TagDatabase,
} from "../types/api";

/**
 * API base URL.
 * In dev mode, Vite proxies /api to the FastAPI server (see vite.config.ts).
 * In production, the FastAPI server serves both the SPA and the API.
 */
const BASE = "/api";

async function request<T>(
  path: string,
  options?: RequestInit,
): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API error ${res.status}: ${body}`);
  }
  return res.json() as Promise<T>;
}

/** List available models and their loading status. */
export function fetchModels(): Promise<ModelInfo[]> {
  return request<ModelInfo[]>("/models");
}

/** Submit a generation request. Returns immediately with a job ID. */
export function submitGeneration(
  req: GenerateRequest,
): Promise<GenerateResponse> {
  return request<GenerateResponse>("/generate", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

/** Poll job status. */
export function fetchJobStatus(jobId: string): Promise<JobInfo> {
  return request<JobInfo>(`/status/${jobId}`);
}

/** Get the URL for downloading generated audio. */
export function getAudioUrl(jobId: string): string {
  return `${BASE}/audio/${jobId}`;
}

/** Get the URL for downloading generated MIDI. */
export function getMidiUrl(jobId: string): string {
  return `${BASE}/midi/${jobId}`;
}

/** Get AI prompt suggestions. */
export function suggestPrompts(
  prompt: string,
  count = 4,
): Promise<PromptAnalysis> {
  return request<PromptAnalysis>("/suggest", {
    method: "POST",
    body: JSON.stringify({ prompt, count }),
  });
}

/** Separate a job's audio into stems. */
export function separateStems(jobId: string): Promise<StemResult> {
  return request<StemResult>(`/separate/${jobId}`, {
    method: "POST",
  });
}

/** List all saved presets. */
export function fetchPresets(): Promise<PresetInfo[]> {
  return request<PresetInfo[]>("/presets");
}

/** Save current params as a named preset. */
export function savePreset(
  name: string,
  params: GenerateRequest,
): Promise<{ saved: string }> {
  return request<{ saved: string }>(`/presets/${encodeURIComponent(name)}`, {
    method: "POST",
    body: JSON.stringify(params),
  });
}

/** Load a preset by name. */
export function loadPreset(name: string): Promise<GenerateRequest> {
  return request<GenerateRequest>(`/presets/${encodeURIComponent(name)}`);
}

// ---------------------------------------------------------------------------
// Phase 7b: LLM Enhancement, Memory, Settings
// ---------------------------------------------------------------------------

/** Enhance a prompt via LLM or template fallback. */
export function enhancePrompt(
  prompt: string,
  includeMemory = true,
): Promise<EnhanceResponse> {
  return request<EnhanceResponse>("/enhance", {
    method: "POST",
    body: JSON.stringify({ prompt, include_memory: includeMemory }),
  });
}

/** Get the tag database for autocomplete. */
export function fetchTags(): Promise<TagDatabase> {
  return request<TagDatabase>("/tags");
}

/** List discovered LLM models. */
export function fetchLLMModels(): Promise<LLMModelInfo[]> {
  return request<LLMModelInfo[]>("/llm/models");
}

/** Select an LLM model. */
export function selectLLMModel(
  modelId: string,
): Promise<{ status: string }> {
  return request<{ status: string }>("/llm/select", {
    method: "POST",
    body: JSON.stringify({ model_id: modelId }),
  });
}

/** Get LLM status. */
export function fetchLLMStatus(): Promise<LLMStatus> {
  return request<LLMStatus>("/llm/status");
}

/** Get prompt memory. */
export function fetchMemory(): Promise<PromptMemoryData> {
  return request<PromptMemoryData>("/memory");
}

/** Clear prompt memory. */
export function clearMemory(): Promise<{ status: string }> {
  return request<{ status: string }>("/memory", { method: "DELETE" });
}

/** Export prompt memory as downloadable JSON. */
export function getMemoryExportUrl(): string {
  return `${BASE}/memory/export`;
}

/** Import prompt memory from file. */
export async function importMemory(
  file: File,
): Promise<{ status: string }> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${BASE}/memory/import`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API error ${res.status}: ${body}`);
  }
  return res.json();
}

/** Get server settings. */
export function fetchServerSettings(): Promise<ServerSettings> {
  return request<ServerSettings>("/settings");
}

/** Update server settings. */
export function updateServerSettings(
  settings: Partial<ServerSettings>,
): Promise<ServerSettings> {
  return request<ServerSettings>("/settings", {
    method: "POST",
    body: JSON.stringify(settings),
  });
}
