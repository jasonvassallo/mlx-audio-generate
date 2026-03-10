import type {
  GenerateRequest,
  GenerateResponse,
  JobInfo,
  ModelInfo,
  PromptAnalysis,
  StemResult,
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
