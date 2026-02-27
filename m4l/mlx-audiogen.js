/**
 * MLX Audio Generate — Node for Max HTTP Client
 *
 * Thin client that communicates with the mlx-audiogen-server to generate
 * audio from text prompts inside Ableton Live via Max for Live.
 *
 * Architecture:
 *   Max for Live UI (dials, text) → Node for Max (this script)
 *     → HTTP POST to localhost:8420/api/generate
 *     → Poll /api/status/{id} until done
 *     → Download WAV from /api/audio/{id}
 *     → Write to temp file → output path to Max for drag-to-track
 *
 * Messages from Max:
 *   generate <prompt>        — Start generation with current settings
 *   model <name>             — Set model type (musicgen / stable_audio)
 *   seconds <float>          — Set duration
 *   temperature <float>      — Set temperature (musicgen)
 *   top_k <int>              — Set top-k (musicgen)
 *   guidance <float>         — Set guidance coefficient
 *   steps <int>              — Set diffusion steps (stable_audio)
 *   cfg_scale <float>        — Set CFG scale (stable_audio)
 *   seed <int>               — Set random seed (-1 for random)
 *   server <host:port>       — Set server address
 *   style_audio <path>       — Set style reference audio path
 *   style_coef <float>       — Set style coefficient
 *   melody <path>            — Set melody audio path
 *
 * Messages to Max:
 *   status <text>            — Status updates for UI display
 *   progress <0-100>         — Generation progress percentage
 *   audio <filepath>         — Path to generated WAV file
 *   error <text>             — Error message
 *   models <json>            — Available models list
 */

const http = require("http");
const https = require("https");
const fs = require("fs");
const path = require("path");
const os = require("os");
const Max = require("max-api");

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let serverHost = "127.0.0.1";
let serverPort = 8420;

const params = {
  model: "musicgen",
  prompt: "",
  seconds: 5.0,
  temperature: 1.0,
  top_k: 250,
  guidance_coef: 3.0,
  steps: 8,
  cfg_scale: 6.0,
  seed: -1,
  melody_path: null,
  style_audio_path: null,
  style_coef: 5.0,
};

let generating = false;
let pollTimer = null;

// Output directory for generated WAVs
const outputDir = path.join(os.tmpdir(), "mlx-audiogen");
if (!fs.existsSync(outputDir)) {
  fs.mkdirSync(outputDir, { recursive: true });
}

// ---------------------------------------------------------------------------
// Max Message Handlers
// ---------------------------------------------------------------------------

Max.addHandler("generate", (...args) => {
  const prompt = args.join(" ");
  if (!prompt.trim()) {
    Max.outlet("error", "Empty prompt");
    return;
  }
  params.prompt = prompt;
  startGeneration();
});

Max.addHandler("model", (name) => {
  params.model = name;
  Max.outlet("status", `Model: ${name}`);
});

Max.addHandler("seconds", (val) => {
  params.seconds = Math.max(0.1, Math.min(300, parseFloat(val) || 5));
  Max.outlet("status", `Duration: ${params.seconds}s`);
});

Max.addHandler("temperature", (val) => {
  params.temperature = Math.max(0.01, parseFloat(val) || 1.0);
});

Max.addHandler("top_k", (val) => {
  params.top_k = Math.max(1, parseInt(val) || 250);
});

Max.addHandler("guidance", (val) => {
  params.guidance_coef = Math.max(0, parseFloat(val) || 3.0);
});

Max.addHandler("steps", (val) => {
  params.steps = Math.max(1, Math.min(1000, parseInt(val) || 8));
});

Max.addHandler("cfg_scale", (val) => {
  params.cfg_scale = Math.max(0, parseFloat(val) || 6.0);
});

Max.addHandler("seed", (val) => {
  params.seed = parseInt(val) || -1;
});

Max.addHandler("server", (addr) => {
  const parts = addr.split(":");
  serverHost = parts[0] || "127.0.0.1";
  serverPort = parseInt(parts[1]) || 8420;
  Max.outlet("status", `Server: ${serverHost}:${serverPort}`);
});

Max.addHandler("style_audio", (filepath) => {
  params.style_audio_path = filepath || null;
  Max.outlet("status", filepath ? `Style: ${path.basename(filepath)}` : "Style: none");
});

Max.addHandler("style_coef", (val) => {
  params.style_coef = Math.max(0, parseFloat(val) || 5.0);
});

Max.addHandler("melody", (filepath) => {
  params.melody_path = filepath || null;
  Max.outlet("status", filepath ? `Melody: ${path.basename(filepath)}` : "Melody: none");
});

Max.addHandler("list_models", () => {
  fetchModels();
});

// ---------------------------------------------------------------------------
// HTTP Helpers
// ---------------------------------------------------------------------------

function httpRequest(method, urlPath, body) {
  return new Promise((resolve, reject) => {
    const options = {
      hostname: serverHost,
      port: serverPort,
      path: urlPath,
      method: method,
      headers: { "Content-Type": "application/json" },
    };

    const req = http.request(options, (res) => {
      let data = "";
      res.on("data", (chunk) => (data += chunk));
      res.on("end", () => {
        try {
          resolve({ status: res.statusCode, data: JSON.parse(data) });
        } catch {
          resolve({ status: res.statusCode, data: data });
        }
      });
    });

    req.on("error", (err) => reject(err));
    req.setTimeout(5000, () => {
      req.destroy();
      reject(new Error("Request timeout"));
    });

    if (body) req.write(JSON.stringify(body));
    req.end();
  });
}

function downloadFile(urlPath, outputPath) {
  return new Promise((resolve, reject) => {
    const options = {
      hostname: serverHost,
      port: serverPort,
      path: urlPath,
      method: "GET",
    };

    const req = http.request(options, (res) => {
      if (res.statusCode !== 200) {
        reject(new Error(`Download failed: HTTP ${res.statusCode}`));
        return;
      }
      const file = fs.createWriteStream(outputPath);
      res.pipe(file);
      file.on("finish", () => {
        file.close();
        resolve(outputPath);
      });
    });

    req.on("error", (err) => reject(err));
    req.setTimeout(30000, () => {
      req.destroy();
      reject(new Error("Download timeout"));
    });
    req.end();
  });
}

// ---------------------------------------------------------------------------
// Generation Flow
// ---------------------------------------------------------------------------

async function startGeneration() {
  if (generating) {
    Max.outlet("error", "Generation already in progress");
    return;
  }

  generating = true;
  Max.outlet("status", "Submitting...");
  Max.outlet("progress", 0);

  try {
    // Build request body
    const body = {
      model: params.model,
      prompt: params.prompt,
      seconds: params.seconds,
      temperature: params.temperature,
      top_k: params.top_k,
      guidance_coef: params.guidance_coef,
      steps: params.steps,
      cfg_scale: params.cfg_scale,
    };

    if (params.seed >= 0) body.seed = params.seed;
    if (params.melody_path) body.melody_path = params.melody_path;
    if (params.style_audio_path) body.style_audio_path = params.style_audio_path;
    if (params.style_coef !== 5.0) body.style_coef = params.style_coef;

    // Submit generation request
    const res = await httpRequest("POST", "/api/generate", body);

    if (res.status !== 200) {
      throw new Error(res.data.detail || `Server error: ${res.status}`);
    }

    const jobId = res.data.id;
    Max.outlet("status", `Generating... (${jobId})`);

    // Poll for completion
    await pollUntilDone(jobId);
  } catch (err) {
    Max.outlet("error", err.message);
    Max.outlet("status", "Error");
  } finally {
    generating = false;
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  }
}

function pollUntilDone(jobId) {
  return new Promise((resolve, reject) => {
    let elapsed = 0;
    const interval = 500; // Poll every 500ms

    pollTimer = setInterval(async () => {
      try {
        const res = await httpRequest("GET", `/api/status/${jobId}`);

        if (res.status !== 200) {
          clearInterval(pollTimer);
          reject(new Error(`Status check failed: ${res.status}`));
          return;
        }

        const job = res.data;
        elapsed += interval;

        // Estimate progress (rough: based on elapsed time vs expected)
        const expectedMs = params.seconds * 1000 * 2; // ~2x realtime
        const progress = Math.min(95, Math.round((elapsed / expectedMs) * 100));
        Max.outlet("progress", progress);

        if (job.status === "done") {
          clearInterval(pollTimer);
          Max.outlet("progress", 100);
          Max.outlet("status", "Downloading audio...");

          // Download the WAV file
          const timestamp = Date.now();
          const filename = `${params.model}_${jobId}_${timestamp}.wav`;
          const outputPath = path.join(outputDir, filename);

          await downloadFile(`/api/audio/${jobId}`, outputPath);

          Max.outlet("status", "Done!");
          Max.outlet("audio", outputPath);
          resolve();
        } else if (job.status === "error") {
          clearInterval(pollTimer);
          reject(new Error(job.error || "Generation failed"));
        }
        // else: still running/queued — continue polling
      } catch (err) {
        clearInterval(pollTimer);
        reject(err);
      }
    }, interval);

    // Timeout after 10 minutes
    setTimeout(() => {
      if (pollTimer) {
        clearInterval(pollTimer);
        reject(new Error("Generation timed out (10 min)"));
      }
    }, 600000);
  });
}

async function fetchModels() {
  try {
    const res = await httpRequest("GET", "/api/models");
    if (res.status === 200) {
      Max.outlet("models", JSON.stringify(res.data));
      Max.outlet("status", `${res.data.length} model(s) available`);
    }
  } catch (err) {
    Max.outlet("error", `Cannot reach server: ${err.message}`);
  }
}

// ---------------------------------------------------------------------------
// Startup
// ---------------------------------------------------------------------------

Max.post("MLX Audio Generate — Node for Max client loaded");
Max.post(`Server: ${serverHost}:${serverPort}`);
Max.outlet("status", "Ready");
