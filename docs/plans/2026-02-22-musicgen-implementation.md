# MusicGen MLX Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Port Meta's MusicGen to MLX for native Apple Silicon audio generation, loading from HuggingFace safetensors without any PyTorch dependency.

**Architecture:** MusicGen uses three components: (1) T5 encoder for text conditioning (shared module, already built), (2) autoregressive transformer decoder with KV cache that generates audio tokens using a codebook delay pattern, and (3) EnCodec audio codec that converts tokens back to waveform. We inline Apple's proven MLX EnCodec implementation as a shared module, and load weights from HF `model.safetensors` format (pre-split QKV, no PyTorch needed).

**Tech Stack:** MLX, safetensors, HuggingFace Hub, transformers (tokenizer only), numpy

**Reference:** Apple's `mlx-examples/musicgen/` (Apache 2.0) — proven MLX patterns for KV cache, Metal LSTM kernel, attention, and generation loop.

---

### Task 1: EnCodec Shared Module

**Files:**
- Create: `mlx_audio_generate/shared/encodec.py`
- Modify: `mlx_audio_generate/shared/__init__.py`

**What:** Port Apple's `encodec.py` (741 lines) into our shared module. This includes:
- Custom Metal LSTM kernel (`_lstm_kernel` + `lstm_custom` + `LSTM` class)
- `EncodecConv1d` / `EncodecConvTranspose1d` (with causal/asymmetric padding)
- `EncodecResnetBlock`, `EncodecEncoder`, `EncodecDecoder`
- `EncodecResidualVectorQuantizer` (RVQ with encode + decode)
- `EncodecModel` with `from_pretrained()` loading from `mlx-community/encodec-32khz-float32`
- `preprocess_audio()` utility

Key adaptations from Apple's code:
- Keep it essentially identical — this is battle-tested MLX code
- Ensure `from_pretrained()` works with our `shared.hub.download_model()` if possible
- Add proper docstrings

**Verify:** `python -c "from mlx_audio_generate.shared.encodec import EncodecModel; print('OK')"`

**Commit:** `feat: add EnCodec shared module (ported from mlx-examples)`

---

### Task 2: MusicGen Config

**Files:**
- Create: `mlx_audio_generate/models/musicgen/config.py`

**What:** Config dataclass that reads from HuggingFace's `config.json`. Needs to handle the nested structure:
```json
{
  "text_encoder": {"d_model": 768, "_name_or_path": "t5-base", ...},
  "audio_encoder": {"codebook_size": 2048, "sampling_rate": 32000, ...},
  "decoder": {"hidden_size": 1024, "num_hidden_layers": 24, "num_attention_heads": 16, "ffn_dim": 4096, "num_codebooks": 4, ...}
}
```

Classes: `DecoderConfig`, `AudioEncoderConfig`, `TextEncoderConfig`, `MusicGenConfig` — each with `from_dict()`.

**Verify:** `python -c "from mlx_audio_generate.models.musicgen.config import MusicGenConfig; c = MusicGenConfig(); print(c.decoder.hidden_size)"`

**Commit:** `feat: add MusicGen config`

---

### Task 3: MusicGen Transformer Decoder

**Files:**
- Create: `mlx_audio_generate/models/musicgen/transformer.py`

**What:** The autoregressive transformer with KV cache support. Components:

1. `KVCache` — Pre-allocated cache with step-size growth (from Apple's implementation)
2. `MultiHeadAttention` — Separate q/k/v projections (no bias), uses `mx.fast.scaled_dot_product_attention`
3. `TransformerBlock` — Pre-norm (norm1 → self_attn, norm_cross → cross_attn, norm2 → FFN with GELU)
4. `create_sin_embedding()` — Sinusoidal positional embeddings (not learned, not RoPE)

Weight key mapping from HF checkpoint:
- `decoder.model.decoder.layers.{i}.self_attn.{q,k,v,out}_proj.weight`
- `decoder.model.decoder.layers.{i}.encoder_attn.{q,k,v,out}_proj.weight`
- `decoder.model.decoder.layers.{i}.fc1.weight`, `fc2.weight`
- `decoder.model.decoder.layers.{i}.self_attn_layer_norm.weight`
- `decoder.model.decoder.layers.{i}.encoder_attn_layer_norm.weight`
- `decoder.model.decoder.layers.{i}.final_layer_norm.weight`

**Verify:** Instantiate decoder block, pass dummy tensor through

**Commit:** `feat: add MusicGen transformer decoder with KV cache`

---

### Task 4: MusicGen Model (Embeddings + LM Heads + Generation)

**Files:**
- Create: `mlx_audio_generate/models/musicgen/model.py`

**What:** The main model class that ties embeddings, transformer, and output heads together.

1. `MusicGenModel` class:
   - `self.embed_tokens` — List of K `nn.Embedding(codebook_size+1, hidden_size)` (one per codebook)
   - `self.layers` — List of `TransformerBlock`
   - `self.layer_norm` — Final `nn.LayerNorm`
   - `self.lm_heads` — List of K `nn.Linear(hidden_size, codebook_size)` (one per codebook)
   - `self.enc_to_dec_proj` — `nn.Linear(768, hidden_size)` for T5 → decoder projection

2. `__call__()` — Forward pass: sum embeddings → add sinusoidal pos → transformer → norm → logits

3. `top_k_sampling()` — Compiled sampling function (from Apple's implementation)

4. `generate()` method:
   - Takes text conditioning tensor + generation params
   - Runs autoregressive loop with KV cache
   - Applies delay pattern masking at each step
   - Undoes delay pattern after generation
   - Returns token sequence

Weight keys from HF:
- `decoder.model.decoder.embed_tokens.{k}.weight`
- `decoder.model.decoder.layer_norm.weight`
- `decoder.lm_heads.{k}.weight`
- `enc_to_dec_proj.weight`, `enc_to_dec_proj.bias`

**Verify:** Instantiate model, run forward pass with dummy tokens + conditioning

**Commit:** `feat: add MusicGen model with generation loop`

---

### Task 5: MusicGen Pipeline

**Files:**
- Create: `mlx_audio_generate/models/musicgen/pipeline.py`

**What:** High-level pipeline that loads T5 + decoder + EnCodec and orchestrates generation.

1. `MusicGenPipeline` class:
   - `from_pretrained(weights_dir)` — Loads all three components from split safetensors
   - `generate(prompt, seconds, ...)` — Full text-to-audio:
     1. Tokenize + T5 encode text → conditioning
     2. Project conditioning via `enc_to_dec_proj`
     3. Setup CFG (batch conditional + unconditional)
     4. Run `model.generate()` to get audio tokens
     5. Decode tokens via EnCodec RVQ → waveform
     6. Return audio array

**Verify:** Import test passes

**Commit:** `feat: add MusicGen pipeline`

---

### Task 6: MusicGen Weight Conversion

**Files:**
- Create: `mlx_audio_generate/models/musicgen/convert.py`

**What:** Convert HF `model.safetensors` into our split format.

HF checkpoint structure:
- `text_encoder.*` → `t5.safetensors` (T5 weights, key remapping for our T5 module)
- `decoder.model.decoder.*` → `decoder.safetensors` (transformer decoder)
- `decoder.lm_heads.*` → `decoder.safetensors` (LM heads, merge with decoder)
- `enc_to_dec_proj.*` → `decoder.safetensors` (projection layer)
- `audio_encoder.*` — SKIP (we load EnCodec from mlx-community separately)

Key remapping for decoder:
- Strip `decoder.model.decoder.` prefix → `layers.{i}.self_attn.q_proj.weight` etc.
- `decoder.model.decoder.layer_norm.weight` → `layer_norm.weight`
- `decoder.lm_heads.{k}.weight` → `lm_heads.{k}.weight`
- `decoder.model.decoder.embed_tokens.{k}.weight` → `embed_tokens.{k}.weight`

Key remapping for T5:
- Strip `text_encoder.` prefix
- Map `encoder.block.{i}.layer.0.SelfAttention.*` → `encoder.block.{i}.self_attn.*`
- Map `encoder.block.{i}.layer.0.layer_norm.*` → `encoder.block.{i}.self_attn_norm.*`
- Map `encoder.block.{i}.layer.1.DenseReluDense.*` → `encoder.block.{i}.ff.*`
- Map `encoder.block.{i}.layer.1.layer_norm.*` → `encoder.block.{i}.ff_norm.*`

Also saves: `config.json` (model config), `t5_config.json`, tokenizer

**Verify:** Run conversion on `facebook/musicgen-small`, check output files exist

**Commit:** `feat: add MusicGen weight conversion`

---

### Task 7: Wire Up Exports & CLI

**Files:**
- Modify: `mlx_audio_generate/models/musicgen/__init__.py`
- Verify: `mlx_audio_generate/cli/generate.py` (already has musicgen dispatch)
- Verify: `mlx_audio_generate/cli/convert.py` (already has musicgen dispatch)

**What:** Update `__init__.py` to export `MusicGenPipeline`, `MusicGenModel`, `MusicGenConfig`, `convert_musicgen`. Verify CLI stubs work with the new imports.

**Verify:** `python -c "from mlx_audio_generate.models.musicgen import MusicGenPipeline; print('OK')"`

**Commit:** `feat: wire up MusicGen exports and CLI`

---

### Task 8: Integration Smoke Test

**What:** Run through the full pipeline without real weights to verify all shapes and connections work:
1. Instantiate all components with default configs
2. Run T5 forward pass → conditioning tensor
3. Run decoder forward pass → logits
4. Verify shape: `(B, 1, codebook_size, num_codebooks)`
5. Verify EnCodec loads from mlx-community repo
6. Verify CLI `--help` works

**Commit:** `test: add MusicGen integration smoke test`
