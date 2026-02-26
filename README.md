# mlx-audio-generate

Text-to-audio generation on Apple Silicon using [MLX](https://github.com/ml-explore/mlx). Supports **MusicGen** (including melody and stereo variants) and **Stable Audio Open**.

Runs entirely on-device via Metal GPU — no cloud API needed.

## Supported Models

| Model | Variants | Output | Sample Rate | Architecture |
|-------|----------|--------|-------------|--------------|
| MusicGen | small, medium, large | Mono | 32 kHz | Autoregressive (T5 + Transformer + EnCodec) |
| MusicGen Stereo | small, medium, large | Stereo | 32 kHz | Autoregressive (8 codebooks) |
| MusicGen Melody | base, large | Mono | 32 kHz | Autoregressive + Chroma conditioning |
| Stable Audio Open | small | Stereo | 44.1 kHz | Diffusion (T5 + DiT + Oobleck VAE) |
| Stable Audio Open | 1.0 | Stereo | 44.1 kHz | Diffusion (larger DiT, dual time conditioning) |

## Quick Start

```bash
# Install
git clone https://github.com/jasonvassallo/mlx-audio-generate
cd mlx-audio-generate
uv sync

# Convert weights (one-time per model variant)
uv run mlx-audio-convert --model facebook/musicgen-small --output ./converted/musicgen-small

# Generate audio
uv run mlx-audio-generate \
  --model musicgen \
  --prompt "happy upbeat rock song with electric guitar" \
  --seconds 10 \
  --weights-dir ./converted/musicgen-small \
  --output my_song.wav
```

### Stable Audio Example

```bash
# Convert weights
uv run mlx-audio-convert --model stabilityai/stable-audio-open-small --output ./converted/stable-audio

# Generate (stereo, 44.1kHz)
uv run mlx-audio-generate \
  --model stable_audio \
  --prompt "ambient electronic pad with warm reverb" \
  --seconds 15 \
  --steps 100 \
  --cfg-scale 7.0 \
  --weights-dir ./converted/stable-audio \
  --output ambient.wav
```

### Melody Conditioning Example

MusicGen melody variants can condition generation on an existing audio file, extracting its pitch contour (chromagram) to guide the output:

```bash
# Convert a melody variant
uv run mlx-audio-convert --model facebook/musicgen-melody --output ./converted/musicgen-melody

# Generate with melody conditioning
uv run mlx-audio-generate \
  --model musicgen \
  --prompt "orchestral arrangement with strings" \
  --melody my_humming.wav \
  --seconds 10 \
  --weights-dir ./converted/musicgen-melody \
  --output orchestral.wav
```

The `--melody` flag accepts any WAV file. The pipeline extracts a 12-bin chromagram (one-hot pitch class per frame) and uses it as additional cross-attention conditioning alongside the text prompt. Melody variants also work without `--melody` for text-only generation.

## Requirements

- Python 3.11+
- Apple Silicon Mac (M1/M2/M3/M4)
- [uv](https://github.com/astral-sh/uv) package manager (recommended)

For converting models that only provide PyTorch `.bin` weights (musicgen-medium, musicgen-large):
```bash
uv sync --extra convert   # installs torch for .bin file loading
```

## CLI Parameters

### Generation (`mlx-audio-generate`)

| Parameter | MusicGen | Stable Audio | Default | Description |
|-----------|:--------:|:------------:|---------|-------------|
| `--model` | required | required | — | `musicgen` or `stable_audio` |
| `--prompt` | required | required | — | Text description of desired audio |
| `--seconds` | yes | yes | 5.0 | Duration (max ~30s for MusicGen, ~47s for Stable Audio) |
| `--output` | yes | yes | auto | Output WAV file path |
| `--seed` | yes | yes | random | Random seed for reproducibility |
| `--weights-dir` | yes | yes | — | Path to converted weights directory |
| `--temperature` | yes | — | 1.0 | Sampling temperature (higher = more creative) |
| `--top-k` | yes | — | 250 | Top-k sampling candidates |
| `--guidance-coef` | yes | — | 3.0 | Classifier-free guidance scale |
| `--melody` | yes | — | — | Audio file for melody conditioning (melody variants only) |
| `--steps` | — | yes | 8 | Number of diffusion steps |
| `--cfg-scale` | — | yes | 6.0 | CFG guidance scale |
| `--sampler` | — | yes | euler | ODE sampler (`euler` or `rk4`) |
| `--negative-prompt` | — | yes | "" | Negative prompt for CFG |

### Conversion (`mlx-audio-convert`)

| Parameter | Description |
|-----------|-------------|
| `--model` | HuggingFace repo ID (e.g., `facebook/musicgen-small`) |
| `--output` | Output directory for converted weights |
| `--dtype` | Optional: `float16`, `bfloat16`, or `float32` |
| `--trust-remote-code` | Allow non-whitelisted repo IDs |

#### Supported Repos

**MusicGen (mono):** `facebook/musicgen-small`, `facebook/musicgen-medium`, `facebook/musicgen-large`

**MusicGen Stereo:** `facebook/musicgen-stereo-small`, `facebook/musicgen-stereo-medium`, `facebook/musicgen-stereo-large`

**MusicGen Melody:** `facebook/musicgen-melody`, `facebook/musicgen-melody-large`

**Stable Audio:** `stabilityai/stable-audio-open-small`, `stabilityai/stable-audio-open-1.0`

> **Note:** Some HF repos (musicgen-medium, musicgen-large) only provide `pytorch_model.bin` files instead of safetensors. The converter handles both formats automatically, but PyTorch must be installed (`uv sync --extra convert`).

> **Note:** `stabilityai/stable-audio-open-1.0` is a gated model. You must accept the license agreement on the [HuggingFace model page](https://huggingface.co/stabilityai/stable-audio-open-1.0) before converting.

## Architecture

```
mlx_audio_generate/
├── shared/           # T5 encoder, EnCodec, hub utils, audio I/O
├── models/
│   ├── musicgen/     # Autoregressive: T5 -> transformer -> EnCodec decode
│   │   └── chroma.py # Chromagram extraction for melody conditioning
│   └── stable_audio/ # Diffusion: T5 -> DiT (rectified flow) -> VAE decode
└── cli/              # Unified CLI for generation and conversion
```

**MusicGen**: Text -> T5 encode -> autoregressive transformer with KV cache + classifier-free guidance + codebook delay pattern -> top-k sampling -> EnCodec decode -> 32kHz WAV

**MusicGen Melody**: Text -> T5 encode + chromagram from audio -> cross-attention conditioning -> same pipeline as above

**Stable Audio**: Text -> T5 encode + time conditioning -> rectified flow ODE sampling through DiT -> Oobleck VAE decode -> 44.1kHz stereo WAV

## Development

```bash
# Install with dev dependencies
uv sync

# Lint and format
uv run ruff check .
uv run ruff format .

# Run tests
uv run pytest

# Type checking
uv run mypy mlx_audio_generate/

# Security audit
uv run bandit -r mlx_audio_generate/ -c pyproject.toml
uv run pip-audit

# Import smoke test (no weights needed)
uv run python -c "from mlx_audio_generate.models.musicgen import MusicGenPipeline; print('OK')"
```

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Links

- [GitHub](https://github.com/jasonvassallo/mlx-audio-generate)
- [HuggingFace](https://huggingface.co/jasonvassallo/mlx-audio-generate)
- [MusicGen paper](https://arxiv.org/abs/2306.05284)
- [Stable Audio Open paper](https://arxiv.org/abs/2407.14358)
