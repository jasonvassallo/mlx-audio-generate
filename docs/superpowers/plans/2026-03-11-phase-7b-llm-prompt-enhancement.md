# Phase 7b: LLM Prompt Enhancement Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add local LLM prompt enhancement, prompt memory, tag autocomplete, and a Settings tab to the mlx-audiogen web UI.

**Architecture:** Backend adds `mlx-lm` as optional dependency for local LLM inference with lazy loading + 5-min idle unload. Prompt memory stored in `~/.mlx-audiogen/prompt_memory.json` (max 2000 entries). Tag autocomplete served from a static database. Frontend adds Settings tab, enhance preview card, and inline autocomplete.

**Tech Stack:** Python (FastAPI, mlx-lm, Pydantic), TypeScript (React 19, Zustand 5, Tailwind CSS v4)

**Spec:** `docs/superpowers/specs/2026-03-11-phase-7b-llm-prompt-enhancement-design.md`

---

## File Structure

### New Files
| Path | Responsibility |
|------|---------------|
| `tests/test_phase7b.py` | All Phase 7b unit + integration tests |
| `web/src/components/LLMSettingsPanel.tsx` | Settings tab: LLM model dropdown, AI enhance toggle, history context slider, memory management |
| `web/src/components/TagAutocomplete.tsx` | Inline autocomplete dropdown below prompt textarea |
| `web/src/components/EnhancePreview.tsx` | Inline card: enhanced prompt + Accept/Edit/Use Original buttons |

### Modified Files
| Path | Changes |
|------|---------|
| `pyproject.toml` | Add `llm` optional extra with `mlx-lm` dependency |
| `mlx_audiogen/shared/prompt_suggestions.py` | Add `TAG_DATABASE`, `PromptMemory` class, `discover_mlx_models()`, `enhance_with_llm()` |
| `mlx_audiogen/server/app.py` | Add 11 new endpoints, lazy LLM loading, idle unload timer, auto-save to memory in generate handler |
| `web/src/types/api.ts` | Add `EnhanceResponse`, `AnalysisTags`, `LLMModelInfo`, `SettingsData`, `TagEntry`, `PromptMemoryData` types |
| `web/src/api/client.ts` | Add typed wrappers for enhance, tags, llm/*, memory/*, settings endpoints |
| `web/src/store/useStore.ts` | Add server settings state, enhance flow, tag cache, memory state; expand `activeTab` type |
| `web/src/App.tsx` | Add Settings tab, render LLMSettingsPanel, wire up server settings loading |
| `web/src/components/PromptInput.tsx` | Integrate TagAutocomplete |
| `web/src/components/GenerateButton.tsx` | Integrate AI enhance toggle + enhance flow |

---

## Chunk 1: Backend Foundation (Python)

### Task 1: Add `mlx-lm` optional dependency

**Files:**
- Modify: `pyproject.toml:35-46`

- [ ] **Step 1: Add llm extra to pyproject.toml**

In `pyproject.toml`, after the `server` optional dependency group (line 42), add:

```toml
llm = [
    "mlx-lm>=0.22.0",
]
```

- [ ] **Step 2: Sync and verify**

Run: `uv sync --extra llm --extra server --extra dev`
Expected: Installs mlx-lm and its dependencies successfully.

- [ ] **Step 3: Verify import**

Run: `uv run python -c "import mlx_lm; print('mlx-lm', mlx_lm.__version__)"`
Expected: Prints version without error.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat: add mlx-lm as optional dependency for LLM prompt enhancement"
```

---

### Task 2: Add TAG_DATABASE to prompt_suggestions.py

**Files:**
- Modify: `mlx_audiogen/shared/prompt_suggestions.py`
- Test: `tests/test_phase7b.py`

- [ ] **Step 1: Write failing test for TAG_DATABASE**

Create `tests/test_phase7b.py`:

```python
"""Tests for Phase 7b: LLM prompt enhancement, prompt memory, tag autocomplete."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Tag Database
# ---------------------------------------------------------------------------


def test_tag_database_has_all_categories():
    from mlx_audiogen.shared.prompt_suggestions import TAG_DATABASE

    assert set(TAG_DATABASE.keys()) == {"genre", "mood", "instrument", "era", "production"}


def test_tag_database_each_category_nonempty():
    from mlx_audiogen.shared.prompt_suggestions import TAG_DATABASE

    for category, tags in TAG_DATABASE.items():
        assert len(tags) >= 10, f"Category '{category}' has too few tags: {len(tags)}"


def test_tag_database_entries_are_strings():
    from mlx_audiogen.shared.prompt_suggestions import TAG_DATABASE

    for category, tags in TAG_DATABASE.items():
        for tag in tags:
            assert isinstance(tag, str), f"Non-string tag in '{category}': {tag}"
            assert len(tag) > 0, f"Empty tag in '{category}'"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_phase7b.py::test_tag_database_has_all_categories -v`
Expected: FAIL — `TAG_DATABASE` not defined.

- [ ] **Step 3: Implement TAG_DATABASE**

Add to `mlx_audiogen/shared/prompt_suggestions.py` after the `INSTRUMENTS` dict (after line 137):

```python
# Era/style descriptors
ERAS = [
    "80s", "90s", "70s", "60s", "2000s", "2010s",
    "vintage", "modern", "retro", "futuristic", "classic",
    "Y2K", "art deco", "baroque", "contemporary",
]

# Unified tag database for autocomplete (all categories)
TAG_DATABASE: dict[str, list[str]] = {
    "genre": list(GENRES),
    "mood": list(MOODS),
    "instrument": [
        inst for instruments in INSTRUMENTS.values() for inst in instruments
    ],
    "era": list(ERAS),
    "production": list(PRODUCTION),
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_phase7b.py -k "tag_database" -v`
Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add mlx_audiogen/shared/prompt_suggestions.py tests/test_phase7b.py
git commit -m "feat: add TAG_DATABASE for autocomplete with 5 categories"
```

---

### Task 3: Add PromptMemory class

**Files:**
- Modify: `mlx_audiogen/shared/prompt_suggestions.py`
- Test: `tests/test_phase7b.py`

- [ ] **Step 1: Write failing tests for PromptMemory**

Append to `tests/test_phase7b.py`:

```python
# ---------------------------------------------------------------------------
# Prompt Memory
# ---------------------------------------------------------------------------


@pytest.fixture
def memory_dir(tmp_path):
    """Create a temp directory to use instead of ~/.mlx-audiogen/."""
    return tmp_path


def test_prompt_memory_init_creates_empty(memory_dir):
    from mlx_audiogen.shared.prompt_suggestions import PromptMemory

    mem = PromptMemory(memory_dir / "prompt_memory.json")
    assert mem.history == []
    assert mem.style_profile["generation_count"] == 0


def test_prompt_memory_add_entry(memory_dir):
    from mlx_audiogen.shared.prompt_suggestions import PromptMemory

    mem = PromptMemory(memory_dir / "prompt_memory.json")
    mem.add_entry("dark ambient pad", "musicgen", {"seconds": 10})
    assert len(mem.history) == 1
    assert mem.history[0]["prompt"] == "dark ambient pad"
    assert mem.style_profile["generation_count"] == 1


def test_prompt_memory_persist_and_load(memory_dir):
    from mlx_audiogen.shared.prompt_suggestions import PromptMemory

    path = memory_dir / "prompt_memory.json"
    mem = PromptMemory(path)
    mem.add_entry("synthwave arpeggio", "musicgen", {"seconds": 5})
    mem.save()

    mem2 = PromptMemory(path)
    assert len(mem2.history) == 1
    assert mem2.history[0]["prompt"] == "synthwave arpeggio"


def test_prompt_memory_eviction_at_max(memory_dir):
    from mlx_audiogen.shared.prompt_suggestions import PromptMemory

    mem = PromptMemory(memory_dir / "prompt_memory.json", max_entries=5)
    for i in range(7):
        mem.add_entry(f"prompt {i}", "musicgen", {})
    assert len(mem.history) == 5
    # Oldest evicted — newest kept
    assert mem.history[0]["prompt"] == "prompt 2"
    assert mem.history[-1]["prompt"] == "prompt 6"


def test_prompt_memory_style_profile_derivation(memory_dir):
    from mlx_audiogen.shared.prompt_suggestions import PromptMemory

    mem = PromptMemory(memory_dir / "prompt_memory.json")
    mem.add_entry("dark ambient pad, warm analog", "musicgen", {"seconds": 10})
    mem.add_entry("dark electronic, synth bass", "musicgen", {"seconds": 8})
    mem.add_entry("ambient dreamy, synth pad", "musicgen", {"seconds": 12})

    profile = mem.style_profile
    assert profile["generation_count"] == 3
    assert "ambient" in profile["top_genres"]
    assert profile["preferred_duration"] == 10  # median of [10, 8, 12]


def test_prompt_memory_enhanced_prompt_stored(memory_dir):
    from mlx_audiogen.shared.prompt_suggestions import PromptMemory

    mem = PromptMemory(memory_dir / "prompt_memory.json")
    mem.add_entry("lo-fi", "musicgen", {}, enhanced_prompt="lo-fi chill beats, warm vinyl")
    assert mem.history[0]["enhanced_prompt"] == "lo-fi chill beats, warm vinyl"


def test_prompt_memory_recent_prompts(memory_dir):
    from mlx_audiogen.shared.prompt_suggestions import PromptMemory

    mem = PromptMemory(memory_dir / "prompt_memory.json")
    for i in range(10):
        mem.add_entry(f"prompt {i}", "musicgen", {})
    recent = mem.recent_prompts(5)
    assert len(recent) == 5
    assert recent[0] == "prompt 9"  # newest first
    assert recent[4] == "prompt 5"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_phase7b.py -k "prompt_memory" -v`
Expected: FAIL — `PromptMemory` not defined.

- [ ] **Step 3: Implement PromptMemory class**

Add to `mlx_audiogen/shared/prompt_suggestions.py` after `TAG_DATABASE`:

```python
import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


class PromptMemory:
    """Manages prompt history and derived style profile.

    Stores history entries and auto-derives a style profile (top genres,
    moods, instruments) from the history using analyze_prompt().
    """

    def __init__(
        self, path: Path | None = None, max_entries: int = 2000
    ):
        self._path = path or (Path.home() / ".mlx-audiogen" / "prompt_memory.json")
        self._max_entries = max_entries
        self.history: list[dict] = []
        self.style_profile: dict = {
            "top_genres": [],
            "top_moods": [],
            "top_instruments": [],
            "preferred_duration": 0,
            "generation_count": 0,
        }
        self._load()

    def _load(self) -> None:
        """Load from disk if file exists."""
        if self._path.is_file():
            try:
                data = json.loads(self._path.read_text())
                self.history = data.get("history", [])
                # Always re-derive profile from history
                self._derive_profile()
            except (json.JSONDecodeError, OSError):
                pass

    def save(self) -> None:
        """Persist to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {"history": self.history, "style_profile": self.style_profile}
        self._path.write_text(json.dumps(data, indent=2))

    def add_entry(
        self,
        prompt: str,
        model: str,
        params: dict,
        enhanced_prompt: str | None = None,
    ) -> None:
        """Append a generation entry and re-derive the style profile."""
        entry: dict = {
            "prompt": prompt,
            "model": model,
            "params": params,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if enhanced_prompt:
            entry["enhanced_prompt"] = enhanced_prompt
        self.history.append(entry)

        # Evict oldest if over limit
        if len(self.history) > self._max_entries:
            self.history = self.history[-self._max_entries :]

        self._derive_profile()
        self.save()

    def recent_prompts(self, count: int = 50) -> list[str]:
        """Return the N most recent prompt strings, newest first."""
        if count <= 0:
            # 0 = all history
            prompts = [e["prompt"] for e in reversed(self.history)]
        else:
            prompts = [e["prompt"] for e in reversed(self.history)][:count]
        return prompts

    def clear(self) -> None:
        """Clear all history and reset profile."""
        self.history = []
        self.style_profile = {
            "top_genres": [],
            "top_moods": [],
            "top_instruments": [],
            "preferred_duration": 0,
            "generation_count": 0,
        }
        self.save()

    def to_dict(self) -> dict:
        """Return serializable dict."""
        return {"history": self.history, "style_profile": self.style_profile}

    def _derive_profile(self) -> None:
        """Re-derive style profile from full history."""
        genre_counter: Counter[str] = Counter()
        mood_counter: Counter[str] = Counter()
        instrument_counter: Counter[str] = Counter()
        durations: list[float] = []

        for entry in self.history:
            analysis = analyze_prompt(entry["prompt"], count=0)
            genre_counter.update(analysis["genres"])
            mood_counter.update(analysis["moods"])
            instrument_counter.update(analysis["instruments"])
            secs = entry.get("params", {}).get("seconds")
            if secs is not None:
                durations.append(float(secs))

        self.style_profile = {
            "top_genres": [g for g, _ in genre_counter.most_common(5)],
            "top_moods": [m for m, _ in mood_counter.most_common(5)],
            "top_instruments": [i for i, _ in instrument_counter.most_common(5)],
            "preferred_duration": (
                int(statistics.median(durations)) if durations else 0
            ),
            "generation_count": len(self.history),
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_phase7b.py -k "prompt_memory" -v`
Expected: All 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add mlx_audiogen/shared/prompt_suggestions.py tests/test_phase7b.py
git commit -m "feat: add PromptMemory class with history, profile derivation, persistence"
```

---

### Task 4: Add discover_mlx_models() and enhance_with_llm()

**Files:**
- Modify: `mlx_audiogen/shared/prompt_suggestions.py`
- Test: `tests/test_phase7b.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_phase7b.py`:

```python
# ---------------------------------------------------------------------------
# MLX Model Discovery
# ---------------------------------------------------------------------------


def test_discover_mlx_models_empty_dir(tmp_path):
    from mlx_audiogen.shared.prompt_suggestions import discover_mlx_models

    models = discover_mlx_models([tmp_path])
    assert models == []


def test_discover_mlx_models_valid_model(tmp_path):
    from mlx_audiogen.shared.prompt_suggestions import discover_mlx_models

    # Create a fake MLX model directory
    model_dir = tmp_path / "mlx-community" / "Qwen3.5-9B-6bit"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text('{"model_type": "qwen2"}')
    (model_dir / "model.safetensors").write_bytes(b"fake")
    (model_dir / "tokenizer_config.json").write_text("{}")

    models = discover_mlx_models([tmp_path])
    assert len(models) == 1
    assert models[0]["id"] == "mlx-community/Qwen3.5-9B-6bit"
    assert models[0]["name"] == "Qwen3.5-9B-6bit"
    assert "path" not in models[0]  # no path exposed


def test_discover_mlx_models_filters_non_llm(tmp_path):
    from mlx_audiogen.shared.prompt_suggestions import discover_mlx_models

    # Create a model dir WITHOUT tokenizer (like EnCodec)
    model_dir = tmp_path / "mlx-community" / "encodec-32khz"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text('{"model_type": "encodec"}')
    (model_dir / "model.safetensors").write_bytes(b"fake")
    # No tokenizer_config.json or tokenizer.json

    models = discover_mlx_models([tmp_path])
    assert models == []


def test_discover_mlx_models_hf_snapshot_structure(tmp_path):
    from mlx_audiogen.shared.prompt_suggestions import discover_mlx_models

    # HF cache: models--org--name/snapshots/<hash>/
    model_root = tmp_path / "models--mlx-community--Qwen3.5-9B-6bit"
    snapshot = model_root / "snapshots" / "abc123def"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text('{"model_type": "qwen2"}')
    (snapshot / "model.safetensors").write_bytes(b"fake")
    (snapshot / "tokenizer_config.json").write_text("{}")

    models = discover_mlx_models([tmp_path])
    assert len(models) == 1
    assert models[0]["id"] == "mlx-community/Qwen3.5-9B-6bit"


# ---------------------------------------------------------------------------
# LLM Enhancement
# ---------------------------------------------------------------------------


def test_enhance_with_llm_fallback_no_model():
    from mlx_audiogen.shared.prompt_suggestions import enhance_with_llm

    result = enhance_with_llm("dark ambient", model_path=None)
    assert result["original"] == "dark ambient"
    assert result["used_llm"] is False
    # Falls back to template engine
    assert len(result["enhanced"]) > len("dark ambient")


def test_enhance_with_llm_mock_success():
    from mlx_audiogen.shared.prompt_suggestions import enhance_with_llm

    with patch("mlx_audiogen.shared.prompt_suggestions._run_llm_inference") as mock_llm:
        mock_llm.return_value = "dark ambient pad, warm analog, slow tempo, reverb-drenched"
        result = enhance_with_llm("dark ambient", model_path="/fake/path")
        assert result["used_llm"] is True
        assert "reverb-drenched" in result["enhanced"]
        assert result["original"] == "dark ambient"


def test_enhance_with_llm_timeout_fallback():
    from mlx_audiogen.shared.prompt_suggestions import enhance_with_llm

    with patch("mlx_audiogen.shared.prompt_suggestions._run_llm_inference") as mock_llm:
        mock_llm.side_effect = TimeoutError("LLM timed out")
        result = enhance_with_llm("dark ambient", model_path="/fake/path")
        assert result["used_llm"] is False
        assert result["warning"] is not None
        assert "timeout" in result["warning"].lower() or "timed out" in result["warning"].lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_phase7b.py -k "discover_mlx or enhance_with" -v`
Expected: FAIL — functions not defined.

- [ ] **Step 3: Implement discover_mlx_models()**

Add to `mlx_audiogen/shared/prompt_suggestions.py`:

```python
def discover_mlx_models(
    scan_paths: list[Path] | None = None,
) -> list[dict]:
    """Scan filesystem for valid MLX LLM model directories.

    A valid model dir has: config.json, *.safetensors, and a tokenizer file.
    Returns list of dicts with 'id', 'name', 'size_gb', 'source' (no paths).
    """
    if scan_paths is None:
        scan_paths = [
            Path.home() / ".cache" / "huggingface" / "hub",
            Path.home() / "Library" / "Caches" / "huggingface" / "hub",
            Path.home() / ".lmstudio" / "hub" / "models",
        ]

    found: dict[str, dict] = {}  # id -> info (dedup by id)

    for base in scan_paths:
        if not base.is_dir():
            continue
        _scan_dir_for_models(base, base, found)

    return list(found.values())


def _scan_dir_for_models(
    directory: Path, scan_root: Path, found: dict[str, dict]
) -> None:
    """Recursively scan a directory for MLX model dirs."""
    try:
        children = list(directory.iterdir())
    except PermissionError:
        return

    # Check if this directory IS a model dir
    if _is_valid_llm_dir(directory):
        model_id = _derive_model_id(directory, scan_root)
        if model_id and model_id not in found:
            size_gb = _estimate_size_gb(directory)
            source = "lmstudio" if ".lmstudio" in str(scan_root) else "huggingface"
            found[model_id] = {
                "id": model_id,
                "name": model_id.split("/")[-1] if "/" in model_id else model_id,
                "size_gb": round(size_gb, 1),
                "source": source,
            }
        return  # Don't recurse into model dirs

    # HF snapshot resolution: models--org--name/snapshots/<hash>/
    for child in children:
        if not child.is_dir():
            continue
        if child.name.startswith("models--"):
            snapshots_dir = child / "snapshots"
            if snapshots_dir.is_dir():
                # Use the latest snapshot (last in sorted order)
                snapshot_dirs = sorted(
                    [s for s in snapshots_dir.iterdir() if s.is_dir()]
                )
                if snapshot_dirs:
                    _scan_dir_for_models(snapshot_dirs[-1], scan_root, found)
        else:
            # Recurse max 2 levels deep
            rel = child.relative_to(scan_root)
            if len(rel.parts) < 3:
                _scan_dir_for_models(child, scan_root, found)


def _is_valid_llm_dir(directory: Path) -> bool:
    """Check if a directory contains a valid MLX LLM model."""
    has_config = (directory / "config.json").is_file()
    has_safetensors = any(directory.glob("*.safetensors"))
    has_tokenizer = (
        (directory / "tokenizer_config.json").is_file()
        or (directory / "tokenizer.json").is_file()
    )
    return has_config and has_safetensors and has_tokenizer


def _derive_model_id(model_dir: Path, scan_root: Path) -> str | None:
    """Derive a human-readable model identifier from the path."""
    # Try to get org/name from path structure
    try:
        rel = model_dir.relative_to(scan_root)
    except ValueError:
        rel = Path(model_dir.name)

    parts = rel.parts

    # HF snapshot: models--org--name/snapshots/<hash> -> org/name
    for i, part in enumerate(parts):
        if part.startswith("models--"):
            segments = part.split("--")
            if len(segments) >= 3:
                return f"{segments[1]}/{'/'.join(segments[2:])}"

    # Direct structure: org/name/ or just name/
    if len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1]}"
    if len(parts) == 1:
        return parts[0]

    return None


def _estimate_size_gb(model_dir: Path) -> float:
    """Estimate model size from safetensors files."""
    total = sum(f.stat().st_size for f in model_dir.glob("*.safetensors"))
    return total / (1024**3)
```

- [ ] **Step 4: Implement enhance_with_llm() and _run_llm_inference()**

Add to `mlx_audiogen/shared/prompt_suggestions.py`:

```python
import signal


_LLM_SYSTEM_PROMPT = """You are a music prompt engineer for AI audio generation models (MusicGen, Stable Audio).
Given a user's prompt, enhance it with rich musical descriptors including genre, mood,
instrumentation, tempo, production style, and era details.
Keep the user's core intent and artistic direction. Output ONLY the enhanced prompt
as a single line, nothing else. Do not add explanations or formatting.

{memory_context}"""


def enhance_with_llm(
    prompt: str,
    model_path: str | None = None,
    memory_context: str = "",
    timeout: int = 30,
) -> dict:
    """Enhance a prompt using a local MLX LLM, with template fallback.

    Returns dict with: original, enhanced, analysis_tags, used_llm, warning.
    """
    analysis = analyze_prompt(prompt, count=0)
    analysis_tags = {
        "genres": analysis["genres"],
        "moods": analysis["moods"],
        "instruments": analysis["instruments"],
        "missing": analysis["missing"],
    }

    if model_path is None:
        # No LLM available — use template fallback
        suggestions = suggest_refinements(prompt, count=1)
        return {
            "original": prompt,
            "enhanced": suggestions[0] if suggestions else prompt,
            "analysis_tags": analysis_tags,
            "used_llm": False,
            "warning": None,
        }

    try:
        system = _LLM_SYSTEM_PROMPT.format(memory_context=memory_context)
        enhanced = _run_llm_inference(prompt, system, model_path, timeout)
        # Truncate to 2000 chars for safety
        enhanced = enhanced[:2000].strip()
        if not enhanced:
            raise ValueError("LLM returned empty response")
        return {
            "original": prompt,
            "enhanced": enhanced,
            "analysis_tags": analysis_tags,
            "used_llm": True,
            "warning": None,
        }
    except TimeoutError:
        suggestions = suggest_refinements(prompt, count=1)
        return {
            "original": prompt,
            "enhanced": suggestions[0] if suggestions else prompt,
            "analysis_tags": analysis_tags,
            "used_llm": False,
            "warning": "LLM timed out, used template suggestions",
        }
    except Exception as e:
        suggestions = suggest_refinements(prompt, count=1)
        return {
            "original": prompt,
            "enhanced": suggestions[0] if suggestions else prompt,
            "analysis_tags": analysis_tags,
            "used_llm": False,
            "warning": f"LLM error: {e}",
        }


def _run_llm_inference(
    prompt: str, system: str, model_path: str, timeout: int = 30
) -> str:
    """Run LLM inference with timeout. Separated for easy mocking."""
    import threading

    result_holder: list[str] = []
    error_holder: list[Exception] = []

    def _infer():
        try:
            from mlx_lm import generate, load

            model, tokenizer = load(model_path)
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]
            chat_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            response = generate(
                model, tokenizer, prompt=chat_prompt, max_tokens=512
            )
            result_holder.append(response)
        except Exception as e:
            error_holder.append(e)

    thread = threading.Thread(target=_infer, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        raise TimeoutError(f"LLM inference exceeded {timeout}s timeout")
    if error_holder:
        raise error_holder[0]
    if not result_holder:
        raise RuntimeError("LLM produced no output")
    return result_holder[0]
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_phase7b.py -k "discover_mlx or enhance_with" -v`
Expected: All 7 tests PASS.

- [ ] **Step 6: Run full suite**

Run: `uv run ruff format . && uv run ruff check . && uv run mypy mlx_audiogen/ && uv run bandit -r mlx_audiogen/ -c pyproject.toml && uv run pip-audit && uv run pytest`
Expected: All pass.

- [ ] **Step 7: Commit**

```bash
git add mlx_audiogen/shared/prompt_suggestions.py tests/test_phase7b.py
git commit -m "feat: add MLX model discovery and LLM prompt enhancement with fallback"
```

---

### Task 5: Add server endpoints (enhance, tags, llm, memory, settings)

**Files:**
- Modify: `mlx_audiogen/server/app.py`
- Test: `tests/test_phase7b.py`

- [ ] **Step 1: Write failing integration tests**

Append to `tests/test_phase7b.py`:

```python
from mlx_audiogen.server.app import app, _jobs, _weights_dirs


@pytest.fixture(autouse=True)
def _clean_server_state():
    _jobs.clear()
    _weights_dirs.clear()
    yield
    _jobs.clear()
    _weights_dirs.clear()


@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# Server Endpoints
# ---------------------------------------------------------------------------


def test_enhance_endpoint_template_fallback(client):
    """POST /api/enhance falls back to template when no LLM."""
    resp = client.post("/api/enhance", json={"prompt": "dark ambient"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["original"] == "dark ambient"
    assert data["used_llm"] is False
    assert "enhanced" in data
    assert "analysis_tags" in data


def test_tags_endpoint(client):
    """GET /api/tags returns all tag categories."""
    resp = client.get("/api/tags")
    assert resp.status_code == 200
    data = resp.json()
    assert set(data.keys()) == {"genre", "mood", "instrument", "era", "production"}
    for category, tags in data.items():
        assert isinstance(tags, list)
        assert len(tags) > 0


def test_llm_models_endpoint(client):
    """GET /api/llm/models returns a list (possibly empty)."""
    resp = client.get("/api/llm/models")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_llm_status_endpoint(client):
    """GET /api/llm/status returns status dict."""
    resp = client.get("/api/llm/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "model_id" in data
    assert "loaded" in data


def test_settings_get_defaults(client):
    """GET /api/settings returns defaults."""
    resp = client.get("/api/settings")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ai_enhance"] is True
    assert data["history_context_count"] == 50


def test_settings_update(client):
    """POST /api/settings persists changes."""
    resp = client.post(
        "/api/settings",
        json={"ai_enhance": False, "history_context_count": 25},
    )
    assert resp.status_code == 200
    # Verify persisted
    resp2 = client.get("/api/settings")
    data = resp2.json()
    assert data["ai_enhance"] is False
    assert data["history_context_count"] == 25


def test_memory_lifecycle(client, tmp_path):
    """GET/DELETE /api/memory lifecycle."""
    resp = client.get("/api/memory")
    assert resp.status_code == 200

    resp = client.delete("/api/memory")
    assert resp.status_code == 200

    resp = client.get("/api/memory")
    data = resp.json()
    assert data["history"] == []


def test_memory_export(client):
    """GET /api/memory/export returns JSON file."""
    resp = client.get("/api/memory/export")
    assert resp.status_code == 200
    assert "application/json" in resp.headers.get("content-type", "")


def test_memory_import_validation(client):
    """POST /api/memory/import rejects invalid data."""
    # Import with no file
    resp = client.post(
        "/api/memory/import",
        files={"file": ("test.json", b"not json", "application/json")},
    )
    assert resp.status_code == 400
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_phase7b.py -k "endpoint or lifecycle or export or import" -v`
Expected: FAIL — endpoints not defined.

- [ ] **Step 3: Add Pydantic models and all new endpoints to app.py**

Add to `mlx_audiogen/server/app.py` — new Pydantic models after `ModelInfo` (line ~127), then new endpoint sections. This is a large addition. The key additions are:

1. **Pydantic models:** `EnhanceRequest`, `AnalysisTags`, `EnhanceResponse`, `LLMModelInfo`, `LLMSelectRequest`, `SettingsData`
2. **Global state:** `_llm_model_path`, `_llm_model_id`, `_llm_pipeline`, `_llm_last_used`, `_llm_busy`, `_llm_models_cache`, `_prompt_memory`, `_server_settings`
3. **Endpoints:**
   - `POST /api/enhance` — LLM enhancement with template fallback
   - `GET /api/tags` — static tag database
   - `GET /api/llm/models` — list discovered models
   - `POST /api/llm/select` — switch active LLM
   - `GET /api/llm/status` — LLM status
   - `GET /api/memory` — return prompt memory
   - `DELETE /api/memory` — clear memory
   - `GET /api/memory/export` — download JSON
   - `POST /api/memory/import` — upload/restore JSON (with UploadFile)
   - `GET /api/settings` — get settings
   - `POST /api/settings` — update settings
4. **Auto-save in generate handler:** After successful generation, save prompt to memory
5. **LLM idle timer:** Background thread checks every 60s, unloads LLM after 5 min idle
6. **CLI flags:** `--llm-model` and `--llm-idle-timeout` in `main()` and `launch_app()`

Important implementation details:
- Import `UploadFile, File` from fastapi for memory import
- Use `from mlx_audiogen.shared.prompt_suggestions import ...` with lazy imports
- Settings file at `~/.mlx-audiogen/settings.json`
- Memory path at `~/.mlx-audiogen/prompt_memory.json`
- The `_run_generation` function gets a memory save call after `job.status = JobStatus.DONE`

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_phase7b.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Run full suite**

Run: `uv run ruff format . && uv run ruff check . && uv run mypy mlx_audiogen/ && uv run bandit -r mlx_audiogen/ -c pyproject.toml && uv run pip-audit && uv run pytest`
Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add mlx_audiogen/server/app.py tests/test_phase7b.py
git commit -m "feat: add enhance, tags, LLM, memory, and settings API endpoints"
```

---

## Chunk 2: Frontend (TypeScript/React)

### Task 6: Add TypeScript types and API client wrappers

**Files:**
- Modify: `web/src/types/api.ts`
- Modify: `web/src/api/client.ts`

- [ ] **Step 1: Add types to api.ts**

Append to `web/src/types/api.ts`:

```typescript
/** Response from POST /api/enhance. */
export interface AnalysisTags {
  genres: string[];
  moods: string[];
  instruments: string[];
  missing: string[];
}

export interface EnhanceResponse {
  original: string;
  enhanced: string;
  analysis_tags: AnalysisTags;
  used_llm: boolean;
  warning: string | null;
}

/** LLM model info from GET /api/llm/models. */
export interface LLMModelInfo {
  id: string;
  name: string;
  size_gb: number;
  source: "huggingface" | "lmstudio";
}

/** Server-side settings from GET /api/settings. */
export interface ServerSettings {
  llm_model: string | null;
  ai_enhance: boolean;
  history_context_count: number;
}

/** LLM status from GET /api/llm/status. */
export interface LLMStatus {
  model_id: string | null;
  loaded: boolean;
  idle_seconds: number;
  memory_mb: number;
}

/** Tag database from GET /api/tags. */
export type TagDatabase = Record<string, string[]>;

/** Prompt memory from GET /api/memory. */
export interface PromptMemoryData {
  history: Array<{
    prompt: string;
    enhanced_prompt?: string;
    model: string;
    params: Record<string, unknown>;
    timestamp: string;
  }>;
  style_profile: {
    top_genres: string[];
    top_moods: string[];
    top_instruments: string[];
    preferred_duration: number;
    generation_count: number;
  };
}
```

- [ ] **Step 2: Add API client wrappers to client.ts**

Append to `web/src/api/client.ts` and add the new types to the import:

```typescript
// Add to imports at top:
import type {
  // ... existing imports ...
  EnhanceResponse,
  LLMModelInfo,
  LLMStatus,
  ServerSettings,
  TagDatabase,
  PromptMemoryData,
} from "../types/api";

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
export function selectLLMModel(modelId: string): Promise<{ status: string }> {
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
export async function importMemory(file: File): Promise<{ status: string }> {
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
```

- [ ] **Step 3: Verify TypeScript compiles**

Run: `cd web && npx tsc --noEmit`
Expected: No errors.

- [ ] **Step 4: Commit**

```bash
git add web/src/types/api.ts web/src/api/client.ts
git commit -m "feat(web): add TypeScript types and API client for Phase 7b endpoints"
```

---

### Task 7: Update Zustand store with enhance flow and settings

**Files:**
- Modify: `web/src/store/useStore.ts`

- [ ] **Step 1: Expand AppState interface and add new state/actions**

Key additions to the store:
- `activeTab` type expands to `"generate" | "suggest" | "settings"`
- New `serverSettings` state + `loadServerSettings()` + `updateServerSetting()`
- New `enhanceResult` state + `enhancePrompt()` + `clearEnhanceResult()`
- New `tagDatabase` state + `loadTags()`
- New `promptMemory` state + `loadMemory()` + `clearMemory()` + `importMemory()`
- New `llmModels` + `llmStatus` state + `loadLLMModels()` + `selectLLMModel()`
- Modify `generate()` to insert enhance flow when `serverSettings.ai_enhance` is true

- [ ] **Step 2: Verify TypeScript compiles**

Run: `cd web && npx tsc --noEmit`
Expected: No errors.

- [ ] **Step 3: Commit**

```bash
git add web/src/store/useStore.ts
git commit -m "feat(web): add enhance flow, server settings, tags, and memory to Zustand store"
```

---

### Task 8: Create EnhancePreview component

**Files:**
- Create: `web/src/components/EnhancePreview.tsx`

- [ ] **Step 1: Create EnhancePreview.tsx**

Component shows the enhanced prompt with Accept & Generate / Edit / Use Original buttons.
Props from store: `enhanceResult`, `clearEnhanceResult`, `setParam("prompt", ...)`, `generate()`.

Displays:
- Enhanced prompt text (plain text, safe rendering)
- Warning banner if `used_llm === false`
- Three buttons: Accept & Generate (accent), Edit (opens prompt with enhanced text), Use Original (generates with original)

- [ ] **Step 2: Verify TypeScript compiles**

Run: `cd web && npx tsc --noEmit`
Expected: No errors.

- [ ] **Step 3: Commit**

```bash
git add web/src/components/EnhancePreview.tsx
git commit -m "feat(web): add EnhancePreview component for LLM prompt approval flow"
```

---

### Task 9: Create TagAutocomplete component

**Files:**
- Create: `web/src/components/TagAutocomplete.tsx`

- [ ] **Step 1: Create TagAutocomplete.tsx**

Component renders a dropdown below the prompt textarea with filtered, color-coded tag suggestions.

Props: `query: string`, `onSelect: (tag: string) => void`, `onDismiss: () => void`, `visible: boolean`

Behavior:
- Fetches tags from store on mount (cached)
- Filters tags by substring match on `query` (case-insensitive, min 2 chars)
- Shows max 8 results, each with colored dot (amber=genre, emerald=mood, sky=instrument, violet=era, rose=production)
- Tab inserts selected tag, Enter/Escape dismisses
- Click on tag inserts it

- [ ] **Step 2: Verify TypeScript compiles**

Run: `cd web && npx tsc --noEmit`
Expected: No errors.

- [ ] **Step 3: Commit**

```bash
git add web/src/components/TagAutocomplete.tsx
git commit -m "feat(web): add TagAutocomplete with color-coded inline suggestions"
```

---

### Task 10: Create LLMSettingsPanel component

**Files:**
- Create: `web/src/components/LLMSettingsPanel.tsx`

- [ ] **Step 1: Create LLMSettingsPanel.tsx**

Rendered inside the Settings tab content area. Contains:

1. **LLM Model dropdown** — lists `llmModels` from store, shows `name (size_gb GB)`, refresh button, status indicator
2. **AI Enhance toggle** — bound to `serverSettings.ai_enhance`
3. **History Context slider** — range 0-100 + text input, default 50, warning at 160+
4. **Prompt Memory section** — style profile display (colored tags), generation count, Export/Clear/Import buttons
5. **Theme placeholder** — disabled dropdown "Dark (Default)" with tooltip

- [ ] **Step 2: Verify TypeScript compiles**

Run: `cd web && npx tsc --noEmit`
Expected: No errors.

- [ ] **Step 3: Commit**

```bash
git add web/src/components/LLMSettingsPanel.tsx
git commit -m "feat(web): add LLMSettingsPanel with model selector, memory management, settings"
```

---

### Task 11: Integrate components into App, PromptInput, GenerateButton

**Files:**
- Modify: `web/src/App.tsx`
- Modify: `web/src/components/PromptInput.tsx`
- Modify: `web/src/components/GenerateButton.tsx`

- [ ] **Step 1: Update App.tsx**

- Add `"settings"` to TABS array
- Import and render `LLMSettingsPanel` when `activeTab === "settings"`
- Load server settings, tags, LLM models on mount
- Update `setActiveTab` cast to include `"settings"`

- [ ] **Step 2: Update PromptInput.tsx**

- Import `TagAutocomplete`
- Track the current word being typed (split by comma, get last segment)
- Show/hide autocomplete based on word length >= 2
- On tag select: insert tag at cursor with comma separator

- [ ] **Step 3: Update GenerateButton.tsx**

- Import `EnhancePreview`
- Before generation: if `serverSettings.ai_enhance` is true, call `enhancePrompt()` first
- Show `EnhancePreview` when `enhanceResult` is present
- Add compact AI Enhance toggle next to the button

- [ ] **Step 4: Build and verify**

Run: `cd web && npm run build`
Expected: Build succeeds with no errors.

- [ ] **Step 5: Commit**

```bash
git add web/src/App.tsx web/src/components/PromptInput.tsx web/src/components/GenerateButton.tsx
git commit -m "feat(web): integrate enhance preview, tag autocomplete, and settings tab"
```

---

## Chunk 3: Full Suite + Docs + Push

### Task 12: Run full test suite, update docs, commit, push

**Files:**
- Modify: `CLAUDE.md`
- Modify: memory files

- [ ] **Step 1: Run full test suite**

```bash
uv run ruff format .
uv run ruff check .
uv run mypy mlx_audiogen/
uv run bandit -r mlx_audiogen/ -c pyproject.toml
uv run pip-audit
uv run pytest
cd web && npm run build
```

Fix any failures before proceeding.

- [ ] **Step 2: Update CLAUDE.md**

Add to CLAUDE.md:
- New API endpoints table entries (enhance, tags, llm/*, memory/*, settings)
- LLM integration section (lazy loading, idle unload, model discovery)
- Prompt memory section
- Updated Web UI section (Settings tab, enhance flow, tag autocomplete)
- New pyproject.toml `llm` extra

- [ ] **Step 3: Update memory files**

Update `MEMORY.md` and relevant memory files to reflect Phase 7b completion.

- [ ] **Step 4: Final commit and push**

```bash
git add -A
git commit -m "docs: update CLAUDE.md and memory for Phase 7b completion"
git push
```
