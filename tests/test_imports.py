"""Basic import smoke tests — verify package structure without weights."""


def test_import_musicgen_pipeline():
    from mlx_audiogen.models.musicgen import MusicGenPipeline

    assert MusicGenPipeline is not None


def test_import_stable_audio_pipeline():
    from mlx_audiogen.models.stable_audio import StableAudioPipeline

    assert StableAudioPipeline is not None


def test_import_shared_hub():
    from mlx_audiogen.shared.hub import (
        download_model,
        load_all_safetensors,
        load_safetensors,
        save_safetensors,
    )

    assert all(
        callable(fn)
        for fn in [
            download_model,
            load_all_safetensors,
            load_safetensors,
            save_safetensors,
        ]
    )


def test_import_shared_audio_io():
    from mlx_audiogen.shared.audio_io import load_wav, play_audio, save_wav

    assert all(callable(fn) for fn in [save_wav, load_wav, play_audio])


def test_import_version():
    from mlx_audiogen.version import __version__

    assert isinstance(__version__, str)
    assert len(__version__) > 0


def test_available_models():
    from mlx_audiogen.models import AVAILABLE_MODELS

    assert "musicgen" in AVAILABLE_MODELS
    assert "stable_audio" in AVAILABLE_MODELS


# ---------------------------------------------------------------------------
# Model registry (Phase 9b)
# ---------------------------------------------------------------------------


def test_import_model_registry():
    from mlx_audiogen.shared.model_registry import (
        DEFAULT_MODELS_DIR,
        MODEL_REGISTRY,
        list_registry_models,
        resolve_weights_dir,
    )

    assert callable(resolve_weights_dir)
    assert callable(list_registry_models)
    assert isinstance(MODEL_REGISTRY, dict)
    assert DEFAULT_MODELS_DIR is not None


def test_registry_has_all_models():
    from mlx_audiogen.shared.model_registry import MODEL_REGISTRY

    # All 11 MusicGen + 2 Stable Audio + 1 Demucs = 14 entries
    assert len(MODEL_REGISTRY) >= 14
    assert "musicgen-small" in MODEL_REGISTRY
    assert "musicgen-style" in MODEL_REGISTRY
    assert "stable-audio" in MODEL_REGISTRY
    assert "demucs-htdemucs" in MODEL_REGISTRY
    # All values should be HF repo IDs
    for name, repo in MODEL_REGISTRY.items():
        assert "/" in repo, f"Registry entry {name} missing org prefix: {repo}"


def test_list_registry_models():
    from mlx_audiogen.shared.model_registry import list_registry_models

    models = list_registry_models()
    assert isinstance(models, list)
    assert len(models) >= 14
    assert models == sorted(models)  # should be sorted


def test_resolve_weights_dir_existing_path(tmp_path):
    """resolve_weights_dir returns the path directly when it exists."""
    from mlx_audiogen.shared.model_registry import resolve_weights_dir

    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "t5.safetensors").write_bytes(b"")
    result = resolve_weights_dir(str(tmp_path), required_files=["config.json"])
    assert result == tmp_path


def test_resolve_weights_dir_missing_raises():
    """resolve_weights_dir raises FileNotFoundError for unknown model."""
    import pytest

    from mlx_audiogen.shared.model_registry import resolve_weights_dir

    with pytest.raises(FileNotFoundError, match="not found"):
        resolve_weights_dir("/nonexistent/path/not-a-model-name")


def test_resolve_weights_dir_registry_name_without_path():
    """resolve_weights_dir recognizes registry names passed as weights_dir."""
    from mlx_audiogen.shared.model_registry import resolve_weights_dir

    # "musicgen-small" is in registry — if not cached, it will try to download
    # which requires network. Just verify it doesn't raise ValueError about
    # "weights_dir is required" — it should recognize it as a registry name.
    try:
        result = resolve_weights_dir("musicgen-small")
        # If it succeeds, it found cached weights
        assert result.is_dir()
    except (FileNotFoundError, OSError):
        # Download failed (no network) — that's fine for unit tests
        pass
