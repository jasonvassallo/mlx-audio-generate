"""Basic import smoke tests — verify package structure without weights."""


def test_import_musicgen_pipeline():
    from mlx_audio_generate.models.musicgen import MusicGenPipeline

    assert MusicGenPipeline is not None


def test_import_stable_audio_pipeline():
    from mlx_audio_generate.models.stable_audio import StableAudioPipeline

    assert StableAudioPipeline is not None


def test_import_shared_hub():
    from mlx_audio_generate.shared.hub import (
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
    from mlx_audio_generate.shared.audio_io import load_wav, play_audio, save_wav

    assert all(callable(fn) for fn in [save_wav, load_wav, play_audio])


def test_import_version():
    from mlx_audio_generate.version import __version__

    assert isinstance(__version__, str)
    assert len(__version__) > 0


def test_available_models():
    from mlx_audio_generate.models import AVAILABLE_MODELS

    assert "musicgen" in AVAILABLE_MODELS
    assert "stable_audio" in AVAILABLE_MODELS
