"""Pipeline error handling tests.

Verifies helpful error messages for common failures.
"""

import json
import os
import tempfile

import pytest

# ---------------------------------------------------------------------------
# MusicGen pipeline errors
# ---------------------------------------------------------------------------


class TestMusicGenErrors:
    def test_weights_dir_none(self):
        from mlx_audiogen.models.musicgen import MusicGenPipeline

        with pytest.raises(ValueError, match="weights_dir is required"):
            MusicGenPipeline.from_pretrained(None)

    def test_weights_dir_not_exists(self):
        from mlx_audiogen.models.musicgen import MusicGenPipeline

        with pytest.raises(FileNotFoundError, match="Weights directory not found"):
            MusicGenPipeline.from_pretrained("/nonexistent/path")

    def test_missing_config_json(self):
        from mlx_audiogen.models.musicgen import MusicGenPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy weight files but no config
            open(os.path.join(tmpdir, "t5.safetensors"), "w").close()
            open(os.path.join(tmpdir, "decoder.safetensors"), "w").close()
            with pytest.raises(FileNotFoundError, match="config.json"):
                MusicGenPipeline.from_pretrained(tmpdir)

    def test_missing_t5_weights(self):
        from mlx_audiogen.models.musicgen import MusicGenPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create config but missing t5.safetensors
            config = {"decoder": {"hidden_size": 1024, "num_codebooks": 4}}
            with open(os.path.join(tmpdir, "config.json"), "w") as f:
                json.dump(config, f)
            open(os.path.join(tmpdir, "decoder.safetensors"), "w").close()
            with pytest.raises(FileNotFoundError, match="t5.safetensors"):
                MusicGenPipeline.from_pretrained(tmpdir)

    def test_missing_decoder_weights(self):
        from mlx_audiogen.models.musicgen import MusicGenPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"decoder": {"hidden_size": 1024, "num_codebooks": 4}}
            with open(os.path.join(tmpdir, "config.json"), "w") as f:
                json.dump(config, f)
            open(os.path.join(tmpdir, "t5.safetensors"), "w").close()
            with pytest.raises(FileNotFoundError, match="decoder.safetensors"):
                MusicGenPipeline.from_pretrained(tmpdir)

    def test_corrupted_config_json(self):
        from mlx_audiogen.models.musicgen import MusicGenPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "config.json"), "w") as f:
                f.write("not valid json {{{")
            open(os.path.join(tmpdir, "t5.safetensors"), "w").close()
            open(os.path.join(tmpdir, "decoder.safetensors"), "w").close()
            with pytest.raises(json.JSONDecodeError):
                MusicGenPipeline.from_pretrained(tmpdir)


# ---------------------------------------------------------------------------
# Stable Audio pipeline errors
# ---------------------------------------------------------------------------


class TestStableAudioErrors:
    def test_weights_dir_none(self):
        from mlx_audiogen.models.stable_audio import StableAudioPipeline

        with pytest.raises(ValueError, match="weights_dir is required"):
            StableAudioPipeline.from_pretrained(None)

    def test_weights_dir_not_exists(self):
        from mlx_audiogen.models.stable_audio import StableAudioPipeline

        with pytest.raises(FileNotFoundError, match="Weights directory not found"):
            StableAudioPipeline.from_pretrained("/nonexistent/path")

    def test_missing_vae_weights(self):
        from mlx_audiogen.models.stable_audio import StableAudioPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"sample_rate": 44100}
            with open(os.path.join(tmpdir, "config.json"), "w") as f:
                json.dump(config, f)
            for name in [
                "dit.safetensors",
                "t5.safetensors",
                "conditioners.safetensors",
            ]:
                open(os.path.join(tmpdir, name), "w").close()
            with pytest.raises(FileNotFoundError, match="vae.safetensors"):
                StableAudioPipeline.from_pretrained(tmpdir)

    def test_missing_dit_weights(self):
        from mlx_audiogen.models.stable_audio import StableAudioPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"sample_rate": 44100}
            with open(os.path.join(tmpdir, "config.json"), "w") as f:
                json.dump(config, f)
            for name in [
                "vae.safetensors",
                "t5.safetensors",
                "conditioners.safetensors",
            ]:
                open(os.path.join(tmpdir, name), "w").close()
            with pytest.raises(FileNotFoundError, match="dit.safetensors"):
                StableAudioPipeline.from_pretrained(tmpdir)

    def test_missing_t5_weights(self):
        from mlx_audiogen.models.stable_audio import StableAudioPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"sample_rate": 44100}
            with open(os.path.join(tmpdir, "config.json"), "w") as f:
                json.dump(config, f)
            for name in [
                "vae.safetensors",
                "dit.safetensors",
                "conditioners.safetensors",
            ]:
                open(os.path.join(tmpdir, name), "w").close()
            with pytest.raises(FileNotFoundError, match="t5.safetensors"):
                StableAudioPipeline.from_pretrained(tmpdir)

    def test_missing_conditioners_weights(self):
        from mlx_audiogen.models.stable_audio import StableAudioPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"sample_rate": 44100}
            with open(os.path.join(tmpdir, "config.json"), "w") as f:
                json.dump(config, f)
            for name in ["vae.safetensors", "dit.safetensors", "t5.safetensors"]:
                open(os.path.join(tmpdir, name), "w").close()
            with pytest.raises(FileNotFoundError, match="conditioners.safetensors"):
                StableAudioPipeline.from_pretrained(tmpdir)


# ---------------------------------------------------------------------------
# Audio trimming edge cases
# ---------------------------------------------------------------------------


class TestTrimEdgeCases:
    def test_trim_very_short_duration(self):
        """0.01 seconds at 32kHz = 320 samples."""
        import numpy as np

        from mlx_audiogen.server.app import _trim_to_exact_duration

        audio = np.zeros(32000, dtype=np.float32)
        trimmed = _trim_to_exact_duration(audio, 0.01, 32000, 1)
        assert len(trimmed) == 320

    def test_trim_fractional_bpm(self):
        """Odd BPM values produce non-integer sample counts — should still work."""
        import numpy as np

        from mlx_audiogen.server.app import _trim_to_exact_duration

        # 4 bars at 137 BPM = 4 * 4 * (60/137) ≈ 7.00729927... seconds
        target = 4 * 4 * (60 / 137)
        audio = np.zeros(int(8 * 44100), dtype=np.float32)
        trimmed = _trim_to_exact_duration(audio, target, 44100, 1)
        expected = int(round(target * 44100))
        assert len(trimmed) == expected
