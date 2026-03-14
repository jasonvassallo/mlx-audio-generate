"""End-to-end integration tests: MusicGen generate → Demucs separate → validate stems.

These tests require real converted model weights on disk and run on the GPU.
They are marked with ``@pytest.mark.integration`` and excluded from the default
``pytest`` run.  Execute with::

    uv run pytest -m integration -v

"""

import numpy as np
import pytest

MUSICGEN_WEIGHTS = "./converted/musicgen-small"
DEMUCS_WEIGHTS = "./converted/demucs-htdemucs"

EXPECTED_STEMS = {"drums", "bass", "other", "vocals"}


def _weights_available() -> bool:
    """Check that both MusicGen and Demucs converted weights exist locally."""
    from pathlib import Path

    mg = Path(MUSICGEN_WEIGHTS) / "config.json"
    dm = Path(DEMUCS_WEIGHTS) / "config.json"
    return mg.exists() and dm.exists()


skip_no_weights = pytest.mark.skipif(
    not _weights_available(),
    reason="Converted weights not found — run mlx-audiogen-convert first",
)


# ---------------------------------------------------------------------------
# MusicGen → Demucs end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.integration
@skip_no_weights
class TestMusicGenToDemucs:
    """Generate audio with MusicGen, separate with Demucs, validate stems."""

    @pytest.fixture(scope="class")
    def generated_audio(self):
        """Generate a short clip once for the whole test class."""
        from mlx_audiogen.models.musicgen.pipeline import MusicGenPipeline

        pipeline = MusicGenPipeline.from_pretrained(MUSICGEN_WEIGHTS)
        audio = pipeline.generate(
            prompt="upbeat drums and bass",
            seconds=5.0,
            seed=42,
        )
        return audio  # 1-D float32, 32 kHz

    @pytest.fixture(scope="class")
    def separated_stems(self, generated_audio):
        """Separate the generated audio into stems."""
        from mlx_audiogen.models.demucs.pipeline import DemucsPipeline

        pipeline = DemucsPipeline.from_pretrained(DEMUCS_WEIGHTS)
        stems = pipeline.separate(generated_audio, sample_rate=32000)
        return stems

    # -- generation sanity checks --

    def test_generated_audio_is_valid(self, generated_audio):
        assert isinstance(generated_audio, np.ndarray)
        assert generated_audio.dtype == np.float32
        assert generated_audio.ndim >= 1
        assert len(generated_audio) > 0

    def test_generated_audio_duration(self, generated_audio):
        # MusicGen outputs 32 kHz; 5 seconds ≈ 160 000 samples
        total_samples = (
            generated_audio.shape[-1]
            if generated_audio.ndim > 1
            else len(generated_audio)
        )
        duration_s = total_samples / 32000
        assert 4.5 <= duration_s <= 6.5, (
            f"Duration {duration_s:.1f}s outside expected range"
        )

    def test_generated_audio_not_silent(self, generated_audio):
        rms = np.sqrt(np.mean(generated_audio**2))
        assert rms > 1e-4, f"Generated audio is near-silent (RMS={rms:.6f})"

    # -- stem separation checks --

    def test_stem_names(self, separated_stems):
        assert set(separated_stems.keys()) == EXPECTED_STEMS

    def test_stems_are_stereo(self, separated_stems):
        for name, stem in separated_stems.items():
            assert stem.ndim == 2, f"{name} should be (2, T)"
            assert stem.shape[0] == 2, (
                f"{name} has {stem.shape[0]} channels, expected 2"
            )

    def test_stems_are_float32(self, separated_stems):
        for name, stem in separated_stems.items():
            assert stem.dtype == np.float32, f"{name} dtype is {stem.dtype}"

    def test_stems_same_length(self, separated_stems):
        lengths = {name: stem.shape[-1] for name, stem in separated_stems.items()}
        unique_lengths = set(lengths.values())
        assert len(unique_lengths) == 1, f"Stem lengths differ: {lengths}"

    def test_stems_not_all_silent(self, separated_stems):
        """At least some stems should have meaningful energy."""
        active_count = 0
        for name, stem in separated_stems.items():
            rms = np.sqrt(np.mean(stem**2))
            if rms > 1e-4:
                active_count += 1
        assert active_count >= 2, "Fewer than 2 stems have meaningful energy"

    def test_stems_sum_approximates_mix(self, generated_audio, separated_stems):
        """Separated stems should approximately sum to the original mix.

        Stems are returned at the original sample rate (32 kHz), so we
        compare directly against the original audio (duplicated to stereo).
        """
        # Original mono → stereo for comparison with stereo stems
        mono = generated_audio if generated_audio.ndim == 1 else generated_audio[0]
        stereo = np.stack([mono, mono], axis=0)

        # Sum all stems
        stem_sum = sum(separated_stems.values())

        # Trim to same length (resampling round-trip can differ by a few samples)
        min_len = min(stereo.shape[-1], stem_sum.shape[-1])
        stereo = stereo[:, :min_len]
        stem_sum = stem_sum[:, :min_len]

        # Normalise both for comparison
        orig_rms = np.sqrt(np.mean(stereo**2)) + 1e-8
        sum_rms = np.sqrt(np.mean(stem_sum**2)) + 1e-8

        # RMS ratio should be in a reasonable range
        ratio = sum_rms / orig_rms
        assert 0.3 < ratio < 3.0, (
            f"Stem sum RMS ratio to original is {ratio:.2f} — "
            "stems don't approximately reconstruct the mix"
        )


# ---------------------------------------------------------------------------
# Demucs via stem_separator.separate() (high-level API)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@skip_no_weights
class TestStemSeparatorHighLevel:
    """Test the high-level ``separate()`` API with real Demucs weights."""

    def test_separate_returns_demucs_stems(self):
        """separate() with use_demucs=True should return 4 stems, not 3 FFT bands."""
        from mlx_audiogen.shared.stem_separator import separate

        # 2 seconds of a 440 Hz sine + noise (simulates a simple instrument)
        sr = 44100
        t = np.linspace(0, 2, sr * 2, dtype=np.float32)
        audio = (
            0.5 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t))
        ).astype(np.float32)

        stems = separate(audio, sr, use_demucs=True, demucs_weights_dir=DEMUCS_WEIGHTS)

        # Should be Demucs stems, not FFT bands
        assert set(stems.keys()) == EXPECTED_STEMS, (
            f"Got {set(stems.keys())} — separate() likely fell back to FFT"
        )

    def test_separate_basic_fallback(self):
        """With use_demucs=False, should fall back to basic FFT band-split."""
        from mlx_audiogen.shared.stem_separator import separate

        audio = np.random.randn(44100).astype(np.float32) * 0.5
        stems = separate(audio, 44100, use_demucs=False)
        assert set(stems.keys()) == {"bass", "mid", "high"}


# ---------------------------------------------------------------------------
# Server endpoint integration
# ---------------------------------------------------------------------------


@pytest.mark.integration
@skip_no_weights
class TestServerStemEndpoint:
    """Test POST /api/separate/{id} with real Demucs separation."""

    @pytest.fixture()
    def client(self):
        from starlette.testclient import TestClient

        from mlx_audiogen.server.app import app

        return TestClient(app)

    def test_separate_job_returns_demucs_stems(self, client):
        from mlx_audiogen.server.app import GenerateRequest, JobStatus, _Job, _jobs

        # Inject a fake completed job with real-ish audio
        sr = 44100
        t = np.linspace(0, 1, sr, dtype=np.float32)
        audio = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        job = _Job("e2e-sep-1", GenerateRequest(model="musicgen", prompt="test"))
        job.status = JobStatus.DONE
        job.audio = audio
        job.sample_rate = sr
        job.channels = 1
        _jobs["e2e-sep-1"] = job

        res = client.post("/api/separate/e2e-sep-1")
        assert res.status_code == 200
        data = res.json()
        assert "stems" in data
        # Demucs should produce 4 stems
        assert len(data["stems"]) == 4
        for stem_name in EXPECTED_STEMS:
            stem_key = f"e2e-sep-1_stem_{stem_name}"
            assert data["stems"][stem_name] == stem_key

        # Verify each stem is downloadable
        for stem_name in EXPECTED_STEMS:
            stem_id = data["stems"][stem_name]
            audio_res = client.get(f"/api/audio/{stem_id}")
            assert audio_res.status_code == 200
            assert audio_res.headers["content-type"] == "audio/aiff"
            assert len(audio_res.content) > 44  # AIFF header + some data
