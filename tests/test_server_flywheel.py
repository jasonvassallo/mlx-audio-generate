"""Server endpoint tests for flywheel intelligence (Phase 9g-4).

Tests the star, flywheel config, versions, and reset endpoints.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mlx_audiogen.server.app import (
    GenerateRequest,
    JobStatus,
    _Job,
    _jobs,
    _rate_limiter,
    _server_settings,
    _weights_dirs,
    app,
)


@pytest.fixture(autouse=True)
def _clean_state():
    """Reset global server state between tests."""
    import mlx_audiogen.server.app as srv

    _jobs.clear()
    _weights_dirs.clear()
    _rate_limiter._generate_hits.clear()
    _rate_limiter._general_hits.clear()
    srv._flywheel_manager = None
    _server_settings.pop("flywheel", None)
    yield
    _jobs.clear()
    _weights_dirs.clear()
    _rate_limiter._generate_hits.clear()
    _rate_limiter._general_hits.clear()
    srv._flywheel_manager = None
    _server_settings.pop("flywheel", None)


@pytest.fixture
def client():
    return TestClient(app)


def _make_done_job(job_id: str = "test-1") -> _Job:
    """Create a completed job with audio data."""
    req = GenerateRequest(
        model="musicgen",
        prompt="test prompt",
        seconds=5.0,
    )
    job = _Job(job_id, req)
    job.status = JobStatus.DONE
    job.audio = np.zeros(16000, dtype=np.float32)
    job.sample_rate = 16000
    _jobs[job_id] = job
    return job


class TestStarEndpoints:
    def test_star_generation(self, client, tmp_path, monkeypatch):
        """POST /api/star/{id} stars a generation."""
        monkeypatch.setattr(
            "mlx_audiogen.lora.flywheel.DEFAULT_KEPT_DIR", tmp_path / "kept"
        )
        _make_done_job("j1")
        resp = client.post("/api/star/j1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["starred"] is True
        assert data["stars_since_train"] >= 1

    def test_star_missing_job(self, client):
        """POST /api/star/{id} returns 404 for missing job."""
        resp = client.post("/api/star/nonexistent")
        assert resp.status_code == 404

    def test_star_expired_audio(self, client):
        """POST /api/star/{id} returns 410 if audio is gone."""
        req = GenerateRequest(model="musicgen", prompt="test", seconds=5.0)
        job = _Job("j2", req)
        job.status = JobStatus.DONE
        job.audio = None  # expired
        _jobs["j2"] = job
        resp = client.post("/api/star/j2")
        assert resp.status_code == 410

    def test_unstar_generation(self, client, tmp_path, monkeypatch):
        """DELETE /api/star/{id} removes star."""
        monkeypatch.setattr(
            "mlx_audiogen.lora.flywheel.DEFAULT_KEPT_DIR", tmp_path / "kept"
        )
        _make_done_job("j1")
        client.post("/api/star/j1")
        resp = client.delete("/api/star/j1")
        assert resp.status_code == 200
        assert resp.json()["starred"] is False

    def test_starred_field_on_status(self, client, tmp_path, monkeypatch):
        """GET /api/status/{id} includes starred field."""
        monkeypatch.setattr(
            "mlx_audiogen.lora.flywheel.DEFAULT_KEPT_DIR", tmp_path / "kept"
        )
        _make_done_job("j1")
        client.post("/api/star/j1")
        resp = client.get("/api/status/j1")
        assert resp.status_code == 200
        assert resp.json()["starred"] is True

    def test_starred_field_on_jobs_list(self, client, tmp_path, monkeypatch):
        """GET /api/jobs includes starred field."""
        monkeypatch.setattr(
            "mlx_audiogen.lora.flywheel.DEFAULT_KEPT_DIR", tmp_path / "kept"
        )
        _make_done_job("j1")
        client.post("/api/star/j1")
        resp = client.get("/api/jobs")
        assert resp.status_code == 200
        jobs = resp.json()
        assert any(j["starred"] is True for j in jobs)


class TestFlywheelConfig:
    def test_get_config(self, client, tmp_path, monkeypatch):
        """GET /api/flywheel/config returns defaults."""
        monkeypatch.setattr(
            "mlx_audiogen.lora.flywheel.DEFAULT_KEPT_DIR", tmp_path / "kept"
        )
        resp = client.get("/api/flywheel/config")
        assert resp.status_code == 200
        data = resp.json()
        assert data["retrain_threshold"] == 10
        assert data["blend_ratio"] == 80
        assert data["auto_retrain"] is True

    def test_update_config(self, client, tmp_path, monkeypatch):
        """PUT /api/flywheel/config updates settings."""
        monkeypatch.setattr(
            "mlx_audiogen.lora.flywheel.DEFAULT_KEPT_DIR", tmp_path / "kept"
        )
        resp = client.put(
            "/api/flywheel/config",
            json={"retrain_threshold": 20, "blend_ratio": 60},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["retrain_threshold"] == 20
        assert data["blend_ratio"] == 60


class TestFlywheelStatus:
    def test_get_status(self, client, tmp_path, monkeypatch):
        """GET /api/flywheel/status returns current state."""
        monkeypatch.setattr(
            "mlx_audiogen.lora.flywheel.DEFAULT_KEPT_DIR", tmp_path / "kept"
        )
        resp = client.get("/api/flywheel/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "stars_since_train" in data
        assert "retrain_threshold" in data


class TestVersionEndpoints:
    def test_list_versions_empty(self, client, tmp_path, monkeypatch):
        """GET /api/loras/{name}/versions returns empty for unknown adapter."""
        monkeypatch.setattr(
            "mlx_audiogen.lora.flywheel.DEFAULT_LORAS_DIR", tmp_path / "loras"
        )
        monkeypatch.setattr(
            "mlx_audiogen.lora.flywheel.DEFAULT_KEPT_DIR", tmp_path / "kept"
        )
        resp = client.get("/api/loras/nonexistent/versions")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_invalid_name_rejected(self, client):
        """Path traversal in name is rejected."""
        # URL-encoded path traversal to bypass FastAPI path normalization
        resp = client.get("/api/loras/%2e%2e%5cetc/versions")
        assert resp.status_code in (400, 422)


class TestResetEndpoint:
    def test_reset_kept(self, client, tmp_path, monkeypatch):
        """POST /api/flywheel/reset/{name} clears kept gens."""
        monkeypatch.setattr(
            "mlx_audiogen.lora.flywheel.DEFAULT_KEPT_DIR", tmp_path / "kept"
        )
        resp = client.post("/api/flywheel/reset/my-style")
        assert resp.status_code == 200
        assert resp.json()["status"] == "reset"
