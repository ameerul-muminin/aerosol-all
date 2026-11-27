"""GET /api/version answers the judges' "ML model: deployed where?" on the live site itself: the model file
and its fingerprint, the library version it needs, the model behind each horizon, the host and git commit."""
import hashlib
from pathlib import Path

import pytest

from api import service
from api.main import create_app

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def client():
    return create_app().test_client()


def test_version_names_the_model_file_and_its_fingerprint(client):
    r = client.get("/api/version")
    assert r.status_code == 200 and r.headers["Cache-Control"] == "no-store"   # a new deploy shows at once
    m = r.get_json()["model"]
    data = (ROOT / m["file"]).read_bytes()
    assert m["file"] == "models/dengue_model.joblib" and m["bytes"] == len(data)
    assert m["sha256"] == hashlib.sha256(data).hexdigest()


def test_version_names_the_library_and_model_per_horizon(client):
    d = client.get("/api/version").get_json()
    pinned = next(l for l in (ROOT / "requirements.txt").read_text().splitlines() if l.startswith("scikit-learn=="))
    assert d["model"]["scikit_learn"] == pinned.split("==")[1]          # the version the model file needs
    assert set(d["model"]["by_horizon"]) == {"2", "3", "4"}
    assert all(v["model_type"] for v in d["model"]["by_horizon"].values())
    assert "no separate model server" in d["runs_in"]


def test_version_reports_the_render_build(client, monkeypatch):
    monkeypatch.setenv("RENDER", "true")
    monkeypatch.setenv("RENDER_GIT_COMMIT", "abc123def456")
    monkeypatch.setenv("RENDER_GIT_BRANCH", "main")
    service.version.cache_clear()
    try:
        d = client.get("/api/version").get_json()
        assert (d["host"], d["commit"], d["branch"]) == ("render", "abc123def456", "main")
    finally:
        monkeypatch.undo()
        service.version.cache_clear()
    d = client.get("/api/version").get_json()
    assert (d["host"], d["commit"]) == ("local", None)
