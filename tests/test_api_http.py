"""HTTP-level tests with Flask's built-in test client (no server, no network)."""
import pytest

from api.main import create_app


@pytest.fixture(scope="module")
def client():
    return create_app().test_client()


def test_health_and_meta(client):
    assert client.get("/api/health").json == {"status": "ok"}
    m = client.get("/api/meta").json
    assert m["horizons"] == [2, 3, 4] and m["ward_count"] == 60


def test_forecast_ok(client):
    r = client.get("/api/forecast?ward_id=CCC_Ward_01&as_of=2025-07-14&horizon=3")
    assert r.status_code == 200
    assert r.json["risk_tier"] in {"Low", "Medium", "High", "Critical"}
    assert "max-age" in r.headers["Cache-Control"]


@pytest.mark.parametrize("url", [
    "/api/forecast?ward_id=NOPE&horizon=3",
    "/api/forecast?ward_id=CCC_Ward_01&as_of=2016-01-11&horizon=2",
    "/api/ranking?horizon=9",
    "/api/ranking?horizon=abc",
])
def test_bad_input_is_friendly_400(client, url):
    r = client.get(url)
    assert r.status_code == 400 and "error" in r.json


def test_ranking_is_gzipped_when_asked(client):
    r = client.get("/api/ranking?as_of=2025-07-14&horizon=3", headers={"Accept-Encoding": "gzip"})
    assert r.status_code == 200 and r.headers.get("Content-Encoding") == "gzip"


def test_static_pages_served(client):
    assert client.get("/").status_code == 200
    assert client.get("/app/").status_code == 200
    assert client.get("/nope.html").status_code == 404


def test_new_endpoints_over_http(client):
    assert client.get("/api/lag_profile?city=Gazipur").json["peak_weeks"] in (2, 3, 4)
    assert client.get("/api/warnings?as_of=2025-08-18").status_code == 200
    assert client.get("/api/warnings?as_of=2025-08-18&city=Atlantis").status_code == 400
    r = client.get("/api/efficiency")
    assert r.status_code == 200 and r.headers["Cache-Control"] == "no-store"   # live counters
    assert r.json["measured"]["external_api_calls"] == 0
