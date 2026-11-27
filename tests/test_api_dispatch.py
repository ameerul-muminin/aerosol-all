"""GET /api/dispatch and /api/dispatch/brief: ward notices for the dashboard's Dispatch view.
The API must produce exactly the committed exports/ samples, so the screen and the files agree."""
import json
import re
from pathlib import Path

import pytest

from api import service
from api.main import create_app
from src.dispatch import sms_segments

EXPORTS = Path(__file__).resolve().parents[1] / "exports"
SAMPLE = "as_of=2025-09-15&horizon=2&stock_pct=60&crews=20&beds_available_pct=70"   # = exports/README.md command
BENGALI = re.compile(r"[ঀ-৿]")


@pytest.fixture(scope="module")
def client():
    return create_app().test_client()


def get(client, path, query=SAMPLE, **extra):
    q = query + "".join(f"&{k}={v}" for k, v in extra.items())
    return client.get(f"{path}?{q}")


def test_dispatch_happy_path(client):
    r = get(client, "/api/dispatch")
    assert r.status_code == 200
    d = r.get_json()
    assert (d["as_of"], d["horizon"], d["forecast_week"], d["lang"]) == ("2025-09-15", 2, "2025-09-29", "en")
    assert (d["stock_pct"], d["crews"], d["beds_available_pct"]) == (60.0, 20, 70.0)
    notices = d["notices"]
    assert len(notices) == 60 and [n["priority_rank"] for n in notices] == list(range(1, 61))
    for n in notices:
        for item in ("saline", "kits", "beds"):
            assert n[f"{item}_allocated"] + n[f"{item}_deficit"] == n[f"{item}_required"]
        assert n["critical"] == (n["allocation_critical"] or n["unroutable"] > 0)
        assert n["sms_chars"] == len(n["sms"]) and n["sms_segments"] == sms_segments(n["sms"])
        assert n["ward_id"] in n["sms"] and n["ward_id"] in n["whatsapp"]
    s = d["summary"]
    assert s["wards"] == 60 and s["critical"] == sum(n["critical"] for n in notices)
    assert s["unroutable_patients"] == sum(n["unroutable"] for n in notices)
    assert s["max_sms_segments"] == max(n["sms_segments"] for n in notices)
    assert {"stock", "priority", "crews", "bed_capacity", "overflow"} <= set(d["assumptions"])


def test_api_reproduces_the_committed_exports_exactly(client):
    en = get(client, "/api/dispatch", lang="en").get_json()["notices"]
    bn = get(client, "/api/dispatch", lang="bn").get_json()["notices"]
    joined = lambda key: "\n\n\n".join("\n\n".join(n[key] for n in group) for group in (en, bn)) + "\n"
    assert joined("sms") == (EXPORTS / "sms_2025-09-15.txt").read_text(encoding="utf-8")
    assert joined("whatsapp") == (EXPORTS / "whatsapp_2025-09-15.txt").read_text(encoding="utf-8")
    for lang in ("en", "bn"):
        brief = get(client, "/api/dispatch/brief", lang=lang)
        assert brief.status_code == 200 and brief.mimetype == "text/html"
        assert brief.get_data(as_text=True) == (EXPORTS / f"brief_2025-09-15_{lang}.html").read_text(encoding="utf-8")


def test_bangla_notices_are_in_bangla(client):
    d = get(client, "/api/dispatch", lang="bn").get_json()
    assert d["lang"] == "bn"
    assert all(BENGALI.search(n["sms"]) and BENGALI.search(n["whatsapp"]) for n in d["notices"])


def test_city_filters_notices_and_brief_only(client):
    full = get(client, "/api/dispatch").get_json()
    one = get(client, "/api/dispatch", city="Gazipur").get_json()
    assert one["city"] == "Gazipur" and one["notices"]
    assert one["notices"] == [n for n in full["notices"] if n["city"] == "Gazipur"]
    assert one["summary"]["wards"] == len(one["notices"])
    brief = get(client, "/api/dispatch/brief", city="Gazipur").get_data(as_text=True)
    assert all(n["ward_id"] in brief for n in one["notices"])
    assert not any(n["ward_id"] in brief for n in full["notices"] if n["city"] != "Gazipur")


def test_defaults_come_from_the_config(client):
    d = client.get("/api/dispatch?as_of=2025-09-15&stock_pct=60").get_json()
    cfg = json.loads((Path(__file__).resolve().parents[1] / "config" / "operations.json").read_text())
    assert d["horizon"] == 3 and d["lang"] == "en"
    assert d["crews"] == cfg["crews"]["total_default"]["value"]
    assert d["beds_available_pct"] == cfg["bed_capacity"]["beds_available_pct"]["value"]


@pytest.mark.parametrize("path", ["/api/dispatch", "/api/dispatch/brief"])
@pytest.mark.parametrize("query", [
    "as_of=2025-09-15",                                    # stock_pct missing
    "as_of=2025-09-15&stock_pct=20",                       # below 30%
    "as_of=2025-09-15&stock_pct=abc",
    "as_of=2025-09-15&stock_pct=60&lang=fr",
    "as_of=2025-09-15&stock_pct=60&city=Atlantis",
    "as_of=2025-09-15&stock_pct=60&horizon=5",
    "as_of=2025-09-15&stock_pct=60&crews=-1",
    "as_of=2025-09-15&stock_pct=60&beds_available_pct=10",
    "as_of=15/09/2025&stock_pct=60",                       # bad date format
    "as_of=2025-09-16&stock_pct=60",                       # not a Monday
])
def test_bad_input_is_friendly_400(client, path, query):
    r = client.get(f"{path}?{query}")
    assert r.status_code == 400 and r.is_json
    d = r.get_json()
    assert set(d) == {"error"} and d["error"] and "Traceback" not in d["error"]


def test_same_request_twice_gives_identical_answers(client):
    first = json.dumps(get(client, "/api/dispatch", lang="bn").get_json())
    brief = get(client, "/api/dispatch/brief").get_data()
    service._notices.cache_clear()                     # recompute from scratch, not just replay the cache
    assert json.dumps(get(client, "/api/dispatch", lang="bn").get_json()) == first
    assert get(client, "/api/dispatch/brief").get_data() == brief
