"""Phase 2 routes over HTTP (Flask test client, no server, no network):
POST /api/allocate, GET /api/overflow, POST /api/intervention, GET /api/intervention/roi."""
import json

import pytest

from api import service
from api.main import create_app
from src.allocation import load_config

A, H = "2025-07-14", 3
DEPLOY = {"DSCC_Ward_04": 2, "CCC_Ward_01": 1}
OVERFLOW_BUSY = "/api/overflow?as_of=2024-08-05&horizon=2&beds_available_pct=80"   # has transfers


@pytest.fixture(scope="module")
def client():
    return create_app().test_client()


def _plain_json(obj):
    """Only plain JSON types, and every float has at most 2 decimals."""
    if isinstance(obj, dict):
        return all(isinstance(k, str) and _plain_json(v) for k, v in obj.items())
    if isinstance(obj, list):
        return all(_plain_json(v) for v in obj)
    if isinstance(obj, float):
        return round(obj, 2) == obj
    return obj is None or isinstance(obj, (str, int, bool))


def _assumptions_from_config(assumptions: dict) -> bool:
    cfg = load_config()
    return bool(assumptions) and all(entry == cfg[section][key]
                                     for section, keys in assumptions.items() for key, entry in keys.items())


# ---------------------------------------------------------------- happy paths
def test_allocate_happy_path(client):
    r = client.post("/api/allocate", json={"as_of": A, "horizon": H, "stock_pct": 60, "crews": 10})
    assert r.status_code == 200
    d = r.get_json()
    assert d["as_of"] == A and d["horizon"] == H and d["crews"] == 10 and len(d["wards"]) == 60
    t = d["totals"]
    for item in t["items"].values():
        assert item["allocated"] <= item["available"] and item["allocated"] + item["deficit"] == item["required"]
    assert t["crews"]["assigned"] <= 10
    assert [w["priority_rank"] for w in d["wards"]] == list(range(1, 61))
    assert _assumptions_from_config(d["assumptions"]) and "stock_pct_range" in d["assumptions"]["stock"]
    assert _plain_json(d)
    assert "max-age" not in r.headers.get("Cache-Control", "")      # POST scenarios are not cached


def test_allocate_city_filters_rows_only(client):
    full = client.post("/api/allocate", json={"as_of": A, "horizon": H, "stock_pct": 60}).get_json()
    one = client.post("/api/allocate", json={"as_of": A, "horizon": H, "stock_pct": 60, "city": "Gazipur"}).get_json()
    assert one["wards"] and {w["city"] for w in one["wards"]} == {"Gazipur"}
    assert one["wards"] == [w for w in full["wards"] if w["city"] == "Gazipur"]
    assert one["totals"] == full["totals"]


def test_overflow_happy_path(client):
    r = client.get("/api/overflow?as_of=2025-08-04&horizon=2&demand_basis=range_high")
    assert r.status_code == 200
    d = r.get_json()
    assert d["demand_basis"] == "range_high" and d["forecast_week"] == "2025-08-18"
    assert len(d["wards"]) == 60 and {c["city"] for c in d["cities"]} == {w["city"] for w in d["wards"]}
    assert _assumptions_from_config(d["assumptions"]) and "same_city_only" in d["assumptions"]["overflow"]
    assert _plain_json(d)


def test_intervention_happy_path(client):
    r = client.post("/api/intervention", json={"as_of": A, "deployment": DEPLOY, "total_crews": 5})
    assert r.status_code == 200
    d = r.get_json()
    assert d["crews_used"] == 3 and d["total_crews"] == 5
    assert [h["horizon"] for h in d["horizons"]] == [2, 3, 4]
    assert {w["ward_id"] for w in d["wards"]} == set(DEPLOY)
    assert all(h["cases_after"] <= h["cases_before"] for h in d["horizons"])
    assert _assumptions_from_config(d["assumptions"]) and "reduction_per_crew" in d["assumptions"]["crew_effect"]
    assert _plain_json(d)


def test_intervention_roi_happy_path(client):
    r = client.get(f"/api/intervention/roi?as_of={A}&crews_per_ward=2")
    assert r.status_code == 200
    d = r.get_json()
    assert d["crews_per_ward"] == 2 and len(d["wards"]) == 60
    assert [w["rank"] for w in d["wards"]] == list(range(1, 61))
    rois = [w["roi"] for w in d["wards"]]
    assert rois == sorted(rois, reverse=True)
    assert _assumptions_from_config(d["assumptions"])
    assert _plain_json(d)


# ---------------------------------------------------------------- friendly 400s
@pytest.mark.parametrize("method, url, body", [
    ("post", "/api/allocate", {"as_of": A, "horizon": H, "stock_pct": 20}),           # stock below 30%
    ("post", "/api/allocate", {"as_of": A, "horizon": H, "stock_pct": 101}),
    ("post", "/api/allocate", {"as_of": A, "horizon": H}),                            # stock missing
    ("post", "/api/allocate", {"as_of": "14/07/2025", "horizon": H, "stock_pct": 60}),  # bad date
    ("post", "/api/allocate", {"as_of": "2025-07-15", "horizon": H, "stock_pct": 60}),  # not a Monday
    ("post", "/api/allocate", {"as_of": A, "horizon": 5, "stock_pct": 60}),           # horizon 5
    ("post", "/api/allocate", {"as_of": A, "stock_pct": 60, "city": "Atlantis"}),
    ("post", "/api/allocate", {"as_of": A, "stock_pct": 60, "crews": -1}),
    ("post", "/api/allocate", {"as_of": A, "stock_pct": 60, "demand_basis": "nope"}),
    ("post", "/api/allocate", {"as_of": A, "stock_pct": 60, "stok_pct": 60}),         # typo field
    ("post", "/api/allocate", None),                                                  # no JSON body
    ("get", "/api/overflow?as_of=2025-02-30&horizon=2", None),                       # bad date
    ("get", "/api/overflow?as_of=2025-08-04&horizon=5", None),                       # horizon 5
    ("get", "/api/overflow?as_of=2025-08-04&demand_basis=nope", None),
    ("get", "/api/overflow?as_of=2025-08-04&beds_available_pct=10", None),
    ("post", "/api/intervention", {"as_of": A, "deployment": {"NOPE_Ward_99": 1}}),   # unknown ward
    ("post", "/api/intervention", {"as_of": A, "deployment": DEPLOY, "total_crews": 2}),  # crews over total
    ("post", "/api/intervention", {"as_of": A, "deployment": {"DSCC_Ward_04": 9}}),   # over per-ward cap
    ("post", "/api/intervention", {"as_of": A, "deployment": {"DSCC_Ward_04": 1.5}}),
    ("post", "/api/intervention", {"as_of": A, "deployment": {}}),
    ("post", "/api/intervention", {"as_of": "July 14", "deployment": DEPLOY}),        # bad date
    ("get", "/api/intervention/roi?as_of=2025-7-14", None),                          # bad date
    ("get", "/api/intervention/roi?as_of=2025-07-14&crews_per_ward=9", None),
    ("get", "/api/intervention/roi?as_of=2025-07-14&crews_per_ward=two", None),
    # found by the offline API fuzz: these used to crash with HTTP 500
    ("post", "/api/allocate", {"as_of": A, "stock_pct": ""}),
    ("post", "/api/allocate", {"as_of": A, "stock_pct": 60, "city": []}),
    ("post", "/api/allocate", {"as_of": A, "stock_pct": 60, "city": {"a": 1}}),
    ("post", "/api/allocate", {"as_of": A, "stock_pct": 60, "demand_basis": ["range_high"]}),
    ("post", "/api/allocate", {"as_of": A, "stock_pct": 60, "demand_basis": {}}),
])
def test_bad_input_is_friendly_400(client, method, url, body):
    r = client.post(url, json=body) if method == "post" else client.get(url)
    assert r.status_code == 400
    d = r.get_json()
    assert set(d) == {"error"} and d["error"] and "Traceback" not in d["error"]


def test_an_unexpected_bug_still_answers_in_json():
    app = create_app()

    @app.get("/api/boom")
    def boom():
        raise RuntimeError("internal detail that must not leak")

    r = app.test_client().get("/api/boom")
    assert r.status_code == 500 and r.is_json
    assert "internal detail" not in r.get_json()["error"]


def test_specific_messages(client):
    def err(url, body):
        return client.post(url, json=body).get_json()["error"]
    assert "30" in err("/api/allocate", {"as_of": A, "stock_pct": 20})
    assert "NOPE_Ward_99" in err("/api/intervention", {"as_of": A, "deployment": {"NOPE_Ward_99": 1}})
    assert "only 2 are available" in err("/api/intervention", {"as_of": A, "deployment": DEPLOY, "total_crews": 2})
    assert "(2, 3, 4)" in client.get("/api/overflow?as_of=2025-08-04&horizon=5").get_json()["error"]


def test_post_routes_answer_get_with_a_hint(client):
    for url in ("/api/allocate", "/api/intervention"):
        r = client.get(url)
        assert r.status_code == 405 and "POST" in r.get_json()["error"]


def test_intervention_total_crews_defaults_to_config(client):
    total = load_config()["crews"]["total_default"]["value"]
    d = client.post("/api/intervention", json={"as_of": A, "deployment": DEPLOY}).get_json()
    assert d["total_crews"] == total


# ---------------------------------------------------------------- determinism
@pytest.mark.parametrize("method, url, body", [
    ("post", "/api/allocate", {"as_of": A, "horizon": H, "stock_pct": 45, "crews": 7}),
    ("get", OVERFLOW_BUSY, None),
    ("post", "/api/intervention", {"as_of": A, "deployment": DEPLOY}),
    ("get", f"/api/intervention/roi?as_of={A}", None),
])
def test_same_request_twice_gives_identical_json(client, method, url, body):
    def call():
        r = client.post(url, json=body) if method == "post" else client.get(url)
        assert r.status_code == 200
        return json.dumps(r.get_json())
    first = call()
    for fn in (service._allocation, service.overflow, service._simulation, service.intervention_roi):
        fn.cache_clear()                        # recompute from scratch, not just replay the cache
    assert call() == first


# ---------------------------------------------------------------- overflow rules
def test_overflow_transfers_never_cross_cities(client):
    d = client.get(OVERFLOW_BUSY).get_json()
    city_of = {w["ward_id"]: w["city"] for w in d["wards"]}
    assert d["transfers"], "this week should need routing; pick another if the model changes"
    for t in d["transfers"]:
        assert city_of[t["from_ward"]] == city_of[t["to_ward"]] == t["city"]
        assert t["from_ward"] != t["to_ward"] and t["patients"] > 0
    # patients are conserved: everything routed out of a ward lands in a ward of the same city
    for c in d["cities"]:
        moved = sum(t["patients"] for t in d["transfers"] if t["city"] == c["city"])
        wards = [w for w in d["wards"] if w["city"] == c["city"]]
        assert moved == sum(w["routed_out"] for w in wards) == sum(w["routed_in"] for w in wards)



# ---------------------------------------------------------------- simulator sensitivity band
def test_intervention_reports_the_sensitivity_band(client):
    d = client.post("/api/intervention", json={"as_of": A, "deployment": DEPLOY, "total_crews": 5}).get_json()
    for h in d["horizons"]:
        for key in ("cases_prevented", "beds_freed", "saline_saved"):
            assert f"{key}_low" in h and f"{key}_high" in h
        assert h["cases_prevented_low"] <= h["cases_prevented"] + 0.05 <= h["cases_prevented_high"] + 0.1
    assert all("roi_low" in w and "roi_high" in w for w in d["wards"])
    assert d["assumptions"]["crew_effect"]["sensitivity_per_crew"]["value"] == [0.05, 0.15, 0.25]


def test_intervention_roi_reports_the_sensitivity_band(client):
    d = client.get(f"/api/intervention/roi?as_of={A}").get_json()
    assert all("roi_low" in w and "roi_high" in w for w in d["wards"])
    assert "sensitivity_per_crew" in d["assumptions"]["crew_effect"]


# ---------------------------------------------------------------- GET /api/crew_plan (bonus B2)
def test_crew_plan_happy_path(client):
    r = client.get(f"/api/crew_plan?as_of={A}&crews=10")
    assert r.status_code == 200
    d = r.get_json()
    assert _plain_json(d)
    assert d["as_of"] == A and d["crews_available"] == 10
    assert d["crews_used"] == sum(d["plan"].values()) <= 10
    assert d["cases_prevented"] >= d["compare"]["allocation_rule"]["cases_prevented"]
    assert {w["ward_id"] for w in d["wards"]} == set(d["plan"])
    assert _assumptions_from_config(d["assumptions"])


def test_crew_plan_matches_the_module_and_defaults_to_config(client):
    from src.placement import best_placement
    d = client.get(f"/api/crew_plan?as_of={A}").get_json()
    total = load_config()["crews"]["total_default"]["value"]
    assert d["crews_available"] == total
    direct = best_placement(service.features(), A, total)
    assert d["plan"] == direct["plan"] and d["cases_prevented"] == direct["cases_prevented"]


@pytest.mark.parametrize("query, msg", [("crews=-2", "whole number"), ("crews=two", "whole number"),
                                        ("as_of=2025-7-14", "YYYY-MM-DD"), ("as_of=2025-07-15", "Mondays")])
def test_crew_plan_bad_input_is_a_400(client, query, msg):
    r = client.get(f"/api/crew_plan?{query}")
    assert r.status_code == 400 and msg in r.get_json()["error"]


# ---------------------------------------------------------------- simulator input checks (before the cache)
def test_intervention_checks_crews_before_the_cache(client):
    ok = client.post("/api/intervention", json={"as_of": A, "deployment": {"DSCC_Ward_04": 2}})
    assert ok.status_code == 200
    # 2.0 == 2 in a cache key: it must still be refused, not served the cached answer for 2
    r = client.post("/api/intervention", json={"as_of": A, "deployment": {"DSCC_Ward_04": 2.0}})
    assert r.status_code == 400 and "whole number" in r.get_json()["error"]


def test_intervention_ignores_wards_with_zero_crews(client):
    a = client.post("/api/intervention", json={"as_of": A, "deployment": {"DSCC_Ward_04": 2}}).get_json()
    b = client.post("/api/intervention", json={"as_of": A, "deployment": {"DSCC_Ward_04": 2, "CCC_Ward_01": 0}}).get_json()
    assert a == b and [w["ward_id"] for w in b["wards"]] == ["DSCC_Ward_04"]


@pytest.mark.parametrize("url, words", [
    (f"/api/intervention/roi?as_of={A}&crews_per_ward=-1", "from 1 to"),
    (f"/api/crew_plan?as_of={A}&crews=-1", "0 or more"),
])
def test_negative_numbers_get_the_range_message(client, url, words):
    r = client.get(url)
    assert r.status_code == 400 and words in r.get_json()["error"]


def test_crew_plan_says_when_crews_are_capped(client):
    d = client.get(f"/api/crew_plan?as_of={A}&crews=100000").get_json()
    assert d["crews_requested"] == 100000 and d["crews_available"] == d["crews_placeable"] < 100000
    assert d["note"] and str(d["crews_placeable"]) in d["note"]
