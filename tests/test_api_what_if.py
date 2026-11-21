"""POST /api/what_if: typed weekly weather replaces the most recent weeks before features are built;
plain-English 400s; baseline equals /api/forecast; caching/compression behaviour kept."""
import gzip
import json

import pytest

from api.main import create_app
from src.predict import _raw

W, A = "DSCC_Ward_04", "2025-07-21"


@pytest.fixture(scope="module")
def client():
    return create_app().test_client()


def post(client, body, gz=False):
    h = {"Accept-Encoding": "gzip"} if gz else {}
    return client.post("/api/what_if", json=body, headers=h)


def test_baseline_equals_forecast(client):
    for h in (2, 3, 4):
        f = client.get(f"/api/forecast?ward_id={W}&as_of={A}&horizon={h}").get_json()
        d = post(client, {"ward_id": W, "as_of": A, "horizon": h, "rainfall_mm": 150}).get_json()
        for k in ("predicted_cases", "range_low", "range_high", "risk_tier", "top_drivers"):
            assert d["baseline"][k] == f[k]


def test_real_values_reproduce_baseline_same_feature_path(client):
    real = _raw()
    wk = real[(real.ward_id == W) & (real.date == A)].iloc[0]
    d = post(client, {"ward_id": W, "as_of": A, "horizon": 3, "weeks": 1, "rainfall_mm": float(wk.rainfall_mm),
                      "avg_temp_c": float(wk.avg_temp_c), "relative_humidity_pct": float(wk.relative_humidity_pct)}).get_json()
    assert abs(d["scenario"]["predicted_cases"] - d["baseline"]["predicted_cases"]) < 0.02
    assert d["scenario"]["risk_tier"] == d["baseline"]["risk_tier"]


def test_more_rain_and_more_weeks_raise_risk(client):
    dry = post(client, {"ward_id": W, "as_of": A, "horizon": 3, "rainfall_mm": 0}).get_json()
    wet = post(client, {"ward_id": W, "as_of": A, "horizon": 3, "rainfall_mm": 250}).get_json()
    wet1 = post(client, {"ward_id": W, "as_of": A, "horizon": 3, "rainfall_mm": 250, "weeks": 1}).get_json()
    assert wet["scenario"]["predicted_cases"] > dry["scenario"]["predicted_cases"]
    assert wet["scenario"]["predicted_cases"] >= wet1["scenario"]["predicted_cases"]
    assert wet["applied"]["overrides"] == {"rainfall_mm": 250.0} and wet["applied"]["weeks"] == 4
    assert wet["applied"]["weeks_replaced"] == {"from": "2025-06-30", "to": A}
    assert wet["units"]["rainfall_mm"] == "mm per week"
    assert wet["change"]["cases"] == round(wet["scenario"]["predicted_cases"] - wet["baseline"]["predicted_cases"], 2)
    assert {"label", "effect"} <= set(wet["scenario"]["top_drivers"][0])


@pytest.mark.parametrize("body, msg", [
    ({"rainfall_mm": 100}, "ward_id is required"),
    ({"ward_id": "NOPE", "rainfall_mm": 100}, "Unknown ward"),
    ({"ward_id": W}, "at least one weather value"),
    ({"ward_id": W, "rainfall_mm": "200"}, "must be a number"),
    ({"ward_id": W, "rainfall_mm": 500}, "between 0 and 400"),
    ({"ward_id": W, "avg_temp_c": 5}, "between 10 and 45"),
    ({"ward_id": W, "relative_humidity_pct": 120}, "between 0 and 100"),
    ({"ward_id": W, "rainfall_mm": 100, "weeks": 9}, "weeks must be a whole number from 1 to 8"),
    ({"ward_id": W, "rainfall_mm": 100, "weeks": 2.5}, "weeks must be a whole number"),
    ({"ward_id": W, "rainfall_mm": 100, "horizon": 6}, "Horizon"),
    ({"ward_id": W, "rainfall_mm": 100, "as_of": "2016-01-18"}, "history"),
])
def test_plain_english_400s(client, body, msg):
    r = post(client, body)
    assert r.status_code == 400 and msg in r.get_json()["error"]


def test_bad_json_body(client):
    r = client.post("/api/what_if", data="not json", content_type="text/plain")
    assert r.status_code == 400 and "JSON body" in r.get_json()["error"]


def test_missing_ward_id_on_existing_routes(client):
    for url in ("/api/forecast?horizon=3", "/api/history?horizon=3"):
        r = client.get(url)
        assert r.status_code == 400 and r.get_json()["error"] == "ward_id is required."


def test_compression_and_cache_headers(client):
    r = post(client, {"ward_id": W, "as_of": A, "horizon": 3, "rainfall_mm": 200}, gz=True)
    assert r.headers.get("Content-Encoding") == "gzip" and "Cache-Control" not in r.headers
    assert json.loads(gzip.decompress(r.data))["scenario"]["risk_tier"] in {"Low", "Medium", "High", "Critical"}
    g = client.get(f"/api/forecast?ward_id={W}&as_of={A}&horizon=3")
    assert g.headers["Cache-Control"].startswith("public")


# ---------------------------------------------------------------- Step 2 additions
import numpy as np                                                    # noqa: E402
import pandas as pd                                                   # noqa: E402
from src.features import add_lag_features, validate                  # noqa: E402
from src.features import WEATHER                                      # noqa: E402


def test_valid_request_returns_200_with_all_sections(client):
    r = post(client, {"ward_id": W, "as_of": A, "horizon": 2, "relative_humidity_pct": 95})
    assert r.status_code == 200
    d = r.get_json()
    assert {"baseline", "scenario", "change", "applied", "units"} <= set(d)
    for part in ("baseline", "scenario"):
        assert {"predicted_cases", "range_low", "range_high", "risk_tier", "top_drivers"} <= set(d[part])
    assert {"cases", "tier_changed", "drivers"} <= set(d["change"]) and d["applied"]["weeks"] == 4


def test_rainfall_sweep_is_not_constant(client):
    preds = [post(client, {"ward_id": W, "as_of": A, "horizon": 3, "rainfall_mm": mm}).get_json()
             ["scenario"]["predicted_cases"] for mm in (0, 50, 150, 250)]
    assert len(set(preds)) == 4, f"override not reaching the lag features: {preds}"
    assert preds[-1] > preds[0]


def _features_after_override(overrides, weeks=4):
    """Rebuild features exactly like the override path: replace the last `weeks` weeks of this ward's
    raw weather, then run the normal validate + add_lag_features pipeline."""
    raw = _raw()
    ward = raw[raw["ward_id"] == W].copy()
    idx = ward[ward["date"] <= pd.Timestamp(A)].sort_values("date").tail(weeks).index
    for k, v in overrides.items():
        ward.loc[idx, k] = v
    f = add_lag_features(validate(ward))
    return f[f["date"] == pd.Timestamp(A)].iloc[0]


def test_temperature_only_and_humidity_only_change_their_lag_features(client):
    base = _features_after_override({})
    hot = _features_after_override({"avg_temp_c": 40.0})
    humid = _features_after_override({"relative_humidity_pct": 99.0})
    temp_cols = [f"temp_lag{k}" for k in range(4)] + ["temp_mean_0_3"]
    hum_cols = [f"hum_lag{k}" for k in range(4)] + ["hum_mean_0_3"]
    assert all(hot[c] != base[c] for c in temp_cols) and all(hot[c] == base[c] for c in hum_cols)
    assert all(humid[c] != base[c] for c in hum_cols) and all(humid[c] == base[c] for c in temp_cols)
    assert hot["temp_lag4"] == base["temp_lag4"], "weeks beyond the replaced 4 must keep their real values"
    # and the API output moves when only one of them changes
    t = post(client, {"ward_id": W, "as_of": A, "horizon": 2, "avg_temp_c": 40}).get_json()
    h = post(client, {"ward_id": W, "as_of": A, "horizon": 2, "relative_humidity_pct": 99}).get_json()
    assert t["scenario"]["predicted_cases"] != t["baseline"]["predicted_cases"]
    assert h["scenario"]["predicted_cases"] != h["baseline"]["predicted_cases"]


@pytest.mark.parametrize("h", [2, 3, 4])
def test_own_actual_values_reproduce_baseline_every_horizon(client, h):
    wk = _raw()
    wk = wk[(wk.ward_id == W) & (wk.date == A)].iloc[0]
    d = post(client, {"ward_id": W, "as_of": A, "horizon": h, "weeks": 1, "rainfall_mm": float(wk.rainfall_mm),
                      "avg_temp_c": float(wk.avg_temp_c),
                      "relative_humidity_pct": float(wk.relative_humidity_pct)}).get_json()
    assert abs(d["scenario"]["predicted_cases"] - d["baseline"]["predicted_cases"]) < 0.02
    assert d["scenario"]["risk_tier"] == d["baseline"]["risk_tier"] and d["change"]["tier_changed"] is False


# ---------------------------------------------------------------- hardening + richer response
@pytest.mark.parametrize("extra, msg", [
    ({"rain": 200, "rainfall_mm": 10}, "Unknown field(s): rain"),
    ({"rainfall_mm": 100, "horizon": 3.7}, "Horizon"),
    ({"rainfall_mm": 100, "as_of": "21/07/2025"}, "YYYY-MM-DD"),
    ({"rainfall_mm": 100, "as_of": "2025-02-30"}, "real calendar date"),
    ({"rainfall_mm": float("nan")}, "must be a number"),
    ({"rainfall_mm": float("inf")}, "must be a number"),
])
def test_hardened_validation(client, extra, msg):
    r = post(client, {"ward_id": W, **extra})
    assert r.status_code == 400 and msg in r.get_json()["error"]


def test_horizon_as_digit_string_and_normalised_date(client):
    d = post(client, {"ward_id": W, "as_of": " 2025-07-21 ", "horizon": "2", "rainfall_mm": 80}).get_json()
    assert d["horizon"] == 2 and d["as_of"] == A


def test_wrong_method_and_unknown_api_paths_are_json(client):
    r = client.get("/api/what_if")
    assert r.status_code == 405 and "POST" in r.get_json()["error"]
    r = client.get("/api/whatif")
    assert r.status_code == 404 and "Unknown API endpoint" in r.get_json()["error"]
    r = client.get(f"/api/forecast?ward_id={W}&as_of=junk&horizon=3")
    assert r.status_code == 400 and "YYYY-MM-DD" in r.get_json()["error"]


def test_response_shows_replaced_real_weather_and_driver_changes(client):
    d = post(client, {"ward_id": W, "as_of": A, "horizon": 3, "rainfall_mm": 200, "weeks": 3}).get_json()
    real = _raw()
    weeks = real[(real.ward_id == W) & (real.date <= A)].sort_values("date").tail(3)
    rep = d["applied"]["replaced_actual"]["rainfall_mm"]
    assert rep["per_week"] == [round(float(x), 1) for x in weeks["rainfall_mm"]]
    assert rep["mean"] == round(float(weeks["rainfall_mm"].mean()), 1)
    top = d["change"]["drivers"][0]
    assert top["label"].startswith("Rainfall") and top["effect_change"] > 0
    assert abs(sum(x["effect_change"] for x in d["change"]["drivers"]) - d["change"]["cases"]) < 0.05
