"""Service-layer tests for the web API (no HTTP client needed)."""
import json

import pytest

from api import service

TIERS = {"Low", "Medium", "High", "Critical"}


def test_meta_shape():
    m = service.meta()
    assert m["ward_count"] == sum(len(c["wards"]) for c in m["cities"])
    assert m["horizons"] == [2, 3, 4]
    assert m["as_of"]["min"] <= m["as_of"]["default"] <= m["as_of"]["max"]


def test_forecast_contract_and_json():
    f = service.forecast("CCC_Ward_01", "2025-07-14", 3)
    json.dumps(f)  # must be JSON-serialisable
    assert f["risk_tier"] in TIERS
    assert f["range_low"] <= f["predicted_cases"] <= f["range_high"]
    tl = f["rain_timeline"]
    assert tl[-1]["status"] == "forecast" and tl[-1]["week"] == f["forecast_week"]
    assert sum(t["status"] == "unknown" for t in tl) == 2  # 3-week horizon: 2 unknown weeks before it
    assert all(t["rain_mm"] is None for t in tl if t["status"] != "known")
    assert set(f["by_horizon"]) == {"2", "3", "4"}


def test_ranking_counts_match_rows():
    r = service.ranking("2025-07-14", 2)
    json.dumps(r)
    assert sum(r["tier_counts"].values()) == len(r["wards"])
    cases = [w["predicted_cases"] for w in r["wards"]]
    assert cases == sorted(cases, reverse=True)
    g = service.ranking("2025-07-14", 2, city="Gazipur")
    assert {w["city"] for w in g["wards"]} == {"Gazipur"}


@pytest.mark.parametrize("call", [
    lambda: service.forecast("NOPE", "2025-07-14", 3),
    lambda: service.forecast("CCC_Ward_01", "2016-01-11", 2),
    lambda: service.ranking("2025-07-14", 7),
    lambda: service.ranking("2025-07-14", 3, city="Atlantis"),
])
def test_friendly_errors(call):
    with pytest.raises(ValueError):
        call()


def test_metrics_read_from_reports():
    m = service.metrics()
    raw = json.loads(service.METRICS_PATH.read_text())
    for h in m["horizons"]:
        assert h["mae"] == round(raw["horizons"][str(h["horizon"])]["test"]["MAE"], 3)
        base = raw["horizons"][str(h["horizon"])]["test"]["baseline_seasonal_naive"]
        assert h["baseline_rmse"] == round(base["RMSE"], 3)   # accuracy view compares RMSE too


def test_metrics_report_the_deployed_model_type():
    from src.predict import load_bundle
    for h in service.metrics()["horizons"]:
        assert h["model_type"] == load_bundle()["modes"]["weather"][h["horizon"]]["base_kind"]
