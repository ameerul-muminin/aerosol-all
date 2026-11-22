"""Contract tests for the team interface used by the Streamlit app."""
import pandas as pd
import pytest

from src.predict import build_features, predict_risk, predict_all

KEYS = {"ward_id", "forecast_week", "predicted_cases", "range_low", "range_high", "risk_tier", "top_drivers"}
TIERS = {"Low", "Medium", "High", "Critical"}


@pytest.fixture(scope="module")
def feat():
    return build_features()


@pytest.mark.parametrize("h", [2, 3, 4])
def test_predict_risk_contract(feat, h):
    r = predict_risk(feat, "CCC_Ward_01", "2025-07-14", h)
    assert KEYS <= set(r)
    assert r["risk_tier"] in TIERS
    assert 0 <= r["range_low"] <= r["predicted_cases"] <= r["range_high"]
    assert r["forecast_week"] == (pd.Timestamp("2025-07-14") + pd.Timedelta(weeks=h)).date()
    assert 1 <= len(r["top_drivers"]) <= 5
    assert all(isinstance(n, str) and isinstance(v, float) for n, v in r["top_drivers"])


def test_predict_all_has_every_ward_sorted(feat):
    t = predict_all(feat, "2025-07-14", 3)
    assert len(t) == feat["ward_id"].nunique()
    assert t["predicted_cases"].is_monotonic_decreasing
    assert set(t["risk_tier"]) <= TIERS


def test_latest_week_forecasts_into_the_future(feat):
    last = feat["date"].max()
    r = predict_risk(feat, "DNCC_Ward_01" if "DNCC_Ward_01" in set(feat.ward_id) else feat.ward_id.iloc[0], last, 2)
    assert r["forecast_week"] > last.date()


@pytest.mark.parametrize("args, msg", [
    (("NOPE_Ward_99", "2025-07-14", 2), "Unknown ward"),
    (("CCC_Ward_01", "2016-01-18", 2), "history"),
    (("CCC_Ward_01", "2025-07-14", 5), "Horizon"),
    (("CCC_Ward_01", "2030-01-06", 2), "between"),
])
def test_friendly_errors(feat, args, msg):
    with pytest.raises(ValueError, match=msg):
        predict_risk(feat, *args)


def test_served_models_are_single_threaded():
    """Forests are trained on all cores (n_jobs=-1) but served with one thread: tiny batches made the
    thread pool dominate (86 s startup on a 16-core Windows laptop). Predictions do not depend on n_jobs."""
    from src.predict import load_bundle
    found = []
    def walk(o, seen=set()):
        if id(o) in seen or isinstance(o, (str, bytes, int, float)) or type(o).__module__.startswith(("numpy", "pandas")):
            return
        seen.add(id(o))
        if hasattr(o, "predict") and getattr(o, "n_jobs", None) is not None:
            found.append(o.n_jobs)
        for c in (o.values() if isinstance(o, dict) else o if isinstance(o, (list, tuple)) else getattr(o, "__dict__", {}).values()):
            walk(c, seen)
    walk(load_bundle())
    assert found and set(found) == {1}
