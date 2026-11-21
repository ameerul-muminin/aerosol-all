"""Analysis-function checks: new-weather input, what-if, modes; dynamic (not mocked) and safe on bad input."""
import numpy as np
import pandas as pd
import pytest

from src.features import WEATHER
from src.predict import predict, predict_from_weather, what_if, _raw

TIERS = {"Low", "Medium", "High", "Critical"}


def test_predict_outputs_all_wards_with_valid_tiers():
    out = predict("2025-08-04", 2)
    assert len(out) == 60 and set(out["risk_tier"]) <= TIERS
    assert (out["lower_80"] <= out["predicted_cases"]).all()
    assert (out["predicted_cases"] <= out["upper_80"]).all()


def test_predictions_change_with_date():
    off = predict("2025-01-06", 2)["predicted_cases"].mean()
    peak = predict("2025-08-04", 2)["predicted_cases"].mean()
    assert peak > 5 * off


def test_from_weather_matches_predict_on_existing_week():
    raw = _raw()
    week = raw[raw["date"] == "2025-08-04"][["date", "ward_id", *WEATHER]]
    a = predict("2025-08-04", 3).set_index("ward_id").sort_index()
    b = predict_from_weather(week, 3).set_index("ward_id").sort_index()
    assert np.allclose(a["predicted_cases"], b["predicted_cases"])


def test_new_week_after_dataset_end():
    new = pd.DataFrame({"date": "2026-01-05", "ward_id": ["DSCC_Ward_06", "CCC_Ward_03"],
                        "avg_temp_c": 19.0, "max_temp_c": 25.0, "min_temp_c": 14.0,
                        "rainfall_mm": 3.0, "relative_humidity_pct": 65.0})
    out = predict_from_weather(new, 2)
    assert len(out) == 2 and (out["target_date"] == pd.Timestamp("2026-01-19")).all()


def test_weather_input_changes_output():
    dry = what_if("2025-06-02", 2, rainfall_mm=0.0)["predicted_cases"].sum()
    wet = what_if("2025-06-02", 2, rainfall_mm=250.0)["predicted_cases"].sum()
    assert dry != wet


@pytest.mark.parametrize("change, msg", [
    ({"ward_id": "XYZ_Ward_99"}, "unknown ward"),
    ({"rainfall_mm": -5.0}, "rainfall_mm outside"),
    ({"relative_humidity_pct": 140.0}, "relative_humidity_pct outside"),
    ({"date": "2026-03-02"}, "does not line up"),          # gap after 2025-12-29
])
def test_bad_weather_input_rejected(change, msg):
    row = {"date": "2026-01-05", "ward_id": "DSCC_Ward_06", "avg_temp_c": 19.0, "max_temp_c": 25.0,
           "min_temp_c": 14.0, "rainfall_mm": 3.0, "relative_humidity_pct": 65.0, **change}
    with pytest.raises(ValueError, match=msg):
        predict_from_weather(pd.DataFrame([row]), 2)


def test_bad_horizon_and_date_rejected():
    with pytest.raises(ValueError, match="(?i)horizon"):
        predict("2025-08-04", 6)
    with pytest.raises(ValueError, match="no data"):
        predict("2030-01-07", 2)


def test_case_mode_runs_and_differs_from_weather_mode():
    w = predict("2025-08-04", 2).set_index("ward_id")["predicted_cases"].sort_index()
    c = predict("2025-08-04", 2, mode="weather_cases")
    assert len(c) == 60 and set(c["risk_tier"]) <= TIERS and (c["mode"] == "weather_cases").all()
    assert not np.allclose(w, c.set_index("ward_id")["predicted_cases"].sort_index())


def test_case_mode_requires_reported_cases_for_new_weeks():
    row = {"date": "2026-01-05", "ward_id": "DSCC_Ward_06", "avg_temp_c": 19.0, "max_temp_c": 25.0,
           "min_temp_c": 14.0, "rainfall_mm": 3.0, "relative_humidity_pct": 65.0}
    with pytest.raises(ValueError, match="dengue_cases"):
        predict_from_weather(pd.DataFrame([row]), 2, mode="weather_cases")
    out = predict_from_weather(pd.DataFrame([{**row, "dengue_cases": 1}]), 2, mode="weather_cases")
    assert len(out) == 1
    with pytest.raises(ValueError, match="mode"):
        predict("2025-08-04", 2, mode="magic")
