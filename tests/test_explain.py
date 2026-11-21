"""Driver contributions are exact (Shapley efficiency) and use the training-period reference."""
import numpy as np
from src.predict import predict, what_if
from src.explain import GROUP_ORDER


def _drivers(df):
    return df[[c for c in df.columns if c.startswith("driver_")]]


def test_contributions_add_up_to_prediction():
    out = predict("2025-08-04", 2, explain=True)
    assert np.allclose(out["base_cases"] + _drivers(out).sum(axis=1), out["predicted_cases"], atol=1e-6)
    assert set(out["top_driver"]) <= set(GROUP_ORDER) | {"None (below average)"}


def test_case_mode_adds_recent_cases_group():
    out = predict("2025-08-04", 3, mode="weather_cases", explain=True)
    assert "driver_recent_cases" in out.columns
    assert np.allclose(out["base_cases"] + _drivers(out).sum(axis=1), out["predicted_cases"], atol=1e-6)


def test_more_rain_raises_rainfall_driver():
    dry = what_if("2025-07-14", 2, wards=["DSCC_Ward_06"], rainfall_mm=0.0, explain=True)
    wet = what_if("2025-07-14", 2, wards=["DSCC_Ward_06"], rainfall_mm=200.0, explain=True)
    assert wet["driver_rainfall"].iloc[0] != dry["driver_rainfall"].iloc[0]
