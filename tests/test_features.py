"""Checks the lag pipeline: per-ward shifting, no cross-ward bleed, no future data in features."""
import numpy as np
import pandas as pd
import pytest
from src.features import (load_raw, add_lag_features, add_target, feature_columns, build_dataset,
                          validate, anomaly_columns, case_columns)


def test_lags_are_per_ward_and_exact():
    d = add_lag_features(load_raw())
    for ward, g in list(d.groupby("ward_id"))[:5]:
        g = g.sort_values("date")
        assert np.allclose(g["rain_lag3"].iloc[3:].to_numpy(), g["rainfall_mm"].iloc[:-3].to_numpy())
        assert g["rain_lag3"].iloc[:3].isna().all(), "lag must not borrow the previous ward's rows"
        exp = g["rainfall_mm"].shift(2) + g["rainfall_mm"].shift(3) + g["rainfall_mm"].shift(4)
        assert np.allclose(g["rain_sum_2_4"], exp, equal_nan=True)


def test_features_do_not_depend_on_future():
    raw = load_raw()
    cutoff = pd.Timestamp("2020-06-01")
    full = add_lag_features(raw).set_index(["ward_id", "date"])
    trunc = add_lag_features(raw[raw["date"] <= cutoff]).set_index(["ward_id", "date"])
    cols = feature_columns(anomaly=True, cases=True)
    common = trunc.index
    pd.testing.assert_frame_equal(full.loc[common, cols], trunc[cols])


@pytest.mark.parametrize("break_it, msg", [
    (lambda d: d.drop(columns="rainfall_mm"), "missing columns"),
    (lambda d: d.assign(rainfall_mm=d["rainfall_mm"].where(d.index != 5)), "missing values"),
    (lambda d: pd.concat([d, d.iloc[[0]]]), "duplicate"),
    (lambda d: d.drop(index=10), "non-weekly"),
    (lambda d: d.assign(sanitation_score=d["sanitation_score"].where(d.index != 3, 0.01)), "not constant"),
])
def test_validate_rejects_bad_data(break_it, msg):
    with pytest.raises(ValueError, match=msg):
        validate(break_it(load_raw()))


def test_target_is_future_cases_same_ward():
    d = add_target(add_lag_features(load_raw()), 3)
    g = d[d["ward_id"] == d["ward_id"].iloc[0]].sort_values("date")
    assert np.allclose(g["target"].iloc[:-3].to_numpy(), g["dengue_cases"].iloc[3:].to_numpy())
    assert g["target"].iloc[-3:].isna().all()
    b = build_dataset(3)
    assert (b["target_date"] - b["date"] == pd.Timedelta(weeks=3)).all()


def test_rain_anomaly_uses_previous_years_only():
    d = add_lag_features(load_raw())
    g = d[d["ward_id"] == "DSCC_Ward_06"].reset_index(drop=True)
    woy = g["date"].dt.isocalendar().week.astype(int)
    i = g.index[(g["date"].dt.year == 2020) & (woy == 30)][0]
    past = g[(woy == 30) & (g["date"].dt.year < 2020)]["rainfall_mm"]
    assert np.isclose(g.loc[i, "rain_anom_lag0"], g.loc[i, "rainfall_mm"] - past.mean())
    first_year = g[g["date"].dt.year == 2016]
    assert first_year["rain_anom_lag0"].isna().all(), "no history in year 1 -> NaN, not a guess"


def test_case_lags_are_past_cases_and_optional():
    d = add_lag_features(load_raw())
    g = d[d["ward_id"] == "CCC_Ward_01"].sort_values("date")
    assert np.allclose(g["cases_lag2"].iloc[2:], g["dengue_cases"].iloc[:-2])
    assert not set(case_columns()) & set(feature_columns()), "default mode is weather-only"
    assert set(anomaly_columns()) <= set(feature_columns(anomaly=True))
