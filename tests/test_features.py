"""Checks the lag pipeline: per-ward shifting, no cross-ward bleed, no future data in features."""
import numpy as np
import pandas as pd
from src.features import load_raw, add_lag_features, add_target, feature_columns, build_dataset


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
    cols = feature_columns()
    common = trunc.index
    pd.testing.assert_frame_equal(full.loc[common, cols], trunc[cols])


def test_target_is_future_cases_same_ward():
    d = add_target(add_lag_features(load_raw()), 3)
    g = d[d["ward_id"] == d["ward_id"].iloc[0]].sort_values("date")
    assert np.allclose(g["target"].iloc[:-3].to_numpy(), g["dengue_cases"].iloc[3:].to_numpy())
    assert g["target"].iloc[-3:].isna().all()
    b = build_dataset(3)
    assert (b["target_date"] - b["date"] == pd.Timedelta(weeks=3)).all()
