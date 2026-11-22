"""City-wide case rates (used by the 4-week model): correct values, no future data, and the
what-if / new-weather path takes them from ALL wards' reported cases."""
import numpy as np
import pandas as pd

from src.features import load_raw, add_lag_features, CITY_COLUMNS


def test_city_rate_is_city_mean_of_reported_cases_up_to_origin():
    raw = load_raw()
    f = add_lag_features(raw)
    row = f[(f["city"] == "Gazipur") & (f["date"] == pd.Timestamp("2019-08-05"))].iloc[0]
    gz = raw[raw["city"] == "Gazipur"].groupby("date")["dengue_cases"].mean()
    t = pd.Timestamp("2019-08-05")
    past8 = gz[(gz.index <= t)].iloc[-8:]
    assert np.isclose(row["city_rate0"], gz[t])
    assert np.isclose(row["city_rate_mean8"], past8.mean())
    assert np.isclose(row["city_trend"], gz[t] - gz[t - pd.Timedelta(weeks=3)])


def test_city_rates_ignore_future_weeks():
    raw = load_raw()
    cut = pd.Timestamp("2020-06-01")
    full = add_lag_features(raw)
    trunc = add_lag_features(raw[raw["date"] <= cut])
    a = full[full["date"] == cut].set_index("ward_id")[CITY_COLUMNS]
    b = trunc[trunc["date"] == cut].set_index("ward_id")[CITY_COLUMNS]
    pd.testing.assert_frame_equal(a.sort_index(), b.sort_index())


def test_what_if_one_ward_uses_citywide_reported_cases():
    from src.predict import what_if, build_features, predict_risk
    feat = build_features()
    ward, d = feat["ward_id"].iloc[0], "2023-08-07"
    one = what_if(d, 4, wards=[ward])            # no overrides = real weather
    ref = predict_risk(feat, ward, d, 4)
    assert np.isclose(one["predicted_cases"].iloc[0], ref["predicted_cases"], atol=0.006)   # predict_risk rounds to 2 dp
