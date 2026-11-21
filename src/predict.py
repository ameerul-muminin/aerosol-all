"""Inference: load serialized models, compute lag features from raw weather, return risk.

Team interface (used by the Streamlit app; keys/columns are a contract, see CLAUDE.md):
  * build_features()                                    -> featurized DataFrame (cache it)
  * predict_risk(feat_df, ward_id, as_of_date, horizon) -> dict   (one ward, with top_drivers)
  * predict_all(feat_df, as_of_date, horizon)           -> DataFrame (ranking table, with drivers)

Analysis / scenario functions:
  * predict(origin_date, horizon, wards=None)           -> DataFrame for weeks in the dataset
  * predict_from_weather(weather, horizon)              -> NEW weekly weather rows joined onto history
  * what_if(origin_date, horizon, rainfall_mm=...)      -> override one week's weather and re-predict
  explain=True on these three adds exact grouped-Shapley driver columns (src/explain.py).

Every function takes mode="weather" (default: weather + ward attributes + season) or
mode="weather_cases" (optional: also uses reported cases up to the origin week).
"""
from __future__ import annotations
from functools import lru_cache
import joblib
import numpy as np
import pandas as pd

from src.features import (STATIC, WEATHER, HORIZONS, add_lag_features, feature_columns,
                          required_columns, load_raw, validate)
from src.train import BUNDLE, TIER_NAMES, TRAIN_END, to_tier, conformal_interval

# Plain-language driver groups for the team interface. Each group's contribution is measured by
# "what-if" occlusion: replace the group's features with their training-period average and see
# how much the prediction moves. Positive = raises predicted cases, negative = lowers them.
# Columns a horizon's model does not use are skipped (e.g. rain anomaly is used only where CV kept it).
DRIVER_GROUPS = {
    "Rain this week and last week": ["rain_lag0", "rain_lag1", "rain_sum_0_1",
                                     "rain_anom_lag0", "rain_anom_lag1"],
    "Rain 2-4 weeks ago (mosquito breeding window)": ["rain_lag2", "rain_lag3", "rain_lag4", "rain_sum_2_4",
                                                      "rain_anom_lag2", "rain_anom_lag3", "rain_anom_lag4",
                                                      "rain_anom_sum_2_4"],
    "Rain 5-8 weeks ago": ["rain_lag5", "rain_lag6", "rain_lag7", "rain_lag8"],
    "Total rain over the last 1-2 months": ["rain_sum_0_3", "rain_sum_0_8", "rain_anom_sum_0_3"],
    "Temperature (last 4 weeks)": [f"temp_lag{k}" for k in range(5)] + ["temp_mean_0_3"],
    "Humidity (last 4 weeks)": [f"hum_lag{k}" for k in range(5)] + ["hum_mean_0_3"],
    "Time of year (season)": ["week_sin", "week_cos"],
    "Ward water-logging risk": ["water_logging_risk_index"],
    "Ward population density": ["population_density_sqkm"],
    "Ward sanitation score": ["sanitation_score"],
    "Reported cases (last 4 weeks)": [f"cases_lag{k}" for k in range(5)] + ["cases_sum_0_3"],
}
MIN_HISTORY_WEEKS = 8  # rain_lag8 needs 8 earlier weeks

# Plausible physical ranges for user-supplied weather (Bangladesh, weekly aggregates)
RANGES = {"avg_temp_c": (0, 50), "max_temp_c": (0, 55), "min_temp_c": (-5, 45),
          "rainfall_mm": (0, 1500), "relative_humidity_pct": (0, 100)}


@lru_cache(maxsize=1)
def load_bundle():
    """Load the model once. If it can't be unpickled (e.g. a different scikit-learn version),
    retrain from the raw data instead of crashing on a fresh machine."""
    try:
        return joblib.load(BUNDLE)
    except Exception as exc:  # version mismatch or missing file
        print(f"[predict] could not load {BUNDLE.name} ({type(exc).__name__}); retraining...")
        from src.train import main as retrain
        retrain()
        return joblib.load(BUNDLE)


def build_features() -> pd.DataFrame:
    """Featurized dataset the UI passes to predict_risk / predict_all (cache it in the app)."""
    return add_lag_features(load_raw())


@lru_cache(maxsize=1)
def _raw():
    return load_raw()


@lru_cache(maxsize=1)
def _featurized():
    return build_features()


@lru_cache(maxsize=1)
def _reference_values() -> pd.Series:
    """Training-period average of each feature: the 'typical conditions' baseline for drivers."""
    d = _featurized()
    return d.loc[d["date"] <= TRAIN_END, feature_columns(anomaly=True, cases=True)].mean()


def _check_horizon(horizon, mode="weather"):
    b = load_bundle()
    modes = b.get("modes") or {"weather": b["horizons"]}
    if mode not in modes:
        raise ValueError(f"mode must be one of {sorted(modes)}, got {mode!r}")
    hs = modes[mode]
    if horizon not in hs:
        raise ValueError(f"Horizon must be one of {tuple(sorted(hs))} weeks, got {horizon!r}.")
    return hs[horizon]


def _features_of(m) -> list[str]:
    return m.get("features") or load_bundle()["features"]   # per-horizon list (older bundles: global)


def _model_predict(m, d: pd.DataFrame) -> np.ndarray:
    """Model wrappers select their own feature columns; plain estimators get them explicitly."""
    model = m["model"]
    X = d if hasattr(model, "feats") else d[_features_of(m)]
    return np.clip(model.predict(X), 0, None)


# ---------------------------------------------------------------- analysis functions

def _score(rows: pd.DataFrame, horizon: int, mode: str = "weather", explain: bool = False) -> pd.DataFrame:
    """Run the horizon's model on featurized rows (one row per ward at its origin week).
    explain=True adds per-prediction driver contributions (see src/explain.py)."""
    m = _check_horizon(horizon, mode)
    rows = rows.dropna(subset=required_columns(cases=mode == "weather_cases"))
    rows = rows.assign(target_date=rows["date"] + pd.Timedelta(weeks=horizon))
    p = _model_predict(m, rows)
    lo, hi = conformal_interval(p, m["conformal_q"], months=rows["target_date"].dt.month.to_numpy())
    out = rows[["ward_id", "city", "date", "target_date"]].assign(
        horizon_weeks=horizon, mode=mode, predicted_cases=p, lower_80=lo, upper_80=hi,
        risk_tier=[TIER_NAMES[i] for i in to_tier(p, m["tier_cuts"])])
    if explain:
        from src.explain import contributions
        drv = contributions(m["model"], rows, cases=mode == "weather_cases")
        out = pd.concat([out.reset_index(drop=True), drv], axis=1)
    return out.sort_values("predicted_cases", ascending=False).reset_index(drop=True)


def predict(origin_date, horizon: int, wards=None, mode: str = "weather",
            explain: bool = False) -> pd.DataFrame:
    """Forecast cases `horizon` weeks after `origin_date` using dataset weather up to that date."""
    _check_horizon(horizon, mode)
    d = _featurized()
    d = d[d["date"] == pd.Timestamp(origin_date)]
    if d.empty:
        lo, hi = _raw()["date"].min().date(), _raw()["date"].max().date()
        raise ValueError(f"no data for {origin_date}; dataset weeks are Mondays {lo} to {hi}. "
                         "Use predict_from_weather() for new weeks.")
    if wards is not None:
        d = d[d["ward_id"].isin(wards)]
    return _score(d, horizon, mode, explain)


def _validate_weather(weather: pd.DataFrame, cases: bool = False) -> pd.DataFrame:
    num = [*WEATHER, "dengue_cases"] if cases else list(WEATHER)
    need = {"date", "ward_id", *num}
    missing = need - set(weather.columns)
    if missing:
        hint = " (mode='weather_cases' needs reported dengue_cases for each new week)" if cases else ""
        raise ValueError(f"weather input missing columns: {sorted(missing)}{hint}")
    w = weather.copy()
    w["date"] = pd.to_datetime(w["date"])
    for c in num:
        w[c] = pd.to_numeric(w[c], errors="coerce")
    if w[list(need)].isna().any().any():
        raise ValueError("weather input has missing or non-numeric values")
    for c, (lo, hi) in RANGES.items():
        bad = ~w[c].between(lo, hi)
        if bad.any():
            raise ValueError(f"{c} outside plausible range [{lo}, {hi}]: {w.loc[bad, c].tolist()[:5]}")
    if cases and (w["dengue_cases"] < 0).any():
        raise ValueError("dengue_cases must be >= 0")
    if (w["min_temp_c"] > w["max_temp_c"]).any():
        raise ValueError("min_temp_c greater than max_temp_c")
    known = set(_raw()["ward_id"])
    unknown = set(w["ward_id"]) - known
    if unknown:
        raise ValueError(f"unknown ward_id(s): {sorted(unknown)[:5]} (static ward attributes needed)")
    if w.duplicated(["ward_id", "date"]).any():
        raise ValueError("duplicate ward_id/date rows in weather input")
    return w


def predict_from_weather(weather: pd.DataFrame, horizon: int, mode: str = "weather",
                         explain: bool = False) -> pd.DataFrame:
    """Predict from NEW weekly weather rows (columns: date, ward_id + the 5 weather columns).

    Rows can continue the dataset (e.g. 2026-01-05 onward) or replace existing weeks.
    Earlier weeks needed for the lags are taken from each ward's history; static ward
    attributes come from the ward table. Each ward is forecast from its LATEST supplied week.
    In mode="weather_cases" each new row must also carry the week's reported dengue_cases.
    """
    _check_horizon(horizon, mode)
    cases = mode == "weather_cases"
    w = _validate_weather(weather, cases)
    hist = _raw()
    wards = w["ward_id"].unique()
    static = hist.drop_duplicates("ward_id").set_index("ward_id")[["city", *STATIC]]
    w = w.join(static, on="ward_id")
    # history for those wards, minus any weeks the caller is overriding
    h = hist[hist["ward_id"].isin(wards)]
    h = h.merge(w[["ward_id", "date"]], on=["ward_id", "date"], how="left", indicator=True)
    h = h[h["_merge"] == "left_only"].drop(columns="_merge")
    # weather mode: new weeks' cases are unknown and unused by the model (0 is a placeholder
    # only so validate() passes; case features are not inputs in this mode)
    new = w if cases else w.assign(dengue_cases=0)
    combined = pd.concat([h, new], ignore_index=True)
    try:
        combined = validate(combined)
    except ValueError as e:   # e.g. a gap between history and the new weeks
        raise ValueError(f"weather input does not line up with ward history: {e}") from None
    feats = add_lag_features(combined)
    origin = w.groupby("ward_id")["date"].max()
    rows = feats[feats["date"] == feats["ward_id"].map(origin)]
    return _score(rows, horizon, mode, explain)


def what_if(origin_date, horizon: int, wards=None, mode: str = "weather", explain: bool = False,
            **overrides) -> pd.DataFrame:
    """Scenario: take the dataset weather at `origin_date`, override some columns
    (e.g. rainfall_mm=150), and predict. Unspecified columns keep their real values."""
    bad = set(overrides) - set(WEATHER)
    if bad:
        raise ValueError(f"can only override {WEATHER}, got {sorted(bad)}")
    raw = _raw()
    base = raw[raw["date"] == pd.Timestamp(origin_date)]
    if base.empty:
        raise ValueError(f"no data for {origin_date}")
    if wards is not None:
        base = base[base["ward_id"].isin(wards)]
    base = base[["date", "ward_id", *WEATHER, "dengue_cases"]].assign(**overrides)
    return predict_from_weather(base, horizon, mode, explain)


# ---------------------------------------------------------------- team interface

def _check_inputs(feat_df: pd.DataFrame, as_of_date, horizon: int) -> pd.Timestamp:
    if horizon not in HORIZONS:
        raise ValueError(f"Horizon must be one of {HORIZONS} weeks.")
    as_of = pd.Timestamp(as_of_date)
    first, last = feat_df["date"].min(), feat_df["date"].max()
    if as_of < first or as_of > last:
        raise ValueError(f"Pick a date between {first.date()} and {last.date()}.")
    if as_of < first + pd.Timedelta(weeks=MIN_HISTORY_WEEKS):
        raise ValueError(f"Need at least {MIN_HISTORY_WEEKS} weeks of history before this date.")
    if not (feat_df["date"] == as_of).any():
        raise ValueError("No data for that week. Dates are Mondays (weekly data).")
    return as_of


def _drivers(m, X: pd.DataFrame, base_pred: np.ndarray, top_k: int = 5) -> list[list[tuple[str, float]]]:
    """Occlusion attribution per row, vectorised across rows (one model call per group)."""
    ref = _reference_values()
    used = set(X.columns)
    contrib = {}
    for label, cols in DRIVER_GROUPS.items():
        cols = [c for c in cols if c in used]
        if not cols:
            continue
        Xo = X.copy()
        Xo[cols] = ref[cols].values
        contrib[label] = base_pred - _model_predict(m, Xo)
    c = pd.DataFrame(contrib, index=X.index)
    out = []
    for _, row in c.iterrows():
        top = row.reindex(row.abs().sort_values(ascending=False).index)[:top_k]
        out.append([(k, round(float(v), 3)) for k, v in top.items()])
    return out


def predict_all(feat_df: pd.DataFrame, as_of_date, horizon: int, with_drivers: bool = True,
                mode: str = "weather") -> pd.DataFrame:
    """All wards for one as-of date and horizon, sorted by predicted cases (one row per ward)."""
    as_of = _check_inputs(feat_df, as_of_date, horizon)
    m = _check_horizon(horizon, mode)
    feats = _features_of(m)
    d = feat_df[feat_df["date"] == as_of].dropna(subset=required_columns(cases=mode == "weather_cases"))
    X = d[feats]
    p = _model_predict(m, X)
    target_month = (as_of + pd.Timedelta(weeks=horizon)).month
    lo, hi = conformal_interval(p, m["conformal_q"], months=np.full(len(p), target_month))
    out = pd.DataFrame({
        "ward_id": d["ward_id"].values,
        "city": d["city"].values,
        "forecast_week": (as_of + pd.Timedelta(weeks=horizon)).date(),
        "predicted_cases": np.round(p, 2),
        "range_low": np.round(lo, 2),
        "range_high": np.round(hi, 2),
        "risk_tier": [TIER_NAMES[i] for i in to_tier(p, m["tier_cuts"])],
    })
    if with_drivers:
        out["top_drivers"] = _drivers(m, X, p)
    return out.sort_values("predicted_cases", ascending=False).reset_index(drop=True)


def predict_risk(feat_df: pd.DataFrame, ward_id: str, as_of_date, horizon: int,
                 mode: str = "weather") -> dict:
    """One ward: ward_id, forecast_week, predicted_cases, range_low, range_high, risk_tier, top_drivers."""
    if ward_id not in set(feat_df["ward_id"]):
        raise ValueError(f"Unknown ward: {ward_id}")
    as_of = _check_inputs(feat_df, as_of_date, horizon)
    one = feat_df[feat_df["ward_id"] == ward_id]
    res = predict_all(one, as_of, horizon, mode=mode)
    if res.empty:
        raise ValueError("Not enough history for this ward on that date.")
    return res.iloc[0].to_dict()


if __name__ == "__main__":
    import sys
    date = sys.argv[1] if len(sys.argv) > 1 else "2025-07-14"
    for h in HORIZONS:
        print(f"\n== origin {date}, +{h} weeks ==")
        print(predict(date, h).head(5).round(2).to_string(index=False))
    print(f"\n== what-if: 150 mm rain in week {date}, +2 weeks (vs actual) ==")
    real = predict(date, 2).set_index("ward_id")["predicted_cases"]
    wi = what_if(date, 2, rainfall_mm=150.0).set_index("ward_id")
    print(wi.assign(actual_weather_pred=real)[["predicted_cases", "actual_weather_pred", "risk_tier"]]
          .head(5).round(2).to_string())
