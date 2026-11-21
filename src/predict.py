"""Inference: load serialized models, compute lag features from raw weather, return risk.

Team interface (used by the Streamlit app; keys/columns are a contract, see CLAUDE.md):
  * build_features()                                    -> featurized DataFrame (cache it)
  * predict_risk(feat_df, ward_id, as_of_date, horizon) -> dict   (one ward, with top_drivers)
  * predict_all(feat_df, as_of_date, horizon)           -> DataFrame (ranking table, with drivers)

Analysis function:
  * predict(origin_date, horizon, wards=None)           -> DataFrame for weeks in the dataset
"""
from __future__ import annotations
from functools import lru_cache
import joblib
import numpy as np
import pandas as pd

from src.features import HORIZONS, add_lag_features, feature_columns, required_columns, load_raw
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
}
MIN_HISTORY_WEEKS = 8  # rain_lag8 needs 8 earlier weeks

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
    return d.loc[d["date"] <= TRAIN_END, feature_columns(anomaly=True)].mean()


def _check_horizon(horizon):
    hs = load_bundle()["horizons"]
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

def _score(rows: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Run the horizon's model on featurized rows (one row per ward at its origin week)."""
    m = _check_horizon(horizon)
    rows = rows.dropna(subset=required_columns())
    rows = rows.assign(target_date=rows["date"] + pd.Timedelta(weeks=horizon))
    p = _model_predict(m, rows)
    lo, hi = conformal_interval(p, m["conformal_q"])
    out = rows[["ward_id", "city", "date", "target_date"]].assign(
        horizon_weeks=horizon, predicted_cases=p, lower_80=lo, upper_80=hi,
        risk_tier=[TIER_NAMES[i] for i in to_tier(p, m["tier_cuts"])])
    return out.sort_values("predicted_cases", ascending=False).reset_index(drop=True)


def predict(origin_date, horizon: int, wards=None) -> pd.DataFrame:
    """Forecast cases `horizon` weeks after `origin_date` using dataset weather up to that date."""
    _check_horizon(horizon)
    d = _featurized()
    d = d[d["date"] == pd.Timestamp(origin_date)]
    if d.empty:
        lo, hi = _raw()["date"].min().date(), _raw()["date"].max().date()
        raise ValueError(f"no data for {origin_date}; dataset weeks are Mondays {lo} to {hi}.")
    if wards is not None:
        d = d[d["ward_id"].isin(wards)]
    return _score(d, horizon)


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


def predict_all(feat_df: pd.DataFrame, as_of_date, horizon: int, with_drivers: bool = True) -> pd.DataFrame:
    """All wards for one as-of date and horizon, sorted by predicted cases (one row per ward)."""
    as_of = _check_inputs(feat_df, as_of_date, horizon)
    m = _check_horizon(horizon)
    feats = _features_of(m)
    d = feat_df[feat_df["date"] == as_of].dropna(subset=required_columns())
    X = d[feats]
    p = _model_predict(m, X)
    lo, hi = conformal_interval(p, m["conformal_q"])
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


def predict_risk(feat_df: pd.DataFrame, ward_id: str, as_of_date, horizon: int) -> dict:
    """One ward: ward_id, forecast_week, predicted_cases, range_low, range_high, risk_tier, top_drivers."""
    if ward_id not in set(feat_df["ward_id"]):
        raise ValueError(f"Unknown ward: {ward_id}")
    as_of = _check_inputs(feat_df, as_of_date, horizon)
    one = feat_df[feat_df["ward_id"] == ward_id]
    res = predict_all(one, as_of, horizon)
    if res.empty:
        raise ValueError("Not enough history for this ward on that date.")
    return res.iloc[0].to_dict()


if __name__ == "__main__":
    import sys
    date = sys.argv[1] if len(sys.argv) > 1 else "2025-07-14"
    for h in HORIZONS:
        print(f"\n== origin {date}, +{h} weeks ==")
        print(predict(date, h).head(5).round(2).to_string(index=False))
