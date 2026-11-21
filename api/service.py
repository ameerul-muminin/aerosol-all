"""Service layer: turns model outputs into JSON-ready dicts for the web API.

Routes (api/main.py) stay thin; everything here is plain Python and unit-testable.
All numbers come from the trained model (src/predict.py) or reports/ files.
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd

from src.features import ROOT, HORIZONS
from src.predict import build_features, load_bundle, predict_all, MIN_HISTORY_WEEKS

METRICS_PATH = ROOT / "reports" / "metrics.json"
TIERS = ["Low", "Medium", "High", "Critical"]
DEFAULT_AS_OF = "2025-07-14"


# ---------------------------------------------------------------- cached state
# Every response is a pure function of its query, so results are memoised
# (bounded LRU). Errors (ValueError) are not cached.
@lru_cache(maxsize=1)
def features() -> pd.DataFrame:
    return build_features()


def warm_up() -> None:
    """Load model + features once at startup so the first request is fast."""
    load_bundle()
    features()


def _num(x, nd=2):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


def _date(d) -> str:
    return pd.Timestamp(d).date().isoformat()


# ---------------------------------------------------------------- /api/meta
@lru_cache(maxsize=1)
def meta() -> dict:
    f = features()
    first, last = f["date"].min(), f["date"].max()
    earliest = first + pd.Timedelta(weeks=MIN_HISTORY_WEEKS)
    wards = (f[["city", "ward_id"]].drop_duplicates().sort_values(["city", "ward_id"]))
    m = json.loads(METRICS_PATH.read_text())
    tier_cuts = {h: m["horizons"][h]["tier_definition_actual_cases"] for h in m["horizons"]}
    default = pd.Timestamp(DEFAULT_AS_OF)
    return {
        "cities": [{"city": c, "wards": g["ward_id"].tolist()} for c, g in wards.groupby("city")],
        "ward_count": int(f["ward_id"].nunique()),
        "as_of": {"min": _date(earliest), "max": _date(last),
                  "default": _date(default if earliest <= default <= last else last),
                  "step_days": 7},
        "horizons": list(HORIZONS),
        "tiers": TIERS,
        "tier_definition_actual_cases": tier_cuts,
    }


# ---------------------------------------------------------------- /api/ranking
@lru_cache(maxsize=256)
def _ranking_df(as_of: str, horizon: int) -> pd.DataFrame:
    return predict_all(features(), as_of, horizon)


def _row(r) -> dict:
    return {
        "ward_id": r["ward_id"], "city": r["city"],
        "forecast_week": str(r["forecast_week"]),
        "predicted_cases": _num(r["predicted_cases"]),
        "range_low": _num(r["range_low"]), "range_high": _num(r["range_high"]),
        "risk_tier": r["risk_tier"],
        "top_drivers": [{"label": k, "effect": _num(v, 3)} for k, v in r["top_drivers"]],
    }


@lru_cache(maxsize=512)
def ranking(as_of: str, horizon: int, city: str | None = None) -> dict:
    df = _ranking_df(as_of, horizon)
    if city:
        if city not in set(df["city"]):
            raise ValueError(f"Unknown city: {city}")
        df = df[df["city"] == city]
    # expected cases at every horizon for the heat strip
    other = {h: _ranking_df(as_of, h).set_index("ward_id")["predicted_cases"] for h in HORIZONS}
    rows = []
    for _, r in df.iterrows():
        d = _row(r)
        d["by_horizon"] = {str(h): _num(other[h].get(r["ward_id"])) for h in HORIZONS}
        rows.append(d)
    counts = df["risk_tier"].value_counts().reindex(TIERS, fill_value=0)
    by_city = (df.groupby(["city", "risk_tier"]).size().unstack(fill_value=0)
                 .reindex(columns=TIERS, fill_value=0))
    return {
        "as_of": as_of, "horizon": horizon,
        "forecast_week": rows[0]["forecast_week"] if rows else None,
        "tier_counts": {k: int(v) for k, v in counts.items()},
        "by_city": [{"city": c, **{k: int(v) for k, v in row.items()}} for c, row in by_city.iterrows()],
        "wards": rows,
    }


# ---------------------------------------------------------------- /api/forecast
@lru_cache(maxsize=4096)
def forecast(ward_id: str, as_of: str, horizon: int) -> dict:
    f = features()
    if ward_id not in set(f["ward_id"]):
        raise ValueError(f"Unknown ward: {ward_id}")
    df = _ranking_df(as_of, horizon)
    hit = df[df["ward_id"] == ward_id]
    if hit.empty:
        raise ValueError("Not enough history for this ward on that date.")
    out = _row(hit.iloc[0])

    a = pd.Timestamp(as_of)
    fc = a + pd.Timedelta(weeks=horizon)
    w = f[f["ward_id"] == ward_id]
    hist = w[(w["date"] <= a)].tail(MIN_HISTORY_WEEKS + 1)

    # rain timeline: known weeks up to as_of, unknown weeks until the forecast week
    timeline = []
    for _, r in hist.iterrows():
        timeline.append({"week": _date(r["date"]), "rain_mm": _num(r["rainfall_mm"], 1),
                         "cases": int(r["dengue_cases"]), "status": "known"})
    for k in range(1, horizon + 1):
        d = a + pd.Timedelta(weeks=k)
        timeline.append({"week": _date(d), "rain_mm": None, "cases": None,
                         "status": "forecast" if k == horizon else "unknown"})
    for t in timeline:
        wk = pd.Timestamp(t["week"])
        t["in_breeding_window"] = bool(fc - pd.Timedelta(weeks=4) <= wk <= fc - pd.Timedelta(weeks=2))

    last4 = hist.tail(4)
    now = hist.iloc[-1]
    prof = w.iloc[0]
    out.update({
        "as_of": _date(a), "horizon": horizon,
        "rain_timeline": timeline,
        "recent": {
            "rain_last_4_weeks_mm": _num(last4["rainfall_mm"].sum(), 1),
            "humidity_pct": _num(now["relative_humidity_pct"], 1),
            "avg_temp_c": _num(now["avg_temp_c"], 1),
            "cases_this_week": int(now["dengue_cases"]),
        },
        "ward_profile": {
            "water_logging_risk_index": _num(prof["water_logging_risk_index"]),
            "sanitation_score": _num(prof["sanitation_score"]),
            "population_density_sqkm": int(prof["population_density_sqkm"]),
        },
        "by_horizon": {str(h): {"predicted_cases": _num(r["predicted_cases"]), "risk_tier": r["risk_tier"]}
                       for h in HORIZONS
                       for r in [_ranking_df(as_of, h).set_index("ward_id").loc[ward_id]]},
    })
    return out


# ---------------------------------------------------------------- /api/metrics
@lru_cache(maxsize=1)
def metrics() -> dict:
    m = json.loads(METRICS_PATH.read_text())
    per_h = []
    for h in HORIZONS:
        t = m["horizons"][str(h)]["test"]
        tp = pd.read_csv(ROOT / "reports" / f"test_predictions_h{h}.csv")
        err = (tp["pred"] - tp["actual"])
        city = (tp.assign(ae=err.abs(), se=err ** 2).groupby("city")
                  .agg(mae=("ae", "mean"), rmse=("se", lambda s: float(np.sqrt(s.mean())))))
        per_h.append({
            "horizon": h,
            "selected_model": m["horizons"][str(h)]["selected"],
            "mae": _num(t["MAE"], 3), "rmse": _num(t["RMSE"], 3),
            "baseline_mae": _num(t["baseline_seasonal_naive"]["MAE"], 3),
            "interval80_coverage": _num(t["interval80_coverage"], 3),
            "critical_flagged_high_or_above": _num(t.get("critical_recall"), 3),
            "by_city": [{"city": c, "mae": _num(r["mae"], 3), "rmse": _num(r["rmse"], 3)} for c, r in city.iterrows()],
        })
    return {"split": m["split"], "horizons": per_h}


def ward_history(ward_id: str, horizon: int) -> dict:
    """Predicted vs actual with 80% band over the test period, for one ward."""
    tp = pd.read_csv(ROOT / "reports" / f"test_predictions_h{horizon}.csv")
    one = tp[tp["ward_id"] == ward_id]
    if one.empty:
        raise ValueError(f"Unknown ward: {ward_id}")
    return {"ward_id": ward_id, "horizon": horizon,
            "points": [{"week": r.target_date, "actual": _num(r.actual), "predicted": _num(r.pred),
                        "low": _num(r.lo), "high": _num(r.hi)} for r in one.itertuples()]}
