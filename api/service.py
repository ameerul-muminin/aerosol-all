"""Service layer: turns model outputs into JSON-ready dicts for the web API.

Routes (api/main.py) stay thin; everything here is plain Python and unit-testable.
All numbers come from the trained model (src/predict.py) or reports/ files.
"""
from __future__ import annotations

import json
import math
import re
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
    default = meta()["as_of"]["default"]
    for h in HORIZONS:          # pre-compute the default week so the first page view is instant
        _ranking_df(default, h)


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
DRIVER_LABELS = {
    "Rainfall": "Rainfall (lags 0-8 weeks)",
    "Temperature": "Temperature (last 4 weeks)",
    "Humidity": "Humidity (last 4 weeks)",
    "Season": "Time of year",
    "Ward vulnerability": "Ward vulnerability (drainage, crowding, sanitation)",
    "Recent cases": "Recent cases",
}


@lru_cache(maxsize=256)
def _ranking_df(as_of: str, horizon: int) -> pd.DataFrame:
    """All wards for one week/horizon, with exact grouped-Shapley drivers (src/explain.py).
    Contributions add up to (prediction - average-week reference), so the UI can show them honestly."""
    from src.explain import contributions
    from src.features import required_columns
    from src.predict import _check_horizon

    f = features()
    out = predict_all(f, as_of, horizon, with_drivers=False)
    m = _check_horizon(horizon)
    rows = f[f["date"] == pd.Timestamp(as_of)].dropna(subset=required_columns(cases=False)).reset_index(drop=True)
    drv = contributions(m["model"], rows)
    cols = [c for c in drv.columns if c.startswith("driver_")]
    names = {c: next(g for g in DRIVER_LABELS if "driver_" + g.lower().replace(" ", "_") == c) for c in cols}
    per_ward = {}
    for i, w in enumerate(rows["ward_id"]):
        items = sorted(((DRIVER_LABELS[names[c]], float(drv.at[i, c])) for c in cols), key=lambda kv: -abs(kv[1]))
        per_ward[w] = (items, float(drv.at[i, "base_cases"]))
    out["top_drivers"] = out["ward_id"].map(lambda w: per_ward[w][0])
    out["driver_reference_cases"] = out["ward_id"].map(lambda w: per_ward[w][1])
    return out


def _row(r) -> dict:
    return {
        "ward_id": r["ward_id"], "city": r["city"],
        "forecast_week": str(r["forecast_week"]),
        "predicted_cases": _num(r["predicted_cases"]),
        "range_low": _num(r["range_low"]), "range_high": _num(r["range_high"]),
        "risk_tier": r["risk_tier"],
        "top_drivers": [{"label": k, "effect": _num(v, 3)} for k, v in r["top_drivers"]],
        "driver_reference_cases": _num(r.get("driver_reference_cases")),
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
    _require_ward(ward_id)
    f = features()
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
    if not ward_id:
        raise ValueError("ward_id is required.")
    tp = pd.read_csv(ROOT / "reports" / f"test_predictions_h{horizon}.csv")
    one = tp[tp["ward_id"] == ward_id]
    if one.empty:
        raise ValueError(f"Unknown ward: {ward_id}")
    return {"ward_id": ward_id, "horizon": horizon,
            "points": [{"week": r.target_date, "actual": _num(r.actual), "predicted": _num(r.pred),
                        "low": _num(r.lo), "high": _num(r.hi)} for r in one.itertuples()]}


# ---------------------------------------------------------------- /api/what_if
# Typed weekly weather replaces the most recent `weeks` weeks BEFORE lag/rolling features are built
# (src.predict.predict_from_weather -> add_lag_features, the same path as build_features()).
WHAT_IF_RANGES = {"rainfall_mm": (0.0, 400.0), "avg_temp_c": (10.0, 45.0), "relative_humidity_pct": (0.0, 100.0)}
WHAT_IF_UNITS = {"rainfall_mm": "mm per week", "avg_temp_c": "degrees C (weekly mean)",
                 "relative_humidity_pct": "% (weekly mean)", "weeks": "most recent weeks replaced"}
WEEKS_DEFAULT, WEEKS_RANGE = 4, (1, 8)
_LABEL = {"rainfall_mm": "Rainfall", "avg_temp_c": "Average temperature", "relative_humidity_pct": "Humidity"}


WHAT_IF_KEYS = ("ward_id", "as_of", "horizon", "weeks", *WHAT_IF_RANGES)
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def parse_as_of(value) -> str:
    """Strict YYYY-MM-DD (ambiguous forms like 07/08/2025 are rejected); returns the normalised date."""
    v = str(value).strip()
    if not _ISO_DATE.match(v):
        raise ValueError("as_of must be a date in YYYY-MM-DD format, e.g. 2025-07-21.")
    try:
        return pd.Timestamp(v).date().isoformat()
    except ValueError:
        raise ValueError("as_of must be a real calendar date in YYYY-MM-DD format, e.g. 2025-07-21.") from None


def parse_horizon(value) -> int:
    """Whole number 2, 3 or 4 (an int, or a string of digits); 3.7 or true are rejected, never rounded."""
    ok = (isinstance(value, int) and not isinstance(value, bool)) or (isinstance(value, str) and value.strip().isdigit())
    if not ok or int(value) not in HORIZONS:
        raise ValueError("Horizon must be one of (2, 3, 4) weeks.")
    return int(value)


def _require_ward(ward_id: str) -> None:
    if not ward_id:
        raise ValueError("ward_id is required.")
    if ward_id not in set(features()["ward_id"]):
        raise ValueError(f"Unknown ward: {ward_id}")


def validate_what_if(body: dict) -> tuple[dict, int]:
    """Plain-English checks for the typed weather. Returns (overrides, weeks)."""
    overrides = {}
    for k, (lo, hi) in WHAT_IF_RANGES.items():
        if k not in body or body[k] is None:
            continue
        v = body[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"{_LABEL[k]} ({k}) must be a number.")
        if not lo <= float(v) <= hi:
            raise ValueError(f"{_LABEL[k]} ({k}) must be between {lo:g} and {hi:g} ({WHAT_IF_UNITS[k]}).")
        overrides[k] = round(float(v), 2)
    if not overrides:
        raise ValueError("Give at least one weather value to change: rainfall_mm, avg_temp_c or relative_humidity_pct.")
    weeks = body.get("weeks", WEEKS_DEFAULT)
    if isinstance(weeks, bool) or not isinstance(weeks, int) or not WEEKS_RANGE[0] <= weeks <= WEEKS_RANGE[1]:
        raise ValueError(f"weeks must be a whole number from {WEEKS_RANGE[0]} to {WEEKS_RANGE[1]}.")
    return overrides, weeks


def _scenario_row(r) -> dict:
    """Same fields as a forecast row, from src.predict output with Shapley driver columns."""
    cols = [c for c in r.index if c.startswith("driver_")]
    names = {c: next(g for g in DRIVER_LABELS if "driver_" + g.lower().replace(" ", "_") == c) for c in cols}
    drivers = sorted(((DRIVER_LABELS[names[c]], float(r[c])) for c in cols), key=lambda kv: -abs(kv[1]))
    return {"predicted_cases": _num(r["predicted_cases"]), "range_low": _num(r["lower_80"]),
            "range_high": _num(r["upper_80"]), "risk_tier": r["risk_tier"],
            "top_drivers": [{"label": k, "effect": _num(v, 3)} for k, v in drivers],
            "driver_reference_cases": _num(r.get("base_cases"))}


@lru_cache(maxsize=2048)   # key = every input (ward, week, horizon, overrides, weeks)
def _scenario(ward_id: str, as_of: str, horizon: int, overrides: tuple, weeks: int) -> dict:
    from src.features import WEATHER
    from src.predict import _raw, predict_from_weather

    raw = _raw()
    a = pd.Timestamp(as_of)
    w = raw[(raw["ward_id"] == ward_id) & (raw["date"] <= a)].sort_values("date").tail(weeks)
    rows = w[["date", "ward_id", *WEATHER]].assign(**dict(overrides))
    res = predict_from_weather(rows, horizon, explain=True)
    return _scenario_row(res.iloc[0])


def what_if(body: dict) -> dict:
    """POST /api/what_if: baseline (normal forecast) vs scenario (typed weather for the last `weeks` weeks)."""
    unknown = sorted(set(body) - set(WHAT_IF_KEYS))
    if unknown:                                   # typos must not be silently ignored
        raise ValueError(f"Unknown field(s): {', '.join(unknown)}. Allowed: {', '.join(WHAT_IF_KEYS)}.")
    ward_id = str(body.get("ward_id") or "")
    _require_ward(ward_id)
    horizon = parse_horizon(body.get("horizon", 3))
    as_of = parse_as_of(body.get("as_of") or meta()["as_of"]["default"])
    overrides, weeks = validate_what_if(body)
    baseline_full = forecast(ward_id, as_of, horizon)      # cached; validates the date and history
    keys = ("predicted_cases", "range_low", "range_high", "risk_tier", "top_drivers", "driver_reference_cases")
    baseline = {k: baseline_full[k] for k in keys}
    scenario = _scenario(ward_id, as_of, horizon, tuple(sorted(overrides.items())), weeks)
    first = (pd.Timestamp(as_of) - pd.Timedelta(weeks=weeks - 1)).date().isoformat()
    from src.predict import _raw
    raw = _raw()
    replaced = raw[(raw["ward_id"] == ward_id) & (raw["date"] >= pd.Timestamp(first))
                   & (raw["date"] <= pd.Timestamp(as_of))].sort_values("date")
    replaced_actual = {k: {"mean": _num(replaced[k].mean(), 1), "per_week": [_num(x, 1) for x in replaced[k]]}
                       for k in overrides}
    base_eff = {d["label"]: d["effect"] for d in baseline["top_drivers"]}
    scen_eff = {d["label"]: d["effect"] for d in scenario["top_drivers"]}
    driver_change = sorted(((lbl, _num(scen_eff.get(lbl, 0) - base_eff.get(lbl, 0), 3)) for lbl in set(base_eff) | set(scen_eff)),
                           key=lambda kv: -abs(kv[1]))
    return {
        "ward_id": ward_id, "city": baseline_full["city"], "as_of": as_of, "horizon": horizon,
        "forecast_week": baseline_full["forecast_week"],
        "baseline": baseline,
        "scenario": scenario,
        "change": {"cases": _num(scenario["predicted_cases"] - baseline["predicted_cases"]),
                   "tier_changed": scenario["risk_tier"] != baseline["risk_tier"],
                   "drivers": [{"label": k, "effect_change": v} for k, v in driver_change]},
        "applied": {"overrides": overrides, "weeks": weeks, "weeks_replaced": {"from": first, "to": as_of},
                    "replaced_actual": replaced_actual},
        "units": {k: WHAT_IF_UNITS[k] for k in (*overrides, "weeks")},
    }
