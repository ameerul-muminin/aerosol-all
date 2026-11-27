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


# ---------------------------------------------------------------- /api/warnings
@lru_cache(maxsize=512)
def warnings(as_of: str, city: str | None = None) -> dict:
    """Early warnings: wards getting worse over the next 4 weeks. A ward is listed when its risk tier
    at +3 or +4 weeks is higher than at +2 weeks ("rising"), or when it is already High/Critical at
    +2 weeks and its forecast cases go up at every step to +4 weeks ("escalating"). Reuses the cached
    all-ward forecasts for each horizon, so it costs no extra model calls."""
    by_h = {h: _ranking_df(as_of, h).set_index("ward_id") for h in HORIZONS}
    first, last = HORIZONS[0], HORIZONS[-1]
    base = by_h[first]
    if city:
        if city not in set(base["city"]):
            raise ValueError(f"Unknown city: {city}")
        base = base[base["city"] == city]
    lvl = TIERS.index
    rows = []
    for w, r in base.iterrows():
        steps = [{"horizon": h, "forecast_week": str(by_h[h].at[w, "forecast_week"]),
                  "predicted_cases": _num(by_h[h].at[w, "predicted_cases"]),
                  "risk_tier": by_h[h].at[w, "risk_tier"]} for h in HORIZONS]
        start, later = lvl(steps[0]["risk_tier"]), max(lvl(s["risk_tier"]) for s in steps[1:])
        cases = [s["predicted_cases"] for s in steps]
        cases_up = all(b > a for a, b in zip(cases, cases[1:]))
        if later > start:
            reason = "rising"
        elif start >= lvl("High") and cases_up:
            reason = "escalating"
        else:
            continue
        peak = max(steps, key=lambda s: (lvl(s["risk_tier"]), s["predicted_cases"]))
        rows.append({"ward_id": w, "city": r["city"], "reason": reason, "steps": steps,
                     "peak_tier": peak["risk_tier"], "peak_horizon": peak["horizon"]})
    rows.sort(key=lambda x: (-lvl(x["peak_tier"]), -max(s["predicted_cases"] for s in x["steps"])))
    return {"as_of": as_of, "city": city, "horizons": list(HORIZONS), "n_wards": int(len(base)),
            "rule": f"Tier at +{HORIZONS[1]} or +{last} weeks higher than at +{first} weeks, or High/Critical "
                    f"at +{first} weeks with forecast cases rising at every step to +{last} weeks.",
            "wards": rows}


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
def _model_type(h: int) -> dict:
    """What is actually served at horizon h, read from the model bundle. metrics.json's "selected" is the
    training slot name (hgb_poisson), which holds the ensemble or blend when one was adopted."""
    m = load_bundle()["modes"]["weather"][h]
    weights = m.get("ensemble_weights") or {}
    return {"model_type": m.get("base_kind") or m.get("kind"),
            "model_members": [k for k, w in weights.items() if w > 0] or None}


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
            **_model_type(h),
            "mae": _num(t["MAE"], 3), "rmse": _num(t["RMSE"], 3),
            "baseline_mae": _num(t["baseline_seasonal_naive"]["MAE"], 3),
            "baseline_rmse": _num(t["baseline_seasonal_naive"]["RMSE"], 3),
            "interval80_coverage": _num(t["interval80_coverage"], 3),
            "critical_flagged_high_or_above": _num(t.get("critical_recall"), 3),
            "by_city": [{"city": c, "mae": _num(r["mae"], 3), "rmse": _num(r["rmse"], 3)} for c, r in city.iterrows()],
        })
    return {"split": m["split"], "horizons": per_h}


# ---------------------------------------------------------------- /api/lag_profile
LAG_WEEKS = 12


@lru_cache(maxsize=8)   # one entry per city (+ all cities); computed once from the raw data
def lag_profile(city: str | None = None) -> dict:
    """The "breeding clock" in the data: correlation between rain k weeks ago and cases now, k = 0..12.
    Both series are taken as anomalies (minus that ward's average for the same week of the year), so the
    shared monsoon season cannot create the link on its own. Descriptive only; the model does not use it."""
    from src.predict import _raw
    r = _raw().sort_values(["ward_id", "date"])
    if city:
        if city not in set(r["city"]):
            raise ValueError(f"Unknown city: {city}")
    woy = r["date"].dt.isocalendar().week.astype(int)
    key = [r["ward_id"], woy]
    cases = r["dengue_cases"] - r.groupby(key)["dengue_cases"].transform("mean")
    lags = []
    for k in range(LAG_WEEKS + 1):
        rain = r.groupby("ward_id")["rainfall_mm"].shift(k)
        rain = rain - rain.groupby(key).transform("mean")
        sel = (r["city"] == city) if city else slice(None)
        lags.append({"weeks": k, "corr": _num(cases[sel].corr(rain[sel]), 3)})
    peak = max(lags, key=lambda x: x["corr"])
    first, last = r["date"].min(), r["date"].max()
    return {"city": city, "lags": lags, "peak_weeks": peak["weeks"], "peak_corr": peak["corr"],
            "period": {"from": _date(first), "to": _date(last)},
            "n_rows": int(((r["city"] == city) if city else r["city"].notna()).sum()),
            "method": "Pearson correlation of rainfall k weeks earlier with cases now, both minus the "
                      "ward's average for that week of the year."}


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


@lru_cache(maxsize=1)
def training_weather_range() -> dict:
    """Min/max of each what-if input over the TRAINING years (target weeks <= TRAIN_END). Tree models
    cannot extrapolate: beyond these values their response flattens, so the API warns instead of
    silently returning a capped answer."""
    from src.predict import _raw
    from src.train import TRAIN_END
    r = _raw()
    r = r[r["date"] <= TRAIN_END]
    return {k: (float(r[k].min()), float(r[k].max())) for k in WHAT_IF_RANGES}


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
    warnings = []
    for k, v in overrides.items():
        lo, hi = training_weather_range()[k]
        if not lo <= v <= hi:
            warnings.append(f"{_LABEL[k]} {v:g} is outside the range seen in training data ({lo:.1f} to {hi:.1f} "
                            f"{WHAT_IF_UNITS[k].split(' (')[0]}); the model cannot extrapolate, so the effect of "
                            f"values beyond it levels off.")
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
        "warnings": warnings,
    }


# ---------------------------------------------------------------- /api/efficiency
EFFICIENCY_PATH = ROOT / "reports" / "efficiency.json"
# request-level caches: one entry per distinct question (endpoint + inputs)
_ANSWER_CACHES = {"forecast": "forecast", "ranking": "ranking", "warnings": "warnings",
                  "what_if": "_scenario", "metrics": "metrics", "lag_profile": "lag_profile"}
# where the model actually runs: all wards for one week/horizon, or one typed-weather scenario
_MODEL_CACHES = ("_ranking_df", "_scenario")


def efficiency() -> dict:
    """Measured costs (reports/efficiency.json, from scripts/measure_efficiency.py) plus live cache
    counters of this server process since it started. Not cached: the counters change per request."""
    measured = json.loads(EFFICIENCY_PATH.read_text()) if EFFICIENCY_PATH.exists() else None
    per = {}
    for name, fn in _ANSWER_CACHES.items():
        ci = globals()[fn].cache_info()
        per[name] = {"from_cache": ci.hits, "computed": ci.misses}
    hits = sum(v["from_cache"] for v in per.values())
    misses = sum(v["computed"] for v in per.values())
    model_runs = sum(globals()[fn].cache_info().misses for fn in _MODEL_CACHES)
    return {"measured": measured,
            "live": {"answers_from_cache": hits, "answers_computed": misses,
                     "share_from_cache": _num(hits / (hits + misses), 3) if hits + misses else None,
                     "model_runs": model_runs,
                     # all-ward results reused by other questions (a ward's forecast, warnings, other horizons)
                     "model_results_reused": _ranking_df.cache_info().hits,
                     "model_results_needed": _ranking_df.cache_info().hits + _ranking_df.cache_info().misses,
                     "by_endpoint": per}}



# ================================================================ Phase 2 (operations)
# Allocation, overflow routing and the vector-control simulator live in src/ (allocation.py,
# overflow.py, intervention.py). These wrappers only check the request, call them, attach the
# config/operations.json assumptions each one used (so the UI can show them), and convert the
# result to plain JSON (numpy/pandas -> Python, floats rounded to 2 decimals).
def _plain(x):
    """numpy/pandas/date values -> plain JSON types, floats rounded to 2 decimals, NaN -> None."""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray, pd.Series)):
        return [_plain(v) for v in (x.tolist() if isinstance(x, (np.ndarray, pd.Series)) else x)]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return None if not math.isfinite(float(x)) else round(float(x), 2)
    if isinstance(x, (pd.Timestamp,)) or hasattr(x, "isoformat"):
        return _date(x)
    return x


def _records(df: pd.DataFrame) -> list[dict]:
    return [_plain(r) for r in df.to_dict(orient="records")]


# which config/operations.json entries each route uses: {section: [keys]}
_ASSUMPTIONS_USED = {
    "allocate": {"demand": ["basis", "rounding"],
                 "stock": ["full_stock_basis", "full_stock_units", "stock_pct_range"],
                 "priority": ["tier_weight", "tie_break", "fill_rule"],
                 "crews": ["total_default", "max_per_ward", "min_predicted_cases", "assignment_rule"]},
    "overflow": {"demand": ["basis", "rounding"],
                 "bed_capacity": ["rule", "beds_available_pct"],
                 "overflow": ["same_city_only", "sender_order", "receiver_order"]},
    "intervention": {"crew_effect": ["model_input", "reduction_per_crew", "max_total_reduction",
                                     "sensitivity_per_crew"],
                     "crews": ["total_default", "max_per_ward"],
                     "demand": ["rounding"]},
    "intervention_roi": {"crew_effect": ["model_input", "reduction_per_crew", "max_total_reduction",
                                         "sensitivity_per_crew"],
                         "crews": ["max_per_ward"]},
    "crew_plan": {"crew_effect": ["model_input", "reduction_per_crew", "max_total_reduction"],
                  "crews": ["total_default", "max_per_ward", "assignment_rule"]},
}
# a notice combines the allocation plan and the overflow routing, so it relies on both sets of assumptions
_ASSUMPTIONS_USED["dispatch"] = {**_ASSUMPTIONS_USED["allocate"], **{k: v for k, v in _ASSUMPTIONS_USED["overflow"].items()
                                                                     if k != "demand"}}


def _ops_config() -> dict:
    from src.allocation import load_config
    return load_config()


def assumptions(route: str) -> dict:
    """The config/operations.json entries (value, unit, kind, why) a Phase 2 route relies on."""
    cfg = _ops_config()
    return {section: {k: _plain(cfg[section][k]) for k in keys} for section, keys in _ASSUMPTIONS_USED[route].items()}


def _check_fields(body, allowed: tuple, example: str) -> dict:
    if not isinstance(body, dict):
        raise ValueError(f"Send a JSON body, e.g. {example}.")
    unknown = sorted(set(body) - set(allowed))
    if unknown:                                   # typos must not be silently ignored
        raise ValueError(f"Unknown field(s): {', '.join(unknown)}. Allowed: {', '.join(allowed)}.")
    return body


def _check_city(city) -> str | None:
    if city in (None, ""):
        return None
    if not isinstance(city, str):
        raise ValueError("city must be a city name, e.g. \"Gazipur\".")
    if city not in {c["city"] for c in meta()["cities"]}:
        raise ValueError(f"Unknown city: {city}")
    return city


def _check_basis(basis) -> str | None:
    if basis in (None, ""):
        return None
    allowed = _ops_config()["demand"]["basis"]["allowed"]
    if not isinstance(basis, str) or basis not in allowed:
        raise ValueError(f"demand_basis must be one of: {', '.join(allowed)}.")
    return basis


def parse_whole(value, name: str, default=None):
    """Whole number from JSON (int) or a query string (digits); None/"" -> default."""
    if value is None or value == "":
        return default
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a whole number.")
    return value


def parse_number(value, name: str, default=None):
    """Finite number from JSON or a query string; None/"" -> default."""
    if value is None or value == "":
        return default
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            raise ValueError(f"{name} must be a number.") from None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a number.")
    return value


# ---------------------------------------------------------------- POST /api/allocate
ALLOCATE_KEYS = ("as_of", "horizon", "stock_pct", "crews", "demand_basis", "city")


@lru_cache(maxsize=256)    # key = every input; city filtering happens after, on the cached plan
def _allocation(as_of: str, horizon: int, stock_pct: float, crews: int | None, basis: str | None) -> dict:
    from src.allocation import plan, totals
    wards = plan(features(), as_of, horizon, stock_pct, crews, basis)
    return {"rows": _records(wards), "totals": _plain(totals(wards, stock_pct, crews))}


def allocate(body) -> dict:
    """Supplies and crews per ward under a stock level. `city` filters the rows; totals stay region-wide."""
    body = _check_fields(body, ALLOCATE_KEYS, '{"as_of": "2025-07-14", "horizon": 3, "stock_pct": 60}')
    as_of = parse_as_of(body.get("as_of") or meta()["as_of"]["default"])
    horizon = parse_horizon(body.get("horizon", 3))
    if body.get("stock_pct") in (None, ""):
        raise ValueError("stock_pct is required: the share of full stock available, from 30 to 100 (percent).")
    stock_pct = parse_number(body["stock_pct"], "stock_pct")
    crews = parse_whole(body.get("crews"), "crews")
    basis = _check_basis(body.get("demand_basis"))
    city = _check_city(body.get("city"))
    res = _allocation(as_of, horizon, float(stock_pct), crews, basis)
    rows = [r for r in res["rows"] if city is None or r["city"] == city]
    cfg = _ops_config()
    return {"as_of": as_of, "horizon": horizon,
            "forecast_week": res["rows"][0].get("forecast_week") if res["rows"] else None,
            "stock_pct": _plain(float(stock_pct)),
            "crews": res["totals"]["crews"]["available"],
            "demand_basis": basis or cfg["demand"]["basis"]["value"],
            "city": city, "totals": res["totals"], "wards": rows,
            "assumptions": assumptions("allocate")}


# ---------------------------------------------------------------- GET /api/overflow
@lru_cache(maxsize=256)
def overflow(as_of: str, horizon: int, demand_basis: str | None = None, beds_available_pct: float | None = None) -> dict:
    """Bed overflow routed between wards of the same city. Wards and cities as lists (sorted by ID/name)."""
    from src.overflow import overflow_plan
    basis = _check_basis(demand_basis)
    res = overflow_plan(features(), as_of, horizon, basis, beds_available_pct=beds_available_pct)
    city_of = dict(zip(features()["ward_id"], features()["city"]))
    wards = [{"ward_id": w, "city": city_of.get(w), **v} for w, v in sorted(res["wards"].items())]
    cities = [{"city": c, **v} for c, v in sorted(res["cities"].items())]
    return _plain({"as_of": res["as_of"], "horizon": res["horizon"], "forecast_week": res["forecast_week"],
                   "demand_basis": res["demand_basis"], "beds_available_pct": res["beds_available_pct"],
                   "transfers": res["transfers"], "wards": wards, "cities": cities,
                   "assumptions": assumptions("overflow")})


# ---------------------------------------------------------------- POST /api/intervention
INTERVENTION_KEYS = ("as_of", "deployment", "total_crews")


@lru_cache(maxsize=256)    # key = (week, sorted deployment, total crews)
def _simulation(as_of: str, deployment: tuple, total_crews: int) -> dict:
    from src.intervention import simulate
    return _plain(simulate(features(), as_of, dict(deployment), total_crews))


def intervention(body) -> dict:
    """Send crews to wards ({ward_id: crews}); the Phase 1 models are re-run at 2, 3, 4 weeks.
    total_crews defaults to the config's crews.total_default, so a plan cannot use more crews than exist."""
    body = _check_fields(body, INTERVENTION_KEYS, '{"as_of": "2025-07-14", "deployment": {"DSCC_Ward_04": 2}}')
    as_of = parse_as_of(body.get("as_of") or meta()["as_of"]["default"])
    dep = body.get("deployment")
    if not isinstance(dep, dict) or not dep:
        raise ValueError('deployment must map ward IDs to crews, e.g. {"DSCC_Ward_04": 2}.')
    for ward_id in dep:
        _require_ward(ward_id)
    from src.allocation import setting
    total = parse_whole(body.get("total_crews"), "total_crews",
                        default=int(setting(_ops_config(), "crews", "total_default")))
    # values are checked by src.intervention; only hashable ones can be cached, others fail there anyway
    if any(isinstance(v, (dict, list)) for v in dep.values()):
        raise ValueError("Crews per ward must be whole numbers.")
    res = dict(_simulation(as_of, tuple(sorted(dep.items())), total))
    res["assumptions"] = assumptions("intervention")
    return res


# ---------------------------------------------------------------- GET /api/intervention/roi
@lru_cache(maxsize=64)
def intervention_roi(as_of: str, crews_per_ward: int = 1) -> dict:
    """Every ward's cases prevented (weeks +2..+4) per crew if it got `crews_per_ward` crews, best first."""
    from src.intervention import roi_ranking
    rows = roi_ranking(features(), as_of, crews_per_ward)
    return _plain({"as_of": as_of, "crews_per_ward": crews_per_ward, "wards": rows,
                   "assumptions": assumptions("intervention_roi")})


# ---------------------------------------------------------------- GET /api/crew_plan  (bonus B2)
@lru_cache(maxsize=64)
def crew_plan(as_of: str, crews: int | None = None) -> dict:
    """The crew plan that prevents the most cases over weeks +2..+4 (exact knapsack, src/placement.py),
    with the allocation engine's crew rule on the same table for comparison. crews defaults to the config."""
    from src.placement import best_placement
    res = best_placement(features(), as_of, crews)
    res["assumptions"] = assumptions("crew_plan")
    return _plain(res)


# live /api/efficiency counters also cover the Phase 2 answers (each miss runs the model)
_ANSWER_CACHES.update({"allocate": "_allocation", "overflow": "overflow",
                       "intervention": "_simulation", "intervention_roi": "intervention_roi",
                       "crew_plan": "crew_plan"})
_MODEL_CACHES = (*_MODEL_CACHES, "_allocation", "overflow", "_simulation", "intervention_roi", "crew_plan")


# ---------------------------------------------------------------- GET /api/dispatch (+ /brief)
DISPATCH_LANGS = ("en", "bn")


def _check_lang(lang) -> str:
    lang = lang or "en"
    if lang not in DISPATCH_LANGS:
        raise ValueError("lang must be 'en' (English) or 'bn' (Bangla).")
    return lang


@lru_cache(maxsize=64)    # key = every input; city filtering happens after, on the cached notices
def _notices(as_of: str, horizon: int, stock_pct: float, crews: int | None, beds_available_pct: float | None,
             lang: str) -> tuple[list, list]:
    """(raw notices for brief_html, JSON-ready notices with SMS and WhatsApp text), all 60 wards in
    allocation priority order. Same templates as scripts/export_dispatch.py, so the API and exports/ agree."""
    from src.dispatch import build_notices, sms, sms_segments, whatsapp
    raw = build_notices(features(), as_of, horizon, stock_pct, crews, beds_available_pct, lang)
    rows = []
    for n in raw:
        text = sms(n)
        row = {k: v for k, v in n.items() if k not in ("intervention", "lang")}
        row.update(sms=text, sms_chars=len(text), sms_segments=sms_segments(text), whatsapp=whatsapp(n))
        rows.append(_plain(row))
    return raw, rows


def _dispatch_inputs(stock_pct, crews, beds_available_pct, lang, city) -> tuple:
    if stock_pct is None:
        raise ValueError("stock_pct is required: the share of full stock available, from 30 to 100 (percent).")
    return float(stock_pct), crews, beds_available_pct, _check_lang(lang), _check_city(city)


def dispatch(as_of: str, horizon: int, stock_pct, crews: int | None = None, beds_available_pct=None,
             lang: str | None = None, city: str | None = None) -> dict:
    """Ward-specific dispatch notices (SMS, WhatsApp) built from the allocation and overflow plans for these
    settings. `city` filters the notices; the plans themselves always cover all 60 wards."""
    stock_pct, crews, beds_pct, lang, city = _dispatch_inputs(stock_pct, crews, beds_available_pct, lang, city)
    _, rows = _notices(as_of, horizon, stock_pct, crews, beds_pct, lang)
    rows = [r for r in rows if city is None or r["city"] == city]
    from src.allocation import setting
    cfg = _ops_config()
    return {"as_of": as_of, "horizon": horizon, "forecast_week": rows[0]["forecast_week"] if rows else None,
            "lang": lang, "stock_pct": _plain(stock_pct),
            "crews": int(setting(cfg, "crews", "total_default")) if crews is None else crews,
            "beds_available_pct": _plain(float(setting(cfg, "bed_capacity", "beds_available_pct")
                                               if beds_pct is None else beds_pct)),
            "city": city,
            "summary": {"wards": len(rows), "critical": sum(r["critical"] for r in rows),
                        "unroutable_patients": sum(r["unroutable"] for r in rows),
                        "max_sms_segments": max((r["sms_segments"] for r in rows), default=0)},
            "notices": rows, "assumptions": assumptions("dispatch")}


def dispatch_brief(as_of: str, horizon: int, stock_pct, crews: int | None = None, beds_available_pct=None,
                   lang: str | None = None, city: str | None = None) -> str:
    """The printable commissioner brief (HTML, critical wards first) for the same settings as dispatch()."""
    from src.dispatch import brief_html
    stock_pct, crews, beds_pct, lang, city = _dispatch_inputs(stock_pct, crews, beds_available_pct, lang, city)
    raw, _ = _notices(as_of, horizon, stock_pct, crews, beds_pct, lang)
    return brief_html([n for n in raw if city is None or n["city"] == city])


_ANSWER_CACHES["dispatch"] = "_notices"
_MODEL_CACHES = (*_MODEL_CACHES, "_notices")


# ---------------------------------------------------------------- GET /api/supply_plan  (bonus B1)
SUPPLY_KEYS = {"saline": "saline_bags_needed", "kits": "diagnostic_kits_needed", "beds": "hospital_beds_required"}
_ASSUMPTIONS_USED["supply_plan"] = {"demand": ["basis", "rounding"]}


def _need(cases) -> dict:
    from src.supplies import supplies_for
    s = supplies_for(cases)
    return {"cases": int(cases.sum()), **{k: int(s[c].sum()) for k, c in SUPPLY_KEYS.items()}}


def _order(weeks: list[dict]) -> dict:
    """Saline and kits are used up, so the order is their sum over the weeks; beds are reused, so plan for the peak."""
    return {"cases": sum(w["cases"] for w in weeks), "saline": sum(w["saline"] for w in weeks),
            "kits": sum(w["kits"] for w in weeks), "beds_peak": max(w["beds"] for w in weeks)}


@lru_cache(maxsize=64)
def supply_plan(as_of: str, city: str | None = None) -> dict:
    """Order now for weeks +2, +3 and +4: every ward's need in each week, from that horizon's forecast and the
    same case -> supplies table as allocation (src/supplies.py). Built from the cached all-ward forecasts, so it
    adds no model runs. expected = central forecast; cautious = top of the 80% range. Need only: the data has no
    stock on hand, so subtract what is in the store before ordering."""
    from src.supplies import case_count
    city = _check_city(city)
    weeks, wards = [], {}
    for h in HORIZONS:
        df = _ranking_df(as_of, h)
        if city:
            df = df[df["city"] == city]
        df = df.sort_values("ward_id")
        week = {"horizon": h, "forecast_week": str(df["forecast_week"].iloc[0])}
        for basis, col in (("expected", "predicted_cases"), ("cautious", "range_high")):
            cases = case_count(df[col].to_numpy())
            week[basis] = _need(cases)
            for (_, r), c in zip(df.iterrows(), cases):
                w = wards.setdefault(r["ward_id"], {"ward_id": r["ward_id"], "city": r["city"], "weeks": {}})
                cell = w["weeks"].setdefault(str(h), {"forecast_week": str(r["forecast_week"]), "risk_tier": r["risk_tier"]})
                cell[basis] = _need(case_count([c]))
        weeks.append(week)

    def summarise(week_lists: dict) -> dict:
        return {b: _order(week_lists[b]) for b in ("expected", "cautious")}

    rows = []
    for w in wards.values():
        w["order"] = summarise({b: [w["weeks"][str(h)][b] for h in HORIZONS] for b in ("expected", "cautious")})
        rows.append(w)
    rows.sort(key=lambda w: (-w["order"]["expected"]["saline"], w["ward_id"]))
    by_city = []
    for c in sorted({w["city"] for w in rows}):
        mine = [w for w in rows if w["city"] == c]
        per_week = [{"horizon": h, **{b: {k: sum(w["weeks"][str(h)][b][k] for w in mine) for k in ("cases", *SUPPLY_KEYS)}
                                      for b in ("expected", "cautious")}} for h in HORIZONS]
        by_city.append({"city": c, "wards": len(mine), "weeks": per_week,
                        "order": summarise({b: [wk[b] for wk in per_week] for b in ("expected", "cautious")})})
    return {"as_of": as_of, "city": city, "weeks": weeks,
            "order": summarise({b: [wk[b] for wk in weeks] for b in ("expected", "cautious")}),
            "by_city": by_city, "wards": rows,
            "rule": "Need for weeks +2, +3 and +4 from each horizon's forecast. Saline and kits are used up, so the "
                    "order is their 3-week sum; beds are reused, so plan for the peak week. Expected = central "
                    "forecast; cautious = top of the 80% range. Stock on hand is not in the data: subtract it.",
            "assumptions": assumptions("supply_plan")}


_ANSWER_CACHES["supply_plan"] = "supply_plan"


# ---------------------------------------------------------------- GET /api/selfcheck
def selfcheck(as_of: str | None = None, horizon: int = 2) -> dict:
    """Re-run every Phase 2 math check live, with outbound network blocked (src/selfcheck.py).
    Not cached on purpose: each call proves the math again. Default week = the latest in the data."""
    from src.selfcheck import run_checks
    as_of = parse_as_of(as_of or meta()["as_of"]["max"])
    return _plain(run_checks(features(), as_of, horizon))
