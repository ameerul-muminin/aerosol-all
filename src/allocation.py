"""Constrained allocation: forecast cases -> supplies and spraying crews per ward, under scarcity.

    from src.predict import build_features
    from src.allocation import plan, allocate, totals
    feat = build_features()
    wards = plan(feat, "2025-07-14", horizon=3, stock_pct=60, crews=20)   # all 60 wards
    summary = totals(wards, stock_pct=60, crews=20)

Rules (every number they use lives in config/operations.json, with the reason):
- Need per ward = supplies for its forecast case count (src/supplies.py: learned from the data).
- Available per item = floor(stock_pct% x full stock); full stock = this week's total need
  unless a fixed warehouse count is set in the config.
- Priority = tier weight x population_density_sqkm, ties by ward_id. Fill the optional
  whole-unit minimum share in that order, then fill remaining needs by priority until stock
  runs out. Totals never exceed stock, and any ward under 100% on any item is `critical`.
- Crews are handed out one at a time to the ward with the highest
  (priority x predicted cases) / (crews already there + 1); capped per ward; never more than given.
Same inputs -> same output (no randomness, stable ordering).
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from numbers import Real
from pathlib import Path

import numpy as np
import pandas as pd

from src.supplies import SUPPLY_ITEMS, case_count, supplies_for

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "operations.json"

# short names used in output columns: saline_required, kits_deficit, beds_fulfilled_pct, ...
ITEM_KEYS = {"saline_bags_needed": "saline", "diagnostic_kits_needed": "kits", "hospital_beds_required": "beds"}
REQUIRED_COLUMNS = {"ward_id", "city", "predicted_cases", "risk_tier", "population_density_sqkm"}


@lru_cache(maxsize=1)
def _load_config_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def load_config(path: Path = CONFIG_PATH) -> dict:
    """The operations config as a fresh dict (safe to modify in tests)."""
    return json.loads(_load_config_text(str(path)))


def setting(cfg: dict, section: str, key: str):
    return cfg[section][key]["value"]


# ---------------------------------------------------------------- inputs

def _check_stock_pct(stock_pct, cfg) -> float:
    lo, hi = setting(cfg, "stock", "stock_pct_range")
    if isinstance(stock_pct, bool) or not isinstance(stock_pct, Real) or not math.isfinite(stock_pct):
        raise ValueError(f"Stock must be a number between {lo} and {hi} (percent).")
    if not lo <= stock_pct <= hi:
        raise ValueError(f"Stock must be between {lo}% and {hi}%; got {stock_pct:g}%.")
    return float(stock_pct)


def _check_min_share_pct(min_share_pct, cfg) -> float:
    if min_share_pct is None:
        min_share_pct = cfg["priority"].get("min_share_pct", {"value": 0})["value"]
    if (isinstance(min_share_pct, bool) or not isinstance(min_share_pct, Real)
            or not 0 <= min_share_pct <= 50):
        raise ValueError("Minimum share must be a number between 0% and 50%.")
    return float(min_share_pct)


def _check_crews(crews, cfg) -> int:
    if crews is None:
        return int(setting(cfg, "crews", "total_default"))
    if isinstance(crews, bool) or not isinstance(crews, Real) or crews != int(crews) or crews < 0:
        raise ValueError("Number of spraying crews must be a whole number, 0 or more.")
    return int(crews)


def _check_predictions(pred: pd.DataFrame, basis: str, cfg) -> pd.DataFrame:
    if not isinstance(pred, pd.DataFrame) or pred.empty:
        raise ValueError("No ward forecasts to allocate.")
    allowed = cfg["demand"]["basis"]["allowed"]
    if basis not in allowed:
        raise ValueError(f"Demand basis must be one of {allowed}; got {basis!r}.")
    missing = (REQUIRED_COLUMNS | {basis}) - set(pred.columns)
    if missing:
        raise ValueError(f"Forecast table is missing columns: {sorted(missing)}")
    if pred["ward_id"].duplicated().any():
        raise ValueError("Each ward must appear once in the forecast table.")
    weights = setting(cfg, "priority", "tier_weight")
    bad = set(pred["risk_tier"]) - set(weights)
    if bad:
        raise ValueError(f"Unknown risk tier(s): {sorted(bad)}")
    return pred


# ---------------------------------------------------------------- stock

def available_stock(required_totals: dict[str, int], stock_pct: float, cfg: dict | None = None) -> dict[str, int]:
    """Units available per item at this stock level (whole units, rounded down)."""
    cfg = cfg or load_config()
    fixed = setting(cfg, "stock", "full_stock_units")
    out = {}
    for item in SUPPLY_ITEMS:
        full = fixed.get(item)
        full = int(required_totals[item]) if full is None else int(full)
        out[item] = int(math.floor(full * stock_pct / 100 + 1e-9))
    return out


def _greedy_fill(required: np.ndarray, available: int) -> np.ndarray:
    """`required` is already in priority order. Each ward gets min(need, what is left)."""
    before = np.cumsum(required) - required            # stock used by higher-priority wards
    return np.clip(available - before, 0, required)


# ---------------------------------------------------------------- crews

def assign_crews(order: pd.DataFrame, crews: int, cfg: dict) -> np.ndarray:
    """Crews per ward (aligned to `order`'s rows). One crew at a time to the ward with the
    highest weight / (crews there + 1); ties go to the lowest ward_id."""
    c = cfg["crews"]
    cap = int(c["max_per_ward"]["value"])
    min_cases = float(c["min_predicted_cases"]["value"])
    weight = (order["priority_score"] * order["predicted_cases"]).to_numpy(dtype=float)
    eligible = order["predicted_cases"].to_numpy(dtype=float) >= min_cases
    by_id = np.argsort(order["ward_id"].to_numpy(), kind="stable")   # argmax picks the first max
    w, ok = weight[by_id], eligible[by_id]
    given = np.zeros(len(order), dtype=int)
    for _ in range(min(crews, int(ok.sum()) * cap)):
        share = np.where(ok & (given < cap), w / (given + 1), -np.inf)
        given[int(np.argmax(share))] += 1
    out = np.zeros(len(order), dtype=int)
    out[by_id] = given
    return out


# ---------------------------------------------------------------- main entry

def allocate(predictions: pd.DataFrame, stock_pct: float, crews: int | None = None,
             demand_basis: str | None = None, config: dict | None = None,
             min_share_pct: float | None = None) -> pd.DataFrame:
    """One row per ward, in priority order. Columns:
    priority_rank, ward_id, city, risk_tier, population_density_sqkm, priority_score,
    predicted_cases, demand_cases, then for each of saline / kits / beds:
    <item>_required, <item>_guaranteed (first-pass units), <item>_allocated,
    <item>_fulfilled_pct, <item>_deficit;
    fulfilled_pct (worst item), critical (any item under 100%), crews_assigned.
    Bad input raises ValueError with a message fit to show the user."""
    cfg = config or load_config()
    stock_pct = _check_stock_pct(stock_pct, cfg)
    min_share_pct = _check_min_share_pct(min_share_pct, cfg)
    crews = _check_crews(crews, cfg)
    basis = demand_basis or setting(cfg, "demand", "basis")
    pred = _check_predictions(predictions, basis, cfg)

    weights = setting(cfg, "priority", "tier_weight")
    keep = ["ward_id", "city", "risk_tier", "population_density_sqkm", "predicted_cases"]
    keep += [c for c in ("forecast_week", "range_low", "range_high") if c in pred.columns]
    d = pred[keep].copy()
    d["priority_score"] = d["risk_tier"].map(weights).astype(float) * d["population_density_sqkm"].astype(float)
    d = d.sort_values(["priority_score", "ward_id"], ascending=[False, True], kind="stable").reset_index(drop=True)
    d.insert(0, "priority_rank", np.arange(1, len(d) + 1))
    d["demand_cases"] = case_count(pred.set_index("ward_id").loc[d["ward_id"], basis].to_numpy())

    need = supplies_for(d["demand_cases"].to_numpy())
    stock = available_stock({i: int(need[i].sum()) for i in SUPPLY_ITEMS}, stock_pct, cfg)
    pct_cols = []
    for item, k in ITEM_KEYS.items():
        req = need[item].to_numpy(dtype=int)
        minimum = np.floor(req * min_share_pct / 100).astype(int)
        guaranteed = _greedy_fill(minimum, stock[item])
        got = guaranteed + _greedy_fill(req - guaranteed, stock[item] - int(guaranteed.sum()))
        d[f"{k}_required"] = req
        d[f"{k}_guaranteed"] = guaranteed
        d[f"{k}_allocated"] = got
        d[f"{k}_fulfilled_pct"] = np.where(req > 0, np.floor(1000 * got / np.maximum(req, 1)) / 10, 100.0)
        d[f"{k}_deficit"] = req - got
        pct_cols.append(f"{k}_fulfilled_pct")
    d["fulfilled_pct"] = d[pct_cols].min(axis=1)
    d["critical"] = d[[f"{k}_deficit" for k in ITEM_KEYS.values()]].gt(0).any(axis=1)
    d["crews_assigned"] = assign_crews(d, crews, cfg)
    d.attrs["min_share_pct"] = min_share_pct
    return d


def totals(alloc: pd.DataFrame, stock_pct: float, crews: int | None = None, config: dict | None = None) -> dict:
    """Region-wide summary of an `allocate()` result (JSON-ready)."""
    cfg = config or load_config()
    crews = _check_crews(crews, cfg)
    required = {item: int(alloc[f"{k}_required"].sum()) for item, k in ITEM_KEYS.items()}
    stock = available_stock(required, _check_stock_pct(stock_pct, cfg), cfg)
    items = {}
    for item, k in ITEM_KEYS.items():
        items[k] = {"column": item, "required": required[item], "available": stock[item],
                    "allocated": int(alloc[f"{k}_allocated"].sum()),
                    "deficit": int(alloc[f"{k}_deficit"].sum()),
                    "wards_short": int(alloc[f"{k}_deficit"].gt(0).sum())}
    assigned = int(alloc["crews_assigned"].sum())
    return {"stock_pct": float(stock_pct),
            "min_share_pct": _check_min_share_pct(alloc.attrs.get("min_share_pct"), cfg), "items": items,
            "critical_wards": int(alloc["critical"].sum()), "wards": int(len(alloc)),
            "crews": {"available": crews, "assigned": assigned, "unused": crews - assigned}}


def with_density(predictions: pd.DataFrame, feat_df: pd.DataFrame) -> pd.DataFrame:
    """Add population_density_sqkm (fixed per ward in the data) to predict_all() output."""
    dens = feat_df[["ward_id", "population_density_sqkm"]].drop_duplicates("ward_id")
    return predictions.drop(columns="population_density_sqkm", errors="ignore").merge(dens, on="ward_id", how="left")


def plan(feat_df: pd.DataFrame, as_of_date, horizon: int, stock_pct: float, crews: int | None = None,
         demand_basis: str | None = None, config: dict | None = None,
         min_share_pct: float | None = None) -> pd.DataFrame:
    """predict_all() for every ward at (as_of_date, horizon), then allocate()."""
    from src.predict import predict_all      # lazy: allocate() itself needs no model
    pred = predict_all(feat_df, as_of_date, horizon, with_drivers=False)
    return allocate(with_density(pred, feat_df), stock_pct, crews, demand_basis, config, min_share_pct)
