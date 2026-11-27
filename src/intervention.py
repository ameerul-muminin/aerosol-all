"""Vector-control simulator: what do spraying / larvicide crews change 2-4 weeks later?

    from src.predict import build_features
    from src.intervention import simulate, roi_ranking
    feat = build_features()
    simulate(feat, "2025-07-14", {"DSCC_Ward_04": 2, "CCC_Ward_01": 1})   # before/after at h = 2, 3, 4
    roi_ranking(feat, "2025-07-14")                                      # all 60 wards, best ROI first

How it works (model-driven, not a toggle):
1. Crews lower the ward's `water_logging_risk_index` (breeding habitat): each crew by
   `reduction_per_crew` of its value, capped at `max_total_reduction` (config/operations.json,
   "crew_effect" - an assumption, there is no intervention data).
2. The deployed Phase 1 models are re-run on the changed inputs at 2, 3 and 4 weeks.
3. Cases prevented = forecast before - forecast after. Beds and saline come from the
   case -> supplies lookup learned from the data (src/supplies.py).
4. ROI = cases prevented over weeks +2..+4 per crew.

Each ward's forecast uses only its own inputs plus city case rates (reported cases, which a
crew does not change), so a crew in one ward never changes another ward's forecast. That is
why roi_ranking() can score all 60 wards in one model run per horizon.
"""
from __future__ import annotations

from numbers import Integral

import numpy as np
import pandas as pd

from src.allocation import load_config, setting
from src.features import HORIZONS
from src.predict import _check_inputs, predict_all
from src.supplies import case_count, supplies_for

BEDS, SALINE = "hospital_beds_required", "saline_bags_needed"


def _as_of(feat_df: pd.DataFrame, as_of_date) -> pd.Timestamp:
    try:
        return _check_inputs(feat_df, as_of_date, HORIZONS[0])
    except (TypeError, ValueError) as e:
        if "Pick a date" in str(e) or "history" in str(e) or "No data" in str(e):
            raise
        raise ValueError("Dates must look like YYYY-MM-DD (a Monday).") from None


def _check_deployment(deployment, wards: set[str], cfg: dict, total_crews) -> dict[str, int]:
    if not isinstance(deployment, dict) or not deployment:
        raise ValueError("Choose at least one ward and how many crews to send there.")
    cap = int(setting(cfg, "crews", "max_per_ward"))
    out = {}
    for ward_id, crews in deployment.items():
        if ward_id not in wards:
            raise ValueError(f"Unknown ward: {ward_id}")
        if isinstance(crews, bool) or not isinstance(crews, Integral) or not 0 <= crews <= cap:
            raise ValueError(f"Crews for {ward_id} must be a whole number from 0 to {cap}; got {crews!r}.")
        if crews:
            out[ward_id] = int(crews)
    if not out:
        raise ValueError("Send at least one crew.")
    if total_crews is not None:
        if isinstance(total_crews, bool) or not isinstance(total_crews, Integral) or total_crews < 0:
            raise ValueError("Total crews must be a whole number, 0 or more.")
        if sum(out.values()) > total_crews:
            raise ValueError(f"That plan uses {sum(out.values())} crews but only {total_crews} are available.")
    return dict(sorted(out.items()))


def reduction(crews, cfg: dict | None = None):
    """Fraction the water-logging index is lowered by, for a number of crews (scalar or array)."""
    cfg = cfg or load_config()
    per = float(setting(cfg, "crew_effect", "reduction_per_crew"))
    cap = float(setting(cfg, "crew_effect", "max_total_reduction"))
    return np.minimum(per * np.asarray(crews, dtype=float), cap)


def input_floor(feat_df: pd.DataFrame, col: str) -> float:
    """Lowest value of `col` in the training years (read from the data, not hard-coded)."""
    from src.train import TRAIN_END
    train = feat_df.loc[feat_df["date"] <= TRAIN_END, col]
    if train.empty:
        raise ValueError("No training-period rows to find the lowest model input.")
    return float(train.min())


def _before_after(feat_df: pd.DataFrame, as_of: pd.Timestamp, crews: pd.Series, cfg: dict) -> pd.DataFrame:
    """One row per ward: city, crews, the model input before/after, and cases before/after
    per horizon (columns before_h2, after_h2, ...)."""
    col = setting(cfg, "crew_effect", "model_input")
    on_day = (feat_df["date"] == as_of).to_numpy()
    day = feat_df[on_day]
    k = day["ward_id"].map(crews).fillna(0).to_numpy()
    before = day[col].to_numpy(dtype=float)
    # never below the lowest value the models were trained on: past it a tree model just
    # repeats its edge value, so a lower number would claim an effect nobody has seen
    floor = np.minimum(before, input_floor(feat_df, col))
    changed = feat_df.copy()                       # predict_all checks history, so keep all weeks
    changed.loc[on_day, col] = np.maximum(before * (1 - reduction(k, cfg)), floor)
    out = day[["ward_id", "city", col]].rename(columns={col: "input_before"}).set_index("ward_id")
    out["input_after"] = changed[on_day].set_index("ward_id")[col]
    out["crews"] = out.index.map(crews).fillna(0).astype(int)
    for h in HORIZONS:
        b = predict_all(feat_df, as_of, h, with_drivers=False).set_index("ward_id")
        a = predict_all(changed, as_of, h, with_drivers=False).set_index("ward_id")
        out[f"before_h{h}"] = b["predicted_cases"]
        out[f"after_h{h}"] = a["predicted_cases"]
        out[f"week_h{h}"] = b["forecast_week"].astype(str)
    return out


def _load(cases: pd.Series) -> dict[str, int]:
    s = supplies_for(case_count(cases.to_numpy()))
    return {"beds": int(s[BEDS].sum()), "saline": int(s[SALINE].sum())}


def simulate(feat_df: pd.DataFrame, as_of_date, deployment: dict[str, int], total_crews: int | None = None,
             config: dict | None = None) -> dict:
    """Deploy crews ({ward_id: crews}) at `as_of_date`; re-run the model at 2, 3 and 4 weeks.
    Returns (JSON-ready): as_of, crews_used, assumption, `horizons` (region totals before/after:
    cases, beds, saline) and `wards` (each deployed ward, best ROI first).
    Bad input raises ValueError with a message fit to show the user."""
    cfg = config or load_config()
    as_of = _as_of(feat_df, as_of_date)
    plan = _check_deployment(deployment, set(feat_df["ward_id"]), cfg, total_crews)
    t = _before_after(feat_df, as_of, pd.Series(plan, dtype=int), cfg)

    horizons = []
    for h in HORIZONS:
        b, a = t[f"before_h{h}"], t[f"after_h{h}"]
        lb, la = _load(b), _load(a)
        horizons.append({
            "horizon": h, "forecast_week": t[f"week_h{h}"].iloc[0],
            "cases_before": round(float(b.sum()), 2), "cases_after": round(float(a.sum()), 2),
            "cases_prevented": round(float((b - a).sum()), 2),
            "beds_before": lb["beds"], "beds_after": la["beds"], "beds_freed": lb["beds"] - la["beds"],
            "saline_before": lb["saline"], "saline_after": la["saline"], "saline_saved": lb["saline"] - la["saline"],
        })

    wards = [_ward_row(ward_id, r) for ward_id, r in t.loc[list(plan)].iterrows()]
    wards.sort(key=lambda w: (-w["roi"], w["ward_id"]))
    return {"as_of": str(as_of.date()), "crews_used": sum(plan.values()), "total_crews": total_crews,
            "assumption": _assumption(cfg), "horizons": horizons, "wards": wards}


def _ward_row(ward_id: str, r: pd.Series) -> dict:
    by_h = {}
    for h in HORIZONS:
        b, a = float(r[f"before_h{h}"]), float(r[f"after_h{h}"])
        by_h[str(h)] = {"forecast_week": r[f"week_h{h}"], "cases_before": round(b, 2),
                        "cases_after": round(a, 2), "cases_prevented": round(b - a, 2)}
    total = round(sum(v["cases_prevented"] for v in by_h.values()), 2)
    crews = int(r["crews"])
    return {"ward_id": ward_id, "city": r["city"], "crews": crews,
            "input_before": round(float(r["input_before"]), 4), "input_after": round(float(r["input_after"]), 4),
            "by_horizon": by_h, "cases_prevented": total,
            "roi": round(total / crews, 3) if crews else 0.0}


def _assumption(cfg: dict) -> dict:
    e = cfg["crew_effect"]
    return {"model_input": e["model_input"]["value"],
            "reduction_per_crew": e["reduction_per_crew"]["value"],
            "max_total_reduction": e["max_total_reduction"]["value"],
            "note": "Effect size is a planning assumption (no intervention data); see config/operations.json."}


def roi_ranking(feat_df: pd.DataFrame, as_of_date, crews_per_ward: int = 1, config: dict | None = None) -> list[dict]:
    """Every ward's ROI if it got `crews_per_ward` crews: cases prevented over weeks +2..+4
    per crew, best first (ties by ward_id). Wards do not affect each other, so all are scored
    in one model run per horizon."""
    cfg = config or load_config()
    as_of = _as_of(feat_df, as_of_date)
    cap = int(setting(cfg, "crews", "max_per_ward"))
    if isinstance(crews_per_ward, bool) or not isinstance(crews_per_ward, Integral) or not 1 <= crews_per_ward <= cap:
        raise ValueError(f"Crews per ward must be a whole number from 1 to {cap}.")
    wards = sorted(feat_df.loc[feat_df["date"] == as_of, "ward_id"].unique())
    t = _before_after(feat_df, as_of, pd.Series(crews_per_ward, index=wards), cfg)
    rows = [_ward_row(w, r) for w, r in t.iterrows()]
    rows.sort(key=lambda w: (-w["roi"], w["ward_id"]))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return rows
