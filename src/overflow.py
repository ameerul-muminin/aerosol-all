"""Route hospital bed overflow within cities.

    from src.predict import build_features
    from src.overflow import overflow_plan
    plan = overflow_plan(build_features(), "2025-08-04", horizon=2)
    plan["transfers"], plan["wards"], plan["cities"]

Beds needed per ward = hospital_beds_required for the forecast case count
(src/supplies.py, learned from the data). Bed capacity per ward is not in the
data: it comes from the rule in config/operations.json ("bed_capacity").
"""
from __future__ import annotations

from functools import lru_cache
from numbers import Integral

import numpy as np

WARD_KEYS = ("ward_id", "city", "beds_required", "bed_capacity")


def _check_wards(wards) -> None:
    """Friendly ValueError for input the routing cannot trust."""
    seen = set()
    for ward in wards:
        missing = [k for k in WARD_KEYS if k not in ward]
        if missing:
            raise ValueError(f"Ward entry is missing {missing}.")
        if ward["ward_id"] in seen:
            raise ValueError(f"Ward {ward['ward_id']} appears more than once.")
        seen.add(ward["ward_id"])
        for k in ("beds_required", "bed_capacity"):
            v = ward[k]
            if isinstance(v, bool) or not isinstance(v, Integral) or v < 0:
                raise ValueError(f"{k} for ward {ward['ward_id']} must be a whole number, 0 or more; got {v!r}.")


def route_overflow(wards: list[dict]) -> dict:
    """Return transfers and summaries keyed by ward ID and city.

    Each ward: ward_id (unique), city, beds_required and bed_capacity (whole
    numbers >= 0); anything else raises ValueError. Patients only move within
    their own city. Within a city, the ward with the largest overflow sends
    first (ties by ID), so if the city runs out of free beds the patients left
    without a bed are in the wards with the smallest overflow. Each patient
    goes to the receiver with the most free beds left, ties by ID (rules and
    reasons: config/operations.json, "overflow"). City utilisation measures
    original demand as a percentage of capacity and is zero when capacity is zero.
    """
    _check_wards(wards)
    transfers = []
    ward_results = {}
    city_results = {}
    wards_by_city = {}

    for ward in sorted(wards, key=lambda ward: ward["ward_id"]):
        ward_id = ward["ward_id"]
        demand = ward["beds_required"]
        capacity = ward["bed_capacity"]
        wards_by_city.setdefault(ward["city"], []).append(ward_id)
        ward_results[ward_id] = {
            "beds_required": demand,
            "capacity": capacity,
            "routed_out": 0,
            "routed_in": 0,
            "net_deficit": max(0, demand - capacity),
        }

    for city, ward_ids in sorted(wards_by_city.items()):
        headroom = {}
        capacity = demand = unroutable = 0
        for ward_id in ward_ids:
            result = ward_results[ward_id]
            capacity += result["capacity"]
            demand += result["beds_required"]
            free_beds = result["capacity"] - result["beds_required"]
            if free_beds > 0:
                headroom[ward_id] = free_beds

        senders = sorted(ward_ids, key=lambda ward_id: (-ward_results[ward_id]["net_deficit"], ward_id))
        for sender_id in senders:
            sender = ward_results[sender_id]
            if sender["net_deficit"] == 0:
                continue

            receivers = sorted(headroom, key=lambda ward_id: (-headroom[ward_id], ward_id))
            for receiver_id in receivers:
                patients = min(sender["net_deficit"], headroom[receiver_id])
                transfers.append({
                    "from_ward": sender_id,
                    "to_ward": receiver_id,
                    "city": city,
                    "patients": patients,
                })
                sender["routed_out"] += patients
                sender["net_deficit"] -= patients
                ward_results[receiver_id]["routed_in"] += patients
                headroom[receiver_id] -= patients
                if headroom[receiver_id] == 0:
                    del headroom[receiver_id]
                if sender["net_deficit"] == 0:
                    break

            unroutable += sender["net_deficit"]

        city_results[city] = {
            "capacity": capacity,
            "demand": demand,
            "utilisation_pct": 100 * demand / capacity if capacity else 0.0,
            "headroom": sum(headroom.values()),
            "unroutable": unroutable,
        }

    return {"transfers": transfers, "wards": ward_results, "cities": city_results}


@lru_cache(maxsize=1)
def _historic_beds():
    """Weekly beds needed per ward, all years (beds = a fixed function of cases, src/supplies.py)."""
    from src.features import load_raw
    from src.supplies import supplies_for
    raw = load_raw()
    beds = supplies_for(raw["dengue_cases"].to_numpy())["hospital_beds_required"].to_numpy()
    return raw[["date", "ward_id"]].assign(beds=beds)


def bed_capacity(config: dict | None = None) -> dict[str, int]:
    """Beds per ward from the config rule: the chosen quantile of each ward's weekly beds
    needed up to period_end, rounded up, never below minimum_beds."""
    from src.allocation import load_config, setting
    rule = setting(config or load_config(), "bed_capacity", "rule")
    if rule["method"] != "per_ward_quantile":
        raise ValueError(f"Unknown bed capacity method: {rule['method']!r}")
    hist = _historic_beds()
    hist = hist[hist["date"] <= rule["period_end"]]
    q = hist.groupby("ward_id")["beds"].quantile(rule["quantile"])
    cap = np.ceil(q.to_numpy() - 1e-9) if rule["round"] == "up" else np.round(q.to_numpy())
    cap = np.maximum(cap, rule["minimum_beds"]).astype(int)
    return {ward_id: int(c) for ward_id, c in zip(q.index, cap)}


def overflow_plan(feat_df, as_of_date, horizon: int, demand_basis: str | None = None,
                  config: dict | None = None) -> dict:
    """Forecast every ward (predict_all), turn cases into beds needed, route the overflow.
    Returns route_overflow()'s result plus as_of, horizon, forecast_week, demand_basis."""
    from src.allocation import load_config, setting
    from src.predict import predict_all
    from src.supplies import case_count, supplies_for
    cfg = config or load_config()
    basis = demand_basis or setting(cfg, "demand", "basis")
    if basis not in cfg["demand"]["basis"]["allowed"]:
        raise ValueError(f"Demand basis must be one of {cfg['demand']['basis']['allowed']}; got {basis!r}.")
    pred = predict_all(feat_df, as_of_date, horizon, with_drivers=False)
    beds = supplies_for(case_count(pred[basis].to_numpy()))["hospital_beds_required"].to_numpy()
    capacity = bed_capacity(cfg)
    unknown = sorted(set(pred["ward_id"]) - set(capacity))
    if unknown:
        raise ValueError(f"No bed capacity for wards: {unknown}")
    wards = [{"ward_id": w, "city": c, "beds_required": int(b), "bed_capacity": capacity[w]}
             for w, c, b in zip(pred["ward_id"], pred["city"], beds)]
    result = route_overflow(wards)
    result.update(as_of=str(as_of_date)[:10], horizon=int(horizon), demand_basis=basis,
                  forecast_week=str(pred["forecast_week"].iloc[0]))
    return result
