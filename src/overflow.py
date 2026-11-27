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


def usable_capacity(capacity: dict[str, int], beds_available_pct, config: dict | None = None) -> dict[str, int]:
    """Beds that can take dengue patients: floor(capacity x pct / 100), at least 1 bed.
    pct must be within the config range (30-100); the rest of the beds serve other patients."""
    from src.allocation import load_config
    cfg = config or load_config()
    entry = cfg["bed_capacity"]["beds_available_pct"]
    lo, hi = entry["range"]
    pct = entry["value"] if beds_available_pct is None else beds_available_pct
    if isinstance(pct, bool) or not isinstance(pct, (int, float)) or not np.isfinite(pct) or not lo <= pct <= hi:
        raise ValueError(f"Beds available must be between {lo}% and {hi}%; got {pct!r}.")
    return {w: max(1, int(np.floor(c * pct / 100 + 1e-9))) for w, c in capacity.items()}


def _city_escalations(cities: dict) -> list[dict]:
    """Report outstanding bed demand and other cities' unreserved headroom."""
    by_headroom = sorted(cities, key=lambda city: (-cities[city]["headroom"], city))
    escalations = []
    for city, summary in sorted(cities.items()):
        unroutable = summary["unroutable"]
        if unroutable == 0:
            continue
        escalations.append({
            "city": city,
            "unroutable_patients": unroutable,
            "surge_beds_needed": unroutable,
            "suggestions": [
                {"city": other, "headroom": cities[other]["headroom"]}
                for other in by_headroom if other != city and cities[other]["headroom"] > 0
            ],
            "label": "requires city-level approval; not routed",
        })
    return escalations


def overflow_plan(feat_df, as_of_date, horizon: int, demand_basis: str | None = None,
                  config: dict | None = None, beds_available_pct: float | None = None) -> dict:
    """Forecast every ward (predict_all), turn cases into beds needed, route the overflow.
    `beds_available_pct` (30-100, default from config) is the share of each ward's beds free for
    dengue. Returns route_overflow()'s result plus as_of, horizon, forecast_week, demand_basis,
    beds_available_pct; each ward and city also gets full_capacity (the config rule) and
    available_capacity (after the slider; the same as "capacity", which routing used).
    Each city with unroutable patients gets an escalation with its surge bed need
    and advisory headroom from other cities.
    """
    from src.allocation import load_config, setting
    from src.predict import predict_all
    from src.supplies import case_count, supplies_for
    cfg = config or load_config()
    basis = demand_basis or setting(cfg, "demand", "basis")
    if basis not in cfg["demand"]["basis"]["allowed"]:
        raise ValueError(f"Demand basis must be one of {cfg['demand']['basis']['allowed']}; got {basis!r}.")
    pct = setting(cfg, "bed_capacity", "beds_available_pct") if beds_available_pct is None else beds_available_pct
    full = bed_capacity(cfg)
    capacity = usable_capacity(full, pct, cfg)     # check the slider before the slow part
    pred = predict_all(feat_df, as_of_date, horizon, with_drivers=False)
    beds = supplies_for(case_count(pred[basis].to_numpy()))["hospital_beds_required"].to_numpy()
    unknown = sorted(set(pred["ward_id"]) - set(capacity))
    if unknown:
        raise ValueError(f"No bed capacity for wards: {unknown}")
    wards = [{"ward_id": w, "city": c, "beds_required": int(b), "bed_capacity": capacity[w]}
             for w, c, b in zip(pred["ward_id"], pred["city"], beds)]
    result = route_overflow(wards)
    # "capacity" is what routing used (= available); full_capacity is the rule before the slider
    for ward_id, w in result["wards"].items():
        w["full_capacity"], w["available_capacity"] = full[ward_id], capacity[ward_id]
    for city, c in result["cities"].items():
        ids = [w["ward_id"] for w in wards if w["city"] == city]
        c["full_capacity"] = sum(full[i] for i in ids)
        c["available_capacity"] = sum(capacity[i] for i in ids)
    result["escalations"] = _city_escalations(result["cities"])
    result.update(as_of=str(as_of_date)[:10], horizon=int(horizon), demand_basis=basis,
                  forecast_week=str(pred["forecast_week"].iloc[0]), beds_available_pct=float(pct))
    return result
