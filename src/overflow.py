"""Route hospital bed overflow within cities."""
from numbers import Integral

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
