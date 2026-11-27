import json

import pytest

from src.overflow import route_overflow


def ward(ward_id, demand, capacity, city="Dhaka North"):
    return {
        "ward_id": ward_id,
        "city": city,
        "beds_required": demand,
        "bed_capacity": capacity,
    }


def test_transfers_stay_within_city():
    result = route_overflow([
        ward("north-sender", 8, 3),
        ward("north-receiver", 0, 2),
        ward("south-receiver", 0, 10, "Dhaka South"),
    ])

    assert result["transfers"] == [{
        "from_ward": "north-sender",
        "to_ward": "north-receiver",
        "city": "Dhaka North",
        "patients": 2,
    }]
    assert result["cities"]["Dhaka South"]["headroom"] == 10


def test_patients_are_conserved_across_transfers_and_deficits():
    wards = [ward("a", 9, 2), ward("b", 6, 4), ward("c", 1, 5), ward("d", 0, 3)]
    result = route_overflow(wards)
    transferred = sum(transfer["patients"] for transfer in result["transfers"])
    summaries = result["wards"].values()

    assert transferred == sum(item["routed_out"] for item in summaries) == 7
    assert transferred == sum(item["routed_in"] for item in summaries)
    admitted = 0
    for item in summaries:
        local_load = item["beds_required"] - item["routed_out"] + item["routed_in"]
        treated = local_load - item["net_deficit"]
        assert 0 <= treated <= item["capacity"]
        admitted += treated
    assert admitted + result["cities"]["Dhaka North"]["unroutable"] == 16


def test_unroutable_overflow_is_reported_per_ward_and_city():
    result = route_overflow([ward("a", 9, 4), ward("b", 1, 3)])

    assert result["wards"]["a"] == {
        "beds_required": 9, "capacity": 4, "routed_out": 2, "routed_in": 0, "net_deficit": 3,
    }
    assert result["cities"]["Dhaka North"] == {
        "capacity": 7, "demand": 10, "utilisation_pct": 1000 / 7, "headroom": 0, "unroutable": 3,
    }


def test_routing_prioritises_current_headroom_and_breaks_ties_by_ward_id():
    wards = [ward("e", 0, 5), ward("b", 2, 0), ward("d", 0, 5), ward("a", 2, 0), ward("c", 0, 4)]
    result = route_overflow(wards)

    assert json.dumps(result) == json.dumps(route_overflow(list(reversed(wards))))
    assert [(item["from_ward"], item["to_ward"]) for item in result["transfers"]] == [
        ("a", "d"), ("b", "e"),
    ]


def test_zero_demand_with_and_without_capacity():
    result = route_overflow([ward("a", 0, 4), ward("b", 0, 0, "Gazipur")])

    assert result["transfers"] == []
    assert result["wards"]["b"] == {
        "beds_required": 0, "capacity": 0, "routed_out": 0, "routed_in": 0, "net_deficit": 0,
    }
    for city, capacity in [("Dhaka North", 4), ("Gazipur", 0)]:
        assert result["cities"][city] == {
            "capacity": capacity, "demand": 0, "utilisation_pct": 0.0,
            "headroom": capacity, "unroutable": 0,
        }


def test_largest_overflow_is_helped_first_when_the_city_runs_out():
    # A is short 1, Z is short 9, only 3 free beds in M: Z must not be the one left behind by name order
    result = route_overflow([ward("A", 4, 3), ward("Z", 12, 3), ward("M", 0, 3)])

    assert result["transfers"][0] == {"from_ward": "Z", "to_ward": "M", "city": "Dhaka North", "patients": 3}
    assert {k: v["net_deficit"] for k, v in result["wards"].items()} == {"A": 1, "M": 0, "Z": 6}
    assert result["cities"]["Dhaka North"]["unroutable"] == 7


@pytest.mark.parametrize("wards, message", [
    ([ward("a", 5, 2), ward("a", 5, 2)], "more than once"),
    ([ward("a", -3, 2)], "whole number"),
    ([ward("a", 2.5, 1)], "whole number"),
    ([ward("a", 2, True)], "whole number"),
    ([{"ward_id": "a", "city": "Gazipur", "beds_required": 1}], "missing"),
])
def test_bad_input_gives_friendly_errors(wards, message):
    with pytest.raises(ValueError, match=message):
        route_overflow(wards)


# ---------------------------------------------------------------- with the real data and model

@pytest.fixture(scope="module")
def feat():
    from src.predict import build_features
    return build_features()


def test_bed_capacity_follows_the_config_rule():
    import numpy as np
    import pandas as pd
    from src.allocation import load_config
    from src.overflow import bed_capacity
    from src.supplies import ONSITE_XLSX

    cap = bed_capacity()
    assert len(cap) == 60 and all(isinstance(c, int) and c >= 2 for c in cap.values())
    # recompute straight from the data's own hospital_beds_required column
    rule = load_config()["bed_capacity"]["rule"]["value"]
    d = pd.read_excel(ONSITE_XLSX, usecols=["date", "ward_id", "hospital_beds_required"])
    q = d[d["date"] <= rule["period_end"]].groupby("ward_id")["hospital_beds_required"].quantile(rule["quantile"])
    assert cap == {w: max(int(np.ceil(v - 1e-9)), rule["minimum_beds"]) for w, v in q.items()}


def test_lower_quantile_never_gives_more_beds():
    from src.allocation import load_config
    from src.overflow import bed_capacity
    cfg = load_config()
    cfg["bed_capacity"]["rule"]["value"]["quantile"] = 0.8
    low, high = bed_capacity(cfg), bed_capacity()
    assert all(low[w] <= high[w] for w in high)


@pytest.mark.parametrize("basis", ["predicted_cases", "range_high"])
def test_overflow_plan_on_real_forecasts(feat, basis):
    from src.overflow import overflow_plan
    r = overflow_plan(feat, "2025-08-04", 2, basis)
    assert len(r["wards"]) == 60 and r["forecast_week"] == "2025-08-18"
    city_of = feat.drop_duplicates("ward_id").set_index("ward_id")["city"]
    for t in r["transfers"]:
        assert city_of[t["from_ward"]] == city_of[t["to_ward"]] == t["city"]
    for city, c in r["cities"].items():
        wards = [w for w, city_w in city_of.items() if city_w == city]
        assert c["demand"] == sum(r["wards"][w]["beds_required"] for w in wards)
        assert c["unroutable"] == sum(r["wards"][w]["net_deficit"] for w in wards)
    assert json.dumps(r) == json.dumps(overflow_plan(feat, "2025-08-04", 2, basis))   # deterministic


def test_overflow_plan_bad_basis(feat):
    from src.overflow import overflow_plan
    with pytest.raises(ValueError, match="Demand basis"):
        overflow_plan(feat, "2025-08-04", 2, "range_mid")


# ---------------------------------------------------------------- beds available for dengue (slider)

def test_usable_capacity_floors_and_keeps_one_bed():
    from src.overflow import usable_capacity
    assert usable_capacity({"a": 5, "b": 2, "c": 1}, 50) == {"a": 2, "b": 1, "c": 1}
    assert usable_capacity({"a": 5, "b": 3}, 100) == {"a": 5, "b": 3}
    assert usable_capacity({"a": 10}, 30) == {"a": 3}


@pytest.mark.parametrize("pct", [29, 101, "50", True, float("nan")])
def test_beds_available_pct_out_of_range(feat, pct):
    from src.overflow import overflow_plan
    with pytest.raises(ValueError, match="Beds available"):
        overflow_plan(feat, "2025-08-04", 2, beds_available_pct=pct)


def test_default_beds_available_is_100_pct(feat):
    from src.overflow import overflow_plan
    a, b = overflow_plan(feat, "2025-08-04", 2), overflow_plan(feat, "2025-08-04", 2, beds_available_pct=100)
    assert a == b and a["beds_available_pct"] == 100.0


@pytest.mark.parametrize("date, pct", [("2024-08-19", 50), ("2025-09-15", 70)])
def test_fewer_beds_produces_intra_city_transfers(feat, date, pct):
    """A whole city is often over capacity at once (dengue peaks are city-wide), so the test
    uses weeks where some wards still have room: found by scanning Jun 2024 - Nov 2025."""
    from src.overflow import bed_capacity, overflow_plan, usable_capacity
    r = overflow_plan(feat, date, 2, beds_available_pct=pct)
    assert r["transfers"]
    city_of = feat.drop_duplicates("ward_id").set_index("ward_id")["city"]
    for t in r["transfers"]:
        assert city_of[t["from_ward"]] == city_of[t["to_ward"]] == t["city"]
    cap = usable_capacity(bed_capacity(), pct)
    assert all(w["capacity"] == cap[k] for k, w in r["wards"].items())
    for k, w in r["wards"].items():                       # nobody is sent past a receiver's capacity
        assert w["beds_required"] - w["routed_out"] + w["routed_in"] - w["net_deficit"] <= w["capacity"]


def test_70_pct_on_2025_09_15_routes_every_overflow_patient(feat):
    from src.overflow import overflow_plan
    r = overflow_plan(feat, "2025-09-15", 2, beds_available_pct=70)
    assert sum(t["patients"] for t in r["transfers"]) == 5
    assert sum(c["unroutable"] for c in r["cities"].values()) == 0


def test_full_and_available_capacity_are_reported_per_ward_and_city(feat):
    from src.overflow import bed_capacity, overflow_plan, usable_capacity
    r = overflow_plan(feat, "2024-08-19", 2, beds_available_pct=50)
    full, avail = bed_capacity(), usable_capacity(bed_capacity(), 50)
    assert r["beds_available_pct"] == 50.0
    for k, w in r["wards"].items():
        assert (w["full_capacity"], w["available_capacity"], w["capacity"]) == (full[k], avail[k], avail[k])
    for city, c in r["cities"].items():
        ids = [k for k in r["wards"] if feat.loc[feat["ward_id"] == k, "city"].iloc[0] == city]
        assert c["full_capacity"] == sum(full[i] for i in ids)
        assert c["available_capacity"] == c["capacity"] == sum(avail[i] for i in ids)
        assert c["available_capacity"] < c["full_capacity"]


def test_100_pct_gives_the_same_routing_as_before_the_slider(feat):
    """'Before' = routing straight on the full capacity rule, without the slider."""
    from src.overflow import bed_capacity, overflow_plan, route_overflow
    from src.predict import predict_all
    from src.supplies import case_count, supplies_for
    r = overflow_plan(feat, "2025-08-04", 2, beds_available_pct=100)
    pred = predict_all(feat, "2025-08-04", 2, with_drivers=False)
    beds = supplies_for(case_count(pred["predicted_cases"].to_numpy()))["hospital_beds_required"]
    cap = bed_capacity()
    old = route_overflow([{"ward_id": w, "city": c, "beds_required": int(b), "bed_capacity": cap[w]}
                          for w, c, b in zip(pred["ward_id"], pred["city"], beds)])
    assert r["transfers"] == old["transfers"]
    for k, w in old["wards"].items():
        assert {key: r["wards"][k][key] for key in w} == w
        assert r["wards"][k]["full_capacity"] == r["wards"][k]["available_capacity"]
    for city, c in old["cities"].items():
        assert {key: r["cities"][city][key] for key in c} == c


def test_patients_are_conserved_with_fewer_beds(feat):
    from src.overflow import overflow_plan
    r = overflow_plan(feat, "2024-08-19", 2, beds_available_pct=50)
    assert r["transfers"]
    wards = r["wards"].values()
    assert sum(t["patients"] for t in r["transfers"]) == sum(w["routed_out"] for w in wards) \
        == sum(w["routed_in"] for w in wards)
    admitted = sum(w["beds_required"] - w["routed_out"] + w["routed_in"] - w["net_deficit"] for w in wards)
    stranded = sum(c["unroutable"] for c in r["cities"].values())
    assert admitted + stranded == sum(w["beds_required"] for w in wards)


def test_2025_07_14_at_50_pct_cannot_route_because_every_ward_is_full(feat):
    """Asked for as a 'produces transfers' test, but it cannot: at 50% no ward in any city has a
    free bed, so nobody can be moved. Pinned here so the claim is never made by mistake."""
    from src.overflow import overflow_plan
    r = overflow_plan(feat, "2025-07-14", 2, beds_available_pct=50)
    assert not any(w["available_capacity"] > w["beds_required"] for w in r["wards"].values())
    assert r["transfers"] == []
    assert sum(c["unroutable"] for c in r["cities"].values()) == \
        sum(max(0, w["beds_required"] - w["available_capacity"]) for w in r["wards"].values())
