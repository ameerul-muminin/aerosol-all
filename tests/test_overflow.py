import json

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
