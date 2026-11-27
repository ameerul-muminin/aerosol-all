"""Judges: "What if the patient is sent and the hospital is full?" Planned arrivals never exceed a receiver's
beds; each transfer names a backup ward in the same city (most free beds left); a city with no free bed is full."""
import pytest

from api.main import create_app
from api.service import backup_ward

BUSY = "/api/overflow?as_of=2024-08-05&horizon=2&beds_available_pct=80"     # transfers in 4 cities
SQUEEZE = "/api/overflow?as_of=2025-07-14&horizon=2&beds_available_pct=50"  # every city over capacity


@pytest.fixture(scope="module")
def client():
    return create_app().test_client()


def test_planned_arrivals_never_exceed_beds_and_free_beds_add_up(client):
    for url in (BUSY, SQUEEZE):
        d = client.get(url).get_json()
        for w in d["wards"]:
            after = w["beds_required"] - w["routed_out"] + w["routed_in"]
            if w["routed_in"]:
                assert after <= w["capacity"], w                 # a receiver is never planned over capacity
            assert w["free_after"] == max(0, w["capacity"] - after) >= 0
        for c in d["cities"]:
            assert sum(w["free_after"] for w in d["wards"] if w["city"] == c["city"]) == c["headroom"]
            assert c["full"] == (c["headroom"] == 0)
            if c["unroutable"]:
                assert c["full"]                                  # patients are unroutable only when the city is full


def test_every_transfer_has_the_best_backup_in_its_own_city(client):
    d = client.get(BUSY).get_json()
    free = {w["ward_id"]: w for w in d["wards"]}
    assert d["transfers"] and d["if_full"]
    for t in d["transfers"]:
        others = [w for w in d["wards"] if w["city"] == t["city"] and w["free_after"] > 0
                  and w["ward_id"] not in (t["from_ward"], t["to_ward"])]
        if others:
            b = free[t["backup_ward"]]
            assert b["city"] == t["city"] and b["ward_id"] not in (t["from_ward"], t["to_ward"])
            assert t["backup_free_beds"] == b["free_after"] == max(w["free_after"] for w in others)
        else:
            assert t["backup_ward"] is None and t["backup_free_beds"] == 0
            assert next(c for c in d["cities"] if c["city"] == t["city"])["headroom"] <= 1   # only the receiver had room


def test_backup_choice_rules():
    w = lambda i, city, free: {"ward_id": i, "city": city, "free_after": free}
    wards = [w("A", "X", 0), w("B", "X", 3), w("C", "X", 2), w("D", "X", 2), w("E", "Y", 9)]
    t = {"from_ward": "A", "to_ward": "B", "city": "X"}
    assert backup_ward(t, wards) == {"backup_ward": "C", "backup_free_beds": 2}   # most free, ties by id, never B
    assert backup_ward({**t, "to_ward": "C"}, wards) == {"backup_ward": "B", "backup_free_beds": 3}
    full = [w("A", "X", 0), w("B", "X", 1), w("E", "Y", 9)]                      # never across cities
    assert backup_ward(t, full) == {"backup_ward": None, "backup_free_beds": 0}
