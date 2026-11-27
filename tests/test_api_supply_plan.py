"""GET /api/supply_plan (bonus B1): order now for weeks +2, +3 and +4, per ward, city and region.
It must agree with the allocation and overflow modules, add no model runs, and follow its order rule."""
import json

import pandas as pd
import pytest

from api import service
from api.main import create_app

A = "2025-07-14"
ITEMS = ("cases", "saline", "kits", "beds")
BASES = ("expected", "cautious")


@pytest.fixture(scope="module")
def client():
    return create_app().test_client()


@pytest.fixture(scope="module")
def plan(client):
    r = client.get(f"/api/supply_plan?as_of={A}")
    assert r.status_code == 200
    return r.get_json()


def test_three_weeks_ahead_for_every_ward(plan):
    assert [w["horizon"] for w in plan["weeks"]] == [2, 3, 4]
    assert [w["forecast_week"] for w in plan["weeks"]] == [
        (pd.Timestamp(A) + pd.Timedelta(weeks=h)).date().isoformat() for h in (2, 3, 4)]
    assert len(plan["wards"]) == 60 and plan["city"] is None
    assert {"demand"} == set(plan["assumptions"]) and plan["rule"]


def test_region_city_and_ward_totals_add_up(plan):
    for i, h in enumerate((2, 3, 4)):
        for b in BASES:
            for k in ITEMS:
                wards = sum(w["weeks"][str(h)][b][k] for w in plan["wards"])
                cities = sum(c["weeks"][i][b][k] for c in plan["by_city"])
                assert plan["weeks"][i][b][k] == wards == cities, (h, b, k)


def test_order_rule_sum_for_consumables_peak_for_beds(plan):
    for unit in (plan, *plan["by_city"], *plan["wards"]):
        weeks = unit["weeks"] if isinstance(unit["weeks"], list) else [unit["weeks"][str(h)] for h in (2, 3, 4)]
        for b in BASES:
            o = unit["order"][b]
            assert o["saline"] == sum(w[b]["saline"] for w in weeks)
            assert o["kits"] == sum(w[b]["kits"] for w in weeks)
            assert o["cases"] == sum(w[b]["cases"] for w in weeks)
            assert o["beds_peak"] == max(w[b]["beds"] for w in weeks)


def test_cautious_plan_is_never_below_expected(plan):
    for w in plan["wards"]:
        for h in ("2", "3", "4"):
            for k in ITEMS:
                assert w["weeks"][h]["cautious"][k] >= w["weeks"][h]["expected"][k], (w["ward_id"], h, k)


def test_agrees_with_allocation_and_overflow(client, plan):
    by_ward = {w["ward_id"]: w for w in plan["wards"]}
    alloc = client.post("/api/allocate", json={"as_of": A, "horizon": 3, "stock_pct": 100}).get_json()
    for r in alloc["wards"]:                        # same week, same table: allocation's need = plan's week +3
        cell = by_ward[r["ward_id"]]["weeks"]["3"]["expected"]
        assert (r["saline_required"], r["kits_required"], r["beds_required"]) == (cell["saline"], cell["kits"], cell["beds"])
    flow = client.get(f"/api/overflow?as_of={A}&horizon=2").get_json()
    for r in flow["wards"]:                         # overflow's beds needed = plan's week +2
        assert r["beds_required"] == by_ward[r["ward_id"]]["weeks"]["2"]["expected"]["beds"]


def test_city_filter_matches_that_citys_totals(client, plan):
    one = client.get(f"/api/supply_plan?as_of={A}&city=Gazipur").get_json()
    assert one["city"] == "Gazipur" and {w["city"] for w in one["wards"]} == {"Gazipur"}
    gaz = next(c for c in plan["by_city"] if c["city"] == "Gazipur")
    assert one["order"] == gaz["order"] and len(one["wards"]) == gaz["wards"]


def test_adds_no_model_runs_once_forecasts_are_cached(client):
    week = "2025-08-04"
    for h in (2, 3, 4):
        client.get(f"/api/ranking?as_of={week}&horizon={h}")      # the shared all-ward forecasts
    runs = service._ranking_df.cache_info().misses
    assert client.get(f"/api/supply_plan?as_of={week}").status_code == 200
    assert service._ranking_df.cache_info().misses == runs


@pytest.mark.parametrize("query", ["as_of=2025-07-15", "as_of=14/07/2025", "as_of=1999-01-04",
                                   f"as_of={A}&city=Atlantis"])
def test_bad_input_is_friendly_400(client, query):
    r = client.get(f"/api/supply_plan?{query}")
    assert r.status_code == 400 and set(r.get_json()) == {"error"}


def test_same_request_twice_gives_identical_json(client, plan):
    service.supply_plan.cache_clear()
    service._ranking_df.cache_clear()             # recompute the forecasts too, not just replay them
    again = client.get(f"/api/supply_plan?as_of={A}").get_json()
    assert json.dumps(again) == json.dumps(plan)
