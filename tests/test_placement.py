"""Bonus B2 (src/placement.py): the crew plan that prevents the most cases."""
import itertools
import json
import random

import pytest

from src.intervention import roi_ranking, simulate
from src.placement import best_placement, optimise

PEAK = "2025-07-14"


def brute_force(table, crews):
    wards = sorted(table)
    best = None
    for ks in itertools.product(*[range(len(table[w])) for w in wards]):
        if sum(ks) <= crews:
            value = round(sum(table[w][k] for w, k in zip(wards, ks)), 2)
            if best is None or value > best:
                best = value
    return best


@pytest.mark.parametrize("seed", range(30))
def test_optimise_matches_brute_force_even_without_diminishing_returns(seed):
    rng = random.Random(seed)
    table = {}
    for i in range(rng.randint(1, 5)):
        gains = [round(rng.uniform(-0.5, 3), 2) for _ in range(3)]     # any shape, even negative
        table[f"W{i}"] = [0.0] + list(itertools.accumulate(gains))
    crews = rng.randint(0, 9)
    plan = optimise(table, crews)
    assert sum(plan.values()) <= crews
    assert round(sum(table[w][k] for w, k in plan.items()), 2) == brute_force(table, crews)


def test_optimise_prefers_fewer_crews_and_lower_ward_id_on_ties():
    table = {"B": [0, 2.0, 2.0], "A": [0, 2.0, 2.0]}
    assert optimise(table, 1) == {"A": 1}                 # tie: the lower ward_id
    assert optimise(table, 4) == {"A": 1, "B": 1}         # a 2nd crew adds nothing, so it is not used
    assert optimise(table, 0) == {}


# ---------------------------------------------------------------- with the real model

@pytest.fixture(scope="module")
def feat():
    from src.predict import build_features
    return build_features()


@pytest.fixture(scope="module")
def best20(feat):
    return best_placement(feat, PEAK, 20)


def test_plan_respects_the_crews_and_the_ward_cap(best20):
    from src.allocation import load_config
    cap = load_config()["crews"]["max_per_ward"]["value"]
    assert best20["crews_used"] == sum(best20["plan"].values()) <= 20
    assert max(best20["plan"].values()) <= cap
    json.dumps(best20)


def test_plan_total_is_what_the_simulator_reports(feat, best20):
    s = simulate(feat, PEAK, best20["plan"], total_crews=20)
    assert round(sum(h["cases_prevented"] for h in s["horizons"]), 2) == pytest.approx(best20["cases_prevented"], abs=0.05)


def test_beats_or_equals_the_simple_plans(feat, best20):
    assert best20["cases_prevented"] >= best20["compare"]["allocation_rule"]["cases_prevented"]
    top = roi_ranking(feat, PEAK)[:20]                     # one crew to each of the 20 best-ROI wards
    s = simulate(feat, PEAK, {w["ward_id"]: 1 for w in top}, total_crews=20)
    assert best20["cases_prevented"] >= round(sum(h["cases_prevented"] for h in s["horizons"]), 2) - 0.05


def test_more_crews_never_prevent_fewer_cases(feat):
    totals = [best_placement(feat, PEAK, c)["cases_prevented"] for c in (0, 1, 5, 10)]
    assert totals[0] == 0 and totals == sorted(totals)


def test_deterministic(feat, best20):
    assert json.dumps(best_placement(feat, PEAK, 20)) == json.dumps(best20)


def test_off_season_prevents_almost_nothing(feat):
    assert best_placement(feat, "2025-02-03", 20)["cases_prevented"] < 1


@pytest.mark.parametrize("crews", [-1, 2.5, True, "10"])
def test_bad_crews_give_friendly_errors(feat, crews):
    with pytest.raises(ValueError, match="whole number"):
        best_placement(feat, PEAK, crews)


def test_bad_date_gives_friendly_error(feat):
    with pytest.raises(ValueError, match="Mondays"):
        best_placement(feat, "2025-07-15", 5)
