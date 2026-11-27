"""Constrained allocation (src/allocation.py) and its config."""
import numpy as np
import pandas as pd
import pytest

from src.allocation import ITEM_KEYS, allocate, available_stock, load_config, plan, totals

TIERS = ["Low", "Medium", "High", "Critical"]
KEYS = list(ITEM_KEYS.values())


def make_pred(n=12, seed=0):
    rng = np.random.default_rng(seed)
    cases = np.round(rng.gamma(1.5, 6, n), 2)
    return pd.DataFrame({
        "ward_id": [f"W_{i:02d}" for i in range(n)],
        "city": rng.choice(["Dhaka North", "Gazipur"], n),
        "predicted_cases": cases,
        "range_low": np.round(cases * 0.5, 2),
        "range_high": np.round(cases * 1.6 + 1, 2),
        "risk_tier": rng.choice(TIERS, n),
        "population_density_sqkm": rng.integers(18_000, 75_000, n),
    })


def short_wards(a, k):
    return list(a.loc[a[f"{k}_deficit"] > 0, "priority_rank"])


# ---------------------------------------------------------------- supplies

def test_full_stock_means_no_deficit_and_nothing_critical():
    a = allocate(make_pred(), 100, crews=5)
    for k in KEYS:
        assert (a[f"{k}_allocated"] == a[f"{k}_required"]).all()
        assert (a[f"{k}_fulfilled_pct"] == 100).all()
    assert not a["critical"].any()


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("pct", [30, 37.5, 50, 64, 85, 99, 100])
def test_totals_never_exceed_stock(seed, pct):
    a = allocate(make_pred(25, seed), pct, crews=7)
    t = totals(a, pct, crews=7)
    for k in KEYS:
        assert (a[f"{k}_allocated"] >= 0).all()
        assert (a[f"{k}_allocated"] <= a[f"{k}_required"]).all()
        assert t["items"][k]["allocated"] <= t["items"][k]["available"]
        assert (a[f"{k}_required"] - a[f"{k}_allocated"] == a[f"{k}_deficit"]).all()
    assert t["crews"]["assigned"] <= 7


@pytest.mark.parametrize("seed", range(5))
def test_scarcity_hits_only_the_lowest_priority_wards(seed):
    a = allocate(make_pred(25, seed), 30, crews=0)
    for k in KEYS:
        short = short_wards(a, k)
        assert short, f"30% stock should leave someone short of {k}"
        # after the first short ward, every ward that needs this item is short too
        tail = a[a["priority_rank"] >= short[0]]
        assert (tail.loc[tail[f"{k}_required"] > 0, f"{k}_deficit"] > 0).all()
        # and everyone above it is fully served
        assert (a.loc[a["priority_rank"] < short[0], f"{k}_deficit"] == 0).all()


def test_stock_is_used_up_when_short():
    pct = 40
    a = allocate(make_pred(25, 1), pct, crews=0)
    t = totals(a, pct, crews=0)
    for k in KEYS:
        assert t["items"][k]["allocated"] == t["items"][k]["available"]


def test_priority_is_tier_times_density():
    p = pd.DataFrame({"ward_id": ["A", "B", "C"], "city": "Gazipur", "predicted_cases": [5.0, 5.0, 5.0],
                      "risk_tier": ["Critical", "Low", "High"],
                      "population_density_sqkm": [20_000, 70_000, 30_000]})
    a = allocate(p, 100, crews=0)
    # C: 3 x 30k = 90k  >  A: 4 x 20k = 80k  >  B: 1 x 70k = 70k
    assert list(a["ward_id"]) == ["C", "A", "B"]
    assert list(a["priority_score"]) == [90_000, 80_000, 70_000]


def test_ties_broken_by_ward_id_and_input_order_does_not_matter():
    p = make_pred(20, 3)
    p["risk_tier"], p["population_density_sqkm"] = "High", 40_000          # all tied
    a = allocate(p, 45, crews=9)
    assert list(a["ward_id"]) == sorted(p["ward_id"])
    b = allocate(p.sample(frac=1, random_state=7), 45, crews=9)
    pd.testing.assert_frame_equal(a, b)


def test_critical_flag_when_any_item_under_100():
    a = allocate(make_pred(25, 2), 55, crews=0)
    pct = a[[f"{k}_fulfilled_pct" for k in KEYS]]
    assert (a["critical"] == (pct < 100).any(axis=1)).all()
    assert (a["fulfilled_pct"] == pct.min(axis=1)).all()
    assert a["critical"].any()


def test_ward_with_zero_need_counts_as_fulfilled():
    p = make_pred(5)
    p.loc[0, "predicted_cases"] = 0.2
    a = allocate(p, 30, crews=0).set_index("ward_id")
    assert a.loc["W_00", "demand_cases"] == 0
    assert a.loc["W_00", "saline_fulfilled_pct"] == 100 and not a.loc["W_00", "critical"]


def test_range_high_basis_plans_for_more():
    p = make_pred(15, 4)
    mid = allocate(p, 100, 0).set_index("ward_id")
    high = allocate(p, 100, 0, demand_basis="range_high").set_index("ward_id")
    assert (high["saline_required"] >= mid["saline_required"]).all()
    assert high["saline_required"].sum() > mid["saline_required"].sum()


def test_fixed_warehouse_stock_from_config():
    cfg = load_config()
    cfg["stock"]["full_stock_units"]["value"]["saline_bags_needed"] = 50
    assert available_stock({"saline_bags_needed": 999, "diagnostic_kits_needed": 10,
                            "hospital_beds_required": 4}, 50, cfg) == \
        {"saline_bags_needed": 25, "diagnostic_kits_needed": 5, "hospital_beds_required": 2}
    a = allocate(make_pred(25, 5), 50, crews=0, config=cfg)
    assert a["saline_allocated"].sum() <= 25


# ---------------------------------------------------------------- crews

@pytest.mark.parametrize("crews", [0, 1, 4, 13, 1000])
def test_crews_never_exceed_the_total_or_the_ward_cap(crews):
    p = make_pred(20, 6)
    a = allocate(p, 80, crews=crews)
    cfg = load_config()
    cap, min_cases = cfg["crews"]["max_per_ward"]["value"], cfg["crews"]["min_predicted_cases"]["value"]
    assert a["crews_assigned"].sum() <= crews
    assert a["crews_assigned"].max() <= cap
    assert (a.loc[a["predicted_cases"] < min_cases, "crews_assigned"] == 0).all()
    eligible = int((a["predicted_cases"] >= min_cases).sum())
    assert a["crews_assigned"].sum() == min(crews, eligible * cap)


def test_first_crew_goes_to_highest_priority_times_cases():
    p = pd.DataFrame({"ward_id": ["A", "B", "C"], "city": "Gazipur", "predicted_cases": [2.0, 10.0, 0.4],
                      "risk_tier": ["Critical", "Medium", "Critical"],
                      "population_density_sqkm": [60_000, 30_000, 70_000]})
    a = allocate(p, 100, crews=1).set_index("ward_id")
    # A: 240k x 2 = 480k, B: 60k x 10 = 600k, C below 1 case -> not eligible
    assert a["crews_assigned"].to_dict() == {"B": 1, "A": 0, "C": 0}
    a = allocate(p, 100, crews=2).set_index("ward_id")
    assert a["crews_assigned"].to_dict() == {"B": 1, "A": 1, "C": 0}   # B's 2nd crew = 300k < A's 480k


# ---------------------------------------------------------------- bad input

@pytest.mark.parametrize("kwargs, msg", [
    ({"stock_pct": 29}, "between 30% and 100%"),
    ({"stock_pct": 101}, "between 30% and 100%"),
    ({"stock_pct": "60"}, "must be a number"),
    ({"stock_pct": float("nan")}, "must be a number"),
    ({"stock_pct": 60, "crews": -1}, "whole number"),
    ({"stock_pct": 60, "crews": 2.5}, "whole number"),
    ({"stock_pct": 60, "demand_basis": "range_mid"}, "Demand basis"),
])
def test_bad_input_gives_friendly_errors(kwargs, msg):
    with pytest.raises(ValueError, match=msg):
        allocate(make_pred(), **kwargs)


def test_bad_tables_give_friendly_errors():
    p = make_pred()
    with pytest.raises(ValueError, match="missing columns"):
        allocate(p.drop(columns="population_density_sqkm"), 60)
    with pytest.raises(ValueError, match="once"):
        allocate(pd.concat([p, p.head(1)]), 60)
    with pytest.raises(ValueError, match="Unknown risk tier"):
        allocate(p.assign(risk_tier="Extreme"), 60)
    with pytest.raises(ValueError, match="No ward forecasts"):
        allocate(p.head(0), 60)


# ---------------------------------------------------------------- config

def test_every_assumption_has_a_reason_and_a_kind():
    cfg = load_config()
    entries = [(s, k, v) for s, sec in cfg.items() if not s.startswith("_") for k, v in sec.items()]
    assert len(entries) >= 10
    for s, k, v in entries:
        assert {"value", "kind", "why"} <= set(v), f"{s}.{k}"
        assert v["kind"] in {"assumption", "data_rule"}, f"{s}.{k}"
        assert len(v["why"]) > 40, f"{s}.{k} needs a real reason"


# ---------------------------------------------------------------- with the real model

@pytest.fixture(scope="module")
def feat():
    from src.predict import build_features
    return build_features()


def test_plan_on_real_forecasts(feat):
    full = plan(feat, "2025-07-14", 3, 100, crews=20)
    assert len(full) == feat["ward_id"].nunique() == 60
    assert not full["critical"].any()
    assert full["crews_assigned"].sum() == 20
    short = plan(feat, "2025-07-14", 3, 30, crews=20)
    t = totals(short, 30, crews=20)
    for k in KEYS:
        assert t["items"][k]["allocated"] <= t["items"][k]["available"]
    assert short["critical"].any()
    # same call twice -> identical (offline determinism)
    pd.testing.assert_frame_equal(short, plan(feat, "2025-07-14", 3, 30, crews=20))


def test_zero_minimum_share_preserves_greedy_fill():
    predictions = make_pred(25)
    default = allocate(predictions, 60)
    explicit = allocate(predictions, 60, min_share_pct=0)
    pd.testing.assert_frame_equal(default, explicit)
    summary = totals(explicit, 60)
    assert summary["min_share_pct"] == 0
    for item in KEYS:
        available = summary["items"][item]["available"]
        for row in explicit.to_dict("records"):
            expected = min(available, row[f"{item}_required"])
            assert row[f"{item}_allocated"] == expected
            assert row[f"{item}_guaranteed"] == 0
            available -= expected


def test_minimum_share_and_priority_at_sixty_percent_stock():
    predictions = make_pred(25, 3)
    predictions["risk_tier"], predictions["population_density_sqkm"] = "High", 40_000
    allocation = allocate(predictions, 60, min_share_pct=20)
    shuffled = predictions.sample(frac=1, random_state=7)
    pd.testing.assert_frame_equal(allocation, allocate(shuffled, 60, min_share_pct=20))
    summary = totals(allocation, 60)
    assert summary["min_share_pct"] == 20
    for item in KEYS:
        required = allocation[f"{item}_required"]
        guaranteed = allocation[f"{item}_guaranteed"]
        allocated = allocation[f"{item}_allocated"]
        # Whole units round down, including for wards that need fewer than five.
        pd.testing.assert_series_equal(guaranteed, required // 5, check_names=False)
        assert (guaranteed <= allocated).all()
        assert (allocated <= required).all()
        assert allocated.sum() <= summary["items"][item]["available"]
        residual = required - guaranteed
        top_up = allocated - guaranteed
        second_pass_fill = top_up[residual > 0] / residual[residual > 0]
        assert second_pass_fill.is_monotonic_decreasing


def test_minimum_share_cannot_exceed_fixed_stock():
    cfg = load_config()
    cfg["priority"]["min_share_pct"]["value"] = 20
    cfg["stock"]["full_stock_units"]["value"]["saline_bags_needed"] = 4
    predictions = make_pred(3).assign(predicted_cases=20)
    allocation = allocate(predictions, 60, config=cfg)
    assert allocation["saline_guaranteed"].tolist() == [2, 0, 0]
    assert allocation["saline_allocated"].tolist() == [2, 0, 0]
    assert totals(allocation, 60, config=cfg)["min_share_pct"] == 20


@pytest.mark.parametrize("pct", [-1, 51, "20", True, float("nan")])
def test_invalid_minimum_share(pct):
    with pytest.raises(ValueError, match="Minimum share must be a number between 0% and 50%"):
        allocate(make_pred(), 60, min_share_pct=pct)


def test_plan_passes_minimum_share_through(feat):
    allocation = plan(feat, "2025-07-14", 2, 60, min_share_pct=20)
    assert totals(allocation, 60)["min_share_pct"] == 20
    assert allocation["saline_guaranteed"].sum() > 0
