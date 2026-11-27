"""Bonus B2: optimal crew placement - which wards get the crews to prevent the most cases?

    from src.predict import build_features
    from src.placement import best_placement
    best_placement(build_features(), "2025-07-14", crews=20)

How:
1. A crew changes only its own ward's forecast (tests/test_intervention.py proves it), so each
   ward's cases prevented with 1, 2, ... max_per_ward crews can be measured for all 60 wards in
   one batched model run per crew count (src/intervention.py, same crew effect and floor).
2. Choosing how many crews each ward gets is then a small knapsack: maximise the total cases
   prevented over weeks +2..+4 with at most `crews` crews. `optimise()` solves it exactly by
   dynamic programming (60 wards x crews x 4 choices), so the plan is the best possible under
   the model and the configured crew effect - not a greedy guess, even if gains are not
   diminishing. Ties: fewer crews first, then the lower ward_id gets the crew.
3. The result is compared with the allocation engine's crew rule (src/allocation.assign_crews)
   on the same table, and can be re-checked with intervention.simulate().
"""
from __future__ import annotations

from numbers import Integral

import pandas as pd

from src.allocation import load_config, setting
from src.intervention import _as_of, _before_after, _ward_row

SCALE = 100      # cases are rounded to 0.01; integers make the optimisation exact and tie-safe


def prevented_table(feat_df: pd.DataFrame, as_of, config: dict | None = None) -> tuple[dict, dict]:
    """({ward_id: [0, cases prevented with 1 crew, with 2, ...]}, {ward_id: city}).
    Cases prevented are summed over weeks +2, +3, +4, exactly as intervention.simulate reports them."""
    cfg = config or load_config()
    cap = int(setting(cfg, "crews", "max_per_ward"))
    wards = sorted(feat_df.loc[feat_df["date"] == as_of, "ward_id"].unique())
    table = {w: [0.0] for w in wards}
    city = {}
    for k in range(1, cap + 1):
        t = _before_after(feat_df, as_of, pd.Series(k, index=wards), cfg)
        for w, r in t.iterrows():
            table[w].append(_ward_row(w, r)["cases_prevented"])
            city[w] = r["city"]
    return table, city


def optimise(table: dict[str, list[float]], crews: int) -> dict[str, int]:
    """Exact best split of `crews` over wards. table[w][k] = cases prevented with k crews
    (table[w][0] == 0). Returns {ward_id: crews} for wards that get at least one crew."""
    wards = sorted(table)
    gain = {w: [round(v * SCALE) for v in table[w]] for w in wards}
    # best[c] = (total gain, -crews used) using exactly the wards seen so far and at most c crews
    best = [(0, 0)] * (crews + 1)
    choice = []
    for w in wards:
        new, pick = [], []
        for c in range(crews + 1):
            options = [((best[c - k][0] + gain[w][k], best[c - k][1] - k), -k, k)
                       for k in range(min(c, len(gain[w]) - 1) + 1)]
            value, _, k = max(options)                 # more cases, then fewer crews, then fewer here
            new.append(value)
            pick.append(k)
        best, choice = new, choice + [pick]
    plan, c = {}, crews
    for w, pick in zip(reversed(wards), reversed(choice)):
        k = pick[c]
        if k:
            plan[w] = k
        c -= k
    return dict(sorted(plan.items()))


def _total(table: dict, plan: dict[str, int]) -> float:
    return round(sum(table[w][k] for w, k in plan.items()), 2)


def best_placement(feat_df: pd.DataFrame, as_of_date, crews: int | None = None, horizon: int = 2,
                   config: dict | None = None) -> dict:
    """The crew plan that prevents the most cases over weeks +2..+4 (JSON-ready):
    plan {ward_id: crews}, crews_used, cases_prevented, per-ward rows, and the same table's
    total for the allocation engine's crew rule at `horizon` (for comparison).
    At most wards x max_per_ward crews can be placed: a bigger request is capped, and
    crews_requested, crews_placeable and `note` say so (crews_available is the capped number).
    Bad input raises ValueError with a message fit to show the user."""
    cfg = config or load_config()
    if crews is None:
        crews = int(setting(cfg, "crews", "total_default"))
    if isinstance(crews, bool) or not isinstance(crews, Integral) or crews < 0:
        raise ValueError("Number of spraying crews must be a whole number, 0 or more.")
    as_of = _as_of(feat_df, as_of_date)
    table, city = prevented_table(feat_df, as_of, cfg)
    cap = len(next(iter(table.values()))) - 1
    requested, placeable = int(crews), cap * len(table)
    crews = min(requested, placeable)                  # more crews than wards x cap cannot be placed
    plan = optimise(table, crews)

    from src.allocation import plan as allocation_plan
    rule = allocation_plan(feat_df, as_of, horizon, 100, crews, config=cfg)
    rule_plan = {w: int(k) for w, k in zip(rule["ward_id"], rule["crews_assigned"]) if k}

    rows = [{"ward_id": w, "city": city[w], "crews": k, "cases_prevented": table[w][k],
             "roi": round(table[w][k] / k, 3), "next_crew_gain": round(table[w][k + 1] - table[w][k], 2)
             if k < cap else None} for w, k in plan.items()]
    rows.sort(key=lambda r: (-r["cases_prevented"], r["ward_id"]))
    best, by_rule = _total(table, plan), _total(table, rule_plan)
    note = (f"Only {placeable} crews can be placed ({len(table)} wards x {cap} per ward); "
            f"the other {requested - placeable} were left out." if requested > placeable else None)
    return {"as_of": str(as_of.date()), "crews_available": crews, "crews_requested": requested,
            "crews_placeable": placeable, "note": note, "crews_used": sum(plan.values()),
            "plan": plan, "cases_prevented": best, "wards": rows,
            "compare": {"allocation_rule": {"plan": rule_plan, "cases_prevented": by_rule,
                                            "horizon": horizon},
                        "gain_over_allocation_rule": round(best - by_rule, 2)},
            "method": "exact knapsack over each ward's cases prevented with 0..max_per_ward crews "
                      "(weeks +2..+4, configured crew effect); wards are independent"}
