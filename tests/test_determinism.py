"""F3: same inputs -> same output, for every Phase 2 entry point (offline determinism).

Each function is called twice with the same inputs, and once with the rows of the feature
table shuffled; all three results must be identical (compared as JSON for dicts/lists).
"""
import json

import pandas as pd
import pytest

from src.allocation import plan
from src.intervention import roi_ranking, simulate
from src.overflow import overflow_plan

AS_OF, H = "2025-07-14", 3


@pytest.fixture(scope="module")
def feat():
    from src.predict import build_features
    return build_features()


@pytest.fixture(scope="module")
def shuffled(feat):
    s = feat.sample(frac=1, random_state=11)
    assert not s.index.equals(feat.index)
    return s


def _same(a, b):
    if isinstance(a, pd.DataFrame):
        pd.testing.assert_frame_equal(a, b)
    else:
        assert json.dumps(a) == json.dumps(b)


CALLS = {
    "allocation.plan": lambda f: plan(f, AS_OF, H, stock_pct=45, crews=17),
    "overflow.overflow_plan": lambda f: overflow_plan(f, "2025-09-15", 2, beds_available_pct=70),
    "intervention.simulate": lambda f: simulate(f, AS_OF, {"DSCC_Ward_12": 2, "GCC_Ward_03": 1,
                                                          "DSCC_Ward_14": 3}, total_crews=10),
    "intervention.roi_ranking": lambda f: roi_ranking(f, AS_OF, crews_per_ward=2),
}


@pytest.mark.parametrize("name", list(CALLS))
def test_twice_gives_identical_output(feat, name):
    _same(CALLS[name](feat), CALLS[name](feat))


@pytest.mark.parametrize("name", list(CALLS))
def test_shuffled_rows_give_identical_output(feat, shuffled, name):
    _same(CALLS[name](feat), CALLS[name](shuffled))


def test_allocate_and_route_ignore_input_row_order(feat):
    """The pure functions under plan()/overflow_plan(), fed reordered rows directly."""
    from src.allocation import allocate, with_density
    from src.overflow import route_overflow
    from src.predict import predict_all
    pred = with_density(predict_all(feat, AS_OF, H, with_drivers=False), feat)
    _same(allocate(pred, 60, 12), allocate(pred.iloc[::-1], 60, 12))
    wards = [{"ward_id": w, "city": c, "beds_required": i % 5, "bed_capacity": 2}
             for i, (w, c) in enumerate(zip(pred["ward_id"], pred["city"]))]
    _same(route_overflow(wards), route_overflow(wards[::-1]))
