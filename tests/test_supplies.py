"""Case -> supplies lookup (src/supplies.py) against the Phase 2 data."""
import numpy as np
import pandas as pd
import pytest

from src.supplies import (ONSITE_XLSX, SUPPLY_ITEMS, case_count, learn_ratios, learn_table,
                          load_model, round_half_up, supplies_for)


@pytest.fixture(scope="module")
def data():
    return pd.read_excel(ONSITE_XLSX, usecols=["dengue_cases", *SUPPLY_ITEMS])


def test_reproduces_every_row_of_the_data(data):
    got = supplies_for(data["dengue_cases"].to_numpy())
    assert len(got) == len(data) == 31_320
    for item in SUPPLY_ITEMS:
        np.testing.assert_array_equal(got[item].to_numpy(), data[item].to_numpy(), err_msg=item)


def test_ratio_alone_also_matches_every_seen_count():
    table, ratios = load_model()
    c = table.index.to_numpy()
    for item in SUPPLY_ITEMS:
        np.testing.assert_array_equal(round_half_up(ratios[item] * c), table[item].to_numpy(), err_msg=item)


def test_unseen_counts_use_the_ratio():
    table, ratios = load_model()
    for n in (35, 40, 60, 200):
        assert n not in table.index
        assert supplies_for(n) == {i: int(round_half_up(ratios[i] * n)) for i in SUPPLY_ITEMS}


def test_supplies_never_decrease_as_cases_rise():
    got = supplies_for(np.arange(0, 120))
    for item in SUPPLY_ITEMS:
        assert got[item].is_monotonic_increasing, item


def test_zero_and_negative_and_fractional_cases():
    assert supplies_for(0) == {i: 0 for i in SUPPLY_ITEMS}
    assert supplies_for(-3) == {i: 0 for i in SUPPLY_ITEMS}
    np.testing.assert_array_equal(case_count([0.49, 0.5, 2.5, 3.4]), [0, 1, 3, 3])
    assert supplies_for(11.6) == supplies_for(12)


def test_learn_table_rejects_data_that_is_not_a_function_of_cases():
    bad = pd.DataFrame({"dengue_cases": [2, 2], "saline_bags_needed": [5, 6],
                        "diagnostic_kits_needed": [2, 2], "hospital_beds_required": [1, 1]})
    with pytest.raises(ValueError, match="not a fixed function"):
        learn_table(bad)


def test_ratios_are_close_to_the_eyeballed_values():
    r = learn_ratios(load_model()[0])
    assert r["saline_bags_needed"] == pytest.approx(2.39, abs=0.02)
    assert r["diagnostic_kits_needed"] == pytest.approx(1.16, abs=0.02)
    assert r["hospital_beds_required"] == pytest.approx(0.30, abs=0.02)
