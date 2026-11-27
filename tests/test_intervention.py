"""Vector-control simulator (src/intervention.py): model-driven before/after and ROI ranking."""
import json

import pytest

from src.allocation import load_config
from src.intervention import input_floor, reduction, roi_ranking, sensitivity_configs, simulate

PEAK, QUIET = "2025-07-14", "2025-02-03"


@pytest.fixture(scope="module")
def feat():
    from src.predict import build_features
    return build_features()


@pytest.fixture(scope="module")
def ranking(feat):
    return roi_ranking(feat, PEAK)


def test_reduction_per_crew_and_cap():
    cfg = load_config()
    per = cfg["crew_effect"]["reduction_per_crew"]["value"]
    cap = cfg["crew_effect"]["max_total_reduction"]["value"]
    assert reduction(0) == 0
    assert reduction(1) == pytest.approx(per)
    assert reduction(2) == pytest.approx(2 * per)
    assert reduction(10) == pytest.approx(cap)


def test_crews_change_the_model_input_and_the_forecast_only_goes_down(feat, ranking):
    plan = {w["ward_id"]: 2 for w in ranking[:3]}
    s = simulate(feat, PEAK, plan)
    for w in s["wards"]:
        floor = input_floor(feat, "water_logging_risk_index")
        expected = max(w["input_before"] * (1 - reduction(2)), floor)
        assert w["input_after"] == pytest.approx(expected, abs=1e-4)
        for h in ("2", "3", "4"):
            v = w["by_horizon"][h]
            assert v["cases_after"] <= v["cases_before"]
        assert w["cases_prevented"] > 0
    for h in s["horizons"]:
        assert h["cases_after"] < h["cases_before"]
        assert h["beds_after"] <= h["beds_before"] and h["saline_after"] <= h["saline_before"]
        assert h["beds_freed"] == h["beds_before"] - h["beds_after"]
        # wards without crews are untouched, so the region change is the deployed wards' change
        ward_sum = sum(w["by_horizon"][str(h["horizon"])]["cases_prevented"] for w in s["wards"])
        assert h["cases_prevented"] == pytest.approx(ward_sum, abs=0.05)


def test_one_wards_crew_does_not_change_another_ward(feat, ranking):
    """roi_ranking scores all wards in one run; that is only valid if wards are independent."""
    for w in ranking[:2] + ranking[30:31]:
        alone = simulate(feat, PEAK, {w["ward_id"]: 1})["wards"][0]
        assert alone["cases_prevented"] == pytest.approx(w["cases_prevented"], abs=0.02)


def test_more_crews_prevent_more_but_each_extra_crew_less(feat, ranking):
    ward = ranking[0]["ward_id"]
    runs = [simulate(feat, PEAK, {ward: k})["wards"][0] for k in (1, 2, 3)]
    prevented = [r["cases_prevented"] for r in runs]
    assert prevented[0] < prevented[1] < prevented[2]
    assert runs[0]["roi"] > runs[2]["roi"]


def test_zero_effect_assumption_prevents_nothing(feat, ranking):
    cfg = load_config()
    cfg["crew_effect"]["reduction_per_crew"]["value"] = 0.0
    s = simulate(feat, PEAK, {ranking[0]["ward_id"]: 3}, config=cfg)
    assert all(h["cases_prevented"] == 0 for h in s["horizons"])


def test_off_season_crews_prevent_little(feat):
    r = roi_ranking(feat, QUIET)
    assert max(w["cases_prevented"] for w in r) < 0.5


def test_ranking_covers_every_ward_best_first_and_is_deterministic(feat, ranking):
    assert len(ranking) == 60
    assert [w["rank"] for w in ranking] == list(range(1, 61))
    keys = [(-w["roi"], w["ward_id"]) for w in ranking]
    assert keys == sorted(keys)
    assert all(w["crews"] == 1 and w["roi"] == w["cases_prevented"] for w in ranking)
    assert json.dumps(ranking) == json.dumps(roi_ranking(feat, PEAK))


def test_simulate_output_is_json_ready_and_complete(feat, ranking):
    s = simulate(feat, PEAK, {ranking[0]["ward_id"]: 1, ranking[1]["ward_id"]: 2}, total_crews=5)
    json.dumps(s)
    assert s["crews_used"] == 3 and s["total_crews"] == 5
    assert [h["horizon"] for h in s["horizons"]] == [2, 3, 4]
    assert [h["forecast_week"] for h in s["horizons"]] == ["2025-07-28", "2025-08-04", "2025-08-11"]
    assert s["assumption"]["model_input"] == "water_logging_risk_index"


@pytest.mark.parametrize("plan, kwargs, msg", [
    ({}, {}, "at least one ward"),
    ({"Nowhere_Ward_99": 1}, {}, "Unknown ward"),
    ({"CCC_Ward_01": 4}, {}, "0 to 3"),
    ({"CCC_Ward_01": 1.5}, {}, "0 to 3"),
    ({"CCC_Ward_01": True}, {}, "0 to 3"),
    ({"CCC_Ward_01": 0}, {}, "at least one crew"),
    ({"CCC_Ward_01": 3, "CCC_Ward_02": 3}, {"total_crews": 5}, "only 5 are available"),
])
def test_bad_plans_give_friendly_errors(feat, plan, kwargs, msg):
    with pytest.raises(ValueError, match=msg):
        simulate(feat, PEAK, plan, **kwargs)


@pytest.mark.parametrize("date, msg", [("not-a-date", "YYYY-MM-DD"), ("2030-01-07", "Pick a date"),
                                       ("2025-07-15", "Mondays")])
def test_bad_dates_give_friendly_errors(feat, date, msg):
    with pytest.raises(ValueError, match=msg):
        simulate(feat, date, {"CCC_Ward_01": 1})


@pytest.mark.parametrize("k", [0, 4, 1.0])
def test_bad_crews_per_ward_in_ranking(feat, k):
    with pytest.raises(ValueError, match="1 to 3"):
        roi_ranking(feat, PEAK, crews_per_ward=k)


def test_input_never_goes_below_the_training_minimum(feat):
    col = "water_logging_risk_index"
    train_min = feat.loc[feat["date"] <= "2021-12-31", col].min()
    assert input_floor(feat, col) == train_min
    wards = feat.drop_duplicates("ward_id").set_index("ward_id")[col]
    low = wards.idxmin()                                   # already at the floor
    near = wards[wards * (1 - reduction(3)) < train_min].drop(low).index[0]   # would cross it
    s = simulate(feat, PEAK, {low: 3, near: 3})
    by_id = {w["ward_id"]: w for w in s["wards"]}
    assert by_id[low]["input_after"] == pytest.approx(train_min)
    assert by_id[low]["cases_prevented"] == 0          # no room to lower, so no claimed effect
    assert by_id[near]["input_after"] == pytest.approx(train_min)
    assert by_id[near]["input_before"] > train_min
    for w in s["wards"]:
        assert w["input_after"] >= train_min - 1e-9
    # the floor is reported, so the UI can say "no further effect possible here"
    assert by_id[low]["floor_hit"] and by_id[near]["floor_hit"]
    assert s["floor_hit_wards"] == sorted([low, near])
    assert s["assumption"]["input_floor"] == pytest.approx(train_min)


def test_floor_not_reported_when_the_cut_stays_above_it(feat, ranking):
    s = simulate(feat, PEAK, {ranking[0]["ward_id"]: 1})
    assert s["floor_hit_wards"] == [] and not s["wards"][0]["floor_hit"]


# ---------------------------------------------------------------- sensitivity band

BAND_TOL = 0.05     # tree forecasts are rounded to 0.01 per ward and horizon


def test_sensitivity_configs_take_the_ends_of_the_range():
    cfg = load_config()
    low, high = sensitivity_configs(cfg)
    values = cfg["crew_effect"]["sensitivity_per_crew"]["value"]
    assert values == [0.05, 0.15, 0.25]
    assert low["crew_effect"]["reduction_per_crew"]["value"] == min(values)
    assert high["crew_effect"]["reduction_per_crew"]["value"] == max(values)
    assert cfg["crew_effect"]["reduction_per_crew"]["value"] == 0.15          # central untouched
    cfg["crew_effect"]["sensitivity_per_crew"]["value"] = [0.05, "lots"]
    with pytest.raises(ValueError, match="sensitivity_per_crew"):
        sensitivity_configs(cfg)


def test_simulate_reports_a_band_per_horizon(feat, ranking):
    s = simulate(feat, PEAK, {w["ward_id"]: 2 for w in ranking[:3]})
    for h in s["horizons"]:
        for key in ("cases_prevented", "beds_freed", "saline_saved"):
            low, mid, high = h[f"{key}_low"], h[key], h[f"{key}_high"]
            assert low <= mid + BAND_TOL and mid <= high + BAND_TOL, (h["horizon"], key, low, mid, high)
        assert h["cases_prevented_low"] < h["cases_prevented_high"]
    for w in s["wards"]:
        assert w["roi_low"] <= w["roi"] + BAND_TOL <= w["roi_high"] + 2 * BAND_TOL
    assert s["assumption"]["sensitivity_per_crew"] == [0.05, 0.15, 0.25]


def test_central_value_is_unchanged_by_the_band(feat, ranking):
    """The central figures must equal a run with the configured 0.15 and no band at all."""
    cfg = load_config()
    cfg["crew_effect"]["sensitivity_per_crew"]["value"] = [0.15]
    plan = {ranking[0]["ward_id"]: 2}
    a, b = simulate(feat, PEAK, plan), simulate(feat, PEAK, plan, config=cfg)
    for ha, hb in zip(a["horizons"], b["horizons"]):
        assert ha["cases_prevented"] == hb["cases_prevented"] == hb["cases_prevented_low"] == hb["cases_prevented_high"]


def test_roi_ranking_has_an_ordered_band_for_every_ward(ranking):
    assert all({"roi_low", "roi_high", "cases_prevented_low", "cases_prevented_high"} <= set(w) for w in ranking)
    for w in ranking:
        assert w["roi_low"] <= w["roi"] + BAND_TOL and w["roi"] <= w["roi_high"] + BAND_TOL, w["ward_id"]
    assert ranking[0]["roi_low"] < ranking[0]["roi"] < ranking[0]["roi_high"]


def test_band_still_respects_the_training_floor(feat):
    col = "water_logging_risk_index"
    low_ward = feat.drop_duplicates("ward_id").set_index("ward_id")[col].idxmin()
    s = simulate(feat, PEAK, {low_ward: 3})
    w = s["wards"][0]
    assert w["floor_hit"] and w["cases_prevented"] == w["cases_prevented_low"] == w["cases_prevented_high"] == 0
