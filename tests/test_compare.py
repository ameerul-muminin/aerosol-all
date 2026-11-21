"""Baselines compute what they claim; classification metrics are correct on a hand-checked case."""
import numpy as np
import pandas as pd

from src.features import build_dataset, load_raw
from src.compare_models import Naive, LastYear
from src.evaluate import classification_metrics, outbreak_errors


def test_naive_is_cases_at_forecast_origin():
    d = build_dataset(2, cases=True)
    assert np.allclose(Naive().predict(d), d["dengue_cases"])


def test_last_year_is_same_ward_52_weeks_before_target():
    raw = load_raw()
    d = build_dataset(3, cases=True)
    row = d[(d["ward_id"] == "CCC_Ward_01") & (d["target_date"] == pd.Timestamp("2021-08-02"))]
    expect = raw[(raw["ward_id"] == "CCC_Ward_01") & (raw["date"] == pd.Timestamp("2020-08-03"))]["dengue_cases"]
    assert LastYear(raw).predict(row)[0] == expect.item()


def test_classification_metrics_hand_checked():
    cuts = [2, 9, 15]                       # Medium >= 2, High >= 9, Critical >= 15
    y = np.array([0, 0, 3, 10, 20, 20])     # true tiers: L L M H C C
    p = np.array([0, 3, 3, 10, 20, 1])      # pred tiers: L M M H C L
    m = classification_metrics(y, p, cuts, cuts)
    assert m["per_tier"]["Critical"]["recall"] == 0.5 and m["per_tier"]["Critical"]["precision"] == 1.0
    assert m["alert_high_or_critical"]["recall"] == 2 / 3
    assert np.isclose(m["accuracy"], 4 / 6)
    assert 0 <= m["critical_detection"]["average_precision"] <= 1


def test_outbreak_errors_bias_sign():
    d = pd.DataFrame({"target": [0.0, 20.0], "target_date": pd.to_datetime(["2024-01-01", "2024-08-05"])})
    out = outbreak_errors(d, np.array([0.0, 12.0]), [2, 9, 15])
    assert out["outbreak_critical"]["n"] == 1
    assert out["outbreak_critical"]["mean_bias_pred_minus_actual"] == -8.0   # under-prediction is negative
