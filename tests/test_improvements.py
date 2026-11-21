"""Improvement round 2 building blocks: weights, isotonic wrapper, season-aware ranges."""
import numpy as np
import pandas as pd

from src.features import build_dataset, feature_columns
from src.train import Calibrated, HGB, conformal_fit, conformal_interval, split, split_weight_params


def test_recency_ramp_and_combined_weights():
    d = pd.DataFrame({"target": [0, 20, 0, 20],
                      "target_date": pd.to_datetime(["2016-06-06", "2016-08-01", "2021-06-07", "2021-08-02"])})
    params, w = split_weight_params({"max_iter": 10, "recency_weight": 2.0}, d)
    assert params == {"max_iter": 10}
    assert np.allclose(w, [1, 1, 3, 3]), "oldest year 1, newest 1 + r"
    _, w2 = split_weight_params({"recency_weight": 2.0, "outbreak_weight": 4.0, "outbreak_threshold": 9}, d)
    assert np.allclose(w2, [1, 4, 3, 12]), "weights multiply"
    assert split_weight_params({"max_iter": 10}, d)[1] is None


def test_calibrated_wrapper_predicts_valid_counts():
    tr, va, _ = split(build_dataset(2))
    m = Calibrated(HGB, feature_columns(), max_iter=30, max_leaf_nodes=7).fit(tr)
    p = m.predict(va.iloc[:500])
    assert np.isfinite(p).all() and (p >= 0).all()
    assert np.all(np.diff(m.iso.predict(np.linspace(0, 20, 50))) >= 0), "recalibration is monotone"


def test_season_aware_ranges_and_fallback():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 12, 4000)
    months = rng.integers(1, 13, 4000)
    y = rng.poisson(p * np.where(np.isin(months, [7, 8, 9, 10]), 1.6, 1.0))   # peak months noisier
    q = conformal_fit(p, y, 0.80, months=months)
    assert any(isinstance(k, str) for k in q) and any(isinstance(k, int) for k in q)
    lo_peak, hi_peak = conformal_interval(np.array([8.0]), q, months=np.array([8]))
    lo_off, hi_off = conformal_interval(np.array([8.0]), q, months=np.array([2]))
    assert hi_peak[0] - lo_peak[0] > hi_off[0] - lo_off[0], "peak-season range is wider"
    lo, hi = conformal_interval(np.array([8.0]), q)               # no months -> bin-only fallback
    assert lo[0] <= 8.0 <= hi[0]
