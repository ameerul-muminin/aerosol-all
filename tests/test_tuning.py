"""Part G building blocks: tuning knobs, outbreak weights, blend, global explanations."""
import numpy as np

from src.features import build_dataset, feature_columns
from src.train import HGB, Blend, split
from src.explain import global_contributions, partial_dependence


def _data():
    tr, va, _ = split(build_dataset(2))
    return tr[tr["target_date"].dt.year >= 2019], va.iloc[:600]


def test_outbreak_weights_change_the_model():
    tr, va = _data()
    f = feature_columns()
    plain = HGB(f, max_iter=40, max_leaf_nodes=7).fit(tr).predict(va)
    weighted = HGB(f, max_iter=40, max_leaf_nodes=7, outbreak_weight=4.0, outbreak_threshold=9).fit(tr).predict(va)
    assert not np.allclose(plain, weighted)
    assert weighted.mean() > plain.mean(), "up-weighting outbreak weeks should raise predictions on average"


def test_tuning_knobs_are_accepted():
    tr, va = _data()
    m = HGB(feature_columns(), max_iter=30, max_leaf_nodes=15, learning_rate=0.1, min_samples_leaf=80,
            l2_regularization=3.0, max_features=0.7).fit(tr)
    assert np.isfinite(m.predict(va)).all()


def test_blend_predicts_valid_counts():
    tr, va = _data()
    p = Blend(feature_columns(anomaly=True), max_iter=30, max_leaf_nodes=7).fit(tr).predict(va)
    assert np.isfinite(p).all() and (p >= 0).all()


def test_global_explanations_well_formed():
    tr, va = _data()
    m = HGB(feature_columns(), max_iter=40, max_leaf_nodes=7).fit(tr)
    g = global_contributions(m, va)
    assert set(g) >= {"rainfall", "season", "ward_vulnerability"}
    pdp = partial_dependence(m, va, n_grid=5, max_rows=200)
    rain = pdp["Rain 2-4 weeks ago, total mm (breeding window)"]
    assert len(rain) >= 3 and all(np.isfinite(pt["mean_predicted_cases"]) for pt in rain)
