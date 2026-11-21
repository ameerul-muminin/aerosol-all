"""Part H: weighted ensemble applies its fixed weights and is rebuildable from saved settings."""
import numpy as np

from src.features import build_dataset, feature_columns
from src.train import Ensemble, deployed_model, split


def test_weighted_ensemble_is_weighted_sum_of_members():
    tr, va, _ = split(build_dataset(2))
    tr, va = tr[tr["target_date"].dt.year >= 2019], va.iloc[:400]
    w = {"hgb": 0.6, "extra_trees": 0.0, "random_forest": 0.25, "poisson_glm": 0.15}
    m = Ensemble(feature_columns(), w, max_iter=30, max_leaf_nodes=7).fit(tr)
    assert set(m.members_) == {"hgb", "random_forest", "poisson_glm"}, "zero-weight members are skipped"
    expect = sum(w[n] * np.clip(mm.predict(va), 0, None) for n, mm in m.members_.items())
    assert np.allclose(m.predict(va), expect) and (m.predict(va) >= 0).all()


def test_deployed_model_rebuilds_ensemble():
    spec = {"kind": "weighted_ensemble", "features": feature_columns(), "hyperparameters": {"max_iter": 20},
            "ensemble_weights": {"hgb": 1.0}}
    assert isinstance(deployed_model(spec), Ensemble)
