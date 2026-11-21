"""Split discipline: chronological train/val/test and time-ordered CV folds never look ahead."""
import pandas as pd
from src.features import build_dataset
from src.train import split, time_folds, CV_YEARS, TRAIN_END, VAL_END


def test_split_is_chronological_and_disjoint():
    tr, va, te = split(build_dataset(2))
    assert tr["target_date"].max() <= TRAIN_END < va["target_date"].min()
    assert va["target_date"].max() <= VAL_END < te["target_date"].min()
    assert len(tr) + len(va) + len(te) == len(build_dataset(2))


def test_cv_folds_train_only_on_earlier_years():
    tr, _, _ = split(build_dataset(3))
    folds = list(time_folds(tr))
    assert [y for y, _, _ in folds] == list(CV_YEARS)
    for y, fit, val in folds:
        assert fit["target_date"].dt.year.max() < y
        assert (val["target_date"].dt.year == y).all()
        assert fit.index.intersection(val.index).empty
        assert val["target_date"].max() <= TRAIN_END, "CV never touches validation/test years"


def test_selected_features_are_valid_and_ordered():
    import json
    from src.features import feature_columns
    from src.train import SELECTED
    if not SELECTED.exists():
        return
    allowed = feature_columns(anomaly=True)
    for h, feats in json.loads(SELECTED.read_text()).items():
        assert feats and len(set(feats)) == len(feats)
        assert set(feats) <= set(allowed), "selection may only remove features"
        assert feats == [f for f in allowed if f in feats], "original column order is kept"


def test_feature_list_matches_saved_models():
    import json
    import joblib
    from src.train import BUNDLE, FEATURE_LIST
    bundle, listed = joblib.load(BUNDLE), json.loads(FEATURE_LIST.read_text())
    for mode, hs in bundle["modes"].items():
        for h, m in hs.items():
            assert listed[mode][str(h)]["features"] == list(m["features"]) == list(m["model"].feats)


def test_hgb_wrapper_skips_columns_empty_in_training():
    """A CV fold trained on 2016 alone has all-NaN rain-anomaly columns; newer scikit-learn
    rejects those, so the wrapper must skip them at fit and predict."""
    import numpy as np
    from src.features import feature_columns
    from src.train import HGB
    d = build_dataset(2)
    early = d[d["target_date"].dt.year < 2017]
    feats = feature_columns(anomaly=True)
    assert early["rain_anom_lag0"].isna().all()
    m = HGB(feats, max_iter=20, max_leaf_nodes=7).fit(early)
    assert not any(f.startswith("rain_anom") for f in m.used_)
    p = m.predict(d[d["target_date"].dt.year == 2017])
    assert np.isfinite(p).all() and (p >= 0).all()
