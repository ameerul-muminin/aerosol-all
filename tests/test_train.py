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
