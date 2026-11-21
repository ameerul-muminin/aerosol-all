"""Inference: load serialized models, compute lag features from raw weather, return risk."""
from __future__ import annotations
from functools import lru_cache
import joblib
import numpy as np
import pandas as pd

from src.features import add_lag_features, load_raw, feature_columns
from src.train import BUNDLE, TIER_NAMES, to_tier, conformal_interval


@lru_cache(maxsize=1)
def load_bundle():
    return joblib.load(BUNDLE)


@lru_cache(maxsize=1)
def _featurized():
    return add_lag_features(load_raw())


def predict(origin_date, horizon: int, wards=None) -> pd.DataFrame:
    """Forecast cases `horizon` weeks after `origin_date` using weather up to that date."""
    b = load_bundle()
    m = b["horizons"][horizon]
    d = _featurized()
    d = d[d["date"] == pd.Timestamp(origin_date)]
    if wards is not None:
        d = d[d["ward_id"].isin(wards)]
    d = d.dropna(subset=feature_columns())
    X = d[b["features"]]
    p = np.clip(m["model"].predict(X), 0, None)
    lo, hi = conformal_interval(p, m["conformal_q"])
    out = d[["ward_id", "city", "date"]].assign(
        target_date=d["date"] + pd.Timedelta(weeks=horizon), horizon_weeks=horizon,
        predicted_cases=p, lower_80=lo, upper_80=hi,
        risk_tier=[TIER_NAMES[i] for i in to_tier(p, m["tier_cuts"])])
    return out.sort_values("predicted_cases", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    import sys
    date = sys.argv[1] if len(sys.argv) > 1 else "2025-07-14"
    for h in (2, 3, 4):
        print(f"\n== origin {date}, +{h} weeks ==")
        print(predict(date, h).head(5).round(2).to_string(index=False))
