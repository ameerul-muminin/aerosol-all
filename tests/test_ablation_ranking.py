"""Ablation removes the right feature groups; ranking metrics are correct on a hand-checked week."""
import numpy as np
import pandas as pd

from src.ablation import variants
from src.features import STATIC, feature_columns
from src.ranking_eval import week_scores


def test_ablation_variants_remove_only_their_group():
    f = feature_columns(anomaly=True)
    v = variants(f)
    assert not any(c.startswith(("rain_", "temp_", "hum_")) for c in v["without_weather"])
    assert set(STATIC) <= set(v["without_weather"]) and {"week_sin", "week_cos"} <= set(v["without_weather"])
    assert not set(STATIC) & set(v["without_ward_attributes"]) and len(v["without_ward_attributes"]) == len(f) - 3


def test_ranking_metrics_hand_checked():
    wards = [f"W{i:02d}" for i in range(60)]
    actual = np.r_[np.arange(60, 50, -1), np.zeros(50)]          # W00..W09 are the true top-10
    pred = np.r_[np.arange(60, 55, -1), np.zeros(50), np.arange(5, 0, -1)]   # predicted top-10 = W00..W04 + W55..W59
    d = pd.DataFrame({"ward_id": wards, "target": actual, "score": pred,
                      "target_date": pd.Timestamp("2024-08-05")})
    r = week_scores(d, "score")
    assert r["weeks"] == 1 and r["precision_at_10"] == 0.5 and r["top5_captured"] == 1.0
