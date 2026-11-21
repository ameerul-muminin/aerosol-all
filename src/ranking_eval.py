"""Ward-ranking accuracy: does the model put the right wards at the top? (report only)

Public health officers act on the RANKING of wards, not exact counts. For every target week in the
outbreak season (Jul-Oct), compare the 60-ward ranking from the train-only deployed model with the
actual case counts:
  precision_at_10   share of the predicted top-10 that are in the actual top-10
                    (actual top-10 = wards with cases >= the 10th-highest count, so ties are kept)
  top5_captured     share of the actual top-5 wards that appear in the predicted top-10
  spearman          rank correlation across all 60 wards
Reference rankings: seasonal average (ward x week-of-year mean over training years), static
vulnerability (water-logging index), and random (expected precision@10 = 10/60).
Reported for validation (2022-23) and test (2024-25); nothing is chosen from these numbers.
Writes reports/ranking_eval.json. Run: python -m src.ranking_eval
"""
from __future__ import annotations
import json
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.features import ROOT, HORIZONS, build_dataset, add_lag_features, load_raw
from src.train import BUNDLE, SeasonalNaive, deployed_model, split

OUT = ROOT / "reports" / "ranking_eval.json"
K = 10


def week_scores(df, score_col):
    rows = []
    for _, g in df[df["target_date"].dt.month.between(7, 10)].groupby("target_date"):
        actual, pred = g["target"].to_numpy(), g[score_col].to_numpy()
        if actual.max() == 0:
            continue
        top_pred = set(g["ward_id"].iloc[np.argsort(-pred, kind="stable")[:K]])
        top_actual = set(g["ward_id"][actual >= np.sort(actual)[-K]])
        top5 = set(g["ward_id"][actual >= np.sort(actual)[-5]])
        rho = spearmanr(pred, actual).correlation if np.std(pred) > 0 else np.nan
        rows.append({"precision_at_10": len(top_pred & top_actual) / K,
                     "top5_captured": len(top_pred & top5) / len(top5), "spearman": rho})
    w = pd.DataFrame(rows)
    return {"weeks": int(len(w)), **{c: float(w[c].mean()) for c in ("precision_at_10", "top5_captured", "spearman")}}


def main():
    bundle = joblib.load(BUNDLE)
    featurized = add_lag_features(load_raw())
    report = {"what": __doc__.split("Writes")[0].strip(), "random_precision_at_10": K / 60, "horizons": {}}
    for h in HORIZONS:
        m = bundle["modes"]["weather"][h]
        tr, va, te = split(build_dataset(h, featurized, featurized=True, cases=True))
        model, naive = deployed_model(m).fit(tr), SeasonalNaive().fit(tr)
        out = {}
        for name, d in (("validation", va), ("test", te)):
            d = d.assign(model=model.predict(d), seasonal=naive.predict(d), vulnerability=d["water_logging_risk_index"])
            out[name] = {s: week_scores(d, s) for s in ("model", "seasonal", "vulnerability")}
        report["horizons"][h] = out
        for name in ("validation", "test"):
            r = out[name]
            print(f"h={h}w {name:<10} precision@10: model {r['model']['precision_at_10']:.2f} | seasonal "
                  f"{r['seasonal']['precision_at_10']:.2f} | vulnerability {r['vulnerability']['precision_at_10']:.2f} "
                  f"| random {K/60:.2f}   top-5 captured: model {r['model']['top5_captured']:.2f}   "
                  f"spearman {r['model']['spearman']:.2f}", flush=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
