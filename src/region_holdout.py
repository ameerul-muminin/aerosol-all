"""C4: region hold-out experiment (generalization to an unseen city).

For each city C: fit on the OTHER four cities with target weeks <= 2023 (train + val years),
evaluate on city C in 2024-2025. Holding out time as well as space keeps the experiment
free of same-week weather leakage (weather is a near-identical regional series across cities).

Compared against:
  in_distribution  same model family, trained on ALL cities <= 2023, scored on city C 2024-25
                   (the gap shows the cost of never having seen the city)
  regional_naive   city-agnostic baseline usable for a new city: mean cases per ISO week over
                   the other cities' wards (it cannot use city C's own history)

Uses the hyperparameters chosen by time-ordered CV in train.py (read from the saved bundle),
so nothing here is tuned on the hold-out data. Writes reports/region_holdout.json.
"""
from __future__ import annotations
import json
import joblib
import numpy as np
import pandas as pd

from src.features import ROOT, HORIZONS, build_dataset, feature_columns
from src.train import BUNDLE, TRAIN_END, VAL_END, deployed_model, scores, tier_thresholds, to_tier, calibrate_pred_cuts

OUT = ROOT / "reports" / "region_holdout.json"


def regional_naive(fit_rows, eval_rows):
    w_fit = fit_rows["target_date"].dt.isocalendar().week.astype(int)
    m = fit_rows.assign(w=w_fit).groupby("w")["target"].mean()
    w_eval = eval_rows["target_date"].dt.isocalendar().week.astype(int)
    return w_eval.map(m).fillna(0).to_numpy()


def critical_flagged(y, p, actual_cuts, pred_cuts):
    """Share of truly Critical ward-weeks the model flags High or Critical."""
    t, q = to_tier(y, actual_cuts), to_tier(p, pred_cuts)
    return float(np.mean(q[t == 3] >= 2)) if (t == 3).any() else None


def main():
    bundle = joblib.load(BUNDLE)
    report = {"design": "fit on 4 cities (target<=2023), test on held-out city (2024-2025)", "horizons": {}}
    for h in HORIZONS:
        cfg = bundle["horizons"][h].get("hyperparameters") or {}
        feats = bundle["horizons"][h].get("features") or feature_columns()
        d = build_dataset(h)
        past, future = d[d["target_date"] <= VAL_END], d[d["target_date"] > VAL_END]
        in_dist = deployed_model(bundle["horizons"][h]).fit(past)
        actual_cuts = tier_thresholds(past["target"])
        rows = {}
        for city in sorted(d["city"].unique()):
            fit = past[past["city"] != city]
            ev = future[future["city"] == city]
            m = deployed_model(bundle["horizons"][h]).fit(fit)
            p_hold, p_in = m.predict(ev), in_dist.predict(ev)
            # tier mapping calibrated the same way as train.py, using only the other cities:
            # fit on target years <= 2021, calibrate on their 2022-2023 predictions
            early, late = fit[fit["target_date"] <= TRAIN_END], fit[fit["target_date"] > TRAIN_END]
            pred_cuts = calibrate_pred_cuts(deployed_model(bundle["horizons"][h]).fit(early).predict(late), late["target"], actual_cuts)
            pred_cuts_in = bundle["horizons"][h]["tier_cuts"]
            y = ev["target"]
            rows[city] = {
                "n_test": len(ev), "n_wards": int(ev["ward_id"].nunique()),
                "held_out": {**scores(y, p_hold), "critical_flagged": critical_flagged(y, p_hold, actual_cuts, pred_cuts)},
                "in_distribution": {**scores(y, p_in), "critical_flagged": critical_flagged(y, p_in, actual_cuts, pred_cuts_in)},
                "regional_naive": scores(y, regional_naive(fit, ev)),
            }
        mean = {k: {m_: float(np.mean([r[k][m_] for r in rows.values()])) for m_ in ("MAE", "RMSE")}
                for k in ("held_out", "in_distribution", "regional_naive")}
        report["horizons"][h] = {"cities": rows, "mean": mean}
        print(f"h={h}w  mean MAE: held-out city {mean['held_out']['MAE']:.3f} | "
              f"in-distribution {mean['in_distribution']['MAE']:.3f} | regional naive {mean['regional_naive']['MAE']:.3f}")
        for c, r in rows.items():
            print(f"   {c:<12} held-out MAE {r['held_out']['MAE']:.2f} (in-dist {r['in_distribution']['MAE']:.2f}, "
                  f"naive {r['regional_naive']['MAE']:.2f})  Critical flagged {r['held_out']['critical_flagged']}")
    OUT.write_text(json.dumps(report, indent=2))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
