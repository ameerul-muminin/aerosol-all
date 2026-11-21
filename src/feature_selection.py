"""D12: feature selection decided by time-ordered CV, with a kept/removed log.

Rules (fixed before running; TRAIN period only, validation/test never used):
  1. Low variance: a feature that is constant on train is removed.
  2. Near-duplicates: for every pair with |Pearson r| > 0.95 on train (strongest first), try
     removing the member less correlated with the target; skip if its partner is already gone.
  3. A removal is kept only if the CV mean Poisson deviance (same 5 expanding yearly folds,
     same hyperparameters as the saved model) does NOT get worse. Ties favour the simpler set.

Starts from each horizon's current feature list in the saved bundle. Writes:
  models/selected_features.json         final ordered feature list per horizon (train.py uses it)
  reports/feature_selection_log.json    every candidate, its partner, r, CV before/after, decision
Run: python -m src.feature_selection    (then python -m src.train to refit on the selection)
"""
from __future__ import annotations
import json
import joblib
import numpy as np

from src.features import ROOT, HORIZONS, build_dataset, add_lag_features, load_raw
from src.train import BUNDLE, HGB, split, cross_validate

CORR_THRESHOLD = 0.95
SELECTED = ROOT / "models" / "selected_features.json"
LOG = ROOT / "reports" / "feature_selection_log.json"


def cv_dev(feats, cfg, tr):
    return cross_validate(lambda: HGB(feats, **cfg), tr)["mean"]["PoissonDev"]


def select(feats, cfg, tr):
    X = tr[feats]
    log = {"start_features": len(feats), "corr_threshold": CORR_THRESHOLD}
    std = X.std()
    low_var = [f for f in feats if not np.isfinite(std[f]) or std[f] == 0]
    current = [f for f in feats if f not in low_var]
    log["low_variance_removed"] = low_var

    corr = X[current].corr().abs()
    target_r = X[current].corrwith(tr["target"]).abs()
    pairs = [(a, b, float(corr.loc[a, b])) for i, a in enumerate(current) for b in current[i + 1:]
             if corr.loc[a, b] > CORR_THRESHOLD]
    pairs.sort(key=lambda p: -p[2])
    log["correlated_pairs"] = [{"a": a, "b": b, "r": round(r, 4)} for a, b, r in pairs]

    best = cv_dev(current, cfg, tr)
    log["cv_poisdev_start"] = best
    trials = []
    for a, b, r in pairs:
        if a not in current or b not in current:
            continue                       # partner already removed: keep the survivor
        drop, keep = (a, b) if target_r[a] < target_r[b] else (b, a)
        trial = [f for f in current if f != drop]
        dev = cv_dev(trial, cfg, tr)
        accepted = dev <= best
        trials.append({"candidate": drop, "partner": keep, "r": round(r, 4),
                       "cv_before": best, "cv_after": dev, "decision": "removed" if accepted else "kept"})
        if accepted:
            current, best = trial, dev
    log["trials"] = trials
    log["cv_poisdev_final"] = best
    log["final_features"] = len(current)
    log["removed"] = low_var + [t["candidate"] for t in trials if t["decision"] == "removed"]
    return current, log


def main():
    bundle = joblib.load(BUNDLE)
    featurized = add_lag_features(load_raw())
    selected, logs = {}, {}
    for h in HORIZONS:
        m = bundle["modes"]["weather"][h] if "modes" in bundle else bundle["horizons"][h]
        feats, cfg = list(m["features"]), m["hyperparameters"]
        tr, _, _ = split(build_dataset(h, featurized, featurized=True, cases=True))
        selected[h], logs[h] = select(feats, cfg, tr)
        lg = logs[h]
        print(f"h={h}w  {lg['start_features']} -> {lg['final_features']} features | "
              f"{len(lg['correlated_pairs'])} pairs with |r|>{CORR_THRESHOLD} | "
              f"CV PoisDev {lg['cv_poisdev_start']:.4f} -> {lg['cv_poisdev_final']:.4f} | removed: {lg['removed']}")
    SELECTED.write_text(json.dumps({str(h): f for h, f in selected.items()}, indent=2))
    LOG.write_text(json.dumps({"rules": __doc__.split("Starts from")[0].strip(), "horizons": logs}, indent=2))
    print(f"saved {SELECTED} and {LOG}")


if __name__ == "__main__":
    main()
