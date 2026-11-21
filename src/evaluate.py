"""Validation data for the dashboard (bonus: chronological validation error dashboard).

Writes:
  reports/validation_errors.csv
      MAE / RMSE / n by source x horizon x city x year x month, where source is each
      time-ordered CV fold (cv_2017 .. cv_2021), the validation block (2022-23) and the test
      block (2024-25). sum_abs_err and sum_sq_err are included so the dashboard can
      re-aggregate any slice exactly (e.g. all cities, whole year).
  reports/feature_importance.json
      Grouped permutation importance on VALIDATION with the train-only model: shuffle one
      driver group's columns together, measure the rise in Poisson deviance (5 repeats).

Uses the deployed weather-mode hyperparameters and feature lists from the saved bundle;
it never changes the model or any threshold.
"""
from __future__ import annotations
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_poisson_deviance

from src.features import ROOT, HORIZONS, build_dataset, add_lag_features, load_raw
from src.train import BUNDLE, HGB, split, time_folds
from src.explain import group_of, GROUP_ORDER

ERRORS = ROOT / "reports" / "validation_errors.csv"
IMPORTANCE = ROOT / "reports" / "feature_importance.json"
SEED = 0


def _rows(source, kind, h, d, pred):
    e = d["target"].to_numpy() - pred
    return pd.DataFrame({"source": source, "kind": kind, "horizon_weeks": h, "city": d["city"].to_numpy(),
                         "year": d["target_date"].dt.year.to_numpy(), "month": d["target_date"].dt.month.to_numpy(),
                         "abs_err": np.abs(e), "sq_err": e ** 2})


def _aggregate(raw):
    g = raw.groupby(["source", "kind", "horizon_weeks", "city", "year", "month"], as_index=False)
    out = g.agg(n=("abs_err", "size"), sum_abs_err=("abs_err", "sum"), sum_sq_err=("sq_err", "sum"))
    out["mae"] = out["sum_abs_err"] / out["n"]
    out["rmse"] = np.sqrt(out["sum_sq_err"] / out["n"])
    return out


def permutation_importance(model, d, repeats=5):
    rng = np.random.default_rng(SEED)
    y = d["target"].to_numpy()
    base = mean_poisson_deviance(y, np.clip(model.predict(d), 1e-3, None))
    groups = [g for g in GROUP_ORDER if any(group_of(f) == g for f in model.feats)]
    res = {}
    for g in groups:
        cols = [f for f in model.feats if group_of(f) == g]
        rises = []
        for _ in range(repeats):
            shuffled = d.copy()
            idx = rng.permutation(len(d))
            shuffled[cols] = d[cols].to_numpy()[idx]   # one permutation for the whole group
            dev = mean_poisson_deviance(y, np.clip(model.predict(shuffled), 1e-3, None))
            rises.append(dev - base)
        res[g] = {"deviance_increase_mean": float(np.mean(rises)), "deviance_increase_std": float(np.std(rises))}
    return {"baseline_poisson_deviance": float(base), "groups": res}


def main():
    bundle = joblib.load(BUNDLE)
    featurized = add_lag_features(load_raw())
    raw_errors, importance = [], {}
    for h in HORIZONS:
        m = bundle["horizons"][h]
        feats, cfg = m["features"], m["hyperparameters"]
        d = build_dataset(h, featurized, featurized=True, cases=True)   # same rows as train.py
        tr, va, te = split(d)
        for y, fit, val in time_folds(tr):
            raw_errors.append(_rows(f"cv_{y}", "cv", h, val, HGB(feats, **cfg).fit(fit).predict(val)))
        model_tr = HGB(feats, **cfg).fit(tr)                            # train-only model
        raw_errors.append(_rows("validation", "validation", h, va, model_tr.predict(va)))
        raw_errors.append(_rows("test", "test", h, te, model_tr.predict(te)))
        importance[h] = permutation_importance(model_tr, va)
        top = sorted(importance[h]["groups"].items(), key=lambda kv: -kv[1]["deviance_increase_mean"])
        print(f"h={h}w importance (validation deviance increase): " +
              ", ".join(f"{g} {v['deviance_increase_mean']:.3f}" for g, v in top))
    table = _aggregate(pd.concat(raw_errors, ignore_index=True))
    table.to_csv(ERRORS, index=False)
    IMPORTANCE.write_text(json.dumps({"method": "grouped permutation on validation, train-only model, 5 repeats",
                                      "horizons": importance}, indent=2))
    summ = table.groupby(["horizon_weeks", "kind"])[["n", "sum_abs_err", "sum_sq_err"]].sum()
    summ = pd.DataFrame({"MAE": summ.sum_abs_err / summ.n, "RMSE": np.sqrt(summ.sum_sq_err / summ.n)})
    print(summ.round(3).to_string())
    print(f"saved {ERRORS} ({len(table)} rows) and {IMPORTANCE}")


if __name__ == "__main__":
    main()
