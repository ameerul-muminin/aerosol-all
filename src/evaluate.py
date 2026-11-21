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
  reports/classification_metrics.json   (F1)
      Risk tiers as a classification task on validation and test: per-tier precision, recall,
      F1, macro-F1, confusion matrix, "alert" (High or Critical) precision/recall, and a
      precision-recall curve + average precision for detecting Critical weeks (score =
      predicted cases). sMAPE is not reported: 58% of weeks have 0 cases, where it is undefined.
  reports/outbreak_errors.json          (F4)
      MAE / RMSE / mean bias on all weeks, Jul-Oct, and outbreak weeks (actual >= High and
      >= Critical case thresholds), validation and test.

Uses the deployed weather-mode hyperparameters and feature lists from the saved bundle;
it never changes the model or any threshold.
"""
from __future__ import annotations
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, mean_poisson_deviance, precision_recall_curve,
                             precision_recall_fscore_support, r2_score)

from src.features import ROOT, HORIZONS, build_dataset, add_lag_features, load_raw
from src.train import BUNDLE, HGB, TIER_NAMES, split, time_folds, to_tier
from src.explain import group_of, GROUP_ORDER

ERRORS = ROOT / "reports" / "validation_errors.csv"
IMPORTANCE = ROOT / "reports" / "feature_importance.json"
CLASSIFICATION = ROOT / "reports" / "classification_metrics.json"
OUTBREAK = ROOT / "reports" / "outbreak_errors.json"
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


def classification_metrics(y, p, actual_cuts, pred_cuts):
    """Tiers as classes: truth from actual cases (training-percentile definitions), prediction
    from predicted cases (validation-calibrated cut points), exactly as the app shows them."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    t, q = to_tier(y, actual_cuts), to_tier(p, pred_cuts)
    prec, rec, f1, sup = precision_recall_fscore_support(t, q, labels=[0, 1, 2, 3], zero_division=0)
    alert_true, alert_pred = t >= 2, q >= 2
    crit = (y >= actual_cuts[2]).astype(int)
    out = {
        "per_tier": {TIER_NAMES[i]: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i]),
                                     "support": int(sup[i])} for i in range(4)},
        "macro_f1": float(np.mean(f1)),
        "accuracy": float(np.mean(t == q)),
        "confusion_true_rows_pred_cols": pd.crosstab(t, q).reindex(index=range(4), columns=range(4),
                                                                   fill_value=0).values.tolist(),
        "alert_high_or_critical": {
            "recall": float(alert_pred[alert_true].mean()) if alert_true.any() else None,
            "precision": float(alert_true[alert_pred].mean()) if alert_pred.any() else None},
        "r2_cases": float(r2_score(y, p)),
    }
    if crit.any():
        P, R, T = precision_recall_curve(crit, p)
        keep = np.unique(np.linspace(0, len(T) - 1, min(40, len(T))).astype(int))
        out["critical_detection"] = {
            "average_precision": float(average_precision_score(crit, p)), "prevalence": float(crit.mean()),
            "pr_curve": [{"threshold_cases": float(T[i]), "precision": float(P[i]), "recall": float(R[i])}
                         for i in keep]}
    return out


def outbreak_errors(d, p, actual_cuts):
    y = d["target"].to_numpy(dtype=float)
    masks = {"all_weeks": np.ones(len(y), bool),
             "jul_oct": d["target_date"].dt.month.between(7, 10).to_numpy(),
             "outbreak_high_plus": y >= actual_cuts[1],
             "outbreak_critical": y >= actual_cuts[2]}
    out = {}
    for k, m in masks.items():
        e = p[m] - y[m]
        out[k] = {"n": int(m.sum()), "MAE": float(np.abs(e).mean()), "RMSE": float(np.sqrt((e ** 2).mean())),
                  "mean_bias_pred_minus_actual": float(e.mean())} if m.any() else {"n": 0}
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
    raw_errors, importance, classification, outbreak = [], {}, {}, {}
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
        p_va, p_te = model_tr.predict(va), model_tr.predict(te)
        cuts_a, cuts_p = m["tier_actual_cuts"], m["tier_cuts"]
        classification[h] = {"validation": classification_metrics(va["target"], p_va, cuts_a, cuts_p),
                             "test": classification_metrics(te["target"], p_te, cuts_a, cuts_p)}
        outbreak[h] = {"validation": outbreak_errors(va, p_va, cuts_a), "test": outbreak_errors(te, p_te, cuts_a)}
        c = classification[h]["test"]
        print(f"   test: macro-F1 {c['macro_f1']:.2f}, alert recall {c['alert_high_or_critical']['recall']:.2f} / "
              f"precision {c['alert_high_or_critical']['precision']:.2f}, Critical AP "
              f"{c['critical_detection']['average_precision']:.2f}, outbreak-week bias "
              f"{outbreak[h]['test']['outbreak_critical']['mean_bias_pred_minus_actual']:+.1f} cases")
        top = sorted(importance[h]["groups"].items(), key=lambda kv: -kv[1]["deviance_increase_mean"])
        print(f"h={h}w importance (validation deviance increase): " +
              ", ".join(f"{g} {v['deviance_increase_mean']:.3f}" for g, v in top))
    table = _aggregate(pd.concat(raw_errors, ignore_index=True))
    table.to_csv(ERRORS, index=False)
    IMPORTANCE.write_text(json.dumps({"method": "grouped permutation on validation, train-only model, 5 repeats",
                                      "horizons": importance}, indent=2))
    CLASSIFICATION.write_text(json.dumps({"note": "tiers as classes; train-only model; sMAPE omitted (58% zero weeks)",
                                          "horizons": classification}, indent=2))
    OUTBREAK.write_text(json.dumps({"note": "pred - actual; outbreak = actual >= High / Critical case thresholds",
                                    "horizons": outbreak}, indent=2))
    summ = table.groupby(["horizon_weeks", "kind"])[["n", "sum_abs_err", "sum_sq_err"]].sum()
    summ = pd.DataFrame({"MAE": summ.sum_abs_err / summ.n, "RMSE": np.sqrt(summ.sum_sq_err / summ.n)})
    print(summ.round(3).to_string())
    print(f"saved {ERRORS} ({len(table)} rows) and {IMPORTANCE}")


if __name__ == "__main__":
    main()
