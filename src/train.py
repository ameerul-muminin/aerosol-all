"""Train per-horizon dengue models with a chronological split; save a joblib bundle + metrics.

Split (by TARGET week, never random):
  train: 2016-2021   fit models
  val:   2022-2023   model selection, early stopping, risk-tier thresholds
  test:  2024-2025   touched once, for the reported final numbers
"""
from __future__ import annotations
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance

from src.features import ROOT, HORIZONS, build_dataset, feature_columns

TRAIN_END, VAL_END = pd.Timestamp("2021-12-31"), pd.Timestamp("2023-12-31")
TIER_NAMES = ["Low", "Medium", "High", "Critical"]
SEED = 42
BUNDLE = ROOT / "models" / "dengue_model.joblib"
METRICS = ROOT / "reports" / "metrics.json"


def split(d):
    t = d["target_date"]
    return d[t <= TRAIN_END], d[(t > TRAIN_END) & (t <= VAL_END)], d[t > VAL_END]


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def scores(y, p):
    return {"MAE": float(mean_absolute_error(y, p)), "RMSE": rmse(y, p),
            "PoissonDev": float(mean_poisson_deviance(y, np.clip(p, 1e-3, None)))}


def hgb(loss="poisson", **kw):
    return HistGradientBoostingRegressor(loss=loss, max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                         min_samples_leaf=40, l2_regularization=1.0, random_state=SEED, **kw)


class SeasonalNaive:
    """Baseline: mean cases for that ward and ISO week-of-year over the training years."""
    def fit(self, d):
        w = d["target_date"].dt.isocalendar().week.astype(int)
        self.m = d.assign(w=w).groupby(["ward_id", "w"])["target"].mean()
        return self

    def predict(self, d):
        w = d["target_date"].dt.isocalendar().week.astype(int)
        idx = pd.MultiIndex.from_arrays([d["ward_id"], w])
        return self.m.reindex(idx).fillna(0).to_numpy()


def tier_thresholds(y_train):
    """Tier DEFINITIONS in actual cases/ward-week, from quantiles of TRAIN actuals:
    Medium >= p70, High >= p90, Critical >= p97."""
    return [float(np.quantile(y_train, q)) for q in (0.70, 0.90, 0.97)]


def calibrate_pred_cuts(val_pred, val_actual, actual_cuts):
    """Predicted means are shrunk toward the centre, so applying actual-case cuts to them
    under-calls High/Critical. Map each actual cut to the prediction quantile that yields
    the same tier frequency on VALIDATION (quantile matching)."""
    val_actual = np.asarray(val_actual)
    return [float(np.quantile(val_pred, np.mean(val_actual < c))) for c in actual_cuts]


CONF_BINS = (1.0, 5.0)   # Mondrian bins on predicted cases: <1 (off-season), 1-5, >=5 (season)


def _conf_q(p, y, coverage):
    scale = np.sqrt(p + 1.0)
    a, n = (1 - coverage) / 2, len(y)
    lvl = min(1.0, np.ceil((n + 1) * (1 - a)) / n)   # finite-sample correction
    return float(np.quantile((p - y) / scale, lvl)), float(np.quantile((y - p) / scale, lvl))


def conformal_fit(val_pred, val_actual, coverage=0.80):
    """Split-conformal intervals, asymmetric and variance-scaled (count noise ~ sqrt(mean)),
    calibrated SEPARATELY per predicted-value bin (Mondrian). A single global calibration is
    dominated by off-season near-zero weeks and under-covers the outbreak season.
    Uses VALIDATION residuals of a train-only model. Returns {bin_index: (q_lo, q_hi)}."""
    p, y = np.asarray(val_pred), np.asarray(val_actual)
    b = np.digitize(p, CONF_BINS)
    return {int(k): _conf_q(p[b == k], y[b == k], coverage) for k in np.unique(b)}


def conformal_interval(pred, q):
    pred = np.clip(np.asarray(pred, dtype=float), 0, None)
    b = np.digitize(pred, CONF_BINS)
    fallback = q[max(q)]
    qlo = np.array([q.get(int(k), fallback)[0] for k in b])
    qhi = np.array([q.get(int(k), fallback)[1] for k in b])
    scale = np.sqrt(pred + 1.0)
    return np.clip(pred - qlo * scale, 0, None), pred + qhi * scale


def to_tier(pred, cuts):
    return np.digitize(pred, cuts)  # 0..3


def main():
    feats = feature_columns()
    bundle = {"features": feats, "horizons": {}, "tier_names": TIER_NAMES}
    report = {"split": {"train": "target<=2021-12-31", "val": "2022-01..2023-12", "test": "2024-01..2025-12"},
              "horizons": {}}
    t0 = time.time()
    for h in HORIZONS:
        d = build_dataset(h)
        tr, va, te = split(d)
        Xtr, Xva, Xte = tr[feats], va[feats], te[feats]
        ytr, yva, yte = tr["target"], va["target"], te["target"]

        # --- model selection on VALIDATION only ---
        cands = {
            "seasonal_naive": SeasonalNaive().fit(tr),
            "poisson_glm": make_pipeline(StandardScaler(), PoissonRegressor(alpha=1.0, max_iter=500)).fit(
                np.log1p(Xtr.clip(lower=0)), ytr),
            "hgb_poisson": hgb().fit(Xtr, ytr),
        }
        def pred(name, X, d_):
            m = cands[name]
            if name == "seasonal_naive": return m.predict(d_)
            if name == "poisson_glm": return m.predict(np.log1p(X.clip(lower=0)))
            return m.predict(X)
        val = {n: scores(yva, pred(n, Xva, va)) for n in cands}
        best = min(val, key=lambda n: val[n]["PoissonDev"])

        # --- refit chosen family on train+val for deployment; quantile models for 80% interval ---
        trva = pd.concat([tr, va])
        final = hgb().fit(trva[feats], trva["target"]) if best == "hgb_poisson" else cands[best]
        actual_cuts = tier_thresholds(ytr)
        p_va = pred(best, Xva, va)
        cuts = calibrate_pred_cuts(p_va, yva, actual_cuts)
        conf_q = conformal_fit(p_va, yva, 0.80)

        # --- final test numbers (train-only models for an honest estimate) ---
        p_te = pred(best, Xte, te)
        lo, hi = conformal_interval(p_te, conf_q)
        tier_true, tier_pred = to_tier(yte, actual_cuts), to_tier(p_te, cuts)
        test = scores(yte, p_te)
        test["baseline_seasonal_naive"] = scores(yte, cands["seasonal_naive"].predict(te))
        test["interval80_coverage"] = float(np.mean((yte >= lo) & (yte <= hi)))
        test["tier_exact_acc"] = float(np.mean(tier_true == tier_pred))
        test["tier_within_one"] = float(np.mean(np.abs(tier_true - tier_pred) <= 1))
        test["critical_recall"] = float(np.mean(tier_pred[tier_true == 3] >= 2)) if (tier_true == 3).any() else None
        test["tier_confusion"] = pd.crosstab(tier_true, tier_pred).reindex(index=range(4), columns=range(4), fill_value=0).values.tolist()
        ins = te["target_date"].dt.month.between(7, 10).to_numpy()
        test["interval80_coverage_jul_oct"] = float(np.mean(((yte >= lo) & (yte <= hi))[ins]))
        # per-month errors for the validation dashboard
        te_out = te[["ward_id", "city", "target_date"]].assign(actual=yte.values, pred=p_te, lo=lo, hi=hi)
        te_out.to_csv(ROOT / "reports" / f"test_predictions_h{h}.csv", index=False)

        bundle["horizons"][h] = {"model": final, "conformal_q": conf_q, "tier_cuts": cuts, "tier_actual_cuts": actual_cuts, "kind": best}
        report["horizons"][h] = {"selected": best, "validation": val, "test": test, "tier_cuts_on_prediction": cuts, "tier_definition_actual_cases": actual_cuts, "conformal_q": conf_q,
                                 "n": {"train": len(tr), "val": len(va), "test": len(te)}}
        print(f"h={h}w selected={best}  val PoisDev: " +
              ", ".join(f"{n}={v['PoissonDev']:.3f}" for n, v in val.items()))
        print(f"   test MAE={test['MAE']:.3f} RMSE={test['RMSE']:.3f} (naive MAE={test['baseline_seasonal_naive']['MAE']:.3f}) "
              f"cov80={test['interval80_coverage']:.2f}/JulOct={test['interval80_coverage_jul_oct']:.2f} tier_acc={test['tier_exact_acc']:.2f} "
              f"within1={test['tier_within_one']:.2f} crit_recall={test['critical_recall']}")
    BUNDLE.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, BUNDLE, compress=3)
    METRICS.write_text(json.dumps(report, indent=2))
    print(f"saved {BUNDLE} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
