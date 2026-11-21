"""Train per-horizon dengue models with a chronological split; save a joblib bundle + metrics.

Split (by TARGET week, never random):
  train: 2016-2021   fit models; time-ordered CV inside it picks hyperparameters + model family
  val:   2022-2023   held-out confirmation, risk-tier calibration, conformal intervals
  test:  2024-2025   touched once, for the reported final numbers; never used to choose anything

Everything learned from data (tier definitions, scaler, model parameters) is fitted on train
only; calibration uses validation. Lags and week-of-year are fixed transforms, not learned.
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

from src.features import ROOT, HORIZONS, build_dataset, feature_columns, case_columns, add_lag_features, load_raw

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


CV_YEARS = (2017, 2018, 2019, 2020, 2021)   # validation year of each expanding-window fold
# Grid trimmed after a wider CV search (150/300/600 trees x 7/15/31 leaves) never chose
# 600 trees or 31 leaves: keeps training fast without changing what CV can select.
HGB_GRID = [{"max_iter": n, "max_leaf_nodes": l} for n in (150, 300) for l in (7, 15)]


def hgb(max_iter=300, max_leaf_nodes=15, loss="poisson", **kw):
    # early_stopping=False: sklearn's default early stopping holds out a RANDOM 10% of rows,
    # which would shuffle time inside training. Tree count is chosen by time-ordered CV instead.
    return HistGradientBoostingRegressor(loss=loss, max_iter=max_iter, learning_rate=0.05,
                                         max_leaf_nodes=max_leaf_nodes, min_samples_leaf=40,
                                         l2_regularization=1.0, early_stopping=False,
                                         random_state=SEED, **kw)


def time_folds(tr):
    """Expanding-window folds by TARGET year inside the training period:
    fit on years < Y, validate on year Y, for Y in CV_YEARS. Equivalent to TimeSeriesSplit,
    but cut on dates: rows are stacked ward by ward, so row-count splits would cut mid-history."""
    yr = tr["target_date"].dt.year
    for y in CV_YEARS:
        yield y, tr[yr < y], tr[yr == y]


class HGB:
    def __init__(self, feats, **params):
        self.feats, self.params = feats, params

    def fit(self, d):
        self.m = hgb(**self.params).fit(d[self.feats], d["target"])
        return self

    def predict(self, d):
        return self.m.predict(d[self.feats])


class GLM:
    """Poisson regression on log1p features; scaler is inside the pipeline, so it is fitted
    on the training fold only."""
    def __init__(self, feats):
        self.feats = feats

    def fit(self, d):
        self.m = make_pipeline(StandardScaler(), PoissonRegressor(alpha=1.0, max_iter=500)).fit(
            np.log1p(d[self.feats].clip(lower=0)), d["target"])
        return self

    def predict(self, d):
        return self.m.predict(np.log1p(d[self.feats].clip(lower=0)))


def cross_validate(make_model, tr):
    folds = []
    for y, a, b in time_folds(tr):
        p = make_model().fit(a).predict(b)
        folds.append({"val_year": y, "n_train": len(a), "n_val": len(b), **scores(b["target"], p)})
    mean = {k: float(np.mean([f[k] for f in folds])) for k in ("MAE", "RMSE", "PoissonDev")}
    return {"folds": folds, "mean": mean}


class SeasonalNaive:
    """Baseline: mean cases for that ward and ISO week-of-year over the years it is fitted on."""
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


MODES = ("weather", "weather_cases")   # default weather-only; optional + reported-case history


def test_metrics(te, p_te, conf_q, actual_cuts, cuts, baseline=None):
    """Final held-out numbers. Called once per mode/horizon; nothing downstream reads them to
    make a choice."""
    yte = te["target"].to_numpy()
    lo, hi = conformal_interval(p_te, conf_q)
    tier_true, tier_pred = to_tier(yte, actual_cuts), to_tier(p_te, cuts)
    ins = te["target_date"].dt.month.between(7, 10).to_numpy()
    out = scores(yte, p_te)
    if baseline is not None:
        out["baseline_seasonal_naive"] = scores(yte, baseline)
    out.update({
        "interval80_coverage": float(np.mean((yte >= lo) & (yte <= hi))),
        "interval80_coverage_jul_oct": float(np.mean(((yte >= lo) & (yte <= hi))[ins])),
        "tier_exact_acc": float(np.mean(tier_true == tier_pred)),
        "tier_within_one": float(np.mean(np.abs(tier_true - tier_pred) <= 1)),
        "critical_recall": float(np.mean(tier_pred[tier_true == 3] >= 2)) if (tier_true == 3).any() else None,
        "tier_confusion": pd.crosstab(tier_true, tier_pred).reindex(index=range(4), columns=range(4),
                                                                   fill_value=0).values.tolist(),
    })
    return out, lo, hi


def main():
    feats = feature_columns()
    bundle = {"tier_names": TIER_NAMES, "modes": {m: {} for m in MODES}}
    report = {"split": {"train": "target<=2021-12-31", "val": "2022-01..2023-12", "test": "2024-01..2025-12",
                        "cv": f"expanding window inside train, validation years {list(CV_YEARS)}"},
              "horizons": {}, "modes": {"weather_cases": {}}}
    t0 = time.time()
    featurized = add_lag_features(load_raw())     # lags computed once, reused for every horizon
    for h in HORIZONS:
        d = build_dataset(h, featurized, featurized=True, cases=True)
        tr, va, te = split(d)
        yva = va["target"]
        trva = pd.concat([tr, va])

        # --- C3: time-ordered CV inside TRAIN: tune HGB, then pick the model family ---
        hgb_cv = {f"{c['max_iter']}x{c['max_leaf_nodes']}": cross_validate(lambda c=c: HGB(feats, **c), tr)
                  for c in HGB_GRID}
        best_cfg_key = min(hgb_cv, key=lambda k: hgb_cv[k]["mean"]["PoissonDev"])
        best_cfg = next(c for c in HGB_GRID if f"{c['max_iter']}x{c['max_leaf_nodes']}" == best_cfg_key)
        # rain-anomaly features: kept only if they improve CV deviance at the chosen config
        feats_anom = feature_columns(anomaly=True)
        cv_anom = cross_validate(lambda: HGB(feats_anom, **best_cfg), tr)
        use_anomaly = cv_anom["mean"]["PoissonDev"] < hgb_cv[best_cfg_key]["mean"]["PoissonDev"]
        feats_h = feats_anom if use_anomaly else feats
        makers = {"seasonal_naive": SeasonalNaive, "poisson_glm": lambda: GLM(feats),
                  "hgb_poisson": lambda: HGB(feats_h, **best_cfg)}
        cv = {"seasonal_naive": cross_validate(makers["seasonal_naive"], tr),
              "poisson_glm": cross_validate(makers["poisson_glm"], tr),
              "hgb_poisson": cv_anom if use_anomaly else hgb_cv[best_cfg_key]}
        best = min(cv, key=lambda n: cv[n]["mean"]["PoissonDev"])

        # --- held-out VALIDATION: confirm choice, calibrate tiers + intervals (train-only models) ---
        cands = {n: makers[n]().fit(tr) for n in makers}
        val = {n: scores(yva, cands[n].predict(va)) for n in cands}
        p_va = cands[best].predict(va)
        actual_cuts = tier_thresholds(tr["target"])
        cuts = calibrate_pred_cuts(p_va, yva, actual_cuts)
        conf_q = conformal_fit(p_va, yva, 0.80)

        # --- deployment model: same family + hyperparameters, refit on train+val ---
        final = makers[best]().fit(trva)

        # --- final test numbers (train-only model, honest out-of-sample estimate) ---
        p_te = cands[best].predict(te)
        test, lo, hi = test_metrics(te, p_te, conf_q, actual_cuts, cuts,
                                    baseline=cands["seasonal_naive"].predict(te))
        te[["ward_id", "city", "target_date"]].assign(actual=te["target"].values, pred=p_te, lo=lo, hi=hi) \
            .to_csv(ROOT / "reports" / f"test_predictions_h{h}.csv", index=False)

        feats_final = feats_h if best == "hgb_poisson" else feats
        bundle["modes"]["weather"][h] = {"model": final, "conformal_q": conf_q, "tier_cuts": cuts,
                                         "tier_actual_cuts": actual_cuts, "kind": best,
                                         "hyperparameters": best_cfg, "features": feats_final}
        report["horizons"][h] = {"selected": best, "hyperparameters": best_cfg, "cv": cv,
                                 "rain_anomaly": {"used": bool(use_anomaly),
                                                  "cv_poisdev_without": hgb_cv[best_cfg_key]["mean"]["PoissonDev"],
                                                  "cv_poisdev_with": cv_anom["mean"]["PoissonDev"]},
                                 "hgb_grid_cv_mean": {k: v["mean"] for k, v in hgb_cv.items()},
                                 "validation": val, "test": test, "tier_cuts_on_prediction": cuts,
                                 "tier_definition_actual_cases": actual_cuts, "conformal_q": conf_q,
                                 "n": {"train": len(tr), "val": len(va), "test": len(te)}}

        # --- optional mode: same features + reported-case history, same hyperparameters ---
        feats_c = feats_final + case_columns()
        make_c = lambda: HGB(feats_c, **best_cfg)
        cv_c = cross_validate(make_c, tr)
        m_c = make_c().fit(tr)
        p_va_c = m_c.predict(va)
        cuts_c = calibrate_pred_cuts(p_va_c, yva, actual_cuts)
        conf_c = conformal_fit(p_va_c, yva, 0.80)
        test_c, _, _ = test_metrics(te, m_c.predict(te), conf_c, actual_cuts, cuts_c)
        bundle["modes"]["weather_cases"][h] = {"model": make_c().fit(trva), "conformal_q": conf_c,
                                               "tier_cuts": cuts_c, "tier_actual_cuts": actual_cuts,
                                               "kind": "hgb_poisson", "hyperparameters": best_cfg,
                                               "features": feats_c}
        report["modes"]["weather_cases"][h] = {"cv": cv_c, "validation": scores(yva, p_va_c), "test": test_c,
                                               "tier_cuts_on_prediction": cuts_c, "conformal_q": conf_c}

        print(f"h={h}w selected={best} {best_cfg} rain_anomaly={use_anomaly} "
              f"(CV {hgb_cv[best_cfg_key]['mean']['PoissonDev']:.3f} -> {cv_anom['mean']['PoissonDev']:.3f})\n   CV PoisDev: " +
              ", ".join(f"{n}={v['mean']['PoissonDev']:.3f}" for n, v in cv.items()) +
              f", +cases={cv_c['mean']['PoissonDev']:.3f}"
              " | val PoisDev: " + ", ".join(f"{n}={v['PoissonDev']:.3f}" for n, v in val.items()))
        print(f"   test MAE={test['MAE']:.3f} RMSE={test['RMSE']:.3f} (naive MAE={test['baseline_seasonal_naive']['MAE']:.3f}) "
              f"cov80={test['interval80_coverage']:.2f}/JulOct={test['interval80_coverage_jul_oct']:.2f} tier_acc={test['tier_exact_acc']:.2f} "
              f"within1={test['tier_within_one']:.2f} crit_recall={test['critical_recall']}")
        print(f"   +cases mode test MAE={test_c['MAE']:.3f} RMSE={test_c['RMSE']:.3f} crit_recall={test_c['critical_recall']}")
    bundle["horizons"] = bundle["modes"]["weather"]   # backward-compatible default
    BUNDLE.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, BUNDLE, compress=3)
    METRICS.write_text(json.dumps(report, indent=2))
    print(f"saved {BUNDLE} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    # Import main from the package module so pickled wrapper classes are recorded as
    # src.train.HGB (loadable anywhere), not __main__.HGB.
    from src.train import main as _main
    _main()
