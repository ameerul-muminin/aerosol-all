"""Parts E/F: baselines and a model comparison on identical splits and features.

Every model uses each horizon's deployed feature list and the same 5 expanding yearly CV folds
inside 2016-2021 (src/train.py `time_folds`), then is fitted on all of train and scored on
validation (2022-23). The test set is NOT used to compare models.

Baselines (E1-E3): naive (next = this week's cases), same week last year, ward x week-of-year
mean, Ridge. Models (F2, scikit-learn only, no new dependencies): Lasso, ElasticNet, Poisson GLM,
decision tree, random forest, extra trees, gradient boosting, HistGradientBoosting, deployed model.
Non-HGB models get a median imputer + scaler fitted on each training fold only (rain-anomaly
columns are empty in the first year).

Deployment rule, fixed BEFORE running: the deployed model (built exactly as saved: HGB, outbreak-
weighted HGB, or the HGB + extra-trees blend adopted by src/tuning.py) is replaced only if another
model's CV Poisson deviance is more than 2% lower AND its saved size is < 20 MB AND it scores all
60 wards in < 0.5 s. Writes reports/model_comparison.json.
Run: python -m src.compare_models
"""
from __future__ import annotations
import io
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, PoissonRegressor, Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src.features import ROOT, HORIZONS, build_dataset, add_lag_features, load_raw
from src.train import BUNDLE, HGB, SEED, SeasonalNaive, deployed_model, split, time_folds, scores

OUT = ROOT / "reports" / "model_comparison.json"
SIZE_LIMIT_MB, LATENCY_LIMIT_S, MIN_GAIN = 20.0, 0.5, 0.02


class Naive:
    """E1: next = this week's reported cases at the forecast origin (cases_lag0)."""
    learns = False

    def fit(self, d):
        return self

    def predict(self, d):
        return d["cases_lag0"].to_numpy(dtype=float)


class LastYear:
    """E2: same ward, same week one year (52 weeks) before the target week."""
    learns = False

    def __init__(self, raw):
        self.lookup = raw.set_index(["ward_id", "date"])["dengue_cases"]

    def fit(self, d):
        return self

    def predict(self, d):
        idx = pd.MultiIndex.from_arrays([d["ward_id"], d["target_date"] - pd.Timedelta(weeks=52)])
        return self.lookup.reindex(idx).fillna(0).to_numpy(dtype=float)


class Sk:
    """Any scikit-learn regressor on the feature list, with a train-only imputer (+ scaler)."""
    def __init__(self, feats, est, scale=True):
        self.feats, self.est, self.scale = feats, est, scale

    def fit(self, d):
        steps = [SimpleImputer(strategy="median")] + ([StandardScaler()] if self.scale else []) + [self.est]
        self.m = make_pipeline(*steps).fit(d[self.feats], d["target"])
        return self

    def predict(self, d):
        return np.clip(self.m.predict(d[self.feats]), 0, None)


def candidates(feats, cfg, raw, deployed=None):
    return {
        # baselines
        "naive_last_week": (lambda: Naive(), "baseline"),
        "same_week_last_year": (lambda: LastYear(raw), "baseline"),
        "seasonal_naive_mean": (lambda: SeasonalNaive(), "baseline"),
        "ridge": (lambda: Sk(feats, Ridge(alpha=1.0)), "baseline"),
        # linear / count models
        "lasso": (lambda: Sk(feats, Lasso(alpha=0.05, max_iter=5000)), "model"),
        "elastic_net": (lambda: Sk(feats, ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=5000)), "model"),
        "poisson_glm": (lambda: Sk(feats, PoissonRegressor(alpha=1.0, max_iter=1000)), "model"),
        # trees
        "decision_tree": (lambda: Sk(feats, DecisionTreeRegressor(criterion="poisson", max_depth=8,
                                                                   min_samples_leaf=50, random_state=SEED),
                                     scale=False), "model"),
        "random_forest": (lambda: Sk(feats, RandomForestRegressor(n_estimators=200, min_samples_leaf=20,
                                                                   max_features=0.5, n_jobs=-1, random_state=SEED),
                                     scale=False), "model"),
        "extra_trees": (lambda: Sk(feats, ExtraTreesRegressor(n_estimators=200, min_samples_leaf=20,
                                                               max_features=0.5, n_jobs=-1, random_state=SEED),
                                   scale=False), "model"),
        "gradient_boosting": (lambda: Sk(feats, GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                                                           learning_rate=0.05, random_state=SEED),
                                         scale=False), "model"),
        "hist_gradient_boosting": (lambda: HGB(feats, **{k: v for k, v in cfg.items()
                                                         if not k.startswith("outbreak_")}), "model"),
        "deployed": (lambda: deployed_model(deployed), "deployed"),
    }


def _metrics(y, p):
    out = scores(y, p)
    out["R2"] = float(r2_score(y, p))
    return out


def evaluate(make, tr, va):
    folds, gaps = [], []
    for y, fit, val in time_folds(tr):
        m = make().fit(fit)
        v, t = _metrics(val["target"], m.predict(val)), _metrics(fit["target"], m.predict(fit))
        folds.append({"val_year": y, **v})
        gaps.append(v["PoissonDev"] - t["PoissonDev"])
    cv = {k: float(np.mean([f[k] for f in folds])) for k in ("MAE", "RMSE", "PoissonDev", "R2")}
    m = make().fit(tr)
    train_fit = _metrics(tr["target"], m.predict(tr))
    val = _metrics(va["target"], m.predict(va))
    buf = io.BytesIO()
    joblib.dump(m, buf, compress=3)
    one_week = va[va["date"] == va["date"].max()]
    t0 = time.perf_counter()
    m.predict(one_week)
    latency = time.perf_counter() - t0
    # F5 overfit gap only for models that learn from data (rule baselines have nothing to overfit)
    overfit = ({"train_poisdev": train_fit["PoissonDev"], "cv_poisdev": cv["PoissonDev"],
                "mean_fold_gap_val_minus_train": float(np.mean(gaps))}
               if getattr(m, "learns", True) else None)
    return {"cv_mean": cv, "cv_folds": folds, "validation": val, "overfit": overfit,
            "size_mb": round(buf.tell() / 1e6, 3), "latency_60_wards_s": round(latency, 4)}


def main():
    bundle = joblib.load(BUNDLE)
    raw = load_raw()
    featurized = add_lag_features(raw)
    report = {"rules": __doc__.split("Run:")[0].strip(), "horizons": {}}
    for h in HORIZONS:
        mdl = bundle["modes"]["weather"][h]
        feats, cfg = list(mdl["features"]), mdl["hyperparameters"]
        d = build_dataset(h, featurized, featurized=True, cases=True)
        tr, va, _ = split(d)
        res = {}
        for name, (make, kind) in candidates(feats, cfg, raw, mdl).items():
            t0 = time.time()
            res[name] = {"kind": kind, **evaluate(make, tr, va)}
            print(f"  h={h} {name:<34} CV dev {res[name]['cv_mean']['PoissonDev']:.3f}  "
                  f"MAE {res[name]['cv_mean']['MAE']:.3f}  val dev {res[name]['validation']['PoissonDev']:.3f}  "
                  f"gap {(res[name]['overfit'] or {}).get('mean_fold_gap_val_minus_train', float('nan')):+.3f}  "
                  f"{res[name]['size_mb']:.1f} MB  ({time.time() - t0:.0f}s)", flush=True)
        ranking = sorted(res, key=lambda n: res[n]["cv_mean"]["PoissonDev"])
        dep = "deployed"
        base = res[dep]["cv_mean"]["PoissonDev"]
        challengers = [n for n in ranking if n != dep and res[n]["kind"] != "baseline"
                       and res[n]["cv_mean"]["PoissonDev"] < base * (1 - MIN_GAIN)
                       and res[n]["size_mb"] < SIZE_LIMIT_MB and res[n]["latency_60_wards_s"] < LATENCY_LIMIT_S]
        best_baseline = min((n for n in res if res[n]["kind"] == "baseline"),
                            key=lambda n: res[n]["cv_mean"]["PoissonDev"])
        report["horizons"][h] = {
            "results": res, "ranking_by_cv_poisdev": ranking, "top3": ranking[:3],
            "best_baseline": best_baseline,
            "deployed_beats_all_baselines_cv": all(base < res[n]["cv_mean"]["PoissonDev"]
                                                   for n in res if res[n]["kind"] == "baseline"),
            "decision": (f"replace with {challengers[0]}" if challengers else
                         f"keep deployed model ({mdl.get('kind')}): no model clears the +2% CV / size / speed rule"),
        }
        print(f"h={h}w ranking: {ranking[:4]} | best baseline: {best_baseline} | "
              f"decision: {report['horizons'][h]['decision']}", flush=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
