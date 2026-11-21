"""Part H: ensembles, evaluated without leakage; adopted only if clearly better.

Members (same features/folds as the deployed model): HistGradientBoosting with the deployed
settings, extra trees, random forest, Poisson GLM (imputer/scaler fitted on train folds only).

H4  Diversity: correlation of out-of-fold (OOF) errors between members, on the 5 time folds.
H1  Simple average of the top-2 and top-3 members by OOF CV deviance.
H2  Weighted average: non-negative weights summing to 1, chosen to minimise Poisson deviance on
    OOF predictions (training years only, never validation).
H3  Stacking: Ridge (positive coefficients) on OOF member predictions.
CV scores of H2/H3 are leave-one-fold-out: weights/meta are fitted on the other folds' OOF and
scored on the held-out fold. Validation scores use members fitted on all of train, with
weights/meta fitted on all OOF predictions.

H5 rule (fixed BEFORE running): replace the deployed model at a horizon only if the best ensemble
has CV deviance > 2% lower AND validation deviance lower AND size < 20 MB AND scores 60 wards in
< 0.5 s. Writes reports/ensemble.json; writes models/ensemble.json only if adopted.
Run: python -m src.ensemble
"""
from __future__ import annotations
import io
import json
import time
import joblib
import numpy as np
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import PoissonRegressor, Ridge
from sklearn.metrics import mean_poisson_deviance

from src.compare_models import Sk
from src.features import ROOT, HORIZONS, build_dataset, add_lag_features, load_raw
from src.train import BUNDLE, HGB, SEED, deployed_model, split, time_folds

OUT = ROOT / "reports" / "ensemble.json"
ADOPT = ROOT / "models" / "ensemble.json"


def _dev(y, p):
    return float(mean_poisson_deviance(y, np.clip(p, 1e-3, None)))


def members(feats, cfg):
    return {
        "hgb": lambda: HGB(feats, **cfg),
        "extra_trees": lambda: Sk(feats, ExtraTreesRegressor(n_estimators=200, min_samples_leaf=20, max_features=0.5,
                                                             n_jobs=-1, random_state=SEED), scale=False),
        "random_forest": lambda: Sk(feats, RandomForestRegressor(n_estimators=200, min_samples_leaf=20, max_features=0.5,
                                                                 n_jobs=-1, random_state=SEED), scale=False),
        "poisson_glm": lambda: Sk(feats, PoissonRegressor(alpha=1.0, max_iter=1000)),
    }


def simplex_weights(P, y):
    """Non-negative weights summing to 1 that minimise Poisson deviance of P @ w."""
    k = P.shape[1]
    f = lambda z: _dev(y, P @ (np.exp(z) / np.exp(z).sum()))
    z = minimize(f, np.zeros(k), method="Nelder-Mead", options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-6}).x
    return np.exp(z) / np.exp(z).sum()


def fit_stack(P, y):
    return Ridge(alpha=1.0, positive=True).fit(P, y)


def main():
    t0 = time.time()
    bundle = joblib.load(BUNDLE)
    featurized = add_lag_features(load_raw())
    report, adopted = {"rules": __doc__.split("Run:")[0].strip(), "horizons": {}}, {}
    for h in HORIZONS:
        dep = bundle["modes"]["weather"][h]
        feats, cfg = list(dep["features"]), dict(dep["hyperparameters"])
        tr, va, _ = split(build_dataset(h, featurized, featurized=True, cases=True))
        mk = members(feats, cfg)
        names = list(mk)

        # ---- OOF predictions on the 5 time folds (+ the deployed model for reference)
        fold_id, ys, oof = [], [], {n: [] for n in names + ["deployed"]}
        for k, (_, fit, val) in enumerate(time_folds(tr)):
            fold_id.append(np.full(len(val), k)); ys.append(val["target"].to_numpy())
            for n in names:
                oof[n].append(mk[n]().fit(fit).predict(val))
            oof["deployed"].append(deployed_model(dep).fit(fit).predict(val))
        fold_id, y = np.concatenate(fold_id), np.concatenate(ys)
        oof = {n: np.clip(np.concatenate(v), 0, None) for n, v in oof.items()}
        folds = np.unique(fold_id)
        cv = lambda p: float(np.mean([_dev(y[fold_id == f], p[fold_id == f]) for f in folds]))

        # ---- H4 diversity
        err = {n: y - oof[n] for n in names}
        corr = {a: {b: round(float(np.corrcoef(err[a], err[b])[0, 1]), 3) for b in names} for a in names}

        # ---- members fitted on all train for validation
        full = {n: mk[n]().fit(tr) for n in names}
        val_pred = {n: np.clip(full[n].predict(va), 0, None) for n in names}
        dep_full = deployed_model(dep).fit(tr)
        yv = va["target"].to_numpy()
        res = {n: {"cv": cv(oof[n]), "val": _dev(yv, val_pred[n])} for n in names}
        res["deployed"] = {"cv": cv(oof["deployed"]), "val": _dev(yv, dep_full.predict(va))}
        ranked = sorted(names, key=lambda n: res[n]["cv"])

        # ---- H1 simple averages
        for top in (2, 3):
            pick = ranked[:top]
            res[f"avg_top{top}"] = {"members": pick, "cv": cv(np.mean([oof[n] for n in pick], axis=0)),
                                   "val": _dev(yv, np.mean([val_pred[n] for n in pick], axis=0))}
        # ---- H2 weighted average, H3 stacking: leave-one-fold-out CV, then all-OOF fit for validation
        P, Pv = np.column_stack([oof[n] for n in names]), np.column_stack([val_pred[n] for n in names])
        p_w, p_s = np.zeros_like(y, dtype=float), np.zeros_like(y, dtype=float)
        for f in folds:
            tr_m, te_m = fold_id != f, fold_id == f
            p_w[te_m] = P[te_m] @ simplex_weights(P[tr_m], y[tr_m])
            p_s[te_m] = np.clip(fit_stack(P[tr_m], y[tr_m]).predict(P[te_m]), 0, None)
        w_all, stack_all = simplex_weights(P, y), fit_stack(P, y)
        res["weighted_avg"] = {"weights": dict(zip(names, np.round(w_all, 3).tolist())), "cv": cv(p_w),
                               "val": _dev(yv, Pv @ w_all)}
        res["stack_ridge"] = {"coef": dict(zip(names, np.round(stack_all.coef_, 3).tolist())),
                              "intercept": float(stack_all.intercept_), "cv": cv(p_s),
                              "val": _dev(yv, np.clip(stack_all.predict(Pv), 0, None))}

        # ---- H5 decision
        ens = ["avg_top2", "avg_top3", "weighted_avg", "stack_ridge"]
        best = min(ens, key=lambda n: res[n]["cv"])
        base = res["deployed"]
        size = sum(len((lambda b: (joblib.dump(full[n], b, compress=3), b.getvalue())[1])(io.BytesIO()))
                   for n in names) / 1e6
        week = va[va["date"] == va["date"].max()]
        tl = time.perf_counter(); [full[n].predict(week) for n in names]; lat = time.perf_counter() - tl
        ok = (res[best]["cv"] < base["cv"] * 0.98 and res[best]["val"] < base["val"] and size < 20 and lat < 0.5)
        report["horizons"][h] = {"members_ranked_by_cv": ranked, "error_correlation": corr, "results": res,
                                 "best_ensemble": best, "all_members_size_mb": round(size, 2),
                                 "all_members_latency_s": round(lat, 4), "adopted": bool(ok),
                                 "decision": f"adopt {best}" if ok else "keep deployed model (rule not met)"}
        if ok:
            adopted[str(h)] = {"type": best, **{k: res[best][k] for k in ("members", "weights", "coef", "intercept")
                                                if k in res[best]}}
        member_txt = ", ".join("%s %.3f" % (n, res[n]["cv"]) for n in ranked)
        ens_txt = " | ".join("%s CV %.3f val %.3f" % (n, res[n]["cv"], res[n]["val"]) for n in ens)
        print(f"h={h}w members CV {member_txt} | deployed CV {base['cv']:.3f} val {base['val']:.3f} | "
              f"{ens_txt} | -> {report['horizons'][h]['decision']}", flush=True)
        print(f"   error corr hgb~ET {corr['hgb']['extra_trees']}, hgb~RF {corr['hgb']['random_forest']}, "
              f"hgb~GLM {corr['hgb']['poisson_glm']}; weights {res['weighted_avg']['weights']}", flush=True)
    OUT.write_text(json.dumps(report, indent=2))
    if adopted:
        ADOPT.write_text(json.dumps(adopted, indent=2))
        print(f"adopted -> {ADOPT}")
    print(f"saved {OUT} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
