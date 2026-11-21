"""Part G: hyperparameter search and improvement tests, all on the time-ordered CV folds.

G1/G2/G4  Random search (sklearn ParameterSampler, the engine of RandomizedSearchCV) over
          learning rate, trees, leaves, min samples per leaf, L2 and feature fraction; 24 trials
          per horizon plus the current settings, each scored on the same 5 expanding yearly folds
          inside 2016-2021. Every trial is logged (params, CV deviance, MAE, seconds).
G5        One change at a time against the same reference (tuned or current settings):
          (a) average with extra trees, (b) extra sample weight for outbreak weeks
          (target >= High threshold = training p90), weights 2 and 4.

Rules (fixed BEFORE running; validation 2022-23 only as a "not worse" check; test never used):
  tuning   adopt the best trial if CV Poisson deviance improves by >= 1% AND validation
           deviance is not worse than the current settings.
  blend    adopt if CV deviance improves by > 2% AND size < 20 MB AND 60 wards < 0.5 s.
  weights  adopt the weight with the largest gain if CV MAE on outbreak weeks improves by
           >= 10% AND overall CV deviance worsens by <= 2%.
  If blend and weights both pass, the blend is adopted (primary metric) and weights are only
  reported. If nothing is adopted for any horizon, models/hyperparameters.json is not written
  and train.py keeps its current behaviour.
Writes reports/tuning_log.json, reports/improvement_tests.json and (if adopted)
models/hyperparameters.json. Run: python -m src.tuning, then python -m src.train.
"""
from __future__ import annotations
import io
import json
import time
import joblib
import numpy as np
from sklearn.metrics import mean_poisson_deviance
from sklearn.model_selection import ParameterSampler

from src.features import ROOT, HORIZONS, build_dataset, add_lag_features, load_raw
from src.train import BUNDLE, HYPERPARAMS, SEED, HGB, Blend, split, time_folds, tier_thresholds

TUNING_LOG = ROOT / "reports" / "tuning_log.json"
IMPROVEMENTS = ROOT / "reports" / "improvement_tests.json"
N_TRIALS = 24
SPACE = {"learning_rate": [0.02, 0.03, 0.05, 0.08, 0.1], "max_iter": [150, 300, 500, 800],
         "max_leaf_nodes": [7, 15, 31], "min_samples_leaf": [20, 40, 80, 160],
         "l2_regularization": [0.0, 0.5, 1.0, 3.0, 10.0], "max_features": [0.5, 0.7, 1.0]}


def _dev(y, p):
    return float(mean_poisson_deviance(y, np.clip(p, 1e-3, None)))


def score(make, tr, va, outbreak_thr):
    """CV mean deviance (as in train.py), CV MAE overall and on outbreak weeks, validation deviance."""
    devs, ys, ps = [], [], []
    for _, fit, val in time_folds(tr):
        p = make().fit(fit).predict(val)
        devs.append(_dev(val["target"], p))
        ys.append(val["target"].to_numpy()), ps.append(p)
    y, p = np.concatenate(ys), np.concatenate(ps)
    ob = y >= outbreak_thr
    m = make().fit(tr)
    return {"cv_poisdev": float(np.mean(devs)), "cv_mae": float(np.mean(np.abs(y - p))),
            "cv_outbreak_mae": float(np.mean(np.abs(y[ob] - p[ob]))), "cv_outbreak_n": int(ob.sum()),
            "val_poisdev": _dev(va["target"], m.predict(va)), "_model": m}


def size_latency(m, va):
    buf = io.BytesIO()
    joblib.dump(m, buf, compress=3)
    week = va[va["date"] == va["date"].max()]
    t0 = time.perf_counter()
    m.predict(week)
    return buf.tell() / 1e6, time.perf_counter() - t0


def strip(r):
    return {k: v for k, v in r.items() if not k.startswith("_")}


def main():
    bundle = joblib.load(BUNDLE)
    featurized = add_lag_features(load_raw())
    log, improvements, adopted, any_change = {}, {}, {}, False
    for h in HORIZONS:
        mdl = bundle["modes"]["weather"][h]
        feats, current = list(mdl["features"]), dict(mdl["hyperparameters"])
        tr, va, _ = split(build_dataset(h, featurized, featurized=True, cases=True))
        thr = tier_thresholds(tr["target"])[1]

        # ---- G1/G2/G4: random search, every trial logged
        trials = []
        configs = [current] + list(ParameterSampler(SPACE, n_iter=N_TRIALS, random_state=SEED + h))
        for i, cfg in enumerate(configs):
            t0 = time.time()
            r = strip(score(lambda c=cfg: HGB(feats, **c), tr, va, thr))
            trials.append({"trial": i, "params": cfg, **r, "seconds": round(time.time() - t0, 1),
                           "note": "current settings" if i == 0 else ""})
            print(f"  h={h} trial {i:>2} CV dev {r['cv_poisdev']:.4f} val {r['val_poisdev']:.4f} "
                  f"({trials[-1]['seconds']}s) {cfg}", flush=True)
        base, best = trials[0], min(trials, key=lambda t: t["cv_poisdev"])
        tune_ok = best["cv_poisdev"] <= base["cv_poisdev"] * 0.99 and best["val_poisdev"] <= base["val_poisdev"]
        cfg_h = dict(best["params"]) if tune_ok else current
        ref = best if tune_ok else base
        log[h] = {"n_trials": len(trials), "trials": trials, "current": base, "best": best,
                  "adopted": tune_ok,
                  "decision": "adopt best trial" if tune_ok else "keep current settings (rule not met)"}

        # ---- G5a: average with extra trees
        rb = score(lambda: Blend(feats, **cfg_h), tr, va, thr)
        size, lat = size_latency(rb.pop("_model"), va)
        blend_ok = rb["cv_poisdev"] < ref["cv_poisdev"] * 0.98 and size < 20 and lat < 0.5
        # ---- G5b: outbreak sample weights
        weights = {}
        for w in (2.0, 4.0):
            rw = strip(score(lambda w=w: HGB(feats, **cfg_h, outbreak_weight=w, outbreak_threshold=thr), tr, va, thr))
            rw["passes"] = (rw["cv_outbreak_mae"] <= ref["cv_outbreak_mae"] * 0.90
                            and rw["cv_poisdev"] <= ref["cv_poisdev"] * 1.02)
            weights[str(w)] = rw
        passing = [w for w, r in weights.items() if r["passes"]]
        best_w = min(passing, key=lambda w: weights[w]["cv_outbreak_mae"]) if passing else None
        use_blend = blend_ok
        use_w = best_w if (best_w and not blend_ok) else None
        improvements[h] = {
            "reference": {k: ref[k] for k in ("cv_poisdev", "cv_mae", "cv_outbreak_mae", "val_poisdev")},
            "outbreak_threshold_cases": thr,
            "blend_extra_trees": {**strip(rb), "size_mb": round(size, 2), "latency_s": round(lat, 4),
                                  "adopted": use_blend},
            "outbreak_weights": weights, "weight_adopted": use_w,
        }
        final = dict(cfg_h)
        if use_w:
            final.update(outbreak_weight=float(use_w), outbreak_threshold=float(thr))
        adopted[h] = {"hgb": final, "blend": use_blend}
        any_change |= tune_ok or use_blend or bool(use_w)
        print(f"h={h}w tuning: {log[h]['decision']} (CV {base['cv_poisdev']:.4f} -> best {best['cv_poisdev']:.4f}) | "
              f"blend CV {rb['cv_poisdev']:.4f} adopted={use_blend} | weights "
              + ", ".join(f"w={w}: outbreak MAE {r['cv_outbreak_mae']:.2f} vs {ref['cv_outbreak_mae']:.2f}, "
                          f"CV {r['cv_poisdev']:.4f} pass={r['passes']}" for w, r in weights.items()), flush=True)

    TUNING_LOG.write_text(json.dumps({"rules": __doc__.split("Writes")[0].strip(), "horizons": log}, indent=2))
    IMPROVEMENTS.write_text(json.dumps({"horizons": improvements}, indent=2))
    if any_change:
        HYPERPARAMS.write_text(json.dumps({str(h): v for h, v in adopted.items()}, indent=2))
        print(f"adopted changes -> {HYPERPARAMS} (run python -m src.train)")
    else:
        print("no change passed its rule; models/hyperparameters.json not written")
    print(f"saved {TUNING_LOG} and {IMPROVEMENTS}")


if __name__ == "__main__":
    main()
