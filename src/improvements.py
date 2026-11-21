"""Improvement round 2: four ideas, each adopted only if it passes a rule fixed BEFORE running.
The test set (2024-25) is never used here.

Model-side (sequential: each step starts from whatever the previous step adopted):
  1. Weighting + blend: outbreak weights 2 / 4 on the h=3 blend; outbreak weights 2 / 3 at h=4.
     Rule: CV MAE on outbreak weeks (target >= High threshold) improves >= 10% AND overall CV
     Poisson deviance worsens <= 2% (best passing weight wins).
  2. Recent-year weights: linear ramp, newest training year weighted 1 + r, r in {1, 2}.
     Rule: CV deviance improves >= 1% AND validation deviance is not worse.
Post-processing (on the model after steps 1-2):
  3. Outbreak under-prediction fix: isotonic recalibration fitted on out-of-fold training
     predictions. Rule: validation MAE on outbreak weeks improves >= 10% AND validation deviance
     worsens <= 2%.
  4. Season-aware 80% ranges: calibrate per (prediction bin, Jul-Oct vs rest). Checked by
     calibrating on 2022 and measuring coverage on 2023 (both validation years).
     Rule (all horizons together): Jul-Oct coverage moves closer to 0.80 on average AND overall
     coverage stays within 0.75-0.85 for every horizon.

Writes reports/improvements_round2.json; updates models/hyperparameters.json (steps 1-2) and
writes models/postprocess.json (steps 3-4) only for adopted changes. Then run python -m src.train.
"""
from __future__ import annotations
import functools
import json
import time
import joblib
import numpy as np
from sklearn.metrics import mean_poisson_deviance

from src.features import ROOT, HORIZONS, build_dataset, add_lag_features, load_raw
from src.train import (BUNDLE, ENSEMBLE, HYPERPARAMS, POSTPROCESS, HGB, Blend, Calibrated, Ensemble, split, time_folds,
                       tier_thresholds, conformal_fit, conformal_interval)

OUT = ROOT / "reports" / "improvements_round2.json"


def _dev(y, p):
    return float(mean_poisson_deviance(y, np.clip(p, 1e-3, None)))


def cv_score(make, tr, thr):
    devs, ys, ps = [], [], []
    for _, fit, val in time_folds(tr):
        p = make().fit(fit).predict(val)
        devs.append(_dev(val["target"], p)); ys.append(val["target"].to_numpy()); ps.append(p)
    y, p = np.concatenate(ys), np.concatenate(ps)
    ob = y >= thr
    return {"cv_poisdev": float(np.mean(devs)), "cv_outbreak_mae": float(np.mean(np.abs(y[ob] - p[ob])))}


def val_score(model, va, thr):
    y, p = va["target"].to_numpy(), model.predict(va)
    ob = y >= thr
    return {"val_poisdev": _dev(y, p), "val_outbreak_mae": float(np.mean(np.abs(y[ob] - p[ob]))),
            "val_mae": float(np.mean(np.abs(y - p)))}


def coverage(p, y, lo, hi, months):
    inside = (y >= lo) & (y <= hi)
    peak = np.isin(months, (7, 8, 9, 10))
    return {"all": float(inside.mean()), "jul_oct": float(inside[peak].mean())}


def main():
    t0 = time.time()
    bundle = joblib.load(BUNDLE)
    tuned = json.loads(HYPERPARAMS.read_text()) if HYPERPARAMS.exists() else {}
    ens = json.loads(ENSEMBLE.read_text()) if ENSEMBLE.exists() else {}
    featurized = add_lag_features(load_raw())
    report, new_tuned, post = {"rules": __doc__.split("Writes")[0].strip(), "horizons": {}}, {}, \
        {"season_intervals": False, "isotonic": {}}
    season_eval = {}
    for h in HORIZONS:
        mdl = bundle["modes"]["weather"][h]
        feats = list(mdl["features"])
        cfg = dict(tuned.get(str(h), {}).get("hgb", mdl["hyperparameters"]))
        blend = bool(tuned.get(str(h), {}).get("blend"))
        w_ens = ens.get(str(h), {}).get("weights")
        # evaluate exactly the deployed model type (Part H weighted ensemble where adopted)
        cls = functools.partial(Ensemble, weights=w_ens) if w_ens else (Blend if blend else HGB)
        tr, va, _ = split(build_dataset(h, featurized, featurized=True, cases=True))
        thr = tier_thresholds(tr["target"])[1]
        rep = {"start": {"settings": dict(cfg), "blend": blend}}

        # ---- 1. weighting (+ blend at h=3, more weights at h=4)
        ref = cv_score(lambda: cls(feats, **cfg), tr, thr)
        rep["reference_cv"] = ref
        trials = {}
        weights = (2.0, 4.0) if h == 3 else (2.0, 3.0) if h == 4 else ()
        if "outbreak_weight" not in cfg:
            for w in weights:
                c = {**cfg, "outbreak_weight": w, "outbreak_threshold": float(thr)}
                r = cv_score(lambda c=c: cls(feats, **c), tr, thr)
                r["passes"] = r["cv_outbreak_mae"] <= ref["cv_outbreak_mae"] * 0.90 and r["cv_poisdev"] <= ref["cv_poisdev"] * 1.02
                trials[str(w)] = r
        passing = [w for w, r in trials.items() if r["passes"]]
        if passing:
            w = min(passing, key=lambda w: trials[w]["cv_outbreak_mae"])
            cfg.update(outbreak_weight=float(w), outbreak_threshold=float(thr))
            ref = {k: trials[w][k] for k in ("cv_poisdev", "cv_outbreak_mae")}
        rep["1_weighting"] = {"trials": trials, "adopted": passing and float(min(passing, key=lambda w: trials[w]["cv_outbreak_mae"]))}

        # ---- 2. recent-year weights
        base_val = val_score(cls(feats, **cfg).fit(tr), va, thr)
        rtrials = {}
        for r_ in (1.0, 2.0):
            c = {**cfg, "recency_weight": r_}
            r = cv_score(lambda c=c: cls(feats, **c), tr, thr)
            r.update(val_score(cls(feats, **c).fit(tr), va, thr))
            r["passes"] = r["cv_poisdev"] <= ref["cv_poisdev"] * 0.99 and r["val_poisdev"] <= base_val["val_poisdev"]
            rtrials[str(r_)] = r
        rpass = [k for k, r in rtrials.items() if r["passes"]]
        if rpass:
            best_r = min(rpass, key=lambda k: rtrials[k]["cv_poisdev"])
            cfg["recency_weight"] = float(best_r)
            base_val = {k: rtrials[best_r][k] for k in ("val_poisdev", "val_outbreak_mae", "val_mae")}
        rep["2_recency"] = {"reference_val": base_val, "trials": rtrials,
                            "adopted": float(min(rpass, key=lambda k: rtrials[k]["cv_poisdev"])) if rpass else False}

        # ---- 3. isotonic recalibration (out-of-fold on train), judged on validation
        model = cls(feats, **cfg).fit(tr)
        base_val = val_score(model, va, thr)
        cal = Calibrated(cls, feats, **cfg).fit(tr)
        cal_val = val_score(cal, va, thr)
        iso_ok = (cal_val["val_outbreak_mae"] <= base_val["val_outbreak_mae"] * 0.90
                  and cal_val["val_poisdev"] <= base_val["val_poisdev"] * 1.02)
        post["isotonic"][str(h)] = bool(iso_ok)
        rep["3_isotonic"] = {"without": base_val, "with": cal_val, "adopted": bool(iso_ok)}
        final_model = cal if iso_ok else model

        # ---- 4. season-aware intervals: calibrate on 2022, check coverage on 2023
        p = final_model.predict(va)
        y, months, yr = va["target"].to_numpy(), va["target_date"].dt.month.to_numpy(), va["target_date"].dt.year.to_numpy()
        a, b = yr == 2022, yr == 2023
        q_old, q_new = conformal_fit(p[a], y[a], 0.80), conformal_fit(p[a], y[a], 0.80, months=months[a])
        cov_old = coverage(p[b], y[b], *conformal_interval(p[b], q_old), months[b])
        cov_new = coverage(p[b], y[b], *conformal_interval(p[b], q_new, months=months[b]), months[b])
        season_eval[h] = {"bin_only": cov_old, "season_aware": cov_new}

        new_tuned[str(h)] = {"hgb": cfg, "blend": blend}
        report["horizons"][h] = rep
        print(f"h={h}w weighting {rep['1_weighting']['adopted']} | recency {rep['2_recency']['adopted']} | "
              f"isotonic {iso_ok} (val outbreak MAE {base_val['val_outbreak_mae']:.2f} -> {cal_val['val_outbreak_mae']:.2f}, "
              f"dev {base_val['val_poisdev']:.3f} -> {cal_val['val_poisdev']:.3f}) | 2023 coverage bin-only "
              f"{cov_old['all']:.2f}/{cov_old['jul_oct']:.2f} vs season {cov_new['all']:.2f}/{cov_new['jul_oct']:.2f}",
              flush=True)

    gap_old = np.mean([abs(v["bin_only"]["jul_oct"] - 0.80) for v in season_eval.values()])
    gap_new = np.mean([abs(v["season_aware"]["jul_oct"] - 0.80) for v in season_eval.values()])
    season_ok = gap_new < gap_old and all(0.75 <= v["season_aware"]["all"] <= 0.85 for v in season_eval.values())
    post["season_intervals"] = bool(season_ok)
    report["4_season_intervals"] = {"per_horizon_2023_coverage": season_eval, "mean_jul_oct_gap_bin_only": gap_old,
                                    "mean_jul_oct_gap_season_aware": gap_new, "adopted": bool(season_ok)}
    OUT.write_text(json.dumps(report, indent=2))
    if new_tuned != {k: {"hgb": v["hgb"], "blend": v["blend"]} for k, v in tuned.items()}:
        HYPERPARAMS.write_text(json.dumps(new_tuned, indent=2))
        print(f"updated {HYPERPARAMS}")
    if season_ok or any(post["isotonic"].values()):
        POSTPROCESS.write_text(json.dumps(post, indent=2))
        print(f"wrote {POSTPROCESS}: {post}")
    print(f"season-aware ranges adopted={season_ok} (mean Jul-Oct gap {gap_old:.3f} -> {gap_new:.3f}); "
          f"saved {OUT} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
