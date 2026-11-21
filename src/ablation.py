"""Ablation: how much do weather lags and ward attributes add? (report only; model unchanged)

The deployed model type for each horizon (same settings, weights, ensemble) is refitted on the
same 5 time-ordered CV folds and on train -> validation, with:
  full          all deployed features
  without_weather rain / temperature / humidity features removed (season + ward attributes left)
  without_ward_attributes  the three ward attributes removed (weather + season left)
The test set is not used. Writes reports/ablation.json. Run: python -m src.ablation
"""
from __future__ import annotations
import json
import time
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance

from src.features import ROOT, HORIZONS, STATIC, build_dataset, add_lag_features, load_raw
from src.train import BUNDLE, deployed_model, split, time_folds

OUT = ROOT / "reports" / "ablation.json"
WEATHER_PREFIXES = ("rain_", "temp_", "hum_")


def variants(feats):
    return {"full": feats,
            "without_weather": [f for f in feats if not f.startswith(WEATHER_PREFIXES)],
            "without_ward_attributes": [f for f in feats if f not in STATIC]}


def score(m, feats, tr, va):
    spec = {**m, "features": feats}
    devs, maes = [], []
    for _, fit, val in time_folds(tr):
        p = deployed_model(spec).fit(fit).predict(val)
        devs.append(mean_poisson_deviance(val["target"], np.clip(p, 1e-3, None)))
        maes.append(mean_absolute_error(val["target"], p))
    pv = deployed_model(spec).fit(tr).predict(va)
    return {"n_features": len(feats), "cv_poisdev": float(np.mean(devs)), "cv_mae": float(np.mean(maes)),
            "val_poisdev": float(mean_poisson_deviance(va["target"], np.clip(pv, 1e-3, None))),
            "val_mae": float(mean_absolute_error(va["target"], pv))}


def main():
    t0 = time.time()
    bundle = joblib.load(BUNDLE)
    featurized = add_lag_features(load_raw())
    report = {"what": __doc__.split("The test")[0].strip(), "horizons": {}}
    for h in HORIZONS:
        m = bundle["modes"]["weather"][h]
        tr, va, _ = split(build_dataset(h, featurized, featurized=True, cases=True))
        res = {name: score(m, f, tr, va) for name, f in variants(list(m["features"])).items()}
        full = res["full"]
        for name in ("without_weather", "without_ward_attributes"):
            r = res[name]
            r["cv_mae_increase_pct"] = round(100 * (r["cv_mae"] / full["cv_mae"] - 1), 1)
            r["cv_poisdev_increase_pct"] = round(100 * (r["cv_poisdev"] / full["cv_poisdev"] - 1), 1)
        report["horizons"][h] = {"model_kind": m.get("kind"), "results": res}
        print(f"h={h}w full CV MAE {full['cv_mae']:.3f} dev {full['cv_poisdev']:.3f} | without weather: MAE "
              f"{res['without_weather']['cv_mae']:.3f} (+{res['without_weather']['cv_mae_increase_pct']}%), dev "
              f"+{res['without_weather']['cv_poisdev_increase_pct']}% | without ward attributes: MAE "
              f"{res['without_ward_attributes']['cv_mae']:.3f} (+{res['without_ward_attributes']['cv_mae_increase_pct']}%)", flush=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(f"saved {OUT} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
