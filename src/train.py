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
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
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
SELECTED = ROOT / "models" / "selected_features.json"   # written by src/feature_selection.py (CV-based)
ENSEMBLE = ROOT / "models" / "ensemble.json"            # written by src/ensemble.py, if adopted
HYPERPARAMS = ROOT / "models" / "hyperparameters.json"  # written by src/tuning.py (CV-based), if adopted
POSTPROCESS = ROOT / "models" / "postprocess.json"      # written by src/improvements.py, if adopted
FEATURE_LIST = ROOT / "reports" / "feature_list.json"   # final ordered inputs per mode/horizon


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


def hgb(max_iter=300, max_leaf_nodes=15, learning_rate=0.05, min_samples_leaf=40,
        l2_regularization=1.0, max_features=1.0, loss="poisson", **kw):
    # early_stopping=False: sklearn's default early stopping holds out a RANDOM 10% of rows,
    # which would shuffle time inside training. Tree count is chosen by time-ordered CV instead.
    return HistGradientBoostingRegressor(loss=loss, max_iter=max_iter, learning_rate=learning_rate,
                                         max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf,
                                         l2_regularization=l2_regularization, max_features=max_features,
                                         early_stopping=False, random_state=SEED, **kw)


def time_folds(tr):
    """Expanding-window folds by TARGET year inside the training period:
    fit on years < Y, validate on year Y, for Y in CV_YEARS. Equivalent to TimeSeriesSplit,
    but cut on dates: rows are stacked ward by ward, so row-count splits would cut mid-history."""
    yr = tr["target_date"].dt.year
    for y in CV_YEARS:
        yield y, tr[yr < y], tr[yr == y]


WEIGHT_KEYS = ("outbreak_weight", "outbreak_threshold", "recency_weight")


def split_weight_params(params, d):
    """Separate the sample-weight options from model settings and build the weights (or None).
      outbreak_weight w, outbreak_threshold t: rows with target >= t count w times (G5).
      recency_weight r: linear ramp by target year within this training set, oldest 1, newest 1 + r."""
    params = dict(params)
    w, thr, r = (params.pop(k, None) for k in WEIGHT_KEYS)
    sw = np.ones(len(d))
    if w and thr is not None:
        sw = sw * np.where(d["target"].to_numpy() >= thr, w, 1.0)
    if r:
        yr = d["target_date"].dt.year.to_numpy()
        span = max(yr.max() - yr.min(), 1)
        sw = sw * (1.0 + r * (yr - yr.min()) / span)
    return params, (sw if (w and thr is not None) or r else None)


class HGB:
    def __init__(self, feats, **params):
        self.feats, self.params = feats, params

    def fit(self, d):
        X = d[self.feats]
        # Columns with no observed value in this training set carry no information. This happens
        # for the rain-anomaly columns when a CV fold trains on 2016 alone (no prior-year history).
        # scikit-learn 1.6 tolerates them but newer versions raise, so they are skipped at fit and
        # at predict. Results are unchanged on 1.6.1 (verified fold by fold).
        self.used_ = [f for f in self.feats if X[f].notna().any()]
        params, sw = split_weight_params(self.params, d)
        self.m = hgb(**params).fit(X[self.used_], d["target"], sample_weight=sw)
        return self

    def predict(self, d):
        return self.m.predict(d[getattr(self, "used_", self.feats)])


class Blend:
    """Optional (G5): average of HistGradientBoosting and extra trees (median imputer fitted on
    the training data only, because extra trees cannot take NaN)."""
    def __init__(self, feats, **params):
        self.feats, self.params = feats, params

    def fit(self, d):
        self.hgb = HGB(self.feats, **self.params).fit(d)
        self.used_ = self.hgb.used_
        self.imp = SimpleImputer(strategy="median").fit(d[self.used_])
        _, sw = split_weight_params(self.params, d)     # same sample weights for both halves
        self.et = ExtraTreesRegressor(n_estimators=200, min_samples_leaf=20, max_features=0.5,
                                      n_jobs=-1, random_state=SEED).fit(self.imp.transform(d[self.used_]),
                                                                        d["target"], sample_weight=sw)
        return self

    def predict(self, d):
        et = np.clip(self.et.predict(self.imp.transform(d[self.used_])), 0, None)
        return 0.5 * (np.clip(self.hgb.predict(d), 0, None) + et)


class _Pipe:
    """A scikit-learn regressor on the feature list with a median imputer (+ optional scaler)
    fitted on the training data only (extra trees / random forest / GLM cannot take NaN)."""
    def __init__(self, feats, est, scale):
        self.feats, self.est, self.scale = feats, est, scale

    def fit(self, d):
        steps = [SimpleImputer(strategy="median")] + ([StandardScaler()] if self.scale else []) + [self.est]
        self.m = make_pipeline(*steps).fit(d[self.feats], d["target"])
        return self

    def predict(self, d):
        return np.clip(self.m.predict(d[self.feats]), 0, None)


class Ensemble:
    """Part H: weighted average of members with FIXED weights, learned by src/ensemble.py on
    out-of-fold training predictions (never on validation). Members with weight 0 are skipped."""
    def __init__(self, feats, weights, **params):
        self.feats, self.weights, self.params = feats, dict(weights), params

    def _member(self, name):
        if name == "hgb":
            return HGB(self.feats, **self.params)
        if name == "extra_trees":
            return _Pipe(self.feats, ExtraTreesRegressor(n_estimators=200, min_samples_leaf=20, max_features=0.5,
                                                         n_jobs=-1, random_state=SEED), scale=False)
        if name == "random_forest":
            return _Pipe(self.feats, RandomForestRegressor(n_estimators=200, min_samples_leaf=20, max_features=0.5,
                                                           n_jobs=-1, random_state=SEED), scale=False)
        if name == "poisson_glm":
            return _Pipe(self.feats, PoissonRegressor(alpha=1.0, max_iter=1000), scale=True)
        raise ValueError(f"unknown ensemble member {name!r}")

    def fit(self, d):
        self.members_ = {n: self._member(n).fit(d) for n, w in self.weights.items() if w > 0}
        self.used_ = self.members_["hgb"].used_ if "hgb" in self.members_ else list(self.feats)
        return self

    def predict(self, d):
        return sum(self.weights[n] * np.clip(m.predict(d), 0, None) for n, m in self.members_.items())


class Calibrated:
    """Optional outbreak fix: isotonic recalibration of a base model's predictions. The mapping is
    fitted on OUT-OF-FOLD predictions from the time-ordered folds inside the training data, so the
    validation years stay untouched for calibrating tiers/intervals and for checking the change."""
    def __init__(self, base_cls, feats, **params):
        self.base_cls, self.feats, self.params = base_cls, feats, params

    def fit(self, d):
        self.base = self.base_cls(self.feats, **self.params).fit(d)
        self.used_ = self.base.used_
        ps, ys = [], []
        for _, a, b in time_folds(d):
            if len(a) and len(b):
                ps.append(self.base_cls(self.feats, **self.params).fit(a).predict(b))
                ys.append(b["target"].to_numpy())
        self.iso = (IsotonicRegression(y_min=0, increasing=True, out_of_bounds="clip")
                    .fit(np.concatenate(ps), np.concatenate(ys)) if ps else None)
        return self

    def predict(self, d):
        p = np.clip(self.base.predict(d), 0, None)
        return self.iso.predict(p) if self.iso is not None else p


def deployed_model(m):
    """Fresh, unfitted model of the same kind, features and settings as a saved bundle entry
    (plain HistGradientBoosting, possibly weighted, the HGB + extra-trees blend, the weighted
    ensemble, optionally wrapped in isotonic recalibration)."""
    kind = m.get("base_kind") or m.get("kind", "")
    params = dict(m["hyperparameters"])
    if kind == "weighted_ensemble":
        cls, params = Ensemble, {**params, "weights": m["ensemble_weights"]}
    else:
        cls = Blend if kind == "blend_hgb_extra_trees" else HGB
    if m.get("calibrated"):
        return Calibrated(cls, list(m["features"]), **params)
    return cls(list(m["features"]), **params)


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


def _season(months):
    return np.where(np.isin(np.asarray(months), (7, 8, 9, 10)), "peak", "off")


def conformal_fit(val_pred, val_actual, coverage=0.80, months=None):
    """Split-conformal intervals, asymmetric and variance-scaled (count noise ~ sqrt(mean)),
    calibrated SEPARATELY per predicted-value bin (Mondrian). A single global calibration is
    dominated by off-season near-zero weeks and under-covers the outbreak season.
    Uses VALIDATION residuals of a train-only model. Returns {bin_index: (q_lo, q_hi)}."""
    p, y = np.asarray(val_pred), np.asarray(val_actual)
    b = np.digitize(p, CONF_BINS)
    q = {int(k): _conf_q(p[b == k], y[b == k], coverage) for k in np.unique(b)}
    if months is not None:   # season-aware: also calibrate per (prediction bin, Jul-Oct vs rest)
        s = _season(months)
        for k in np.unique(b):
            for sea in ("peak", "off"):
                sel = (b == k) & (s == sea)
                if sel.sum() >= 30:          # enough points for a stable 80% quantile
                    q[f"{int(k)}|{sea}"] = _conf_q(p[sel], y[sel], coverage)
    return q


def conformal_interval(pred, q, months=None):
    """80% range. With season-aware quantiles and the target months, use the (bin, season)
    calibration where available; otherwise the prediction-bin calibration."""
    pred = np.clip(np.asarray(pred, dtype=float), 0, None)
    b = np.digitize(pred, CONF_BINS)
    q = {(int(k) if isinstance(k, str) and k.isdigit() else k): v for k, v in q.items()}   # JSON keys
    int_keys = [k for k in q if isinstance(k, int)]
    fallback = q[max(int_keys)]
    if months is not None and any(isinstance(k, str) for k in q):
        s = np.broadcast_to(_season(months), b.shape)
        keys = [f"{int(k)}|{sea}" for k, sea in zip(b, s)]
        qq = [q.get(key, q.get(int(k), fallback)) for key, k in zip(keys, b)]
    else:
        qq = [q.get(int(k), fallback) for k in b]
    qlo = np.array([v[0] for v in qq])
    qhi = np.array([v[1] for v in qq])
    scale = np.sqrt(pred + 1.0)
    return np.clip(pred - qlo * scale, 0, None), pred + qhi * scale


def to_tier(pred, cuts):
    return np.digitize(pred, cuts)  # 0..3


MODES = ("weather", "weather_cases")   # default weather-only; optional + reported-case history


def test_metrics(te, p_te, conf_q, actual_cuts, cuts, baseline=None):
    """Final held-out numbers. Called once per mode/horizon; nothing downstream reads them to
    make a choice."""
    yte = te["target"].to_numpy()
    lo, hi = conformal_interval(p_te, conf_q, months=te["target_date"].dt.month.to_numpy())
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


def load_hyperparameters():
    """Per-horizon settings adopted by src/tuning.py under its pre-set rules, if present:
    {h: {"hgb": {...}, "blend": bool}}."""
    if HYPERPARAMS.exists():
        return {int(h): v for h, v in json.loads(HYPERPARAMS.read_text()).items()}
    return None


def load_postprocess():
    """Post-processing adopted by src/improvements.py: {"season_intervals": bool,
    "isotonic": {"2": bool, ...}}. Defaults to off."""
    if POSTPROCESS.exists():
        return json.loads(POSTPROCESS.read_text())
    return {"season_intervals": False, "isotonic": {}}


def load_selected():
    """Per-horizon feature lists chosen by time-ordered CV (src/feature_selection.py), if present."""
    if SELECTED.exists():
        return {int(h): f for h, f in json.loads(SELECTED.read_text()).items()}
    return None


def write_feature_list(bundle):
    """D14: the exact input columns and their order for every saved model."""
    out = {mode: {str(h): {"n_features": len(m["features"]), "features": list(m["features"])}
                  for h, m in hs.items()} for mode, hs in bundle["modes"].items()}
    FEATURE_LIST.write_text(json.dumps(out, indent=2))


def main():
    feats = feature_columns()
    selected = load_selected()
    tuned = load_hyperparameters()
    ens_cfg = {int(h): v for h, v in json.loads(ENSEMBLE.read_text()).items()} if ENSEMBLE.exists() else {}
    post = load_postprocess()
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
        # With a CV-selected feature list (D12) the grid runs on it and the anomaly decision is
        # already encoded in it; otherwise the base list is tuned and anomaly decided here.
        base = selected[h] if selected else feats
        if tuned:   # settings adopted by src/tuning.py (random search on the same time folds)
            best_cfg, best_cfg_key = dict(tuned[h]["hgb"]), "tuned"
            hgb_cv = {"tuned": cross_validate(lambda: HGB(base, **best_cfg), tr)}
        else:
            hgb_cv = {f"{c['max_iter']}x{c['max_leaf_nodes']}": cross_validate(lambda c=c: HGB(base, **c), tr)
                      for c in HGB_GRID}
            best_cfg_key = min(hgb_cv, key=lambda k: hgb_cv[k]["mean"]["PoissonDev"])
            best_cfg = next(c for c in HGB_GRID if f"{c['max_iter']}x{c['max_leaf_nodes']}" == best_cfg_key)
        use_blend = bool(tuned and tuned[h].get("blend"))
        ens_w = ens_cfg.get(h, {}).get("weights")     # Part H weighted ensemble, if adopted
        if ens_w:
            use_blend = False
        if selected:
            cv_anom = None
            use_anomaly = any(c.startswith("rain_anom") for c in base)
            feats_h = base
        else:
            # rain-anomaly features: kept only if they improve CV deviance at the chosen config
            feats_anom = feature_columns(anomaly=True)
            cv_anom = cross_validate(lambda: HGB(feats_anom, **best_cfg), tr)
            use_anomaly = cv_anom["mean"]["PoissonDev"] < hgb_cv[best_cfg_key]["mean"]["PoissonDev"]
            feats_h = feats_anom if use_anomaly else feats
        makers = {"seasonal_naive": SeasonalNaive, "poisson_glm": lambda: GLM(feats),
                  "hgb_poisson": (lambda: Ensemble(feats_h, ens_w, **best_cfg)) if ens_w
                                 else (lambda: Blend(feats_h, **best_cfg)) if use_blend
                                 else (lambda: HGB(feats_h, **best_cfg))}
        cv = {"seasonal_naive": cross_validate(makers["seasonal_naive"], tr),
              "poisson_glm": cross_validate(makers["poisson_glm"], tr),
              "hgb_poisson": cross_validate(makers["hgb_poisson"], tr) if (use_blend or ens_w)
                             else (cv_anom if (use_anomaly and cv_anom) else hgb_cv[best_cfg_key])}
        best = min(cv, key=lambda n: cv[n]["mean"]["PoissonDev"])

        # --- held-out VALIDATION: confirm choice, calibrate tiers + intervals (train-only models) ---
        base_kind = ("weighted_ensemble" if ens_w and best == "hgb_poisson" else
                     "blend_hgb_extra_trees" if use_blend and best == "hgb_poisson" else best)
        use_iso = bool(post.get("isotonic", {}).get(str(h))) and best == "hgb_poisson"
        if use_iso:   # isotonic recalibration on out-of-fold training predictions (outbreak fix)
            if ens_w:
                makers["hgb_poisson"] = lambda: Calibrated(Ensemble, feats_h, weights=ens_w, **best_cfg)
            else:
                base_cls = Blend if use_blend else HGB
                makers["hgb_poisson"] = lambda: Calibrated(base_cls, feats_h, **best_cfg)
        cands = {n: makers[n]().fit(tr) for n in makers}
        val = {n: scores(yva, cands[n].predict(va)) for n in cands}
        p_va = cands[best].predict(va)
        actual_cuts = tier_thresholds(tr["target"])
        cuts = calibrate_pred_cuts(p_va, yva, actual_cuts)
        months_va = va["target_date"].dt.month.to_numpy() if post.get("season_intervals") else None
        conf_q = conformal_fit(p_va, yva, 0.80, months=months_va)

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
                                         "tier_actual_cuts": actual_cuts,
                                         "kind": ("calibrated_" if use_iso else "") + base_kind,
                                         "base_kind": base_kind, "calibrated": use_iso,
                                         "ensemble_weights": ens_w if (ens_w and best == "hgb_poisson") else None,
                                         "season_intervals": bool(post.get("season_intervals")),
                                         "hyperparameters": best_cfg, "features": feats_final}
        report["horizons"][h] = {"selected": best, "hyperparameters": best_cfg, "cv": cv,
                                 "feature_source": "models/selected_features.json (CV selection)" if selected
                                                   else "default + CV anomaly decision",
                                 "n_features": len(feats_final),
                                 "rain_anomaly": {"used": bool(use_anomaly),
                                                  "cv_poisdev_without": None if cv_anom is None else hgb_cv[best_cfg_key]["mean"]["PoissonDev"],
                                                  "cv_poisdev_with": None if cv_anom is None else cv_anom["mean"]["PoissonDev"]},
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
        conf_c = conformal_fit(p_va_c, yva, 0.80, months=months_va)
        test_c, _, _ = test_metrics(te, m_c.predict(te), conf_c, actual_cuts, cuts_c)
        bundle["modes"]["weather_cases"][h] = {"model": make_c().fit(trva), "conformal_q": conf_c,
                                               "tier_cuts": cuts_c, "tier_actual_cuts": actual_cuts,
                                               "kind": "hgb_poisson", "hyperparameters": best_cfg,
                                               "features": feats_c}
        report["modes"]["weather_cases"][h] = {"cv": cv_c, "validation": scores(yva, p_va_c), "test": test_c,
                                               "tier_cuts_on_prediction": cuts_c, "conformal_q": conf_c}

        print(f"h={h}w selected={best} {best_cfg} features={len(feats_final)} rain_anomaly={use_anomaly}"
              f"{' (from CV selection)' if selected else ''}\n   CV PoisDev: " +
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
    write_feature_list(bundle)
    print(f"saved {BUNDLE} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    # Import main from the package module so pickled wrapper classes are recorded as
    # src.train.HGB (loadable anywhere), not __main__.HGB.
    from src.train import main as _main
    _main()
