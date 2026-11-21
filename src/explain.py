"""Per-prediction drivers: exact grouped Shapley values.

For one ward-week, how many predicted cases does each group of inputs add or remove,
compared with an "average week" reference?

  Groups: Rainfall, Temperature, Humidity, Season, Ward vulnerability (+ Recent cases in
          mode="weather_cases").
  Reference: mean of every feature over the TRAINING period only (dates <= TRAIN_END), so
             nothing from validation/test leaks into explanations.
  Method: interventional Shapley over groups. With G groups there are 2^G on/off
          combinations (32 for weather mode); each is one batched model call, so all 60
          wards are explained in milliseconds without extra libraries.
  Property: contributions add up exactly to (prediction - reference prediction).
"""
from __future__ import annotations
from functools import lru_cache
from itertools import product
from math import factorial
import numpy as np
import pandas as pd

from src.features import STATIC, required_columns
from src.train import TRAIN_END

GROUP_ORDER = ["Rainfall", "Temperature", "Humidity", "Season", "Ward vulnerability", "Recent cases"]


def group_of(feature: str) -> str:
    if feature.startswith("rain_"):
        return "Rainfall"
    if feature.startswith("temp_"):
        return "Temperature"
    if feature.startswith("hum_"):
        return "Humidity"
    if feature in ("week_sin", "week_cos"):
        return "Season"
    if feature in STATIC:
        return "Ward vulnerability"
    if feature.startswith("cases_"):
        return "Recent cases"
    raise ValueError(f"no driver group for feature {feature!r}")


def column_name(group: str) -> str:
    return "driver_" + group.lower().replace(" ", "_")


@lru_cache(maxsize=4)
def _reference(features: tuple, cases: bool) -> pd.Series:
    from src.predict import _featurized   # lazy: avoids a circular import
    d = _featurized()
    d = d[d["date"] <= TRAIN_END].dropna(subset=required_columns(cases))
    return d[list(features)].mean()        # NaN-aware: anomaly features skip year-1 NaNs


def contributions(model, rows: pd.DataFrame, cases: bool = False) -> pd.DataFrame:
    """Return one row per input row: base_cases, driver_<group> columns, top_driver."""
    feats = list(model.feats)
    groups = [g for g in GROUP_ORDER if any(group_of(f) == g for f in feats)]
    G = len(groups)
    member = {g: [f for f in feats if group_of(f) == g] for g in groups}
    ref = _reference(tuple(feats), cases)
    X = rows[feats].reset_index(drop=True)
    n = len(X)
    masks = list(product((0, 1), repeat=G))            # all 2^G coalitions
    # batch: for each coalition, features of "on" groups come from the row, others from ref
    blocks = []
    for mask in masks:
        b = X.copy()
        for g, on in zip(groups, mask):
            if not on:
                b[member[g]] = ref[member[g]].to_numpy()
        blocks.append(b)
    f = model.predict(pd.concat(blocks, ignore_index=True)).reshape(len(masks), n)
    value = {mask: f[i] for i, mask in enumerate(masks)}
    phi = np.zeros((G, n))
    for i in range(G):
        for mask in masks:
            if mask[i]:
                continue
            k = sum(mask)
            w = factorial(k) * factorial(G - k - 1) / factorial(G)
            with_i = mask[:i] + (1,) + mask[i + 1:]
            phi[i] += w * (value[with_i] - value[mask])
    out = pd.DataFrame({column_name(g): phi[i] for i, g in enumerate(groups)})
    out.insert(0, "base_cases", value[tuple([0] * G)])
    top = np.argmax(phi, axis=0)
    out["top_driver"] = [groups[j] if phi[j, r] > 0 else "None (below average)" for r, j in enumerate(top)]
    return out


def describe(row: pd.Series) -> str:
    """Human-readable driver line, e.g. 'Rainfall +6.1 | Season +3.0 | ...'."""
    parts = [(c, row[c]) for c in row.index if c.startswith("driver_")]
    parts.sort(key=lambda kv: -abs(kv[1]))
    return " | ".join(f"{c[7:].replace('_', ' ').capitalize()} {v:+.1f}" for c, v in parts)


if __name__ == "__main__":
    import sys
    from src.predict import predict
    date = sys.argv[1] if len(sys.argv) > 1 else "2025-08-04"
    h = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    out = predict(date, h, explain=True)
    print(f"Origin {date}, +{h} weeks. Average-week reference = {out['base_cases'].iloc[0]:.2f} cases\n")
    for _, r in out.head(5).iterrows():
        print(f"{r['ward_id']:<13} {r['predicted_cases']:5.1f} cases [{r['risk_tier']}]  {describe(r)}")


# ---------------------------------------------------------------- global explanations (G8)

def global_contributions(model, rows: pd.DataFrame, cases: bool = False) -> dict:
    """Mean |contribution| and mean signed contribution per driver group over many ward-weeks."""
    c = contributions(model, rows, cases=cases)
    cols = [k for k in c.columns if k.startswith("driver_")]
    return {k[7:]: {"mean_abs_cases": float(c[k].abs().mean()), "mean_signed_cases": float(c[k].mean())}
            for k in sorted(cols, key=lambda k: -c[k].abs().mean())}


# Partial-dependence targets: name -> how to set the inputs for a value v (keeps linked columns consistent)
def _set_breeding_rain(X, v):
    X[["rain_lag2", "rain_lag3", "rain_lag4"]] = v / 3.0
    X["rain_sum_2_4"] = v
    return X


PDP_TARGETS = {
    "Rain 2-4 weeks ago, total mm (breeding window)": (["rain_lag2", "rain_lag3", "rain_lag4", "rain_sum_2_4"],
                                                       "rain_sum_2_4", _set_breeding_rain),
    "Humidity, 4-week mean %": (["hum_mean_0_3"], "hum_mean_0_3", None),
    "Ward water-logging risk index": (["water_logging_risk_index"], "water_logging_risk_index", None),
    "Ward sanitation score": (["sanitation_score"], "sanitation_score", None),
    "Ward population density per km2": (["population_density_sqkm"], "population_density_sqkm", None),
}


def partial_dependence(model, rows: pd.DataFrame, n_grid: int = 15, max_rows: int = 2000, seed: int = 0) -> dict:
    """Brute-force partial dependence (model-agnostic): set the input(s) to each grid value for a
    sample of real ward-weeks and average the predictions. Grid spans the 5th-95th percentile."""
    feats = set(model.feats)
    sample = rows.sample(min(max_rows, len(rows)), random_state=seed)
    out = {}
    for label, (needed, axis_col, setter) in PDP_TARGETS.items():
        if not set(needed) <= feats:
            continue
        lo, hi = np.nanpercentile(rows[axis_col], [5, 95])
        grid = np.unique(np.round(np.linspace(lo, hi, n_grid), 4))
        curve = []
        for v in grid:
            X = sample.copy()
            X = setter(X, v) if setter else X.assign(**{axis_col: v})
            curve.append({"value": float(v), "mean_predicted_cases": float(np.mean(model.predict(X)))})
        out[label] = curve
    return out
