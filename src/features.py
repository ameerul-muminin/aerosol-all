"""Weather lag pipeline. All features are computed from the raw dataset, per ward_id.

Convention: a feature row is indexed by the *forecast origin* week t. It may only use
information from weeks <= t. The target is dengue_cases at week t + horizon.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_XLSX = ROOT / "hackathon_preli_dataset.xlsx"
CACHE_CSV = ROOT / "data" / "dataset_cache.csv"

HORIZONS = (2, 3, 4)
STATIC = ["population_density_sqkm", "water_logging_risk_index", "sanitation_score"]
WEATHER = ["avg_temp_c", "max_temp_c", "min_temp_c", "rainfall_mm", "relative_humidity_pct"]
RAIN_LAGS = range(0, 9)       # rainfall at t-0 .. t-8 weeks
TEMP_HUM_LAGS = range(0, 5)   # temp / humidity at t-0 .. t-4 weeks


def load_raw(path: Path = DATA_XLSX) -> pd.DataFrame:
    """Read the dataset (xlsx is slow, so cache a CSV copy) and validate it."""
    if CACHE_CSV.exists() and CACHE_CSV.stat().st_mtime >= Path(path).stat().st_mtime:
        df = pd.read_csv(CACHE_CSV, parse_dates=["date"])
    else:
        df = pd.read_excel(path)
        CACHE_CSV.parent.mkdir(exist_ok=True)
        df.to_csv(CACHE_CSV, index=False)
    return validate(df)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Fail loudly if the data violates assumptions the lag logic depends on."""
    need = {"date", "city", "ward_id", "dengue_cases", *STATIC, *WEATHER}
    missing = need - set(df.columns)
    assert not missing, f"missing columns: {missing}"
    assert not df[list(need)].isna().any().any(), "unexpected missing values"
    assert not df.duplicated(["ward_id", "date"]).any(), "duplicate ward/date rows"
    df = df.sort_values(["ward_id", "date"]).reset_index(drop=True)
    gaps = df.groupby("ward_id")["date"].diff().dropna()
    assert (gaps == pd.Timedelta(days=7)).all(), "non-weekly spacing: shift(k) would not mean k weeks"
    assert (df.groupby("ward_id")[STATIC].nunique() == 1).all().all(), "ward attributes not constant"
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-ward multi-week lag and rolling features (no cross-ward leakage)."""
    df = df.sort_values(["ward_id", "date"]).reset_index(drop=True)
    g = df.groupby("ward_id", sort=False)
    new = {}
    for k in RAIN_LAGS:
        new[f"rain_lag{k}"] = g["rainfall_mm"].shift(k)
    for col, short in [("avg_temp_c", "temp"), ("relative_humidity_pct", "hum")]:
        for k in TEMP_HUM_LAGS:
            new[f"{short}_lag{k}"] = g[col].shift(k)
    lagged = pd.DataFrame(new, index=df.index)
    # Rolling accumulations over the breeding-cycle windows
    lagged["rain_sum_0_1"] = lagged[["rain_lag0", "rain_lag1"]].sum(axis=1, skipna=False)
    lagged["rain_sum_2_4"] = lagged[[f"rain_lag{k}" for k in (2, 3, 4)]].sum(axis=1, skipna=False)
    lagged["rain_sum_0_3"] = lagged[[f"rain_lag{k}" for k in range(4)]].sum(axis=1, skipna=False)
    lagged["rain_sum_0_8"] = lagged[[f"rain_lag{k}" for k in range(9)]].sum(axis=1, skipna=False)
    lagged["temp_mean_0_3"] = lagged[[f"temp_lag{k}" for k in range(4)]].mean(axis=1, skipna=False)
    lagged["hum_mean_0_3"] = lagged[[f"hum_lag{k}" for k in range(4)]].mean(axis=1, skipna=False)
    out = pd.concat([df, lagged], axis=1)
    woy = out["date"].dt.isocalendar().week.astype(int)
    out["week_sin"] = np.sin(2 * np.pi * woy / 52.18)
    out["week_cos"] = np.cos(2 * np.pi * woy / 52.18)
    return out


def feature_columns() -> list[str]:
    cols = list(STATIC) + ["week_sin", "week_cos"]
    cols += [f"rain_lag{k}" for k in RAIN_LAGS]
    cols += [f"{s}_lag{k}" for s in ("temp", "hum") for k in TEMP_HUM_LAGS]
    cols += ["rain_sum_0_1", "rain_sum_2_4", "rain_sum_0_3", "rain_sum_0_8", "temp_mean_0_3", "hum_mean_0_3"]
    return cols


def add_target(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """target = cases `horizon` weeks after the forecast origin, within the same ward."""
    df = df.copy()
    df["target_date"] = df["date"] + pd.Timedelta(weeks=horizon)
    df["target"] = df.groupby("ward_id", sort=False)["dengue_cases"].shift(-horizon)
    return df


def build_dataset(horizon: int, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rows with complete features and a known target for `horizon`, plus target_date."""
    df = add_lag_features(load_raw() if df is None else df)
    df = add_target(df, horizon)
    return df.dropna(subset=feature_columns() + ["target"]).reset_index(drop=True)
