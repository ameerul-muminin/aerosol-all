"""Weather lag pipeline. All features are computed from the raw dataset, per ward_id.

Convention: a feature row is indexed by the *forecast origin* week t. It may only use
information from weeks <= t. The target is dengue_cases at week t + horizon.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PHASE1_XLSX = ROOT / "hackathon_preli_dataset.xlsx"
ONSITE_XLSX = ROOT / "bangladesh_dengue_large_dataset_onsite.xlsx"
# Phase 2 data = the same 31,320 rows and Phase 1 columns + the supply columns below.
# Use it when present; the Phase 1 file still works on its own.
DATA_XLSX = ONSITE_XLSX if ONSITE_XLSX.exists() else PHASE1_XLSX
CACHE_CSV = ROOT / "data" / "dataset_cache.csv"
# Supplies needed in the SAME week as the cases (a fixed function of dengue_cases), so they
# are outputs, never model inputs: using them as features would leak the target.
SUPPLY_COLUMNS = ["saline_bags_needed", "diagnostic_kits_needed", "hospital_beds_required"]

HORIZONS = (2, 3, 4)
STATIC = ["population_density_sqkm", "water_logging_risk_index", "sanitation_score"]
WEATHER = ["avg_temp_c", "max_temp_c", "min_temp_c", "rainfall_mm", "relative_humidity_pct"]
RAIN_LAGS = range(0, 9)       # rainfall at t-0 .. t-8 weeks
TEMP_HUM_LAGS = range(0, 5)   # temp / humidity at t-0 .. t-4 weeks
ANOM_LAGS = range(0, 5)       # rain anomaly at t-0 .. t-4 weeks
CASE_LAGS = range(0, 5)       # reported cases at t-0 .. t-4 weeks (optional mode)


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Read the dataset and validate it. The default file (DATA_XLSX) is cached as a CSV,
    because xlsx is slow; any other path is read directly."""
    path = DATA_XLSX if path is None else Path(path)
    if path != DATA_XLSX:
        return validate(pd.read_excel(path))
    df = None
    if CACHE_CSV.exists() and CACHE_CSV.stat().st_mtime >= path.stat().st_mtime:
        df = pd.read_csv(CACHE_CSV, parse_dates=["date"])
        # a cache made from the other file (with/without supply columns) is stale
        has_supplies = set(SUPPLY_COLUMNS) <= set(df.columns)
        if has_supplies != (path == ONSITE_XLSX):
            df = None
    if df is None:
        df = pd.read_excel(path)
        CACHE_CSV.parent.mkdir(exist_ok=True)
        df.to_csv(CACHE_CSV, index=False)
    return validate(df)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Fail loudly (ValueError, not assert: asserts vanish under `python -O`) if the data
    violates assumptions the lag logic depends on."""
    need = {"date", "city", "ward_id", "dengue_cases", *STATIC, *WEATHER}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if df[list(need)].isna().any().any():
        raise ValueError("unexpected missing values")
    if df.duplicated(["ward_id", "date"]).any():
        raise ValueError("duplicate ward/date rows")
    df = df.sort_values(["ward_id", "date"]).reset_index(drop=True)
    gaps = df.groupby("ward_id")["date"].diff().dropna()
    if not (gaps == pd.Timedelta(days=7)).all():
        raise ValueError("non-weekly spacing: shift(k) would not mean k weeks")
    if not (df.groupby("ward_id")[STATIC].nunique() == 1).all().all():
        raise ValueError("ward attributes not constant")
    supplies = [c for c in SUPPLY_COLUMNS if c in df.columns]
    if supplies:                                   # Phase 2 file: all three or none
        if len(supplies) != len(SUPPLY_COLUMNS):
            raise ValueError(f"missing supply columns: {sorted(set(SUPPLY_COLUMNS) - set(supplies))}")
        s = df[SUPPLY_COLUMNS]                     # NaN is fine: what-if weeks have no supplies yet
        if (s < 0).any().any() or (s.fillna(0) % 1 != 0).any().any():
            raise ValueError("supply columns must be whole numbers, 0 or more")
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

    # Rain anomaly = rain minus this ward's normal rain for the same ISO week, where "normal"
    # is the mean over PREVIOUS YEARS ONLY (expanding mean, shifted one year). Nothing is
    # fitted, so it cannot leak; the first year has no history and stays NaN (HGB handles NaN).
    clim = (out.groupby(["ward_id", woy], sort=False)["rainfall_mm"]
               .transform(lambda s: s.shift(1).expanding().mean()))
    anom = out["rainfall_mm"] - clim
    ga = anom.groupby(out["ward_id"], sort=False)
    extra = {f"rain_anom_lag{k}": ga.shift(k) for k in ANOM_LAGS}
    # Reported cases up to the origin week (known at time t; target is >= 2 weeks later)
    gc = out.groupby("ward_id", sort=False)["dengue_cases"]
    extra.update({f"cases_lag{k}": gc.shift(k) for k in CASE_LAGS})
    extra = pd.DataFrame(extra, index=out.index)
    extra["rain_anom_sum_2_4"] = extra[[f"rain_anom_lag{k}" for k in (2, 3, 4)]].sum(axis=1, skipna=False)
    extra["rain_anom_sum_0_3"] = extra[[f"rain_anom_lag{k}" for k in range(4)]].sum(axis=1, skipna=False)
    extra["cases_sum_0_3"] = extra[[f"cases_lag{k}" for k in range(4)]].sum(axis=1, skipna=False)
    out = pd.concat([out, extra], axis=1)
    city = city_case_table(out)
    return out.merge(city, on=["city", "date"], how="left", validate="many_to_one")


CITY_COLUMNS = ["city_rate0", "city_rate1", "city_rate_mean4", "city_rate_mean8", "city_trend"]


def city_case_table(df: pd.DataFrame) -> pd.DataFrame:
    """City-wide outbreak signal per (city, origin week t): average reported cases per ward in
    the city at t, t-1, the last 4 and last 8 weeks, and the 3-week change. One ward's weekly
    count is noisy (58% zeros); the city average is a steadier sign of a building outbreak.
    Uses only cases reported at weeks <= t, so it is known at forecast time."""
    c = (df.groupby(["city", "date"])["dengue_cases"].mean().rename("city_rate0")
           .reset_index().sort_values(["city", "date"]))
    g = c.groupby("city", sort=False)["city_rate0"]
    lag = {k: g.shift(k) for k in range(8)}
    c["city_rate1"] = lag[1]
    c["city_rate_mean4"] = sum(lag[k] for k in range(4)) / 4
    c["city_rate_mean8"] = sum(lag[k] for k in range(8)) / 8
    c["city_trend"] = c["city_rate0"] - lag[3]
    return c[["city", "date", *CITY_COLUMNS]]


def anomaly_columns() -> list[str]:
    return [f"rain_anom_lag{k}" for k in ANOM_LAGS] + ["rain_anom_sum_2_4", "rain_anom_sum_0_3"]


def case_columns() -> list[str]:
    return [f"cases_lag{k}" for k in CASE_LAGS] + ["cases_sum_0_3"]


def feature_columns(anomaly: bool = False, cases: bool = False, city: bool = False) -> list[str]:
    """Model inputs. Default = weather + ward attributes + season. `anomaly` adds rain
    anomalies; `city` adds city-wide reported-case rates (used by the 4-week model, adopted by
    CV); `cases` adds the ward's own reported-case history (optional weather+cases mode)."""
    cols = list(STATIC) + ["week_sin", "week_cos"]
    cols += [f"rain_lag{k}" for k in RAIN_LAGS]
    cols += [f"{s}_lag{k}" for s in ("temp", "hum") for k in TEMP_HUM_LAGS]
    cols += ["rain_sum_0_1", "rain_sum_2_4", "rain_sum_0_3", "rain_sum_0_8", "temp_mean_0_3", "hum_mean_0_3"]
    if anomaly:
        cols += anomaly_columns()
    if city:
        cols += CITY_COLUMNS
    if cases:
        cols += case_columns()
    leak = set(cols) & set(SUPPLY_COLUMNS)
    if leak:                                       # same-week supplies would leak the target
        raise ValueError(f"supply columns must never be model inputs: {sorted(leak)}")
    return cols


def required_columns(cases: bool = False) -> list[str]:
    """Features that must be present for a row to be usable. Anomaly columns are allowed to
    be NaN (first year of history); the model handles missing values natively."""
    return feature_columns(cases=cases)


def add_target(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """target = cases `horizon` weeks after the forecast origin, within the same ward."""
    df = df.copy()
    df["target_date"] = df["date"] + pd.Timedelta(weeks=horizon)
    df["target"] = df.groupby("ward_id", sort=False)["dengue_cases"].shift(-horizon)
    return df


def build_dataset(horizon: int, df: pd.DataFrame | None = None, featurized: bool = False,
                  cases: bool = False) -> pd.DataFrame:
    """Rows with complete features and a known target for `horizon`, plus target_date.
    Pass featurized=True with the output of add_lag_features() to skip recomputing lags."""
    if not featurized:
        df = add_lag_features(load_raw() if df is None else df)
    df = add_target(df, horizon)
    return df.dropna(subset=required_columns(cases) + ["target"]).reset_index(drop=True)
