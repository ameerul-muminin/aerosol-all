"""Data loading and quality checks for the ward-level dengue dataset."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_XLSX = ROOT / "hackathon_preli_dataset.xlsx"
CACHE_CSV = ROOT / "data" / "dataset_cache.csv"

STATIC = ["population_density_sqkm", "water_logging_risk_index", "sanitation_score"]
WEATHER = ["avg_temp_c", "max_temp_c", "min_temp_c", "rainfall_mm", "relative_humidity_pct"]


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

