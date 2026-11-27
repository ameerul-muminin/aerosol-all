"""Case count -> supplies needed, learned from the Phase 2 data.

In `bangladesh_dengue_large_dataset_onsite.xlsx` each supply column is a fixed function of
`dengue_cases`: every row with the same case count has the same saline / kits / beds. So:

1. For every case count seen in the data we keep the exact value (a lookup table).
2. For counts never seen (35-37, 40 and anything above 41), we use one ratio per item,
   fitted on the table by least squares through zero, then rounded half up.
   That same ratio also reproduces every seen count exactly (tested), so the two parts agree.

    from src.supplies import supplies_for, SUPPLY_ITEMS
    supplies_for(12)          -> {"saline_bags_needed": 29, "diagnostic_kits_needed": 14, "hospital_beds_required": 4}
    supplies_for([0, 3.4])    -> DataFrame, one row per input
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ONSITE_XLSX = ROOT / "bangladesh_dengue_large_dataset_onsite.xlsx"
SUPPLY_ITEMS = ["saline_bags_needed", "diagnostic_kits_needed", "hospital_beds_required"]


def round_half_up(x):
    """0.5 -> 1, 2.5 -> 3 (numpy's round would give 0 and 2). Works on scalars and arrays."""
    return np.floor(np.asarray(x, dtype=float) + 0.5).astype(int)


def learn_table(df: pd.DataFrame) -> pd.DataFrame:
    """Case count -> supplies, one row per distinct count. Fails loudly if the data is not a
    function of cases (two rows with the same count but different supplies)."""
    missing = {"dengue_cases", *SUPPLY_ITEMS} - set(df.columns)
    if missing:
        raise ValueError(f"supply data is missing columns: {sorted(missing)}")
    g = df.groupby("dengue_cases")[SUPPLY_ITEMS]
    clash = g.nunique().gt(1).any(axis=1)
    if clash.any():
        raise ValueError(f"supplies are not a fixed function of cases at counts {list(clash[clash].index)}")
    return g.first().astype(int).sort_index()


def learn_ratios(table: pd.DataFrame) -> dict[str, float]:
    """Supplies per case for each item: least squares through zero on the table."""
    c = table.index.to_numpy(dtype=float)
    return {item: float((c * table[item]).sum() / (c * c).sum()) for item in SUPPLY_ITEMS}


@lru_cache(maxsize=1)
def load_model(path: Path = ONSITE_XLSX) -> tuple[pd.DataFrame, dict[str, float]]:
    """(table, ratios) learned from the onsite dataset. Cached: the xlsx takes a few seconds."""
    if not Path(path).exists():
        raise ValueError(f"Supply data not found: {Path(path).name}. Put it in the repo root.")
    table = learn_table(pd.read_excel(path, usecols=["dengue_cases", *SUPPLY_ITEMS]))
    return table, learn_ratios(table)


def case_count(predicted) -> np.ndarray:
    """Forecasts are expected values (e.g. 3.4); supplies are bought for whole patients.
    Negative values are clipped to 0, then rounded half up."""
    return round_half_up(np.clip(np.asarray(predicted, dtype=float), 0, None))


def supplies_for(cases, model: tuple[pd.DataFrame, dict[str, float]] | None = None):
    """Supplies for one case count (-> dict) or many (-> DataFrame with `cases` + one column per item)."""
    table, ratios = model or load_model()
    scalar = np.ndim(cases) == 0
    n = case_count(np.atleast_1d(cases))
    out = pd.DataFrame({"cases": n})
    seen = np.isin(n, table.index)
    for item in SUPPLY_ITEMS:
        vals = round_half_up(ratios[item] * n)
        vals[seen] = table[item].reindex(n[seen]).to_numpy()
        out[item] = vals
    if scalar:
        return {item: int(out[item].iloc[0]) for item in SUPPLY_ITEMS}
    return out
