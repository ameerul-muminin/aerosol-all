"""Test-set lock: a SHA-256 fingerprint of the held-out test rows (target weeks 2024-2025).

The test set is used once, for reporting. This file records exactly which rows it contains so
anyone can verify it never changed while models were tuned (tests/test_test_lock.py recomputes it).
Decision-making scripts must not even unpack the test split; that is checked by the same test.
Run: python -m src.test_lock   (only to create the lock; re-running must give the same hashes)
"""
from __future__ import annotations
import hashlib
import json

from src.features import ROOT, HORIZONS, build_dataset, add_lag_features, load_raw
from src.train import split

LOCK = ROOT / "reports" / "test_set_lock.json"


def fingerprint() -> dict:
    featurized = add_lag_features(load_raw())
    out = {}
    for h in HORIZONS:
        _, _, te = split(build_dataset(h, featurized, featurized=True, cases=True))
        rows = te[["ward_id", "date", "target_date", "target"]].sort_values(["ward_id", "target_date"])
        payload = rows.to_csv(index=False, date_format="%Y-%m-%d").encode()
        out[str(h)] = {"sha256": hashlib.sha256(payload).hexdigest(), "rows": int(len(rows)),
                       "first_target_week": str(rows["target_date"].min().date()),
                       "last_target_week": str(rows["target_date"].max().date())}
    return out


if __name__ == "__main__":
    LOCK.write_text(json.dumps({"what": "held-out test rows (ward, origin week, target week, cases) per horizon",
                                "horizons": fingerprint()}, indent=2))
    print(LOCK.read_text())
