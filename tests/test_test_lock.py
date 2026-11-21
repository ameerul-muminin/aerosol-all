"""The held-out test set is locked and decision-making code never touches it."""
import json
import re
from pathlib import Path

from src.test_lock import LOCK, fingerprint

ROOT = Path(__file__).resolve().parents[1]
# Scripts that only REPORT final numbers may read the test split; every other script that calls
# split() chooses something (features, settings, models) and must discard it.
REPORTING_ONLY = {"src/train.py", "src/evaluate.py", "src/test_lock.py", "src/ranking_eval.py"}


def test_test_set_unchanged_since_lock():
    locked = json.loads(LOCK.read_text())["horizons"]
    assert fingerprint() == locked, "test rows changed: re-check that nothing was tuned on them"


def test_decision_scripts_never_unpack_the_test_split():
    scripts = [p for p in ROOT.glob("src/**/*.py")] + [p for p in ROOT.glob("experiments/**/*.py")]
    checked = 0
    for path in scripts:
        rel = path.relative_to(ROOT).as_posix()
        if rel in REPORTING_ONLY:
            continue
        for lhs in re.findall(r"^\s*(.+?)=\s*split\(", path.read_text(), flags=re.M):
            parts = [p.strip() for p in lhs.split(",")]
            assert len(parts) == 3 and parts[2] == "_", f"{rel}: test split must be discarded, got '{lhs.strip()}'"
            checked += 1
    assert checked >= 5, "expected the feature-selection, tuning, improvement, ensemble and comparison scripts"
