"""The efficiency script runs end to end, proves 0 external calls, never overwrites the committed
report unless asked, always restores networking, and `--train` never changes the deployed model."""
import hashlib
import json
import socket

import pytest

from scripts import measure_efficiency as me
from src.train import BUNDLE


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_quick_run_writes_elsewhere_and_reports_real_values(tmp_path):
    committed = me.OUT
    before = _sha(committed) if committed.exists() else None
    out = tmp_path / "eff.json"
    assert me.main(["--out", str(out), "--calls", "2"]) == 0
    r = json.loads(out.read_text())
    assert r["external_api_calls"] == 0 and r["llm_calls"] == 0
    assert r["model_file_mb"] == round(BUNDLE.stat().st_size / 1e6, 2)
    for k in ("model_load_joblib", "lag_features_build_all_wards", "ranking_60_wards_cached_median",
              "forecast_one_ward_uncached_median", "what_if_new_scenario"):
        assert r["seconds"][k] > 0
    assert r["calls_per_median"] == 2 and r["bytes"]["ranking_response_gzip"] < r["bytes"]["ranking_response_raw"]
    assert {"load_avg_1m", "cpu_count"} <= set(r["machine_at_start"])
    assert (_sha(committed) if committed.exists() else None) == before, "committed report must not change"
    assert socket.socket.connect is me._real_connect, "networking must be restored"


def test_network_block_counts_and_is_restored_after_a_crash(monkeypatch):
    me._net_attempts.clear()
    s = socket.socket()
    with pytest.raises(ConnectionRefusedError):
        me._blocked_connect(s, ("example.com", 80))
    s.close()
    assert me._net_attempts == ["('example.com', 80)"]
    from api import service
    monkeypatch.setattr(service, "meta", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        me.measure(calls=1)
    assert socket.socket.connect is me._real_connect


def test_train_timing_restores_models_and_reports(tmp_path, monkeypatch):
    (tmp_path / "models").mkdir(); (tmp_path / "reports").mkdir()
    (tmp_path / "models" / "dengue_model.joblib").write_bytes(b"deployed-model")
    (tmp_path / "reports" / "metrics.json").write_text('{"real": true}')
    monkeypatch.setattr(me, "ROOT", tmp_path)
    from src import train
    def scribble():                                    # a "retrain" that changes the files
        (tmp_path / "models" / "dengue_model.joblib").write_bytes(b"retrained")
        (tmp_path / "reports" / "metrics.json").write_text('{"real": false}')
    monkeypatch.setattr(train, "main", scribble)
    t = me._time_retrain(repeats=3)
    assert (tmp_path / "models" / "dengue_model.joblib").read_bytes() == b"deployed-model"
    assert json.loads((tmp_path / "reports" / "metrics.json").read_text()) == {"real": True}
    assert t["full_retrain_3_horizons_min"] <= t["full_retrain_3_horizons"] <= t["full_retrain_3_horizons_max"]


def test_rejects_bad_arguments():
    with pytest.raises(SystemExit):
        me.main(["--calls", "0"])
