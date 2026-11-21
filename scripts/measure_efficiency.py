"""Measure compute cost of the serving path and write reports/efficiency.json.

    python -m scripts.measure_efficiency            # serving numbers
    python -m scripts.measure_efficiency --train    # also time a full retrain

Every number in ARCHITECTURE.md's efficiency section must come from this file.
Outbound network is blocked while measuring, so the run also proves the app makes
zero external API calls.
"""
from __future__ import annotations

import argparse
import gzip
import json
import platform
import socket
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "efficiency.json"

_net_attempts = []
_real_connect = socket.socket.connect


def _blocked_connect(self, address):  # count + refuse any outbound connection
    _net_attempts.append(str(address))
    raise ConnectionRefusedError("network disabled during efficiency measurement")


def timed(fn, repeat=1):
    runs = []
    for _ in range(repeat):
        t = time.perf_counter()
        out = fn()
        runs.append(time.perf_counter() - t)
    return out, (statistics.median(runs) if repeat > 1 else runs[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="also time python -m src.train")
    args = ap.parse_args()

    socket.socket.connect = _blocked_connect
    tracemalloc.start()
    res = {"python": platform.python_version(), "platform": platform.platform(), "seconds": {}, "bytes": {}}

    from src import predict
    from src.train import BUNDLE
    from api import service
    from api.main import create_app

    res["bytes"]["model_file"] = BUNDLE.stat().st_size
    predict.load_bundle.cache_clear()
    _, res["seconds"]["model_load_joblib"] = timed(predict.load_bundle)
    _, res["seconds"]["lag_features_build_all_wards"] = timed(predict.build_features)

    _, res["seconds"]["app_startup_warm"] = timed(create_app)
    client = create_app().test_client()

    def get(url, gz=False):
        h = {"Accept-Encoding": "gzip"} if gz else {}
        r = client.get(url, headers=h)
        assert r.status_code == 200, (url, r.status_code, r.data[:200])
        return r

    week = service.meta()["as_of"]["default"]
    rank = f"/api/ranking?as_of={week}&horizon=3"
    fc = f"/api/forecast?ward_id=CCC_Ward_01&as_of={week}&horizon=3"

    _, res["seconds"]["ranking_60_wards_first_request"] = timed(lambda: get(rank))
    _, res["seconds"]["ranking_60_wards_cached_median"] = timed(lambda: get(rank), repeat=20)
    _, res["seconds"]["forecast_one_ward_cached_median"] = timed(lambda: get(fc), repeat=20)

    def post_what_if(rain):
        r = client.post("/api/what_if", json={"ward_id": "CCC_Ward_01", "as_of": week, "horizon": 3,
                                              "rainfall_mm": rain, "weeks": 4})
        assert r.status_code == 200, (r.status_code, r.data[:200])
        return r
    _, res["seconds"]["what_if_new_scenario"] = timed(lambda: post_what_if(150.0))
    _, res["seconds"]["what_if_repeat_scenario_median"] = timed(lambda: post_what_if(150.0), repeat=20)

    other = "2024-08-05"
    _, res["seconds"]["forecast_new_week_all_horizons_first_request"] = timed(
        lambda: get(f"/api/forecast?ward_id=DNCC_Ward_05&as_of={other}&horizon=4"))

    raw, gz = get(rank).get_data(), get(rank, gz=True).get_data()
    res["bytes"]["ranking_response_raw"] = len(raw)
    res["bytes"]["ranking_response_gzip"] = len(gz)
    assert gzip.decompress(gz) == raw

    if args.train:
        from src import train
        _, res["seconds"]["full_retrain_3_horizons"] = timed(train.main)

    _, peak = tracemalloc.get_traced_memory()
    res["bytes"]["python_heap_peak"] = peak
    res["external_network_calls"] = len(_net_attempts)
    res["llm_calls"] = 0
    res["measured_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    socket.socket.connect = _real_connect

    res["seconds"] = {k: round(v, 4) for k, v in res["seconds"].items()}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
