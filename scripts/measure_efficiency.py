"""Measure compute cost of the serving path and write reports/efficiency.json.

    python scripts/measure_efficiency.py                      # serving numbers (also: python -m scripts.measure_efficiency)
    python scripts/measure_efficiency.py --train              # also time a full retrain
    python scripts/measure_efficiency.py --train --train-repeats 3   # retrain 3x, report median / min / max
    python scripts/measure_efficiency.py --out /tmp/eff.json --calls 3   # quick run elsewhere (used by the tests)

Every number in ARCHITECTURE.md's efficiency section must come from this file.
Outbound network is blocked while measuring (and always restored afterwards), so the run
also proves the app makes zero external API calls. `--train` backs up models/ and reports/
first and restores them afterwards, so timing a retrain never changes the deployed model.
Timings depend on how busy the machine is: latencies are medians over many calls, the
retrain can be repeated, and the machine load is recorded next to the numbers.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import platform
import shutil
import socket
import statistics
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "efficiency.json"
if str(ROOT) not in sys.path:          # allow `python scripts/measure_efficiency.py` from the repo root
    sys.path.insert(0, str(ROOT))
N_CALLS = 50

_net_attempts: list[str] = []
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


def _load() -> dict:
    """1-minute load average and CPU count, so readers can judge how busy the machine was."""
    try:
        return {"load_avg_1m": round(os.getloadavg()[0], 2), "cpu_count": os.cpu_count()}
    except (AttributeError, OSError):          # Windows has no getloadavg
        return {"load_avg_1m": None, "cpu_count": os.cpu_count()}


def _time_retrain(repeats: int) -> dict:
    """Time src.train.main `repeats` times; models/ and reports/ are restored afterwards."""
    from src import train
    backup = Path(tempfile.mkdtemp(prefix="eff_backup_"))
    dirs = [d for d in (ROOT / "models", ROOT / "reports") if d.exists()]
    for d in dirs:
        shutil.copytree(d, backup / d.name)
    runs = []
    try:
        for _ in range(repeats):
            t = time.perf_counter()
            train.main()
            runs.append(time.perf_counter() - t)
    finally:
        for d in dirs:                                # put the deployed model + reports back exactly
            shutil.rmtree(d)
            shutil.copytree(backup / d.name, d)
        shutil.rmtree(backup, ignore_errors=True)
    out = {"full_retrain_3_horizons": statistics.median(runs)}
    if repeats > 1:
        out.update(full_retrain_3_horizons_min=min(runs), full_retrain_3_horizons_max=max(runs))
    return out


def measure(calls: int = N_CALLS, train: bool = False, train_repeats: int = 1) -> dict:
    res = {"python": platform.python_version(), "platform": platform.platform(), "seconds": {}, "bytes": {},
           "machine_at_start": _load()}
    _net_attempts.clear()
    socket.socket.connect = _blocked_connect
    tracemalloc.start()
    try:
        from src import predict
        from src.train import BUNDLE
        from api import service
        from api.main import create_app

        res["bytes"]["model_file"] = BUNDLE.stat().st_size
        res["model_file_mb"] = round(BUNDLE.stat().st_size / 1e6, 2)          # read from the file on disk
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
        _, res["seconds"]["ranking_60_wards_cached_median"] = timed(lambda: get(rank), repeat=calls)
        _, res["seconds"]["forecast_one_ward_cached_median"] = timed(lambda: get(fc), repeat=calls)

        # forecast logic WITHOUT caches: every call recomputes the model, tiers, range and Shapley drivers
        wards = [w for c in service.meta()["cities"] for w in c["wards"]]
        cold = []
        for i in range(calls):
            service.forecast.cache_clear(); service._ranking_df.cache_clear()
            t = time.perf_counter()
            service.forecast(wards[i % len(wards)], week, 3)
            cold.append(time.perf_counter() - t)
        res["seconds"]["forecast_one_ward_uncached_median"] = statistics.median(cold)
        res["seconds"]["forecast_one_ward_uncached_p95"] = sorted(cold)[max(0, int(0.95 * len(cold)) - 1)]
        res["calls_per_median"] = calls

        def post_what_if(rain):
            r = client.post("/api/what_if", json={"ward_id": "CCC_Ward_01", "as_of": week, "horizon": 3,
                                                  "rainfall_mm": rain, "weeks": 4})
            assert r.status_code == 200, (r.status_code, r.data[:200])
            return r
        _, res["seconds"]["what_if_new_scenario"] = timed(lambda: post_what_if(150.0))
        _, res["seconds"]["what_if_repeat_scenario_median"] = timed(lambda: post_what_if(150.0), repeat=calls)

        # early warnings reuse the per-horizon forecasts above; breeding clock is computed once from raw data
        warn = f"/api/warnings?as_of={week}"
        _, res["seconds"]["warnings_60_wards_first_request"] = timed(lambda: get(warn))
        _, res["seconds"]["warnings_60_wards_cached_median"] = timed(lambda: get(warn), repeat=calls)
        _, res["seconds"]["lag_profile_first_request"] = timed(lambda: get("/api/lag_profile"))
        _, res["seconds"]["lag_profile_cached_median"] = timed(lambda: get("/api/lag_profile"), repeat=calls)

        other = "2024-08-05"
        _, res["seconds"]["forecast_new_week_all_horizons_first_request"] = timed(
            lambda: get(f"/api/forecast?ward_id=DNCC_Ward_05&as_of={other}&horizon=4"))

        raw, gz = get(rank).get_data(), get(rank, gz=True).get_data()
        res["bytes"]["ranking_response_raw"] = len(raw)
        res["bytes"]["ranking_response_gzip"] = len(gz)
        assert gzip.decompress(gz) == raw

        if train:
            res["seconds"].update(_time_retrain(train_repeats))
            res["train_repeats"] = train_repeats

        _, peak = tracemalloc.get_traced_memory()
        res["bytes"]["python_heap_peak"] = peak
        res["external_api_calls"] = len(_net_attempts)        # outbound sockets are blocked and counted
        res["external_network_calls"] = res["external_api_calls"]   # older key, kept for compatibility
        res["llm_calls"] = 0                                  # the app has no LLM client at all
    finally:
        socket.socket.connect = _real_connect                 # always restore networking
        tracemalloc.stop()
    res["machine_at_end"] = _load()
    res["measured_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    res["seconds"] = {k: round(v, 4) for k, v in res["seconds"].items()}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--train", action="store_true", help="also time a full retrain (model files are restored)")
    ap.add_argument("--train-repeats", type=int, default=1, help="retrain this many times; report median/min/max")
    ap.add_argument("--calls", type=int, default=N_CALLS, help="calls per latency median (default 50)")
    ap.add_argument("--out", type=Path, default=OUT, help="where to write the JSON (default reports/efficiency.json)")
    args = ap.parse_args(argv)
    if args.calls < 1 or args.train_repeats < 1:
        ap.error("--calls and --train-repeats must be at least 1")
    res = measure(calls=args.calls, train=args.train, train_repeats=args.train_repeats)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
