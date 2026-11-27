"""Live self-check of the Phase 2 math, run with the network blocked.

    from src.predict import build_features
    from src.selfcheck import run_checks
    run_checks(build_features(), "2025-12-29", 2)
    -> {"passed": bool, "network_blocked": bool, "checks": [{"name", "passed", "detail"}], "seconds"}

Every check recomputes its answer from the real modules (supplies, allocation, overflow,
intervention, dispatch, the saved model) at the requested week; nothing is hard-coded. While the
checks run, outbound network is blocked in-process (see `block_network`), so a pass also shows
that none of this needs the internet. Served at GET /api/selfcheck.
"""
from __future__ import annotations

import contextlib
import ipaddress
import json
import socket
import time

import numpy as np
import pandas as pd

from src.supplies import SUPPLY_ITEMS

STOCK_LEVELS = (30, 60, 100)
BED_LEVELS = (100, 70)
CREWS = 12
TOLERANCE = 0.05          # cases; simulator forecasts are rounded to 0.01


# ---------------------------------------------------------------- network block

class NetworkBlocked(OSError):
    """Raised for any attempt to reach a non-local address while the block is active."""


def _is_local(host) -> bool:
    if host is None:
        return True
    host = host.decode() if isinstance(host, bytes) else str(host)
    if host.lower() in ("localhost", ""):
        return True
    try:
        return ipaddress.ip_address(host.split("%")[0]).is_loopback
    except ValueError:
        return False                      # a hostname other than localhost needs DNS: block it


@contextlib.contextmanager
def block_network():
    """Inside this block, connecting (or resolving a name) to anything but localhost raises
    NetworkBlocked. Unix sockets and loopback still work. The originals are always restored.
    Yields a dict whose "attempts" lists every blocked address."""
    state = {"attempts": []}
    real_connect, real_connect_ex = socket.socket.connect, socket.socket.connect_ex
    real_getaddrinfo, real_create = socket.getaddrinfo, socket.create_connection

    def _guard(address):
        host = address[0] if isinstance(address, tuple) else None    # str/bytes = Unix socket path
        if not _is_local(host):
            state["attempts"].append(str(address))
            raise NetworkBlocked(f"network blocked by selfcheck: {address}")

    def connect(self, address):
        _guard(address)
        return real_connect(self, address)

    def connect_ex(self, address):
        _guard(address)
        return real_connect_ex(self, address)

    def getaddrinfo(host, *args, **kwargs):
        _guard((host, 0))
        return real_getaddrinfo(host, *args, **kwargs)

    def create_connection(address, *args, **kwargs):
        _guard(address)
        return real_create(address, *args, **kwargs)

    socket.socket.connect, socket.socket.connect_ex = connect, connect_ex
    socket.getaddrinfo, socket.create_connection = getaddrinfo, create_connection
    try:
        yield state
    finally:
        socket.socket.connect, socket.socket.connect_ex = real_connect, real_connect_ex
        socket.getaddrinfo, socket.create_connection = real_getaddrinfo, real_create


def network_is_blocked() -> bool:
    """Prove the block works: an outbound connect must fail before it leaves the machine.
    Call it only inside block_network(); the short timeout keeps it from hanging otherwise."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.5)
    try:
        s.connect(("203.0.113.1", 80))   # TEST-NET-3 (RFC 5737): never a real host
    except NetworkBlocked:
        return True
    except OSError:
        return False                      # a real network error means the connect was attempted
    finally:
        s.close()
    return False


# ---------------------------------------------------------------- checks

def _check(name: str, ok, detail: str) -> dict:
    return {"name": name, "passed": bool(ok), "detail": detail}


def _supplies() -> tuple[list[dict], object]:
    from src.features import load_raw
    from src.supplies import supplies_for
    raw = load_raw()
    missing = [c for c in SUPPLY_ITEMS if c not in raw.columns]
    if missing:
        return [_check("supplies table reproduces every dataset row", False,
                       f"dataset has no supply columns {missing}")], None
    got = supplies_for(raw["dengue_cases"].to_numpy())
    same = (got[SUPPLY_ITEMS].to_numpy() == raw[SUPPLY_ITEMS].to_numpy()).all(axis=1)
    return [_check("supplies table reproduces every dataset row", same.all(),
                   f"{int(same.sum())} of {len(raw)} rows match exactly")], got[SUPPLY_ITEMS].to_numpy().tolist()


def _allocation(feat, as_of, horizon) -> tuple[list[dict], object]:
    from src.allocation import ITEM_KEYS, plan, totals
    out, snapshot = [], []
    for pct in STOCK_LEVELS:
        a = plan(feat, as_of, horizon, pct, crews=CREWS)
        t = totals(a, pct, crews=CREWS)
        snapshot.append(a.to_json())
        over = {k: (v["allocated"], v["available"]) for k, v in t["items"].items() if v["allocated"] > v["available"]}
        out.append(_check(f"allocation {pct}%: allocated <= available per item", not over,
                          "; ".join(f"{k} {v['allocated']}/{v['available']}" for k, v in t["items"].items())))
        if pct == 100:
            short = int(a[[f"{k}_deficit" for k in ITEM_KEYS.values()]].sum().sum())
            out.append(_check("allocation 100%: no deficit", short == 0, f"total deficit {short} units"))
        bad = []
        for k in ITEM_KEYS.values():          # wards in priority order; zero-need wards count as 100%
            need = a[a[f"{k}_required"] > 0]
            if not need[f"{k}_fulfilled_pct"].is_monotonic_decreasing:
                bad.append(k)
        out.append(_check(f"allocation {pct}%: higher-priority wards never less fulfilled",
                          not bad, "monotonic for saline, kits, beds" if not bad else f"broken for {bad}"))
        used = int(a["crews_assigned"].sum())
        out.append(_check(f"allocation {pct}%: crews used <= crews available", used <= CREWS,
                          f"{used} of {CREWS} crews"))
    return out, snapshot


def _overflow(feat, as_of, horizon) -> tuple[list[dict], object]:
    from src.overflow import overflow_plan
    out, snapshot = [], []
    city_of = feat.drop_duplicates("ward_id").set_index("ward_id")["city"]
    for pct in BED_LEVELS:
        r = overflow_plan(feat, as_of, horizon, beds_available_pct=pct)
        snapshot.append(r)
        wards = r["wards"]
        moved = sum(t["patients"] for t in r["transfers"])
        admitted = sum(w["beds_required"] - w["routed_out"] + w["routed_in"] - w["net_deficit"] for w in wards.values())
        stranded = sum(c["unroutable"] for c in r["cities"].values())
        need = sum(w["beds_required"] for w in wards.values())
        conserved = (moved == sum(w["routed_out"] for w in wards.values()) == sum(w["routed_in"] for w in wards.values())
                     and admitted + stranded == need)
        out.append(_check(f"overflow {pct}% beds: patients conserved", conserved,
                          f"{need} need a bed = {admitted} admitted + {stranded} unroutable; {moved} moved"))
        cross = [t for t in r["transfers"]
                 if not (city_of[t["from_ward"]] == city_of[t["to_ward"]] == t["city"])]
        out.append(_check(f"overflow {pct}% beds: no transfer crosses cities", not cross,
                          f"{len(r['transfers'])} transfers, {len(cross)} cross-city"))
        full = [w for w, v in wards.items()
                if v["routed_in"] > 0 and v["beds_required"] - v["routed_out"] + v["routed_in"] > v["capacity"]]
        out.append(_check(f"overflow {pct}% beds: no receiver over capacity", not full,
                          f"{sum(1 for v in wards.values() if v['routed_in'] > 0)} receivers checked"))
    return out, snapshot


def _simulator(feat, as_of) -> tuple[list[dict], object]:
    from src.intervention import input_floor, simulate
    col = "water_logging_risk_index"
    day = feat[feat["date"] == pd.Timestamp(as_of)].sort_values("ward_id")
    busiest = day.sort_values(["population_density_sqkm", "ward_id"], ascending=[False, True])["ward_id"].iloc[:2]
    lowest = day.sort_values([col, "ward_id"])["ward_id"].iloc[0]          # exercises the floor
    plan = {busiest.iloc[0]: 2, busiest.iloc[1]: 1, lowest: 3}
    total = 8
    s = simulate(feat, as_of, plan, total_crews=total)
    floor = input_floor(feat, col)
    out = [_check("simulator: crews used <= total", s["crews_used"] <= total, f"{s['crews_used']} of {total} crews")]
    low = [w["ward_id"] for w in s["wards"] if w["input_after"] < floor - 1e-9]
    out.append(_check("simulator: water-logging never below the training floor", not low,
                      f"floor {floor:g}; lowest after {min(w['input_after'] for w in s['wards']):g}; "
                      f"floor reached in {len(s['floor_hit_wards'])} ward(s)"))
    worst = max(v["cases_after"] - v["cases_before"] for w in s["wards"] for v in w["by_horizon"].values())
    worst = max(worst, max(h["cases_after"] - h["cases_before"] for h in s["horizons"]))
    out.append(_check(f"simulator: cases after <= before + {TOLERANCE}", worst <= TOLERANCE,
                      f"largest change after - before: {worst:+.2f} cases"))
    return out, s


def _dispatch(feat, as_of, horizon) -> tuple[list[dict], object]:
    from src.dispatch import TEMPLATES, build_notices, sms
    notices = build_notices(feat, as_of, horizon, 60, CREWS, 70, "en")
    texts = [sms(n) for n in notices]
    bad = [n["ward_id"] for n, t in zip(notices, texts) if not t.isascii() or len(t) > 160]
    out = [_check("dispatch: every English SMS is ASCII and <= 160 chars", not bad,
                  f"{len(texts)} SMS, longest {max(map(len, texts))} chars, {len(bad)} failing")]
    alert = TEMPLATES["en"]["alert"]
    critical = [(n, t) for n, t in zip(notices, texts) if n["critical"]]
    missing = [n["ward_id"] for n, t in critical if alert not in t]
    out.append(_check("dispatch: critical notices contain the deficit alert", not missing,
                      f"{len(critical)} critical notices, {len(missing)} without '{alert}'"))
    return out, texts


def _model() -> list[dict]:
    import joblib
    from src.train import BUNDLE
    if not BUNDLE.exists():
        return [_check("model loads from the local file", False, f"{BUNDLE.name} is missing")]
    bundle = joblib.load(BUNDLE)            # straight from disk, while the network is blocked
    horizons = sorted(bundle["modes"]["weather"])
    return [_check("model loads from the local file", horizons == [2, 3, 4],
                   f"{BUNDLE.name}, {BUNDLE.stat().st_size / 1e6:.1f} MB, horizons {horizons}")]


def _fingerprint(x) -> str:
    return json.dumps(x, sort_keys=True, default=str)


def run_checks(feat_df: pd.DataFrame, as_of_date, horizon: int) -> dict:
    """Run every check at (as_of_date, horizon) with the network blocked. Bad input -> ValueError."""
    from src.predict import _check_inputs
    as_of = _check_inputs(feat_df, as_of_date, horizon)      # friendly ValueError for a bad date/horizon
    start = time.perf_counter()
    checks = []
    with block_network() as net:
        blocked = network_is_blocked()
        net["attempts"].clear()                               # the probe itself is not a real attempt
        checks += _model()
        steps = {
            "supplies": lambda: _supplies(),
            "allocation": lambda: _allocation(feat_df, as_of, horizon),
            "overflow": lambda: _overflow(feat_df, as_of, horizon),
            "simulator": lambda: _simulator(feat_df, as_of),
            "dispatch": lambda: _dispatch(feat_df, as_of, horizon),
        }
        first = {}
        for name, step in steps.items():
            found, first[name] = step()
            checks += found
        for name, step in steps.items():                      # determinism: run everything again
            again = step()[1]
            checks.append(_check(f"determinism: {name} identical on a second run",
                                 _fingerprint(first[name]) == _fingerprint(again), "compared as JSON"))
        checks.append(_check("no network access attempted during the checks", not net["attempts"],
                             f"{len(net['attempts'])} blocked attempt(s)"))
    return {"passed": blocked and all(c["passed"] for c in checks), "network_blocked": blocked,
            "as_of": str(as_of.date()), "horizon": int(horizon),
            "checks": checks, "seconds": round(time.perf_counter() - start, 2)}
