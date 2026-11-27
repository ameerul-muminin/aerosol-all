"""src/selfcheck.py: live math checks with the network blocked, and GET /api/selfcheck."""
import socket

import numpy as np
import pytest

from src.selfcheck import NetworkBlocked, block_network, network_is_blocked, run_checks

PEAK, H = "2025-07-14", 3


@pytest.fixture(scope="module")
def feat():
    from src.predict import build_features
    return build_features()


@pytest.fixture(scope="module")
def peak(feat):
    return run_checks(feat, PEAK, H)


def test_all_checks_pass_on_a_peak_week(peak):
    failed = [c for c in peak["checks"] if not c["passed"]]
    assert peak["passed"] and peak["network_blocked"] and not failed, failed
    assert peak["as_of"] == PEAK and peak["horizon"] == H and peak["seconds"] > 0


def test_every_required_check_is_present(peak):
    names = " | ".join(c["name"] for c in peak["checks"])
    for needed in ["supplies table reproduces every dataset row",
                   "allocation 30%: allocated <= available", "allocation 60%", "allocation 100%: no deficit",
                   "higher-priority wards never less fulfilled", "crews used <= crews available",
                   "overflow 100% beds: patients conserved", "overflow 70% beds: no transfer crosses cities",
                   "no receiver over capacity", "simulator: crews used <= total",
                   "never below the training floor", "simulator: cases after <= before",
                   "English SMS is ASCII and <= 160 chars", "critical notices contain the deficit alert",
                   "determinism: supplies", "determinism: allocation", "determinism: overflow",
                   "determinism: simulator", "determinism: dispatch", "model loads from the local file"]:
        assert needed in names, needed
    assert all(set(c) == {"name", "passed", "detail"} and c["detail"] for c in peak["checks"])


def test_a_broken_allocation_is_caught(feat, monkeypatch):
    """If the fill ignored the stock, the check must fail (the checks are not rubber stamps)."""
    import src.allocation as allocation
    monkeypatch.setattr(allocation, "_greedy_fill", lambda required, available: np.asarray(required))
    r = run_checks(feat, PEAK, H)
    bad = {c["name"] for c in r["checks"] if not c["passed"]}
    assert not r["passed"]
    assert "allocation 30%: allocated <= available per item" in bad


def test_block_network_blocks_outbound_and_restores():
    real = (socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo, socket.create_connection)
    with block_network() as net:
        assert network_is_blocked()
        with pytest.raises(NetworkBlocked):
            socket.create_connection(("example.com", 80), timeout=1)
        with pytest.raises(NetworkBlocked):
            socket.getaddrinfo("example.com", 443)
        assert len(net["attempts"]) == 3          # the probe + the two above
    # all four originals are back (checked by identity: no real connection is ever attempted)
    assert (socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo, socket.create_connection) == real


def test_localhost_still_allowed_while_blocked():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    try:
        with block_network() as net:
            c = socket.create_connection(srv.getsockname(), timeout=2)
            c.close()
            assert net["attempts"] == []
    finally:
        srv.close()


def test_bad_input_gives_friendly_errors(feat):
    with pytest.raises(ValueError, match="Horizon"):
        run_checks(feat, PEAK, 5)
    with pytest.raises(ValueError, match="Mondays"):
        run_checks(feat, "2025-07-15", 2)


# ---------------------------------------------------------------- API

@pytest.fixture(scope="module")
def client():
    from api.main import create_app
    return create_app().test_client()


def test_api_selfcheck_defaults_to_latest_week_and_horizon_2(client):
    from api import service
    r = client.get("/api/selfcheck")
    assert r.status_code == 200
    body = r.get_json()
    assert body["as_of"] == service.meta()["as_of"]["max"] and body["horizon"] == 2
    assert body["passed"] and body["network_blocked"]
    assert r.headers["Cache-Control"] == "no-store"


def test_api_selfcheck_with_params(client):
    body = client.get(f"/api/selfcheck?as_of={PEAK}&horizon={H}").get_json()
    assert body["passed"] and body["as_of"] == PEAK and body["horizon"] == H


@pytest.mark.parametrize("query, msg", [("horizon=7", "Horizon"), ("as_of=14-07-2025", "YYYY-MM-DD"),
                                        ("as_of=2025-07-15", "Mondays"), ("horizon=abc", "Horizon")])
def test_api_selfcheck_bad_input_is_a_400(client, query, msg):
    r = client.get(f"/api/selfcheck?{query}")
    assert r.status_code == 400 and msg in r.get_json()["error"]
