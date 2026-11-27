"""Application rate limiting (api/ratelimit.py): the public API refuses floods with a friendly 429,
per client and for everyone together, and never limits health checks, static files or local use."""
import pytest

from api.main import create_app
from api.ratelimit import RateLimiter, TokenBucket


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def limiter(**kw):
    clock = Clock()
    return RateLimiter(clock=clock, **kw), clock


def test_bucket_allows_a_burst_then_refills_at_the_rate():
    b = TokenBucket(per_min=60, burst=3, now=0)          # one token per second, 3 at most
    assert [b.take(0)[0] for _ in range(3)] == [True, True, True]
    ok, wait = b.take(0)
    assert not ok and wait == pytest.approx(1.0)
    assert b.take(0.5) == (False, pytest.approx(0.5))
    assert b.take(1.0)[0]
    assert b.take(100)[0] and b.tokens == pytest.approx(2)  # refill never goes above the burst


def test_each_client_has_its_own_bucket():
    rl, clock = limiter(per_min=60, burst=2)
    a, b = "203.0.113.1", "198.51.100.2"
    assert [rl.check("/api/meta", "10.0.0.9", a)[0] for _ in range(3)] == [True, True, False]
    assert rl.check("/api/meta", "10.0.0.9", b)[0]       # another client is not affected
    ok, wait, who = rl.check("/api/meta", "10.0.0.9", a)
    assert (ok, wait, who) == (False, 1, "client")
    clock.t += 1
    assert rl.check("/api/meta", "10.0.0.9", a)[0]       # one token back after one second


def test_global_cap_stops_many_spoofed_clients():
    rl, clock = limiter(per_min=600, burst=100, global_per_min=60, global_burst=5)
    results = [rl.check("/api/meta", "10.0.0.9", f"203.0.113.{i}") for i in range(8)]
    assert [r[0] for r in results] == [True] * 5 + [False] * 3
    assert results[-1][2] == "server"
    assert rl.clients["203.0.113.7"].tokens == pytest.approx(100)   # its token was given back: not its fault


def test_never_limited_health_static_and_this_machine():
    rl, _ = limiter(per_min=60, burst=1)
    for _ in range(5):
        assert rl.check("/api/health", "10.0.0.9", "203.0.113.1")[0]
        assert rl.check("/app/", "10.0.0.9", "203.0.113.1")[0]
        assert rl.check("/assets/js/app.js", "10.0.0.9", "203.0.113.1")[0]
        assert rl.check("/api/meta", "127.0.0.1", None)[0]
        assert rl.check("/api/meta", "::1", None)[0]


def test_a_forwarded_loopback_address_is_still_limited():
    rl, _ = limiter(per_min=60, burst=1)          # a proxy header cannot buy the local exemption
    assert [rl.check("/api/meta", "10.0.0.9", "127.0.0.1")[0] for _ in range(2)] == [True, False]


def test_client_is_the_first_forwarded_address():
    assert RateLimiter.client_of("10.0.0.9", "203.0.113.5, 10.0.0.2") == "203.0.113.5"
    assert RateLimiter.client_of("203.0.113.8", None) == "203.0.113.8"
    assert RateLimiter.client_of("127.0.0.1", None) is None


def test_memory_is_bounded(monkeypatch):
    import api.ratelimit as rlmod
    monkeypatch.setattr(rlmod, "MAX_CLIENTS", 3)
    rl, _ = limiter(per_min=60, burst=5, global_burst=100)
    for i in range(10):
        rl.check("/api/meta", "10.0.0.9", f"203.0.113.{i}")
    assert list(rl.clients) == ["203.0.113.7", "203.0.113.8", "203.0.113.9"]


def test_settings_from_environment():
    rl = RateLimiter.from_env({"RATE_LIMIT_PER_MIN": "30", "RATE_LIMIT_BURST": "10"})
    assert (rl.per_min, rl.burst, rl.enabled) == (30, 10, True)
    assert not RateLimiter.from_env({"RATE_LIMIT": "off"}).enabled
    off = RateLimiter.from_env({"RATE_LIMIT": "off", "RATE_LIMIT_BURST": "1"})
    assert all(off.check("/api/meta", "10.0.0.9", "203.0.113.1")[0] for _ in range(5))
    with pytest.raises(ValueError):
        RateLimiter.from_env({"RATE_LIMIT_PER_MIN": "lots"})
    with pytest.raises(ValueError):
        RateLimiter(per_min=0)


@pytest.fixture(scope="module")
def tight_app():
    app = create_app()
    rl, _ = limiter(per_min=60, burst=2)
    app.extensions["rate_limiter"].__dict__.update(rl.__dict__)   # same limiter object the route uses
    return app


def test_http_429_is_friendly_json_with_retry_after(tight_app):
    c = tight_app.test_client()
    who = {"X-Forwarded-For": "203.0.113.50"}
    assert [c.get("/api/meta", headers=who).status_code for _ in range(2)] == [200, 200]
    r = c.get("/api/meta", headers=who)
    assert r.status_code == 429 and r.is_json and r.headers["Retry-After"] == "1"
    assert set(r.get_json()) == {"error"} and "wait 1 second " in r.get_json()["error"]
    assert c.post("/api/allocate", json={"as_of": "2025-07-14", "stock_pct": 60}, headers=who).status_code == 429
    assert c.get("/api/health", headers=who).status_code == 200      # the host's health checks keep working
    assert c.get("/api/meta", headers={"X-Forwarded-For": "198.51.100.60"}).status_code == 200
    assert all(c.get("/api/meta").status_code == 200 for _ in range(5))   # local use is never limited
