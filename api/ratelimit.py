"""Application rate limiting for the public API (standard library only, no extra package).

The live URL is open to anyone, e.g. on public Wi-Fi, so one client must not be able to tie up the server:
- Each client gets a token bucket: every /api request takes one token, and tokens come back at a steady
  rate up to a burst size. An empty bucket answers HTTP 429 with a plain-English message and `Retry-After`;
  the UI shows it like any other API error.
- A global bucket caps all clients together, so spoofed client addresses cannot add up to a flood.
- Never limited: /api/health (the host's health checks) and static files. Requests that reach the app
  straight from this machine (127.0.0.1 / ::1, no proxy header) are not limited either: offline use on an
  officer's own laptop, the test suite and scripts/offline_check.py.
- Behind a proxy (Render), the client is the first address in X-Forwarded-For.

Settings (environment variables, read at startup): RATE_LIMIT=off disables it; RATE_LIMIT_PER_MIN (default
120) and RATE_LIMIT_BURST (default 60) per client; RATE_LIMIT_GLOBAL_PER_MIN (default 1200) and
RATE_LIMIT_GLOBAL_BURST (default 300) for everyone together. Buckets live in this process's memory, so with
several gunicorn workers each worker counts separately.
"""
from __future__ import annotations

import math
import os
import threading
import time
from collections import OrderedDict

EXEMPT_PATHS = {"/api/health"}
LOOPBACK = {"127.0.0.1", "::1", "localhost"}
MAX_CLIENTS = 10_000          # least recently seen clients are forgotten beyond this (bounded memory)


class TokenBucket:
    """`burst` tokens, refilled at `per_min` per minute. take() -> (allowed, seconds until a token is back)."""

    def __init__(self, per_min: float, burst: float, now: float):
        self.rate = per_min / 60.0
        self.burst = float(burst)
        self.tokens = float(burst)
        self.last = now

    def take(self, now: float) -> tuple[bool, float]:
        self.tokens = min(self.burst, self.tokens + (now - self.last) * self.rate)
        self.last = now
        if self.tokens >= 1:
            self.tokens -= 1
            return True, 0.0
        return False, (1 - self.tokens) / self.rate


class RateLimiter:
    def __init__(self, per_min: float = 120, burst: float = 60, global_per_min: float = 1200,
                 global_burst: float = 300, enabled: bool = True, clock=time.monotonic):
        if min(per_min, burst, global_per_min, global_burst) <= 0:
            raise ValueError("Rate limits must be positive numbers.")
        self.per_min, self.burst = per_min, burst
        self.enabled, self.clock = enabled, clock
        self.clients: OrderedDict[str, TokenBucket] = OrderedDict()
        self.everyone = TokenBucket(global_per_min, global_burst, clock())
        self.lock = threading.Lock()

    @classmethod
    def from_env(cls, env=os.environ) -> "RateLimiter":
        def num(name, default):
            try:
                return float(env.get(name, default))
            except ValueError:
                raise ValueError(f"{name} must be a number; got {env.get(name)!r}.") from None
        return cls(per_min=num("RATE_LIMIT_PER_MIN", 120), burst=num("RATE_LIMIT_BURST", 60),
                   global_per_min=num("RATE_LIMIT_GLOBAL_PER_MIN", 1200),
                   global_burst=num("RATE_LIMIT_GLOBAL_BURST", 300),
                   enabled=env.get("RATE_LIMIT", "on").strip().lower() not in {"off", "0", "false", "no"})

    @staticmethod
    def client_of(remote_addr: str | None, forwarded_for: str | None) -> str | None:
        """The client to count, or None for a request that is never limited (straight from this machine)."""
        if forwarded_for:
            first = forwarded_for.split(",")[0].strip()
            return first or (remote_addr or "unknown")
        if (remote_addr or "") in LOOPBACK:
            return None
        return remote_addr or "unknown"

    def check(self, path: str, remote_addr: str | None, forwarded_for: str | None) -> tuple[bool, int, str]:
        """(allowed, Retry-After seconds, who ran out: "client" or "server")."""
        if not self.enabled or not path.startswith("/api/") or path in EXEMPT_PATHS:
            return True, 0, ""
        client = self.client_of(remote_addr, forwarded_for)
        if client is None:
            return True, 0, ""
        with self.lock:
            now = self.clock()
            bucket = self.clients.get(client)
            if bucket is None:
                bucket = self.clients[client] = TokenBucket(self.per_min, self.burst, now)
                if len(self.clients) > MAX_CLIENTS:
                    self.clients.popitem(last=False)
            else:
                self.clients.move_to_end(client)
            ok, wait = bucket.take(now)
            if not ok:
                return False, max(1, math.ceil(wait)), "client"
            ok, wait = self.everyone.take(now)
            if not ok:
                bucket.tokens = min(bucket.burst, bucket.tokens + 1)    # not this client's fault: give it back
                return False, max(1, math.ceil(wait)), "server"
            return True, 0, ""
