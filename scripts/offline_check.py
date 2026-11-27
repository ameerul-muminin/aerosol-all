"""Offline kill-switch check: the whole app must work with outbound network blocked.

    python -m scripts.offline_check            # exit code 0 = passed, 1 = failed

In one process, with outbound network blocked the same way as src/selfcheck.py (socket connect
and DNS lookups to anything but localhost raise), it:
1. starts the Flask app (model + features load inside the block),
2. calls EVERY /api route (GET, and POST with a valid demo body) through the test client,
3. GETs / and /app/ and every web/ file they load, following <link>/<script>/<img> tags, CSS url()
   and JS imports (plus the two language files i18n.js fetches by a computed path),
4. scans web/ for http(s):// URLs a browser would load (src, <link href>, CSS url()/@import,
   JS import/fetch); namespace strings and licence text are not loads and are ignored.
Prints route -> status -> ms, then "OFFLINE CHECK PASSED" or the list of failures.
"""
from __future__ import annotations

import re
import sys
import time
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

from src.selfcheck import block_network, network_is_blocked

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
PAGES = {"/": WEB / "index.html", "/app/": WEB / "app" / "index.html"}
EXTRA_FILES = ["/assets/i18n/en.json", "/assets/i18n/bn.json"]   # fetched as /assets/i18n/${lang}.json

AS_OF, PEAK = "2025-07-14", "2025-09-15"
GET_QUERIES = {   # every GET route needs a known valid request; a new route without one fails the check
    "/api/health": "", "/api/meta": "", "/api/metrics": "", "/api/efficiency": "",
    "/api/forecast": f"?ward_id=DSCC_Ward_04&as_of={AS_OF}&horizon=3",
    "/api/ranking": f"?as_of={AS_OF}&horizon=3",
    "/api/warnings": f"?as_of={AS_OF}",
    "/api/lag_profile": "",
    "/api/history": "?ward_id=DSCC_Ward_04&horizon=3",
    "/api/overflow": f"?as_of={PEAK}&horizon=2&beds_available_pct=70",
    "/api/intervention/roi": f"?as_of={AS_OF}",
    "/api/selfcheck": f"?as_of={AS_OF}&horizon=3",
    "/api/crew_plan": f"?as_of={AS_OF}&crews=20",
}
POST_BODIES = {
    "/api/what_if": {"ward_id": "DSCC_Ward_04", "as_of": AS_OF, "horizon": 3, "rainfall_mm": 200},
    "/api/allocate": {"as_of": AS_OF, "horizon": 3, "stock_pct": 60, "crews": 20},
    "/api/intervention": {"as_of": AS_OF, "deployment": {"DSCC_Ward_04": 2, "CCC_Ward_01": 1}},
}

# a browser LOADS these; plain text, <a href> links and xmlns="http://www.w3.org/..." are not loads
REMOTE_LOADS = [
    re.compile(r"""\bsrc(?:set)?\s*=\s*["']\s*(https?://[^"'\s]+)""", re.I),
    re.compile(r"""<link\b[^>]*\bhref\s*=\s*["'](https?://[^"']+)""", re.I),
    re.compile(r"""url\(\s*["']?(https?://[^"')\s]+)""", re.I),
    re.compile(r"""@import\s+(?:url\()?\s*["']?(https?://[^"')\s;]+)""", re.I),
    re.compile(r"""\bimport\s*(?:\(|[^;]*?\bfrom\s*)?\s*["'](https?://[^"']+)""", re.I),
    re.compile(r"""\bfetch\(\s*["'`](https?://[^"'`]+)""", re.I),
]
SCANNED = {".html", ".css", ".js", ".json", ".svg"}       # licence .txt files are text, never loaded


class _Refs(HTMLParser):
    """Local files an HTML page makes the browser load."""
    def __init__(self):
        super().__init__()
        self.refs = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in ("script", "img", "source", "audio", "video", "iframe") and a.get("src"):
            self.refs.append(a["src"])
        if a.get("srcset"):
            self.refs += [part.strip().split()[0] for part in a["srcset"].split(",") if part.strip()]
        if tag == "link" and a.get("href"):
            self.refs.append(a["href"])


def _local(ref: str) -> bool:
    return not re.match(r"^(?:[a-z]+:|//|#)", ref, re.I)


def _children(url: str, body: bytes) -> list[str]:
    """Files loaded by a CSS or JS file (or HTML page), as absolute URL paths."""
    text = body.decode("utf-8", "replace")
    if url.endswith(".css"):
        refs = re.findall(r"""url\(\s*["']?([^"')]+)""", text) + re.findall(r"""@import\s+["']([^"']+)""", text)
    elif url.endswith(".js"):
        refs = re.findall(r"""\bimport\s*(?:[^;'"]*?\bfrom\s*)?["']([^"']+)["']""", text)
        refs += re.findall(r"""\bimport\(\s*["']([^"']+)["']\s*\)""", text)
    else:
        parser = _Refs()
        parser.feed(text)
        refs = parser.refs
    return [urlparse(urljoin(url, r)).path for r in refs if _local(r)]


def remote_urls(web: Path = WEB) -> list[str]:
    """`file: url` for every remote URL in web/ that a browser would load."""
    found = []
    for path in sorted(p for p in web.rglob("*") if p.is_file() and p.suffix in SCANNED):
        text = path.read_text("utf-8", "replace")
        for pattern in REMOTE_LOADS:
            for m in pattern.finditer(text):
                found.append(f"{path.relative_to(web.parent)}: {m.group(1)}")
    return sorted(set(found))


def _api_routes(app) -> list[tuple[str, str]]:
    out = set()
    for rule in app.url_map.iter_rules():
        if rule.rule.startswith("/api/"):
            for method in rule.methods & {"GET", "POST"}:
                out.add((method, rule.rule))
    return sorted(out, key=lambda x: (x[1], x[0]))


def run() -> tuple[list[dict], list[str], bool]:
    """(rows, failures, network_blocked). Each row: {method, path, status, ms, ok, note}."""
    rows, failures = [], []
    with block_network() as net:
        blocked = network_is_blocked()
        net["attempts"].clear()
        from api.main import create_app
        t = time.perf_counter()
        client = create_app().test_client()
        rows.append({"method": "-", "path": "app startup (model + features)", "status": "ok",
                     "ms": (time.perf_counter() - t) * 1000, "ok": True, "note": ""})

        def call(method, path, body=None, check=None):
            t = time.perf_counter()
            r = client.post(path, json=body) if method == "POST" else client.get(path)
            ms = (time.perf_counter() - t) * 1000
            ok, note = r.status_code == 200, ""
            if ok and check:
                ok, note = check(r)
            rows.append({"method": method, "path": path, "status": r.status_code, "ms": ms, "ok": ok, "note": note})
            if not ok:
                failures.append(f"{method} {path} -> {r.status_code} {note or r.get_data(as_text=True)[:120]}")
            return r

        def json_ok(r):
            try:
                body = r.get_json()
            except Exception:                      # noqa: BLE001 - any parse error is a failure
                return False, "not JSON"
            if r.request.path == "/api/selfcheck" and not body.get("passed"):
                failed = [c["name"] for c in body.get("checks", []) if not c["passed"]]
                return False, f"selfcheck failed: {failed}"
            return True, ""

        for method, rule in _api_routes(client.application):
            if method == "GET" and rule in GET_QUERIES:
                call("GET", rule + GET_QUERIES[rule], check=json_ok)
            elif method == "POST" and rule in POST_BODIES:
                call("POST", rule, POST_BODIES[rule], check=json_ok)
            else:
                failures.append(f"{method} {rule}: no demo request defined in scripts/offline_check.py")

        queue, seen = list(PAGES) + EXTRA_FILES, set()
        while queue:
            path = queue.pop(0)
            if path in seen or path.startswith("/api/"):
                continue
            seen.add(path)
            r = call("GET", path)
            if r.status_code == 200 and path.endswith((".css", ".js", "/", ".html")):
                queue += [p for p in _children(path, r.get_data()) if p not in seen]

        if net["attempts"]:
            failures.append(f"network access attempted: {net['attempts']}")
    if not blocked:
        failures.insert(0, "the network block did not work")
    for item in remote_urls():
        failures.append(f"remote URL a browser would load: {item}")
    return rows, failures, blocked


def main(argv=None) -> int:
    rows, failures, blocked = run()
    width = max(len(f"{r['method']} {r['path']}") for r in rows)
    print(f"Outbound network blocked: {'yes' if blocked else 'NO'}\n")
    print(f"{'route':<{width}}  status      ms")
    for r in rows:
        mark = "" if r["ok"] else "   <-- FAIL " + r["note"]
        print(f"{(r['method'] + ' ' + r['path']).strip('- '):<{width}}  {str(r['status']):>6}  {r['ms']:>6.0f}{mark}")
    n_api = sum(1 for r in rows if r["path"].startswith("/api/"))
    n_web = sum(1 for r in rows if r["method"] == "GET" and not r["path"].startswith("/api/"))
    print(f"\n{n_api} API calls, {n_web} web files, remote URLs in web/: {len(remote_urls())}")
    if failures:
        print("\nOFFLINE CHECK FAILED")
        for f in failures:
            print("  - " + f)
        return 1
    print("\nOFFLINE CHECK PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
