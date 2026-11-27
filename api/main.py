"""HTTP layer (Flask): JSON API + the static web frontend, in one process.

    python -m api            -> http://127.0.0.1:8000        marketing page   (web/index.html)
                                http://127.0.0.1:8000/app/   dashboard        (web/app/index.html)
                                http://127.0.0.1:8000/api/   JSON endpoints

Routes stay thin: all logic lives in api/service.py (pure Python, unit-tested).
The API is stateless (each response depends only on the URL), so responses are
cacheable and the app can be scaled horizontally behind a load balancer.
"""
from __future__ import annotations

import gzip
import time
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory

from api import service

WEB_DIR = Path(__file__).resolve().parents[1] / "web"
CACHE_SECONDS = 3600  # forecasts for a given (ward, week, horizon) never change
LIVE_PATHS = {"/api/health", "/api/efficiency", "/api/selfcheck"}   # live answers: never cached by browsers
POST_EXAMPLES = {   # POST-only routes: a GET gets a 405 with an example body, not the web fallback
    "api/what_if": '{"ward_id": "DSCC_Ward_04", "rainfall_mm": 200}',
    "api/allocate": '{"as_of": "2025-07-14", "horizon": 3, "stock_pct": 60}',
    "api/intervention": '{"as_of": "2025-07-14", "deployment": {"DSCC_Ward_04": 2}}',
}


def create_app() -> Flask:
    app = Flask(__name__, static_folder=None)
    app.json.sort_keys = False

    t = time.perf_counter()
    service.warm_up()  # load model + build lag features once, not per request
    print(f"[api] model + features ready in {time.perf_counter() - t:.2f}s")

    def _horizon() -> int:
        try:
            return int(request.args.get("horizon", "3"))
        except ValueError:
            raise ValueError("Horizon must be one of (2, 3, 4) weeks.")

    def _as_of() -> str:
        return service.parse_as_of(request.args.get("as_of") or service.meta()["as_of"]["default"])

    @app.errorhandler(ValueError)
    def bad_request(err):  # friendly, user-facing message, never a traceback
        return jsonify(error=str(err)), 400

    @app.errorhandler(404)
    def not_found(_):
        return jsonify(error="Not found"), 404

    @app.errorhandler(500)
    def server_error(_):  # last resort for a bug: still JSON the UI can show, never an HTML or traceback page
        return jsonify(error="Something went wrong on the server. Please try again, or change the input."), 500

    @app.after_request
    def compress_and_cache(resp):
        if request.path.startswith("/api/") and resp.status_code == 200 and request.path not in LIVE_PATHS:
            if request.method == "GET":   # scenario POSTs are not cacheable by browsers/CDNs
                resp.headers["Cache-Control"] = f"public, max-age={CACHE_SECONDS}"
            if "gzip" in request.headers.get("Accept-Encoding", "") and resp.content_length and resp.content_length > 1000:
                resp.set_data(gzip.compress(resp.get_data(), compresslevel=6))
                resp.headers["Content-Encoding"] = "gzip"
                resp.headers["Vary"] = "Accept-Encoding"
        return resp

    # ---- JSON API ---------------------------------------------------------------
    @app.get("/api/health")
    def health():
        return jsonify(status="ok")

    @app.get("/api/meta")
    def meta():
        return jsonify(service.meta())

    @app.get("/api/forecast")
    def forecast():
        return jsonify(service.forecast(request.args.get("ward_id", ""), _as_of(), _horizon()))

    @app.post("/api/what_if")
    def what_if():
        """Weather scenario: JSON body {ward_id, as_of?, horizon?, rainfall_mm?, avg_temp_c?,
        relative_humidity_pct?, weeks? (1-8, default 4)}. Returns baseline vs scenario forecast."""
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            raise ValueError("Send a JSON body, e.g. {\"ward_id\": \"DSCC_Ward_04\", \"rainfall_mm\": 200}.")
        return jsonify(service.what_if(body))

    @app.get("/api/ranking")
    def ranking():
        return jsonify(service.ranking(_as_of(), _horizon(), request.args.get("city") or None))

    @app.get("/api/warnings")
    def warnings():
        return jsonify(service.warnings(_as_of(), request.args.get("city") or None))

    @app.get("/api/efficiency")
    def efficiency():
        resp = jsonify(service.efficiency())
        resp.headers["Cache-Control"] = "no-store"     # live counters
        return resp

    @app.get("/api/metrics")
    def metrics():
        return jsonify(service.metrics())

    @app.get("/api/lag_profile")
    def lag_profile():
        return jsonify(service.lag_profile(request.args.get("city") or None))

    @app.get("/api/history")
    def history():
        return jsonify(service.ward_history(request.args.get("ward_id", ""), _horizon()))

    # ---- Phase 2: operations (allocation, overflow routing, vector-control simulator) ----
    @app.post("/api/allocate")
    def allocate():
        """JSON body {as_of, horizon, stock_pct (30-100), crews?, demand_basis?, city?}."""
        return jsonify(service.allocate(request.get_json(silent=True)))

    @app.get("/api/overflow")
    def overflow():
        a = request.args
        return jsonify(service.overflow(_as_of(), service.parse_horizon(a.get("horizon", "3")),
                                        a.get("demand_basis") or None,
                                        service.parse_number(a.get("beds_available_pct"), "beds_available_pct")))

    @app.post("/api/intervention")
    def intervention():
        """JSON body {as_of, deployment: {ward_id: crews}, total_crews?}."""
        return jsonify(service.intervention(request.get_json(silent=True)))

    @app.get("/api/intervention/roi")
    def intervention_roi():
        return jsonify(service.intervention_roi(
            _as_of(), service.parse_whole(request.args.get("crews_per_ward"), "crews_per_ward", default=1)))

    @app.get("/api/selfcheck")
    def selfcheck():
        """Live math checks with the network blocked. ?as_of (default: latest week) &horizon (default 2)."""
        a = request.args
        resp = jsonify(service.selfcheck(a.get("as_of") or None, service.parse_horizon(a.get("horizon", "2"))))
        resp.headers["Cache-Control"] = "no-store"     # a proof must run each time
        return resp

    @app.get("/api/crew_plan")
    def crew_plan():
        """Bonus B2: best placement of ?crews (default: config) at ?as_of, vs the allocation crew rule."""
        return jsonify(service.crew_plan(_as_of(), service.parse_whole(request.args.get("crews"), "crews")))

    def _dispatch_args() -> tuple:
        a = request.args
        return (_as_of(), service.parse_horizon(a.get("horizon", "3")),
                service.parse_number(a.get("stock_pct"), "stock_pct"),
                service.parse_whole(a.get("crews"), "crews"),
                service.parse_number(a.get("beds_available_pct"), "beds_available_pct"),
                a.get("lang") or None, a.get("city") or None)

    @app.get("/api/dispatch")
    def dispatch():
        """Ward notices (SMS, WhatsApp) for ?as_of&horizon&stock_pct (required) [&crews&beds_available_pct&lang&city]."""
        return jsonify(service.dispatch(*_dispatch_args()))

    @app.get("/api/dispatch/brief")
    def dispatch_brief():
        """The printable commissioner brief (HTML, open it and print to PDF), same query as /api/dispatch."""
        return Response(service.dispatch_brief(*_dispatch_args()), mimetype="text/html")

    # ---- static frontend (web/) ----------------------------------------------------
    @app.get("/")
    @app.get("/<path:path>")
    def web(path: str = ""):
        if path.startswith("api/"):                      # API typos get JSON, not the web fallback
            example = POST_EXAMPLES.get(path.rstrip("/"))
            if example:
                return jsonify(error=f"Use POST with a JSON body, e.g. {example}."), 405
            return jsonify(error=f"Unknown API endpoint: /{path}"), 404
        target = WEB_DIR / path
        if path == "" or target.is_dir():
            path = (path.rstrip("/") + "/index.html").lstrip("/")
        return send_from_directory(WEB_DIR, path)

    return app
