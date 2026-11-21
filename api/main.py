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

from flask import Flask, jsonify, request, send_from_directory

from api import service

WEB_DIR = Path(__file__).resolve().parents[1] / "web"
CACHE_SECONDS = 3600  # forecasts for a given (ward, week, horizon) never change


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

    @app.after_request
    def compress_and_cache(resp):
        if request.path.startswith("/api/") and resp.status_code == 200 and request.path != "/api/health":
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

    @app.get("/api/metrics")
    def metrics():
        return jsonify(service.metrics())

    @app.get("/api/lag_profile")
    def lag_profile():
        return jsonify(service.lag_profile(request.args.get("city") or None))

    @app.get("/api/history")
    def history():
        return jsonify(service.ward_history(request.args.get("ward_id", ""), _horizon()))

    # ---- static frontend (web/) ----------------------------------------------------
    @app.get("/")
    @app.get("/<path:path>")
    def web(path: str = ""):
        if path.startswith("api/"):                      # API typos get JSON, not the web fallback
            if path.rstrip("/") == "api/what_if":
                return jsonify(error="Use POST with a JSON body, e.g. "
                                     '{"ward_id": "DSCC_Ward_04", "rainfall_mm": 200}.'), 405
            return jsonify(error=f"Unknown API endpoint: /{path}"), 404
        target = WEB_DIR / path
        if path == "" or target.is_dir():
            path = (path.rstrip("/") + "/index.html").lstrip("/")
        return send_from_directory(WEB_DIR, path)

    return app
