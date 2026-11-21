"""HTTP layer: JSON API + static web frontend, one process.

    python -m api            -> http://127.0.0.1:8000  (marketing page)
                                http://127.0.0.1:8000/app/  (dashboard)
                                http://127.0.0.1:8000/api/...  (JSON)

Starlette (the ASGI toolkit FastAPI is built on) keeps dependencies minimal.
The API is stateless: every response is a pure function of the query, so it can
be cached at the edge and scaled horizontally behind a load balancer.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from api import service

WEB_DIR = Path(__file__).resolve().parents[1] / "web"
CACHE = {"Cache-Control": "public, max-age=3600"}  # forecasts for a past week never change


def _horizon(req: Request) -> int:
    try:
        h = int(req.query_params.get("horizon", "3"))
    except ValueError:
        raise ValueError("horizon must be 2, 3 or 4")
    return h


def _as_of(req: Request) -> str:
    return req.query_params.get("as_of") or service.meta()["as_of"]["default"]


def _json(fn):
    async def endpoint(req: Request):
        try:
            return JSONResponse(fn(req), headers=CACHE)
        except ValueError as err:  # friendly, user-facing message
            return JSONResponse({"error": str(err)}, status_code=400)
        except KeyError as err:
            return JSONResponse({"error": f"Not found: {err}"}, status_code=404)
    return endpoint


async def health(_: Request):
    return JSONResponse({"status": "ok"})


routes = [
    Route("/api/health", health),
    Route("/api/meta", _json(lambda r: service.meta())),
    Route("/api/forecast", _json(lambda r: service.forecast(
        r.query_params.get("ward_id", ""), _as_of(r), _horizon(r)))),
    Route("/api/ranking", _json(lambda r: service.ranking(
        _as_of(r), _horizon(r), r.query_params.get("city") or None))),
    Route("/api/metrics", _json(lambda r: service.metrics())),
    Route("/api/history", _json(lambda r: service.ward_history(
        r.query_params.get("ward_id", ""), _horizon(r)))),
]
if WEB_DIR.exists():
    routes.append(Mount("/", app=StaticFiles(directory=WEB_DIR, html=True), name="web"))


@asynccontextmanager
async def lifespan(_app):
    t = time.perf_counter()
    service.warm_up()
    print(f"[api] model + features ready in {time.perf_counter() - t:.2f}s")
    yield


app = Starlette(routes=routes, lifespan=lifespan, middleware=[Middleware(GZipMiddleware, minimum_size=1000)])
