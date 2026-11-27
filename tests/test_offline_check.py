"""scripts/offline_check.py: every route and web file works with outbound network blocked."""
import pytest

from scripts import offline_check as oc


@pytest.fixture(scope="module")
def result():
    return oc.run()


def test_offline_check_passes(result):
    rows, failures, blocked = result
    assert blocked and failures == [], failures
    assert all(r["ok"] for r in rows)


def test_every_api_route_is_called(result):
    from api.main import create_app
    rows, _, _ = result
    called = {(r["method"], r["path"].split("?")[0]) for r in rows}
    routes = set(oc._api_routes(create_app()))
    assert routes and routes <= called, routes - called


def test_both_pages_and_their_files_are_fetched(result):
    rows, _, _ = result
    paths = {r["path"] for r in rows}
    for needed in ["/", "/app/", "/assets/js/app.js", "/assets/js/landing.js", "/assets/css/base.css",
                   "/assets/i18n/en.json", "/assets/i18n/bn.json"]:
        assert needed in paths, needed
    assert any(p.endswith(".woff2") for p in paths)          # fonts found through CSS url()
    assert any(p.startswith("/assets/js/views/") for p in paths)   # modules found through JS imports


def test_main_prints_table_and_returns_0(capsys):
    assert oc.main() == 0
    out = capsys.readouterr().out
    assert "OFFLINE CHECK PASSED" in out and "Outbound network blocked: yes" in out
    assert "GET /api/selfcheck" in out


def test_a_route_without_a_demo_request_fails(monkeypatch):
    queries = dict(oc.GET_QUERIES)
    del queries["/api/metrics"]
    monkeypatch.setattr(oc, "GET_QUERIES", queries)
    _, failures, _ = oc.run()
    assert any("/api/metrics" in f and "no demo request" in f for f in failures)


def test_remote_url_scan_flags_loads_and_ignores_text(tmp_path):
    web = tmp_path / "web"
    (web / "assets").mkdir(parents=True)
    (web / "index.html").write_text(
        '<link rel="stylesheet" href="https://cdn.example.com/x.css">'
        '<script src="http://cdn.example.com/x.js"></script>'
        '<a href="https://example.com/about">not loaded</a>'
        '<svg xmlns="http://www.w3.org/2000/svg"></svg>')
    (web / "assets" / "a.css").write_text(
        '@import url("https://fonts.example.com/f.css"); .x{background:url(https://img.example.com/a.png)}'
        ".y{background:url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg'%3E\")}")
    (web / "assets" / "a.js").write_text(
        'import { x } from "https://esm.example.com/m.js"; fetch("https://api.example.com/data")')
    (web / "assets" / "OFL.txt").write_text("See https://scripts.sil.org/OFL")
    found = oc.remote_urls(web)
    joined = "\n".join(found)
    for loaded in ["cdn.example.com/x.css", "cdn.example.com/x.js", "fonts.example.com/f.css",
                   "img.example.com/a.png", "esm.example.com/m.js", "api.example.com/data"]:
        assert loaded in joined, loaded
    for ignored in ["example.com/about", "w3.org", "sil.org"]:
        assert ignored not in joined, ignored


def test_real_web_folder_has_no_remote_loads():
    assert oc.remote_urls() == []


def test_children_follow_css_js_and_html():
    assert oc._children("/assets/css/base.css", b'src: url("../fonts/a.woff2"), url(data:x)') == ["/assets/fonts/a.woff2"]
    assert oc._children("/assets/js/app.js", b'import { a } from "./api.js";\nimport("./views/x.js")') == \
        ["/assets/js/api.js", "/assets/js/views/x.js"]
    html = b'<link href="/assets/css/t.css"><script src="/assets/js/app.js"></script><a href="/app/">x</a>'
    assert oc._children("/", html) == ["/assets/css/t.css", "/assets/js/app.js"]
