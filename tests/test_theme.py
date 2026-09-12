from __future__ import annotations

from fastapi.testclient import TestClient

from podaddeduct import db
from podaddeduct.app import app


def test_theme_static_served():
    with TestClient(app) as client:
        css = client.get("/static/theme.css")
        assert css.status_code == 200
        assert '[data-theme="light"]' in css.text
        assert "prefers-color-scheme" in css.text
        js = client.get("/static/theme.js")
        assert js.status_code == 200
        assert "podaddeduct-theme" in js.text


def test_pages_offer_system_light_dark():
    with TestClient(app) as client:
        paths = ["/", "/login"]
        feeds = db.list_feeds()
        if feeds:
            paths.append(f"/shows/{feeds[0].slug}")
            eps = db.list_episodes(feeds[0].id)
            if eps:
                paths.append(f"/episodes/{eps[0].id}")
        for path in paths:
            r = client.get(path)
            assert r.status_code == 200, path
            for choice in ("system", "light", "dark"):
                assert f'data-theme-btn="{choice}"' in r.text, f"{path} missing {choice}"
            assert "theme.css" in r.text and "theme.js" in r.text, path
