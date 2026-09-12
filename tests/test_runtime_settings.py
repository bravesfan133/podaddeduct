from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from podaddeduct import db
from podaddeduct.config import settings


def _setup(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    (tmp_path / "audio").mkdir(exist_ok=True)
    db.init_db()
    return tmp_path


def test_kv_overrides_env_default(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    assert db.models_to_try() == [settings.gemini_model]
    db.set_global_settings({"gemini_model": "gemini-2.0-flash"})
    assert db.models_to_try() == ["gemini-2.0-flash"]
    db.clear_global_settings(["gemini_model"])
    assert db.models_to_try() == [settings.gemini_model]


def test_runtime_coercion(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    assert db.runtime_bool("delete_original_after_cut") is True
    assert db.runtime_int("process_recent", minimum=1) == settings.process_recent
    assert db.runtime_float("min_ad_seconds", minimum=1.0) == settings.min_ad_seconds
    db.set_global_settings({"delete_original_after_cut": "false", "process_recent": "7"})
    assert db.runtime_bool("delete_original_after_cut") is False
    assert db.runtime_int("process_recent", minimum=1) == 7
    db.set_global_settings({"process_recent": "not-a-number"})
    assert db.runtime_int("process_recent", minimum=1) == 1


def test_public_base_prefers_configured(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct.app import public_base
    from starlette.requests import Request

    db.set_global_settings({"public_base_url": "https://podcasts.example.com/"})
    scope = {"type": "http", "headers": [(b"host", b"127.0.0.1:8080")], "query_string": b""}
    assert public_base(Request(scope)) == "https://podcasts.example.com"
    db.clear_global_settings(["public_base_url"])
    assert "192.168" in public_base(Request(scope)) or "127.0.0.1" in public_base(Request(scope))


def test_global_save_accepts_new_fields(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct.app import app

    with TestClient(app) as client:
        r = client.post(
            "/settings/global",
            data={"settings_form": "1", "min_ad_seconds": "12", "gemini_model": "gemini-2.5-flash",
                  "public_base_url": "https://podcasts.example.com",
                  "delete_original_after_cut": "on"},
            follow_redirects=False,
        )
        assert r.status_code == 303
        assert r.headers["location"].startswith("/settings?saved=settings")
    assert db.runtime_float("min_ad_seconds") == 12.0
    assert db.runtime_str("gemini_model") == "gemini-2.5-flash"
    assert db.runtime_bool("delete_original_after_cut") is True
    assert db.runtime_str("public_base_url") == "https://podcasts.example.com"


def test_global_save_rejects_bad_values(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct.app import app

    with TestClient(app) as client:
        r = client.post(
            "/settings/global",
            data={"settings_form": "1", "min_ad_seconds": "abc",
                  "public_base_url": "http://192.168.1.5:8080"},
            follow_redirects=False,
        )
        assert r.status_code == 303
        assert "err=" in r.headers["location"]
        assert r.headers["location"].startswith("/settings?")
    # Nothing persisted for the bad fields (kv stays empty = inherit)
    assert db.get_global_settings()["min_ad_seconds"] == ""
    assert db.get_global_settings()["public_base_url"] == ""


def test_password_set_change_remove(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct.app import app
    from podaddeduct.secrets import get_app_password

    with TestClient(app) as client:
        assert client.get("/").status_code == 200  # no password yet
        # set
        r = client.post("/settings/password", data={"new": "s3cret", "confirm": "s3cret"},
                        follow_redirects=False)
        assert r.status_code == 303 and "saved=password" in r.headers["location"]
        assert get_app_password() == "s3cret"
        # now locked out without cookie (don't follow the redirect)
        assert client.get("/", follow_redirects=False).status_code == 303
        # sign in to keep managing settings
        r = client.post("/login", data={"password": "s3cret"})
        assert r.status_code == 200
        # wrong current rejected
        r = client.post("/settings/password",
                        data={"current": "nope", "new": "x", "confirm": "x"},
                        follow_redirects=False)
        assert "err=" in r.headers["location"]
        assert get_app_password() == "s3cret"
        # change with correct current
        r = client.post("/settings/password",
                        data={"current": "s3cret", "new": "n3w", "confirm": "n3w"},
                        follow_redirects=False)
        assert "saved=password" in r.headers["location"]
        # password change invalidates the old session cookie — sign in again
        r = client.post("/login", data={"password": "n3w"})
        assert r.status_code == 200
        # remove
        r = client.post("/settings/password",
                        data={"current": "n3w", "new": "", "confirm": ""},
                        follow_redirects=False)
        assert "saved=password-off" in r.headers["location"]
        assert get_app_password() == ""
        assert client.get("/").status_code == 200


def test_gemini_test_reports_no_key(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct import seed as seed_mod
    from podaddeduct.app import app

    with (
        patch.object(seed_mod, "resolve_gemini_api_key", return_value=None),
        TestClient(app) as client,
    ):
        r = client.post("/api/gemini-test", json={})
        assert r.status_code == 200
        assert r.json()["ok"] is False


def test_settings_page_renders(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct.app import app

    with TestClient(app) as client:
        r = client.get("/settings")
        assert r.status_code == 200
        for section in ("Storage", "Ad detection", "Processing", "Server", "podaddeduct v", "Gemini"):
            assert section in r.text, section


def test_mutating_routes_need_login(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct.app import app
    from podaddeduct.secrets import set_app_password

    set_app_password("s3cret")
    try:
        with TestClient(app, follow_redirects=False) as client:
            assert client.post("/settings/global", data={}).status_code == 303
            assert client.post("/feeds", data={"upstream_url": "https://x.test/rss"}).status_code == 303
            assert client.post("/settings/gemini-key", data={}).status_code == 303
            assert client.post("/import-opml", files={"file": ("a.opml", b"<opml/>")}).status_code == 303
            assert client.get("/api/search?q=x").status_code == 401
            assert client.get("/api/storage").status_code == 401
            assert client.get("/export.opml").status_code == 303
            # player routes stay open by design
            feed = db.create_feed(slug="open", upstream_url="https://x.test/rss", title="O")
            assert client.get("/feeds/open.xml").status_code in (200, 502)
    finally:
        set_app_password("")


def test_login_throttle(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    import podaddeduct.app as app_mod
    from podaddeduct.app import app
    from podaddeduct.secrets import set_app_password

    app_mod._login_attempts.clear()
    set_app_password("s3cret")
    try:
        with TestClient(app) as client:
            for _ in range(5):
                r = client.post("/login", data={"password": "wrong"})
                assert r.status_code == 200
            r = client.post("/login", data={"password": "wrong"})
            assert r.status_code == 429
            # correct password still works after the window is cleared
            app_mod._login_attempts.clear()
            r = client.post("/login", data={"password": "s3cret"})
            assert r.status_code == 200
    finally:
        set_app_password("")
        app_mod._login_attempts.clear()
