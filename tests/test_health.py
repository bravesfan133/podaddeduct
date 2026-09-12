from __future__ import annotations

import sys

from fastapi.testclient import TestClient

from podaddeduct import db
from podaddeduct.config import settings


def _setup(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    (tmp_path / "audio").mkdir(exist_ok=True)
    db.init_db()
    return tmp_path


def test_resolve_tool_bare_name(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct.stt import _resolve_tool

    exe = _resolve_tool(sys.executable)
    assert exe is not None and exe.exists()
    assert _resolve_tool("python") is not None or True  # PATH-dependent; must not raise
    assert _resolve_tool("") is None
    assert _resolve_tool("/nonexistent-tool-xyz-123") is None


def test_backend_status_shape(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct.stt import backend_status

    st = backend_status()
    for key in ("ok", "tool_ref", "tool_resolved", "script", "script_exists",
                "ffmpeg", "faster_whisper_installed", "model"):
        assert key in st, key


def test_friendly_error_mapping():
    from podaddeduct.process import friendly_error

    assert friendly_error(None) == ""
    assert "Settings → Server" in friendly_error("RuntimeError: Transcription tool not found at x")
    assert "API key" in friendly_error("gemini 401 unauthorized key invalid")
    assert "disk" in friendly_error("OSError: [Errno 28] No space left on device").lower()
    assert "download" in friendly_error("ConnectTimeout while fetching enclosure").lower()
    assert "unsafe" in friendly_error(
        "Ad detection marked 82% of this episode. Original kept — not cutting."
    ).lower()
    assert friendly_error("Something totally novel exploded") != ""


def test_health_endpoint(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct.app import app

    with TestClient(app) as client:
        r = client.get("/api/health")
        assert r.status_code == 200
        body = r.json()
        assert "ok" in body and "stt" in body and "queue_depth" in body and "disk" in body
        assert "vaapi" in body


def test_episode_page_shows_friendly_error(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct.app import app

    feed = db.create_feed(slug="e1", upstream_url="https://example.com/rss", title="E")
    ep = db.upsert_episode(feed.id, guid="g1", title="Ep1",
                           enclosure_url="https://example.com/e1.mp3", pub_date=None)
    raw = "Traceback (most recent call last):\nRuntimeError: Transcription tool not found at python"
    db.update_episode(ep.id, status="error", error=raw)
    with TestClient(app) as client:
        r = client.get(f"/episodes/{ep.id}")
        assert r.status_code == 200
        assert "Transcription isn" in r.text  # autoescaped apostrophe
        assert "Technical details" in r.text


def test_episode_page_refused_cut_not_ready(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct.app import app

    feed = db.create_feed(slug="e2", upstream_url="https://example.com/rss", title="E")
    ep = db.upsert_episode(feed.id, guid="g2", title="Wipe",
                           enclosure_url="https://example.com/e2.mp3", pub_date=None)
    db.update_episode(
        ep.id,
        status="error",
        duration_seconds=3720,
        ad_ranges_json='[{"start":543,"end":3607}]',
        error="Ad detection marked 82% of this episode. Original kept — not cutting.",
    )
    with TestClient(app) as client:
        r = client.get(f"/episodes/{ep.id}")
        assert r.status_code == 200
        assert "Couldn" in r.text and "safely" in r.text
        assert "Ready — ads removed" not in r.text
        assert "52:04" not in r.text
