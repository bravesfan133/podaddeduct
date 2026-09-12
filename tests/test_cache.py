from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from podaddeduct import db
from podaddeduct.config import settings


def _setup(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    (tmp_path / "audio").mkdir(exist_ok=True)
    db.init_db()
    return tmp_path


def test_feed_settings_roundtrip(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    feed = db.create_feed(slug="s1", upstream_url="https://example.com/rss", title="S")
    defaults = db.get_feed_settings(feed)
    assert defaults["auto_download"] is True
    assert defaults["keep_last"] == 5
    assert defaults["mode"] == "cut"
    updated = db.update_feed_settings(feed.id, {"keep_last": 2, "mode": "chapters"})
    assert db.get_feed_settings(updated)["keep_last"] == 2
    assert db.get_feed_settings(updated)["mode"] == "chapters"


def test_global_settings_roundtrip(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    g = db.get_global_settings()
    assert g["max_cache_gb"] == "3.0"
    db.set_global_settings({"max_cache_gb": "1.5", "bogus": "x"})
    assert db.get_global_settings()["max_cache_gb"] == "1.5"


def test_janitor_keep_last(tmp_path, monkeypatch):
    from podaddeduct.retain import run_janitor

    _setup(tmp_path, monkeypatch)
    db.set_global_settings({"max_cache_gb": "10", "delete_after_days": "365"})
    feed = db.create_feed(slug="s2", upstream_url="https://example.com/rss2", title="S2")
    db.update_feed_settings(feed.id, {"keep_last": 2})
    ids = []
    for i in range(4):
        ep = db.upsert_episode(feed.id, guid=f"g{i}", title=f"E{i}",
                               enclosure_url=f"https://e/{i}.mp3", pub_date=None)
        p = tmp_path / "audio" / f"{ep.id}.clean.mp3"
        p.write_bytes(b"x" * 100)
        db.update_episode(ep.id, clean_audio_path=str(p), size_bytes=100, status="ready")
        ids.append(ep.id)
    summary = run_janitor()
    assert summary["evicted"] == 2
    # Newest two kept (highest ids), oldest evicted but rows + marks preserved
    assert db.get_episode(ids[3]).clean_audio_path is not None
    assert db.get_episode(ids[2]).clean_audio_path is not None
    assert db.get_episode(ids[0]).clean_audio_path is None
    assert db.get_episode(ids[0]).status == "pending"


def test_janitor_global_cap(tmp_path, monkeypatch):
    from podaddeduct.retain import run_janitor

    _setup(tmp_path, monkeypatch)
    db.set_global_settings({"max_cache_gb": "0.001", "delete_after_days": "365"})
    feed = db.create_feed(slug="s3", upstream_url="https://example.com/rss3", title="S3")
    db.update_feed_settings(feed.id, {"keep_last": 50})
    for i in range(3):
        ep = db.upsert_episode(feed.id, guid=f"c{i}", title=f"E{i}",
                               enclosure_url=f"https://e/{i}.mp3", pub_date=None)
        p = tmp_path / "audio" / f"{ep.id}.clean.mp3"
        p.write_bytes(b"x" * 500_000)
        db.update_episode(ep.id, clean_audio_path=str(p), size_bytes=500_000, status="ready")
    summary = run_janitor()
    assert summary["evicted"] >= 1
    assert summary["bytes"] <= summary["limit_bytes"] + 500_000


def test_audio_head_queues_nothing(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct.app import app

    feed = db.create_feed(slug="s4", upstream_url="https://example.com/rss4", title="S4")
    ep = db.upsert_episode(feed.id, guid="h1", title="E",
                           enclosure_url="https://example.com/ep.mp3", pub_date=None)
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.head(f"/audio/{ep.id}")
    assert resp.status_code == 200
    # HEAD must not start work
    assert db.get_episode(ep.id).status == "pending"


def test_audio_get_redirects_upstream_while_working(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct.app import app
    from podaddeduct.process import _queued

    feed = db.create_feed(slug="s5", upstream_url="https://example.com/rss5", title="S5")
    ep = db.upsert_episode(feed.id, guid="g1", title="E",
                           enclosure_url="https://example.com/ep.mp3", pub_date=None)
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get(f"/audio/{ep.id}", follow_redirects=False)
    # Plays original immediately while background work starts — never blocks.
    assert resp.status_code == 302
    assert resp.headers["location"] == "https://example.com/ep.mp3"
    _queued.discard(ep.id)


def test_audio_serves_clean_with_size_tracking(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct.app import app

    feed = db.create_feed(slug="s6", upstream_url="https://example.com/rss6", title="S6")
    ep = db.upsert_episode(feed.id, guid="g1", title="E",
                           enclosure_url="https://example.com/ep.mp3", pub_date=None)
    clean = tmp_path / "audio" / f"{ep.id}.clean.mp3"
    clean.write_bytes(b"ID3" + b"x" * 500)
    db.update_episode(ep.id, clean_audio_path=str(clean), status="ready")
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get(f"/audio/{ep.id}")
    assert resp.status_code == 200
    assert db.get_episode(ep.id).size_bytes == 503
