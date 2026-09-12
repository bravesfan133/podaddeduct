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
    updated = db.update_feed_settings(feed.id, {"keep_last": 2, "auto_download": False})
    assert db.get_feed_settings(updated)["keep_last"] == 2
    assert db.get_feed_settings(updated)["auto_download"] is False


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
        # Distinct dates: g3 newest. Ordering must follow pubdate, not row id.
        ep = db.upsert_episode(feed.id, guid=f"g{i}", title=f"E{i}",
                               enclosure_url=f"https://e/{i}.mp3",
                               pub_date=f"Mon, 0{i + 1} Jan 2026 00:00:00 GMT")
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


def test_audio_redirects_while_working(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct.app import app
    from podaddeduct.process import _queued

    feed = db.create_feed(slug="s5", upstream_url="https://example.com/rss5", title="S5")
    ep = db.upsert_episode(feed.id, guid="g1", title="E",
                           enclosure_url="https://example.com/ep.mp3", pub_date=None)
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get(f"/audio/{ep.id}", follow_redirects=False)
    # Overcast treats 503 as "DELETED BY PUBLISHER"; 302 to the publisher
    # enclosure plays immediately while cleaning runs in the background.
    assert resp.status_code == 302
    assert resp.headers["location"] == "https://example.com/ep.mp3"
    assert ep.id in _queued or db.get_episode(ep.id).status in ("pending", "working")
    _queued.discard(ep.id)


def test_audio_redirects_after_janitor_eviction(tmp_path, monkeypatch):
    """Evicted episodes must stay playable via publisher fallback."""
    from podaddeduct.app import app
    from podaddeduct.process import _queued
    from podaddeduct.retain import run_janitor

    _setup(tmp_path, monkeypatch)
    db.set_global_settings({"max_cache_gb": "10", "delete_after_days": "365"})
    feed = db.create_feed(slug="s5b", upstream_url="https://example.com/rss5b", title="S5b")
    db.update_feed_settings(feed.id, {"keep_last": 1})
    older = db.upsert_episode(
        feed.id, guid="old", title="Old",
        enclosure_url="https://example.com/old.mp3",
        pub_date="Mon, 01 Jan 2026 00:00:00 GMT",
    )
    newer = db.upsert_episode(
        feed.id, guid="new", title="New",
        enclosure_url="https://example.com/new.mp3",
        pub_date="Tue, 02 Jan 2026 00:00:00 GMT",
    )
    for ep in (older, newer):
        p = tmp_path / "audio" / f"{ep.id}.clean.mp3"
        p.write_bytes(b"x" * 100)
        db.update_episode(ep.id, clean_audio_path=str(p), size_bytes=100, status="ready")

    run_janitor()
    older = db.get_episode(older.id)
    assert older is not None
    assert older.clean_audio_path is None
    assert older.status == "pending"

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get(f"/audio/{older.id}", follow_redirects=False)
    assert resp.status_code == 302
    assert resp.headers["location"] == "https://example.com/old.mp3"
    _queued.discard(older.id)

    # Newer kept file still serves locally.
    resp2 = client.get(f"/audio/{newer.id}", follow_redirects=False)
    assert resp2.status_code == 200



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
