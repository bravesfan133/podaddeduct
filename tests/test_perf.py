from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from podaddeduct import db
from podaddeduct.config import settings

MIN_RSS = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">
<channel><title>Mini Show</title><link>https://example.com</link>
<description>d</description>
<item><title>Ep Two</title><guid isPermaLink="false">g2</guid>
<pubDate>Tue, 02 Sep 2025 00:00:00 GMT</pubDate>
<enclosure url="https://example.com/e2.mp3" length="10" type="audio/mpeg"/></item>
<item><title>Ep One</title><guid isPermaLink="false">g1</guid>
<pubDate>Mon, 01 Sep 2025 00:00:00 GMT</pubDate>
<enclosure url="https://example.com/e1.mp3" length="10" type="audio/mpeg"/></item>
</channel></rss>"""


def _setup(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    (tmp_path / "audio").mkdir(exist_ok=True)
    db.init_db()
    return tmp_path


def test_batch_upsert_matches_single(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    feed = db.create_feed(slug="b1", upstream_url="https://example.com/rss", title="B")
    items = [
        {"guid": "g1", "title": "E1", "enclosure_url": "https://e/1.mp3",
         "pub_date": "Mon, 01 Sep 2025 00:00:00 GMT"},
        {"guid": "g2", "title": "E2", "enclosure_url": "https://e/2.mp3",
         "pub_date": "Tue, 02 Sep 2025 00:00:00 GMT"},
    ]
    out = db.upsert_episodes_batch(feed.id, items)
    assert [e.guid for e in out] == ["g1", "g2"]
    assert all(e.status == "pending" for e in out)
    assert out[1].pub_ts > out[0].pub_ts > 0
    # Unchanged re-upsert preserves processing state
    db.update_episode(out[0].id, status="ready", ad_ranges_json='[{"start":1,"end":2}]')
    out2 = db.upsert_episodes_batch(feed.id, items)
    assert db.get_episode(out[0].id).status == "ready"
    assert db.get_ad_ranges(db.get_episode(out[0].id)) == [{"start": 1.0, "end": 2.0}]
    assert len(out2) == 2
    # Changed enclosure resets processing state, like upsert_episode
    changed = [dict(items[0], enclosure_url="https://e/1b.mp3"), items[1]]
    db.upsert_episodes_batch(feed.id, changed)
    ep = db.get_episode(out[0].id)
    assert ep.status == "pending" and ep.ad_ranges_json == "[]"
    assert ep.enclosure_url == "https://e/1b.mp3"
    # Empty input is a no-op
    assert db.upsert_episodes_batch(feed.id, []) == []


def test_feed_render_local_ready_only_no_upstream(tmp_path, monkeypatch):
    """Player feed is served from SQLite — never waits on upstream HTTP."""
    _setup(tmp_path, monkeypatch)
    import podaddeduct.app as app_mod
    from podaddeduct.app import app

    app_mod._feed_cache.clear()
    feed = db.create_feed(slug="mini", upstream_url="https://example.com/rss", title="Mini")
    db.update_feed_channel(feed.id, description="Local desc", author="Host")
    feed = db.get_feed(feed.id)
    assert feed

    calls = {"n": 0}

    async def fake_fetch(url):
        calls["n"] += 1
        return MIN_RSS

    # Empty when nothing is ready — still 200, zero upstream calls.
    with (
        patch.object(app_mod, "fetch_feed_bytes", side_effect=fake_fetch),
        TestClient(app) as client,
    ):
        r1 = client.get("/feeds/mini.xml")
        assert r1.status_code == 200
        assert r1.headers.get("cache-control", "").startswith("public")
        assert "<item>" not in r1.text
        assert "Local desc" in r1.text
        assert calls["n"] == 0

        # Cached identical response without upstream.
        r2 = client.get("/feeds/mini.xml")
        assert r2.status_code == 200
        assert r2.text == r1.text
        assert calls["n"] == 0

        # Ready episode appears after cache clear; still no upstream fetch.
        ep = db.upsert_episode(
            feed.id, guid="g2", title="Ep Two",
            enclosure_url="https://example.com/e2.mp3",
            pub_date="Tue, 02 Sep 2025 00:00:00 GMT",
        )
        pending = db.upsert_episode(
            feed.id, guid="g1", title="Ep One",
            enclosure_url="https://example.com/e1.mp3",
            pub_date="Mon, 01 Sep 2025 00:00:00 GMT",
        )
        clean = tmp_path / "audio" / f"{ep.id}.clean.mp3"
        clean.write_bytes(b"x" * 64)
        db.update_episode(ep.id, clean_audio_path=str(clean), status="ready",
                          duration_seconds=100.0, description="Two notes")
        app_mod._feed_cache.clear()
        r3 = client.get("/feeds/mini.xml")
        assert r3.status_code == 200
        assert r3.text.count("<item>") == 1
        assert "Ep Two" in r3.text
        assert "Ep One" not in r3.text
        assert f"/audio/{ep.id}" in r3.text
        assert 'length="64"' in r3.text
        assert pending.status == "pending" or db.get_episode(pending.id).status == "pending"
    assert calls["n"] == 0
    app_mod._feed_cache.clear()


def test_copy_fallback_present(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct.app import app

    with TestClient(app) as client:
        home = client.get("/")
        assert "execCommand" in home.text
