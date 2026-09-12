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


def test_vaapi_available_caches_probe(tmp_path, monkeypatch):
    from podaddeduct import decode as decode_mod

    decode_mod.clear_vaapi_cache()
    calls = {"n": 0}

    def fake_probe():
        calls["n"] += 1
        return True

    monkeypatch.setattr(decode_mod, "_probe_vaapi", fake_probe)
    assert decode_mod.vaapi_available() is True
    assert decode_mod.vaapi_available() is True
    assert calls["n"] == 1
    assert decode_mod.vaapi_available(force=True) is True
    assert calls["n"] == 2


def test_settings_and_health_do_not_reprobe_vainfo(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct import decode as decode_mod
    from podaddeduct.app import app

    decode_mod.clear_vaapi_cache()
    calls = {"n": 0}

    def fake_probe():
        calls["n"] += 1
        return False

    monkeypatch.setattr(decode_mod, "_probe_vaapi", fake_probe)
    with TestClient(app) as client:
        # Lifespan may probe once; reset count after startup.
        startup = calls["n"]
        r = client.get("/settings")
        assert r.status_code == 200
        after_settings = calls["n"]
        h = client.get("/api/health")
        assert h.status_code == 200
        assert "vaapi" in h.json()
        after_health = calls["n"]
    # At most one probe for lifespan; Settings + health must not add more.
    assert after_settings <= startup + 1
    assert after_health == after_settings


def test_backend_status_skips_faster_whisper_import(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct import stt as stt_mod

    real_import = __import__

    def guarded(name, *args, **kwargs):
        if name == "faster_whisper" or (isinstance(name, str) and name.startswith("faster_whisper")):
            raise AssertionError("must not import faster_whisper on health check")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded)
    st = stt_mod.backend_status()
    assert "ok" in st
    assert st["faster_whisper_installed"] is False


def test_home_uses_feed_card_stats_not_full_list(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct.app import app

    feed = db.create_feed(slug="big", upstream_url="https://x.test/rss", title="Big")
    for i in range(80):
        ep = db.upsert_episode(
            feed.id,
            guid=f"g{i}",
            title=f"Ep {i}",
            enclosure_url=f"https://x.test/{i}.mp3",
            pub_date=None,
        )
        if i < 3:
            db.update_episode(ep.id, status="ready", size_bytes=1000)

    with (
        patch.object(db, "list_episodes", side_effect=AssertionError("home must not list_episodes")),
        TestClient(app) as client,
    ):
        r = client.get("/")
        assert r.status_code == 200
        assert "Big" in r.text
        assert "80" in r.text or "3" in r.text


def test_storage_stats_sums_without_walking_sized_rows(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    feed = db.create_feed(slug="s", upstream_url="https://x.test/rss", title="S")
    ep = db.upsert_episode(
        feed.id, guid="g1", title="E", enclosure_url="https://x.test/1.mp3", pub_date=None
    )
    db.update_episode(ep.id, status="ready", size_bytes=2048, audio_path=str(tmp_path / "missing.bin"))
    stats = db.storage_stats()
    assert stats["bytes"] == 2048
    assert stats["per_feed"][feed.id] == 2048


def test_feed_card_stats_counts(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    feed = db.create_feed(slug="c", upstream_url="https://x.test/rss", title="C")
    a = db.upsert_episode(feed.id, guid="a", title="A", enclosure_url="https://x.test/a.mp3", pub_date=None)
    b = db.upsert_episode(feed.id, guid="b", title="B", enclosure_url="https://x.test/b.mp3", pub_date=None)
    db.update_episode(a.id, status="ready")
    db.update_episode(b.id, status="working")
    card = db.feed_card_stats(feed.id)
    assert card["total"] == 2
    assert card["ready"] == 1
    assert card["working"] == 1
    assert card["latest"] is not None
