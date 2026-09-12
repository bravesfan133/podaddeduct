from __future__ import annotations

import asyncio
import time
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from podaddeduct import db
from podaddeduct.config import settings


def _setup(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    (tmp_path / "audio").mkdir(exist_ok=True)
    db.init_db()
    return tmp_path


def _fake_client(monkeypatch, handler):
    import podaddeduct.download as dl

    real = httpx.AsyncClient

    def make(*a, **k):
        k["transport"] = httpx.MockTransport(handler)
        return real(*a, **k)

    monkeypatch.setattr(dl.httpx, "AsyncClient", make)


def test_download_success_writes_marker_and_reports_progress(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct import download as dl

    seen: list[tuple[int, int]] = []

    def handler(request):
        return httpx.Response(200, content=b"x" * 100, headers={"content-length": "100"})

    _fake_client(monkeypatch, handler)
    dest = tmp_path / "audio" / "9.bin"
    out = asyncio.run(dl.download_file("https://example.com/e.mp3", dest,
                                       progress=lambda d, t: seen.append((d, t))))
    assert out == dest
    assert dest.read_bytes() == b"x" * 100
    assert dl.complete_marker_for(dest).exists()
    assert not dest.with_name(dest.name + ".part").exists()
    assert seen and seen[-1] == (100, 100)


def test_download_failure_leaves_no_partial_or_marker(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct import download as dl

    def handler(request):
        raise httpx.ConnectError("boom")

    _fake_client(monkeypatch, handler)
    dest = tmp_path / "audio" / "10.bin"
    try:
        asyncio.run(dl.download_file("https://example.com/e.mp3", dest))
        raise AssertionError("should have raised")
    except httpx.ConnectError:
        pass
    assert not dest.exists()
    assert not dest.with_name(dest.name + ".part").exists()
    assert not dl.complete_marker_for(dest).exists()


def test_stale_partial_without_marker_is_redownloaded(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    import podaddeduct.process as proc
    from podaddeduct import download as dl

    feed = db.create_feed(slug="dl", upstream_url="https://example.com/rss", title="D")
    ep = db.upsert_episode(feed.id, guid="g1", title="E1",
                           enclosure_url="https://example.com/e1.mp3", pub_date=None)
    # Simulate a killed download: bytes on disk, no completeness marker.
    dest = tmp_path / "audio" / f"{ep.id}.bin"
    dest.write_bytes(b"corrupt-partial")
    db.update_episode(ep.id, audio_path=str(dest), status="pending")

    calls: list[str] = []

    async def fake_download(url, d, **kw):
        calls.append(url)
        d.write_bytes(b"fresh-bytes")
        dl.complete_marker_for(d).touch()
        return d

    async def fake_to_thread(func, *a, **k):
        calls.append(f"to_thread:{func.__name__}")
        return None

    monkeypatch.setattr(proc, "download_file", fake_download)
    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)
    asyncio.run(proc.process_episode(ep.id))
    assert calls[0] == "https://example.com/e1.mp3"
    assert dest.read_bytes() == b"fresh-bytes"


def test_trusted_file_with_marker_skips_download(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    import podaddeduct.process as proc
    from podaddeduct import download as dl

    feed = db.create_feed(slug="dl2", upstream_url="https://example.com/rss", title="D")
    ep = db.upsert_episode(feed.id, guid="g1", title="E1",
                           enclosure_url="https://example.com/e1.mp3", pub_date=None)
    dest = tmp_path / "audio" / f"{ep.id}.bin"
    dest.write_bytes(b"good-bytes")
    dl.complete_marker_for(dest).touch()
    db.update_episode(ep.id, audio_path=str(dest), status="pending")

    async def fake_download(url, d, **kw):
        raise AssertionError("should not re-download a marked file")

    seen: list[str] = []

    async def fake_to_thread(func, *a, **k):
        seen.append(func.__name__)
        return None

    monkeypatch.setattr(proc, "download_file", fake_download)
    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)
    asyncio.run(proc.process_episode(ep.id))
    assert seen == ["_detect_and_cut"]


def test_job_tracker_and_describe():
    import podaddeduct.process as proc

    old_current, old_queue = proc._current, proc._queue
    proc._current = None
    proc._queue = None
    try:
        assert proc.describe_job() == ""
        proc._job_start(7)
        assert proc.describe_job() == "Waiting in line…"
        proc._job_stage("downloading", "45/210 MB")
        assert proc.describe_job() == "Downloading… 45/210 MB"
        proc._job_stage("transcribing")
        assert "Transcribing" in proc.describe_job()
        st = proc.worker_state()
        assert st["current"]["episode_id"] == 7
        assert st["current"]["elapsed"] >= 0
        assert st["queued"] == []
        q = proc.get_queue()
        q.put_nowait((1, 99, 555))
        q.put_nowait((1, 100, 556))
        assert proc.queue_position(555) == 1
        assert proc.queue_position(556) == 2
        assert proc.queue_position(999) is None
        proc._job_done(8)  # wrong id: must not clear
        assert proc._current is not None
        proc._job_done(7)
        assert proc._current is None
    finally:
        while not proc.get_queue().empty():
            proc.get_queue().get_nowait()
        proc._current = old_current
        proc._queue = old_queue


def test_status_reports_live_job(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    import podaddeduct.process as proc
    from podaddeduct.app import app

    feed = db.create_feed(slug="st", upstream_url="https://example.com/rss", title="S")
    ep = db.upsert_episode(feed.id, guid="g1", title="E1",
                           enclosure_url="https://example.com/e1.mp3", pub_date=None)
    old = proc._current
    proc._current = {"episode_id": ep.id, "stage": "transcribing",
                     "started_at": time.monotonic(), "detail": ""}
    try:
        with TestClient(app) as client:
            body = client.get(f"/api/episodes/{ep.id}/status").json()
            assert "Transcribing" in body["job_text"]
            assert body["queue_position"] is None
    finally:
        proc._current = old


def test_episode_page_hides_dead_player_while_working(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct.app import app

    feed = db.create_feed(slug="pg", upstream_url="https://example.com/rss", title="P")
    ep = db.upsert_episode(feed.id, guid="g1", title="E1",
                           enclosure_url="https://example.com/e1.mp3", pub_date=None)
    db.update_episode(ep.id, status="pending")
    with TestClient(app) as client:
        # Set working AFTER client startup (lifespan resets stale working→pending).
        db.update_episode(ep.id, status="working")
        r = client.get(f"/episodes/{ep.id}")
        assert r.status_code == 200
        assert "<audio" not in r.text
        assert 'id="liveStatus"' in r.text


def test_janitor_removes_marker(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct import download as dl
    from podaddeduct.retain import run_janitor

    db.set_global_settings({"max_cache_gb": "10", "delete_after_days": "365"})
    feed = db.create_feed(slug="mk", upstream_url="https://example.com/rss", title="M")
    db.update_feed_settings(feed.id, {"keep_last": 1})
    eps = []
    for i in range(2):
        ep = db.upsert_episode(feed.id, guid=f"g{i}", title=f"E{i}",
                               enclosure_url=f"https://e/{i}.mp3",
                               pub_date=f"Mon, 0{i + 1} Jan 2026 00:00:00 GMT")
        p = tmp_path / "audio" / f"{ep.id}.bin"
        p.write_bytes(b"x" * 100)
        dl.complete_marker_for(p).touch()
        db.update_episode(ep.id, audio_path=str(p), size_bytes=100, status="ready")
        eps.append(ep.id)
    run_janitor()
    old = db.get_episode(eps[0])
    assert old.audio_path is None
    assert not (tmp_path / "audio" / f"{eps[0]}.bin.complete").exists()
    assert db.get_episode(eps[1]).audio_path is not None
