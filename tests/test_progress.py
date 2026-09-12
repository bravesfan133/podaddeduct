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
        assert "Waiting on Groq" in proc.describe_job()
        proc._job_stage("encoding", "1/2")
        assert "Compressing audio for Groq" in proc.describe_job()
        assert "1/2" in proc.describe_job()
        proc._job_stage("waiting_groq", "2/2")
        assert "Waiting on Groq" in proc.describe_job()
        proc._job_stage("detecting")
        assert proc.describe_job() == "Waiting on Gemini…"
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
            assert "Waiting on Groq" in body["job_text"]
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


def _make_episode():
    feed = db.create_feed(slug="s-prog", upstream_url="https://example.com/rss", title="S")
    return db.upsert_episode(feed.id, guid="g1", title="E1",
                             enclosure_url="https://example.com/e1.mp3", pub_date=None)


def test_job_progress_tracks_chunks(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct import process as proc_mod

    old = proc_mod._current
    try:
        proc_mod._job_start(99)
        assert proc_mod._current["done"] is None
        proc_mod._job_progress(3, 10)
        assert proc_mod._current["done"] == 3
        assert proc_mod._current["total"] == 10
        assert proc_mod._current["detail"] == "3/10"
        proc_mod._job_stage("cutting")
        assert proc_mod._current["done"] is None
        assert proc_mod._current["total"] is None
    finally:
        proc_mod._current = old


def test_status_reports_progress_fields(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct import process as proc_mod
    from podaddeduct.app import app

    ep = _make_episode()
    old = proc_mod._current
    try:
        with TestClient(app) as client:
            body = client.get(f"/api/episodes/{ep.id}/status").json()
            for key in ("stage", "detail", "done", "total", "elapsed_s", "job_text", "queue_position"):
                assert key in body, key
            proc_mod._job_start(ep.id)
            proc_mod._job_stage("detecting")
            proc_mod._job_progress(4, 10)
            body = client.get(f"/api/episodes/{ep.id}/status").json()
            assert body["stage"] == "detecting"
            assert body["done"] == 4
            assert body["total"] == 10
            assert body["detail"] == "4/10"
            assert body["queue_position"] is None
            assert isinstance(body["elapsed_s"], (int, float))
    finally:
        proc_mod._current = old


def test_episode_page_has_progress_elements(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct.app import app

    ep = _make_episode()
    with TestClient(app) as client:
        db.update_episode(ep.id, status="working")
        html = client.get(f"/episodes/{ep.id}").text
        assert 'id="progWrap"' in html
        assert 'id="progBar"' in html
        assert 'id="liveStatus"' in html
        assert "location.reload()" in html  # still reloads on terminal state


def test_episode_page_shows_progress_when_pending(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct.app import app

    ep = _make_episode()
    with TestClient(app) as client:
        db.update_episode(ep.id, status="pending")
        html = client.get(f"/episodes/{ep.id}").text
        assert 'id="progWrap"' in html
        assert 'id="liveStatus"' in html
        assert "Waiting to clean" in html or "Prepare" in html


def test_download_progress_sets_done_total(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct import process as proc_mod

    old = proc_mod._current
    try:
        proc_mod._job_start(42)
        proc_mod._job_stage("downloading")
        # Simulate the download progress callback.
        if proc_mod._current is not None:
            proc_mod._current["done"] = 50 * 1024**2
            proc_mod._current["total"] = 100 * 1024**2
            proc_mod._current["detail"] = "50/100 MB"
        details = proc_mod.get_queue_details()
        assert details["active"] is True
        assert details["current"]["stage"] == "downloading"
        assert details["current"]["percent"] == 50
        assert details["current"]["job_text"].startswith("Downloading")
    finally:
        proc_mod._current = old


def test_get_queue_details_idle_and_queued(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct import process as proc_mod

    old_current, old_queue, old_recent = proc_mod._current, proc_mod._queue, list(proc_mod._recent_completed)
    proc_mod._current = None
    proc_mod._queue = None
    proc_mod._recent_completed.clear()
    try:
        idle = proc_mod.get_queue_details()
        assert idle["active"] is False
        assert idle["current"] is None
        assert idle["queued"] == []
        assert idle["recent"] == []
        assert idle["queue_depth"] == 0

        feed = db.create_feed(slug="q1", upstream_url="https://example.com/rss", title="Show One")
        ep1 = db.upsert_episode(feed.id, guid="g1", title="Ep A",
                                enclosure_url="https://example.com/a.mp3", pub_date=None)
        ep2 = db.upsert_episode(feed.id, guid="g2", title="Ep B",
                                enclosure_url="https://example.com/b.mp3", pub_date=None)

        proc_mod._job_start(ep1.id)
        proc_mod._job_stage("transcribing")
        q = proc_mod.get_queue()
        q.put_nowait((1, 1, ep2.id))

        details = proc_mod.get_queue_details()
        assert details["active"] is True
        assert details["current"]["episode_id"] == ep1.id
        assert details["current"]["title"] == "Ep A"
        assert details["current"]["feed_title"] == "Show One"
        assert details["current"]["feed_slug"] == "q1"
        assert details["current"]["stage"] == "transcribing"
        assert details["current"]["stage_label"] == "Waiting on Groq"
        assert details["current"]["percent"] is None  # indeterminate
        assert details["queue_depth"] == 1
        assert details["queued"][0]["episode_id"] == ep2.id
        assert details["queued"][0]["position"] == 1
        assert details["queued"][0]["title"] == "Ep B"
    finally:
        while proc_mod._queue is not None and not proc_mod._queue.empty():
            try:
                proc_mod._queue.get_nowait()
            except Exception:
                break
        proc_mod._current = old_current
        proc_mod._queue = old_queue
        proc_mod._recent_completed.clear()
        for item in reversed(old_recent):
            proc_mod._recent_completed.append(item)


def test_record_completed_and_api_queue(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct import process as proc_mod
    from podaddeduct.app import app

    old_current = proc_mod._current
    old_recent = list(proc_mod._recent_completed)
    proc_mod._current = None
    proc_mod._recent_completed.clear()
    try:
        feed = db.create_feed(slug="done", upstream_url="https://example.com/rss", title="Done Show")
        ep = db.upsert_episode(feed.id, guid="g1", title="Finished Ep",
                               enclosure_url="https://example.com/e.mp3", pub_date=None)
        db.update_episode(
            ep.id,
            status="ready",
            ad_ranges_json='[{"start":10,"end":40},{"start":100,"end":130}]',
        )
        proc_mod._record_completed(ep.id)
        details = proc_mod.get_queue_details()
        assert len(details["recent"]) == 1
        assert details["recent"][0]["title"] == "Finished Ep"
        assert details["recent"][0]["ads"] == 2
        assert details["recent"][0]["saved_seconds"] == 60.0

        with TestClient(app) as client:
            r = client.get("/api/queue")
            assert r.status_code == 200
            body = r.json()
            for key in ("active", "queue_depth", "current", "queued", "recent"):
                assert key in body, key
            assert body["recent"][0]["episode_id"] == ep.id

            home = client.get("/")
            assert home.status_code == 200
            assert 'id="queueCard"' in home.text
            assert 'id="queueBody"' in home.text
            assert "/api/queue" in home.text

            show = client.get(f"/shows/{feed.slug}")
            assert show.status_code == 200
            assert 'id="queueCard"' in show.text
            assert "/api/queue" in show.text
    finally:
        proc_mod._current = old_current
        proc_mod._recent_completed.clear()
        for item in reversed(old_recent):
            proc_mod._recent_completed.append(item)


def test_home_renders_with_active_queue(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct import process as proc_mod
    from podaddeduct.app import app

    feed = db.create_feed(slug="live", upstream_url="https://example.com/rss", title="Live")
    ep = db.upsert_episode(feed.id, guid="g1", title="Live Ep",
                           enclosure_url="https://example.com/e.mp3", pub_date=None)
    old = proc_mod._current
    try:
        proc_mod._job_start(ep.id)
        proc_mod._job_stage("detecting")
        proc_mod._job_progress(2, 5)
        with TestClient(app) as client:
            html = client.get("/").text
            assert "Processing" in html
            assert "Live Ep" in html or "queueBody" in html
            body = client.get("/api/queue").json()
            assert body["active"] is True
            assert body["current"]["percent"] == 40
            assert body["current"]["eta_s"] is not None or body["current"]["done"] == 2
    finally:
        proc_mod._current = old
