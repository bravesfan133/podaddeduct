from __future__ import annotations

import asyncio
import json
import logging
import time
import traceback
from pathlib import Path

from . import db
from .config import settings
from .cut import clean_path_for, cut_ads
from .decode import load_mono_pcm
from .download import complete_marker_for, download_file
from .intervals import Interval
from .seed import filter_min_duration, find_ads_with_zen, snap_to_silence
from .stt import transcribe_audio, transcript_path_for

logger = logging.getLogger("podaddeduct.process")

# In-memory view of what the single worker is doing right now, for
# logs/health/episode-page. Shape:
# {"episode_id": int, "stage": str, "started_at": float, "detail": str,
#  "done": int | None, "total": int | None}
_current: dict | None = None


def _job_start(episode_id: int) -> None:
    global _current
    _current = {"episode_id": episode_id, "stage": "queued", "started_at": time.monotonic(), "detail": "",
                "done": None, "total": None}


def _job_stage(stage: str, detail: str = "") -> None:
    if _current is not None:
        _current["stage"] = stage
        _current["detail"] = detail
        _current["started_at"] = time.monotonic()
        if stage != "detecting":
            _current["done"] = None
            _current["total"] = None


def _job_elapsed() -> float:
    if not _current:
        return 0.0
    return max(0.0, time.monotonic() - _current["started_at"])


def _job_progress(done: int, total: int) -> None:
    """Record fine-grained progress (e.g. ad-detection chunk i of n)."""
    if _current is not None:
        _current["done"] = done
        _current["total"] = total
        _current["detail"] = f"{done}/{total}"


def _job_done(episode_id: int) -> None:
    global _current
    if _current and _current.get("episode_id") == episode_id:
        _current = None


def describe_job(job: dict | None = None) -> str:
    """One plain line for the episode page, e.g. 'Downloading… 45/210 MB'."""
    job = job if job is not None else _current
    if not job:
        return ""
    stage = job.get("stage") or ""
    detail = job.get("detail") or ""
    if stage == "queued":
        return "Waiting in line…"
    if stage == "downloading":
        return f"Downloading… {detail}" if detail else "Downloading…"
    if stage == "decoding":
        return "Reading audio…"
    if stage == "transcribing":
        return "Transcribing speech… (slowest step, can take a while)"
    if stage == "detecting":
        return "Finding ads…"
    if stage == "cutting":
        return "Cutting clean file…"
    if stage == "finishing":
        return "Almost done…"
    return "Working…"


def queue_position(episode_id: int) -> int | None:
    """1-based position among waiting jobs, or None if not queued."""
    q = _queue
    if q is None:
        return None
    try:
        waiting = sorted(q._queue)
    except Exception:
        return None
    for i, (_, _, eid) in enumerate(waiting):
        if eid == episode_id:
            return i + 1
    return None


def worker_state() -> dict:
    """Snapshot for /api/health: current job (with elapsed) + waiting ids."""
    q = _queue
    waiting: list[int] = []
    if q is not None:
        try:
            waiting = [eid for _, _, eid in sorted(q._queue)]
        except Exception:
            waiting = []
    current = None
    if _current:
        current = {**_current, "elapsed": round(_job_elapsed(), 1)}
    return {"current": current, "queued": waiting, "alive": _worker_started}

_queue: asyncio.PriorityQueue[tuple[int, int, int]] | None = None
_worker_started = False
_seq = 0
# Dedup: episode ids already queued or being processed. Prevents podcast-app
# refresh storms from flooding the worker.
_queued: set[int] = set()

# Priority levels: a tap in the player jumps ahead of background jobs.
PRIO_TAPPED = 0
PRIO_BACKGROUND = 1


def get_queue() -> asyncio.PriorityQueue[tuple[int, int, int]]:
    global _queue
    if _queue is None:
        _queue = asyncio.PriorityQueue()
    return _queue


async def ensure_worker() -> None:
    global _worker_started
    if _worker_started:
        return
    _worker_started = True
    asyncio.create_task(_worker_loop())


def queue_depth() -> int:
    q = _queue
    return q.qsize() if q else 0


async def enqueue_episode(episode_id: int, *, reseed: bool = False, priority: bool = False) -> bool:
    """Queue an episode unless already queued/working. Returns True if queued.

    priority=True puts a player tap ahead of background jobs.
    """
    global _seq
    if reseed:
        clear_for_redetect(episode_id, clear_transcript=False)
    else:
        ep = db.get_episode(episode_id)
        if not ep:
            return False
        if ep.status == "working":
            return False
        if episode_id in _queued:
            return False
    await ensure_worker()
    _queued.add(episode_id)
    _seq += 1
    await get_queue().put((PRIO_TAPPED if priority else PRIO_BACKGROUND, _seq, episode_id))
    return True


def friendly_error(err: str | None) -> str:
    """One plain sentence for the episode page. Raw traceback stays in logs/DB."""
    text = (err or "").lower()
    if not text.strip():
        return ""
    if "transcription tool not found" in text or "transcription script missing" in text:
        return "Transcription isn't set up — check Settings → Server."
    if "stt failed" in text or "faster_whisper" in text or "whisper" in text:
        return "Speech-to-text failed — check server logs, then hit Prepare to retry."
    if "gemini" in text and ("key" in text or "401" in text or "403" in text or "auth" in text or "api key" in text):
        return "Ad detection needs a valid Gemini API key — check Settings → Ad detection."
    if "ffmpeg" in text:
        return "Audio cutting failed — check server logs, then hit Prepare to retry."
    if "no space" in text or "errno 28" in text or "disk" in text:
        return "Server disk is full — free space or lower storage limits in Settings."
    if "download" in text or "enclosure" in text or "connect" in text or "timeout" in text:
        return "Couldn't download the publisher's file — it usually works on retry. Hit Prepare."
    return "Processing failed — check server logs, then hit Prepare to retry."


async def _worker_loop() -> None:
    q = get_queue()
    while True:
        _, _, episode_id = await q.get()
        _job_start(episode_id)
        try:
            await process_episode(episode_id)
        except Exception:
            logger.exception("Failed processing episode %s", episode_id)
            db.update_episode(episode_id, status="error", error=traceback.format_exc()[-2000:])
        finally:
            _job_done(episode_id)
            _queued.discard(episode_id)
            q.task_done()


def audio_path_for(episode_id: int) -> Path:
    return settings.audio_dir / f"{episode_id}.bin"


def _file_size(path: Path | None) -> int:
    if not path:
        return 0
    try:
        return Path(path).stat().st_size
    except OSError:
        return 0


async def process_episode(episode_id: int) -> None:
    """Download -> detect ads -> snap -> cut -> serve.

    Fast path: if ad marks were saved before (e.g. file was auto-deleted by
    the cache janitor), just re-download and re-cut — no AI cost.
    """
    ep = db.get_episode(episode_id)
    if not ep:
        return
    has_clean = bool(ep.clean_audio_path and Path(ep.clean_audio_path).exists())
    if ep.status == "manual" and has_clean:
        return
    # Already has a clean file on disk — nothing to do.
    if db.served_audio_path(ep) and has_clean:
        if ep.status != "ready":
            db.update_episode(episode_id, status="ready", error=None)
        return
    # Manual marks with evicted files: fall through to re-cut from saved
    # marks below instead of returning early (would 503 forever).

    db.update_episode(episode_id, status="working", error=None)
    audio_path = Path(ep.audio_path) if ep.audio_path else audio_path_for(episode_id)
    if audio_path.exists() and not complete_marker_for(audio_path).exists():
        # Leftover partial from a killed/failed download — never trust it.
        logger.warning("episode %s: dropping incomplete download, re-fetching", episode_id)
        try:
            audio_path.unlink(missing_ok=True)
        except OSError:
            pass
    if not audio_path.exists():
        _job_stage("downloading")
        t0 = time.monotonic()

        def _progress(done: int, total: int) -> None:
            if _current is not None:
                if total:
                    _current["detail"] = f"{done // 1024**2}/{total // 1024**2} MB"
                else:
                    _current["detail"] = f"{done // 1024**2} MB"

        audio_path = await download_file(
            ep.enclosure_url, audio_path_for(episode_id), progress=_progress
        )
        size_mb = audio_path.stat().st_size // 1024**2 if audio_path.exists() else 0
        logger.info("episode %s downloaded %d MB in %.0fs", episode_id, size_mb, time.monotonic() - t0)
        db.update_episode(episode_id, audio_path=str(audio_path))

    saved = db.get_ad_ranges(db.get_episode(episode_id) or ep)
    if saved and not _queued_reseed(episode_id):
        # Re-cut from saved marks (janitor evicted the file earlier).
        _job_stage("cutting")
        t0 = time.monotonic()
        await asyncio.to_thread(_recut_saved, episode_id, audio_path, saved)
        logger.info("episode %s re-cut done in %.0fs", episode_id, time.monotonic() - t0)
    else:
        await asyncio.to_thread(_detect_and_cut, episode_id, audio_path)


def _queued_reseed(episode_id: int) -> bool:
    # reseed() clears ranges before queueing; if ranges are empty we must detect.
    ep = db.get_episode(episode_id)
    if not ep:
        return True
    try:
        data = json.loads(ep.ad_ranges_json or "[]")
    except json.JSONDecodeError:
        return True
    return not (isinstance(data, list) and data)


def _recut_saved(episode_id: int, audio_path: Path, saved: list[dict]) -> None:
    from .chapters import dicts_to_intervals

    logger.info("episode %s re-cutting from %d saved marks (no AI needed)", episode_id, len(saved))
    pcm, sr = load_mono_pcm(audio_path)
    duration = len(pcm) / float(sr)
    ads = dicts_to_intervals([{"start": float(r["start"]), "end": float(r["end"])} for r in saved])
    _finalize(episode_id, audio_path, duration, ads)


def _detect_and_cut(episode_id: int, audio_path: Path) -> None:
    t_all = time.monotonic()
    logger.info("episode %s detecting ads", episode_id)
    db.update_episode(episode_id, status="working", error="Finding ads…")
    _job_stage("decoding")
    pcm, sr = load_mono_pcm(audio_path)
    duration = len(pcm) / float(sr)

    ep = db.get_episode(episode_id)

    # 1) Publisher chapters with Ad/Sponsor titles — cut, no AI.
    from .chapters import try_publisher_chapters

    pub_ads = try_publisher_chapters(ep, duration) if ep is not None else None
    if pub_ads:
        ads = snap_to_silence(pub_ads, pcm, sr, window=db.runtime_float("silence_snap_window", minimum=0.0))
        ads = filter_min_duration(ads, min_seconds=max(3.0, db.runtime_float("min_ad_seconds", minimum=1.0) * 0.5))
        _finalize(episode_id, audio_path, duration, ads)
        logger.info(
            "episode %s done via publisher chapters (%d ads, total %.0fs)",
            episode_id,
            len(ads),
            time.monotonic() - t_all,
        )
        return

    # 2) Transcript → heuristics → LLM on leftovers.
    db.update_episode(episode_id, status="working", error="Transcribing…")
    _job_stage("transcribing")
    t0 = time.monotonic()
    transcript = None
    if ep is not None:
        from .ptranscript import try_publisher_transcript

        transcript = try_publisher_transcript(ep, duration)
        if transcript is not None:
            try:
                dest = transcript_path_for(episode_id)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(json.dumps(transcript, ensure_ascii=False), encoding="utf-8")
            except OSError:
                logger.warning("episode %s could not cache publisher transcript", episode_id)
    if transcript is None:
        transcript = transcribe_audio(audio_path, transcript_path_for(episode_id), force=False)
        transcript.setdefault("source", "local")
    n_sent = len((transcript.get("sentences") or []))
    logger.info("episode %s transcribed %d sentences in %.0fs", episode_id, n_sent, time.monotonic() - t0)
    db.update_episode(episode_id, status="working", error="Finding ads…")
    _job_stage("detecting")
    t0 = time.monotonic()
    detected = find_ads_with_zen(transcript, progress_cb=_job_progress)
    logger.info("episode %s ad detection found %d ranges in %.0fs", episode_id, len(detected), time.monotonic() - t0)
    ads = snap_to_silence(detected, pcm, sr, window=db.runtime_float("silence_snap_window", minimum=0.0))
    ads = filter_min_duration(ads, min_seconds=max(3.0, db.runtime_float("min_ad_seconds", minimum=1.0) * 0.5))

    _finalize(episode_id, audio_path, duration, ads)
    logger.info("episode %s done with %d ad ranges (total %.0fs)", episode_id, len(ads), time.monotonic() - t_all)


def _finalize(
    episode_id: int,
    audio_path: Path,
    duration: float,
    ads: list[Interval],
) -> None:
    from .chapters import intervals_to_dicts

    _job_stage("finishing")
    ad_dicts = intervals_to_dicts(ads)
    fields: dict = {
        "audio_path": str(audio_path),
        "duration_seconds": duration,
        "ad_ranges_json": json.dumps(ad_dicts),
        "status": "ready",
        "error": None,
    }

    clean_path = clean_path_for(episode_id)
    if ads:
        try:
            _job_stage("cutting")
            t0 = time.monotonic()
            cut_ads(audio_path, ads, clean_path, duration=duration)
            logger.info("episode %s ffmpeg cut done in %.0fs", episode_id, time.monotonic() - t0)
            fields["clean_audio_path"] = str(clean_path)
            fields["size_bytes"] = _file_size(clean_path)
        except Exception as exc:
            logger.exception("ffmpeg cut failed for episode %s", episode_id)
            fields["error"] = f"Ads found but the clean file failed: {exc}"[:1500]
            fields["clean_audio_path"] = None
            fields["size_bytes"] = _file_size(audio_path)
    else:
        try:
            clean_path.unlink(missing_ok=True)
        except OSError:
            pass
        fields["clean_audio_path"] = None
        fields["size_bytes"] = _file_size(audio_path)

    # Cache, not archive: drop the big original once the clean file exists.
    if fields.get("clean_audio_path") and db.runtime_bool("delete_original_after_cut"):
        try:
            Path(audio_path).unlink(missing_ok=True)
            fields["audio_path"] = None
        except OSError:
            pass

    db.update_episode(episode_id, **fields)

    # Opportunistic janitor so disk never grows unbounded.
    try:
        from .retain import run_janitor

        run_janitor()
    except Exception:
        logger.exception("janitor failed")


async def apply_manual_ranges(episode_id: int, ad_dicts: list[dict[str, float]]) -> db.Episode:
    """Save hand-edited ad marks and re-cut. Re-downloads original if it was evicted."""
    from .chapters import dicts_to_intervals, intervals_to_dicts

    ep = db.get_episode(episode_id)
    if not ep:
        raise ValueError("Episode not found")
    audio_path = Path(ep.audio_path) if ep.audio_path else audio_path_for(episode_id)
    if audio_path.exists() and not complete_marker_for(audio_path).exists():
        try:
            audio_path.unlink(missing_ok=True)
        except OSError:
            pass
    if not audio_path.exists():
        logger.info("episode %s: re-downloading original for manual cut", episode_id)
        audio_path = await download_file(ep.enclosure_url, audio_path_for(episode_id))
        db.update_episode(episode_id, audio_path=str(audio_path))

    pcm, sr = load_mono_pcm(audio_path)
    duration = len(pcm) / float(sr)
    ads = dicts_to_intervals(ad_dicts)
    clean_path = clean_path_for(episode_id)
    clean_audio_path: str | None = None
    cut_error: str | None = None
    if ads:
        try:
            cut_ads(audio_path, ads, clean_path, duration=duration)
            clean_audio_path = str(clean_path)
        except Exception as exc:
            logger.exception("ffmpeg cut failed for episode %s", episode_id)
            cut_error = f"Marks saved but the clean file failed: {exc}"[:1500]
    else:
        try:
            clean_path.unlink(missing_ok=True)
        except OSError:
            pass

    # Match the normal pipeline: drop the big original once a clean file exists.
    kept_audio: str | None = str(audio_path)
    if clean_audio_path and db.runtime_bool("delete_original_after_cut"):
        try:
            Path(audio_path).unlink(missing_ok=True)
            complete_marker_for(Path(audio_path)).unlink(missing_ok=True)
            kept_audio = None
        except OSError:
            pass

    db.set_ad_ranges(episode_id, intervals_to_dicts(ads), status="manual")
    size = _file_size(Path(clean_audio_path) if clean_audio_path else (Path(kept_audio) if kept_audio else None))
    ep = db.update_episode(
        episode_id,
        duration_seconds=duration,
        audio_path=kept_audio,
        clean_audio_path=clean_audio_path,
        size_bytes=size,
        error=cut_error,
    )
    return ep


def clear_for_redetect(episode_id: int, *, clear_transcript: bool = False) -> None:
    clean = clean_path_for(episode_id)
    try:
        clean.unlink(missing_ok=True)
    except OSError:
        pass
    db.update_episode(
        episode_id,
        status="pending",
        clean_audio_path=None,
        ad_ranges_json="[]",
        error=None,
    )
    if clear_transcript:
        tp = transcript_path_for(episode_id)
        try:
            tp.unlink(missing_ok=True)
        except OSError:
            pass
