from __future__ import annotations

import asyncio
import json
import logging
import traceback
from pathlib import Path

from . import db
from .config import settings
from .cut import clean_path_for, cut_ads
from .decode import load_mono_pcm
from .download import download_file
from .intervals import Interval
from .seed import filter_min_duration, find_ads_with_zen, snap_to_silence
from .stt import transcript_path_for, transcribe_audio

logger = logging.getLogger("podaddeduct.process")

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


async def _worker_loop() -> None:
    q = get_queue()
    while True:
        _, _, episode_id = await q.get()
        try:
            await process_episode(episode_id)
        except Exception:
            logger.exception("Failed processing episode %s", episode_id)
            db.update_episode(episode_id, status="error", error=traceback.format_exc()[-2000:])
        finally:
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
    """Download -> STT -> Zen -> snap -> cut -> serve.

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
    if not audio_path.exists():
        audio_path = await download_file(ep.enclosure_url, audio_path_for(episode_id))
        db.update_episode(episode_id, audio_path=str(audio_path))

    saved = db.get_ad_ranges(db.get_episode(episode_id) or ep)
    if saved and not _queued_reseed(episode_id):
        # Re-cut from saved marks (janitor evicted the file earlier).
        await asyncio.to_thread(_recut_saved, episode_id, audio_path, saved)
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
    logger.info("episode %s detecting ads (STT+Zen)", episode_id)
    db.update_episode(episode_id, status="working", error="Transcribing…")
    pcm, sr = load_mono_pcm(audio_path)
    duration = len(pcm) / float(sr)

    # Per-show mode: "chapters" marks ads without storing a second file.
    ep = db.get_episode(episode_id)
    feed = db.get_feed(ep.feed_id) if ep else None
    mode = db.get_feed_settings(feed).get("mode", "cut") if feed else "cut"

    transcript = transcribe_audio(audio_path, transcript_path_for(episode_id), force=False)
    db.update_episode(episode_id, status="working", error="Finding ads…")
    zen_ads = find_ads_with_zen(transcript)
    ads = snap_to_silence(zen_ads, pcm, sr, window=db.runtime_float("silence_snap_window", minimum=0.0))
    ads = filter_min_duration(ads, min_seconds=max(3.0, db.runtime_float("min_ad_seconds", minimum=1.0) * 0.5))

    if mode == "chapters":
        from .chapters import intervals_to_dicts

        db.update_episode(
            episode_id,
            audio_path=str(audio_path),
            duration_seconds=duration,
            ad_ranges_json=json.dumps(intervals_to_dicts(ads)),
            clean_audio_path=None,
            size_bytes=_file_size(audio_path),
            status="ready",
            error=None,
        )
        logger.info("episode %s marked %d ad ranges (chapters-only)", episode_id, len(ads))
        return

    _finalize(episode_id, audio_path, duration, ads)
    logger.info("episode %s done with %d ad ranges", episode_id, len(ads))


def _finalize(
    episode_id: int,
    audio_path: Path,
    duration: float,
    ads: list[Interval],
) -> None:
    from .chapters import intervals_to_dicts

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
            cut_ads(audio_path, ads, clean_path, duration=duration)
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


def apply_manual_ranges(episode_id: int, ad_dicts: list[dict[str, float]]) -> db.Episode:
    from .chapters import dicts_to_intervals, intervals_to_dicts

    ep = db.get_episode(episode_id)
    if not ep:
        raise ValueError("Episode not found")
    audio_path = Path(ep.audio_path) if ep.audio_path else None
    if not audio_path or not audio_path.exists():
        raise ValueError("Episode file isn't downloaded yet — press Re-check first, then edit marks.")
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
    db.set_ad_ranges(episode_id, intervals_to_dicts(ads), status="manual")
    size = _file_size(Path(clean_audio_path) if clean_audio_path else audio_path)
    ep = db.update_episode(
        episode_id,
        duration_seconds=duration,
        audio_path=str(audio_path),
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
