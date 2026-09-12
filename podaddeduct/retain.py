"""Storage janitor: keep podaddeduct a small cache, never an archive.

Rules (applied oldest-first):
1. Per show: keep only the newest `keep_last` episodes with files.
2. Age: drop files older than `delete_after_days`.
3. Global cap: drop least-recently-served files until under `max_cache_gb`.

Deleting files keeps the episode row + ad marks + transcript, so a later
re-request re-downloads and re-cuts in seconds without extra AI cost
(status goes back to `pending` with ranges preserved).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import db
from .cut import clean_path_for

logger = logging.getLogger("podaddeduct.retain")


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _drop_files(ep: db.Episode, *, reason: str) -> bool:
    from .download import complete_marker_for

    removed = False
    paths: list[str | Path | None] = [ep.clean_audio_path, ep.audio_path]
    if ep.audio_path:
        # Download-completeness marker: without it the next run re-fetches.
        paths.append(complete_marker_for(Path(ep.audio_path)))
    for p in paths:
        if p:
            try:
                Path(p).unlink(missing_ok=True)
                removed = True
            except OSError:
                pass
    # Also drop stray sidecars for this episode id.
    for extra in (
        clean_path_for(ep.id),
        clean_path_for(ep.id).with_name(f"{ep.id}.clean.partial.mp3"),
    ):
        try:
            if extra.exists():
                extra.unlink()
                removed = True
        except OSError:
            pass
    # Keep ad marks + transcript; reset to pending so next play re-cuts fast.
    db.update_episode(
        ep.id,
        audio_path=None,
        clean_audio_path=None,
        size_bytes=0,
        status="pending" if ep.status in {"ready", "manual"} else ep.status,
        error=None if ep.status in {"ready", "manual"} else ep.error,
    )
    logger.info("evicted episode %s (%s): %s", ep.id, ep.title[:60], reason)
    return removed


def run_janitor() -> dict:
    """Enforce per-show, age, and global-cap rules. Returns a summary."""
    g = db.get_global_settings()
    try:
        default_keep = max(1, int(g.get("keep_last_n") or 5))
    except ValueError:
        default_keep = 5
    try:
        max_age_days = max(0, int(g.get("delete_after_days") or 14))
    except ValueError:
        max_age_days = 14
    try:
        cap_bytes = max(1024**2, float(g.get("max_cache_gb") or 3.0) * 1024**3)
    except ValueError:
        cap_bytes = 3.0 * 1024**3

    now = datetime.now(timezone.utc)
    evicted = 0

    for feed in db.list_feeds():
        settings = db.get_feed_settings(feed)
        # "chapters" mode stores ~nothing; still enforce counts for safety.
        keep = max(1, int(settings.get("keep_last", default_keep) or default_keep))
        eps = db.list_episodes(feed.id)
        # Newest first (id DESC). Anything past `keep` with files on disk goes.
        for ep in eps[keep:]:
            if db.served_audio_path(ep):
                if _drop_files(ep, reason=f"keep-last-{keep}"):
                    evicted += 1
        # Age rule within the kept window.
        if max_age_days > 0:
            cutoff = now - timedelta(days=max_age_days)
            for ep in eps[:keep]:
                served = _parse_dt(ep.last_served_at) or _parse_dt(ep.updated_at)
                if served and served < cutoff and db.served_audio_path(ep):
                    if _drop_files(ep, reason=f"older-than-{max_age_days}d"):
                        evicted += 1

    # Global cap: least-recently-served first.
    stats = db.storage_stats()
    if stats["bytes"] > cap_bytes:
        candidates: list[tuple[str, db.Episode]] = []
        for feed in db.list_feeds():
            for ep in db.list_episodes(feed.id):
                if db.served_audio_path(ep):
                    key = ep.last_served_at or ep.updated_at or ""
                    candidates.append((key, ep))
        candidates.sort(key=lambda t: t[0])
        for _, ep in candidates:
            if db.storage_stats()["bytes"] <= cap_bytes:
                break
            if _drop_files(ep, reason="over-cache-cap"):
                evicted += 1

    stats = db.storage_stats()
    return {"evicted": evicted, "bytes": stats["bytes"], "limit_bytes": stats["limit_bytes"]}


def episode_has_saved_marks(ep: db.Episode) -> bool:
    try:
        data = json.loads(ep.ad_ranges_json or "[]")
    except json.JSONDecodeError:
        return False
    return isinstance(data, list) and len(data) > 0
