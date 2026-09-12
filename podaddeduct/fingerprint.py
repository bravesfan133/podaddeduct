"""Chromaprint (fpcalc) library of confirmed ad clips for cross-episode matching."""

from __future__ import annotations

import logging
import shutil
import sqlite3
import subprocess
import tempfile
from pathlib import Path

from . import db
from .config import settings
from .intervals import Interval, merge_intervals

logger = logging.getLogger("podaddeduct.fingerprint")

# Hamming distance threshold for matching 32-bit chromaprint hashes (0–32).
MATCH_MAX_DISTANCE = 8
# Minimum consecutive matching hashes (~0.12s each) ≈ 2s of audio.
MIN_MATCH_HASHES = 16


def fpcalc_bin() -> str | None:
    return shutil.which("fpcalc")


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ad_fingerprints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            feed_id INTEGER NOT NULL,
            duration REAL NOT NULL,
            fingerprint TEXT NOT NULL,
            source_episode_id INTEGER,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_ad_fp_feed ON ad_fingerprints(feed_id)"
    )


def _parse_fpcalc(stdout: str) -> tuple[float, list[int]]:
    duration = 0.0
    raw = ""
    for line in (stdout or "").splitlines():
        if line.startswith("DURATION="):
            try:
                duration = float(line.split("=", 1)[1])
            except ValueError:
                pass
        elif line.startswith("FINGERPRINT="):
            raw = line.split("=", 1)[1].strip()
    if not raw:
        return duration, []
    hashes: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            hashes.append(int(part))
        except ValueError:
            continue
    return duration, hashes


def fingerprint_file(path: Path, *, start: float | None = None, length: float | None = None) -> list[int]:
    """Run fpcalc on a file (optional -ts/-length window). Returns hash ints."""
    tool = fpcalc_bin()
    if not tool or not path.exists():
        return []
    cmd = [tool, "-raw"]
    if start is not None:
        cmd.extend(["-ts", f"{max(0.0, start):.3f}"])
    if length is not None and length > 0:
        cmd.extend(["-length", f"{length:.3f}"])
    cmd.append(str(path))
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        logger.warning("fpcalc failed: %s", (proc.stderr or "")[-300:])
        return []
    _, hashes = _parse_fpcalc(proc.stdout or "")
    return hashes


def fingerprint_episode_window(audio_path: Path, start: float, end: float) -> list[int]:
    length = max(0.5, end - start)
    return fingerprint_file(audio_path, start=start, length=length)


def store_ad_fingerprints(feed_id: int, episode_id: int, audio_path: Path, ads: list[Interval]) -> int:
    """Fingerprint each ad clip and store in the per-show library. Returns count stored."""
    if not fpcalc_bin() or not ads or not audio_path.exists():
        return 0
    stored = 0
    with db.connect() as conn:
        _ensure_schema(conn)
        for ad in ads:
            if ad.duration < 5.0:
                continue
            hashes = fingerprint_episode_window(audio_path, ad.start, ad.end)
            if len(hashes) < MIN_MATCH_HASHES:
                continue
            raw = ",".join(str(h) for h in hashes)
            conn.execute(
                """
                INSERT INTO ad_fingerprints (feed_id, duration, fingerprint, source_episode_id)
                VALUES (?, ?, ?, ?)
                """,
                (feed_id, ad.duration, raw, episode_id),
            )
            stored += 1
    if stored:
        logger.info("stored %d ad fingerprints for feed %s from episode %s", stored, feed_id, episode_id)
    return stored


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _best_match(needle: list[int], haystack: list[int]) -> tuple[int, float] | None:
    """Return (offset_in_hashes, mean_distance) for best alignment of needle in haystack."""
    if len(needle) < MIN_MATCH_HASHES or len(haystack) < len(needle):
        return None
    best_off = -1
    best_dist = 999.0
    limit = len(haystack) - len(needle) + 1
    # Stride for speed on long episodes; refine if needed.
    stride = max(1, len(needle) // 8)
    for off in range(0, limit, stride):
        total = 0
        for i, h in enumerate(needle):
            total += _hamming(h, haystack[off + i])
        mean = total / len(needle)
        if mean < best_dist:
            best_dist = mean
            best_off = off
    if best_off < 0 or best_dist > MATCH_MAX_DISTANCE:
        return None
    # Local refine around best coarse hit.
    refine_lo = max(0, best_off - stride)
    refine_hi = min(limit, best_off + stride + 1)
    for off in range(refine_lo, refine_hi):
        total = 0
        for i, h in enumerate(needle):
            total += _hamming(h, haystack[off + i])
        mean = total / len(needle)
        if mean < best_dist:
            best_dist = mean
            best_off = off
    if best_dist > MATCH_MAX_DISTANCE:
        return None
    return best_off, best_dist


def match_ads_in_episode(feed_id: int, audio_path: Path, duration: float) -> list[Interval]:
    """Slide-match stored ad fingerprints against a new episode. Cheap, no Gemini."""
    tool = fpcalc_bin()
    if not tool or not audio_path.exists() or duration <= 0:
        return []
    with db.connect() as conn:
        _ensure_schema(conn)
        rows = conn.execute(
            "SELECT duration, fingerprint FROM ad_fingerprints WHERE feed_id = ? ORDER BY id DESC LIMIT 40",
            (feed_id,),
        ).fetchall()
    if not rows:
        return []

    # Fingerprint the whole episode once (raw hashes).
    # Cap length for very long shows — fpcalc -length is seconds from start.
    whole = fingerprint_file(audio_path, length=min(duration, 4 * 3600))
    if len(whole) < MIN_MATCH_HASHES * 2:
        return []

    # ~0.123s per chromaprint hash (AcoustID default).
    sec_per_hash = duration / max(1, len(whole))
    hits: list[Interval] = []
    for row in rows:
        needle = []
        for part in (row["fingerprint"] or "").split(","):
            part = part.strip()
            if part:
                try:
                    needle.append(int(part))
                except ValueError:
                    pass
        if len(needle) < MIN_MATCH_HASHES:
            continue
        # Cap needle length so matching stays fast.
        if len(needle) > 500:
            needle = needle[:500]
        found = _best_match(needle, whole)
        if not found:
            continue
        off, dist = found
        start = off * sec_per_hash
        end = start + float(row["duration"] or (len(needle) * sec_per_hash))
        end = min(duration, max(start + 5.0, end))
        if end > start:
            hits.append(Interval(start, end))
            logger.info(
                "fingerprint hit feed=%s at %.1f–%.1f (dist=%.2f)",
                feed_id,
                start,
                end,
                dist,
            )
    return merge_intervals(hits, gap=2.0)
