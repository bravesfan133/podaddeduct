from __future__ import annotations

import json
import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx

from . import db
from .config import settings
from .intervals import Interval, invert_ranges, merge_intervals

logger = logging.getLogger("podaddeduct.chapters")

MAX_DOWNLOAD_BYTES = 2 * 1024 * 1024

# Titles publishers use for skippable ads / sponsor reads.
_AD_TITLE_RE = re.compile(
    r"(?i)^\s*("
    r"ad|ads|advertisement|advertisements|"
    r"sponsor|sponsors|sponsored|"
    r"commercial|promo|promotion|"
    r"mid[- ]?roll|pre[- ]?roll|post[- ]?roll|"
    r"ad break|sponsor read|sponsor message"
    r")\b"
    r"|^\s*(ad|ads|sponsor|sponsored)\s*[:\-—]"
)


def intervals_to_dicts(intervals: list[Interval]) -> list[dict[str, float]]:
    return [{"start": round(i.start, 3), "end": round(i.end, 3)} for i in intervals]


def dicts_to_intervals(ranges: list[dict[str, float]]) -> list[Interval]:
    return [Interval(float(r["start"]), float(r["end"])) for r in ranges]


def is_ad_chapter_title(title: str) -> bool:
    """True when a chapter title clearly marks an ad/sponsor block."""
    t = (title or "").strip()
    if not t:
        return False
    return bool(_AD_TITLE_RE.search(t))


def _localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def parse_chapters_tags(raw: bytes) -> dict[str, str]:
    """Map item guid/enclosure -> podcast:chapters JSON URL."""
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return {}
    out: dict[str, str] = {}
    for item in root.iter("item"):
        keys: set[str] = set()
        chapters_url = ""
        for child in item:
            name = _localname(child.tag).lower()
            if name == "guid" and (child.text or "").strip():
                keys.add(child.text.strip())
            if name == "enclosure" and child.get("url"):
                keys.add(child.get("url", "").strip())
            if name == "chapters":
                url = (child.get("url") or "").strip()
                ctype = (child.get("type") or "").strip().lower()
                if url and (not ctype or "json" in ctype or mime.endswith("+json")):
                    chapters_url = url
        if chapters_url:
            for k in keys:
                if k:
                    out[k] = chapters_url
    return out


def chapters_url_for_entry(entry, enclosure: str, tag_map: dict[str, str]) -> str | None:
    from .feeds import entry_guid

    guid = entry_guid(entry, enclosure)
    return tag_map.get(guid) or tag_map.get(enclosure) or None


def ad_ranges_from_chapters_json(data: dict | list, *, duration: float = 0.0) -> list[Interval]:
    """Extract Ad/Sponsor ranges from Podcasting 2.0 chapters JSON."""
    if isinstance(data, list):
        chapters = data
    elif isinstance(data, dict):
        chapters = data.get("chapters") or []
    else:
        return []
    if not isinstance(chapters, list):
        return []

    # Normalize to (start, end, title) — end may be missing (use next start / duration).
    rows: list[tuple[float, float | None, str]] = []
    for ch in chapters:
        if not isinstance(ch, dict):
            continue
        title = str(ch.get("title") or "")
        try:
            start = float(ch.get("startTime", ch.get("start", 0)))
        except (TypeError, ValueError):
            continue
        end_raw = ch.get("endTime", ch.get("end"))
        end: float | None
        try:
            end = float(end_raw) if end_raw is not None else None
        except (TypeError, ValueError):
            end = None
        rows.append((start, end, title))
    rows.sort(key=lambda r: r[0])

    ads: list[Interval] = []
    for i, (start, end, title) in enumerate(rows):
        if not is_ad_chapter_title(title):
            continue
        if end is None:
            if i + 1 < len(rows):
                end = rows[i + 1][0]
            elif duration > 0:
                end = duration
            else:
                continue
        if end > start:
            ads.append(Interval(start, float(end)))
    return merge_intervals(ads, gap=1.0)


def try_publisher_chapters(episode: db.Episode, duration: float = 0.0) -> list[Interval] | None:
    """Fetch publisher chapters and return Ad ranges, or None if unavailable.

    Empty list means chapters existed but had no Ad/Sponsor titles — caller
    should fall through to transcript detection. None means no chapters URL
    or fetch/parse failed.
    """
    url = (getattr(episode, "chapters_url", None) or "").strip()
    if not url:
        return None
    try:
        with httpx.Client(
            timeout=30.0,
            follow_redirects=True,
            headers={"User-Agent": settings.user_agent},
        ) as client:
            resp = client.get(url)
            resp.raise_for_status()
            if len(resp.content) > MAX_DOWNLOAD_BYTES:
                logger.warning("chapters JSON too large for episode %s", episode.id)
                return None
            data = resp.json()
    except Exception as exc:
        logger.warning("publisher chapters fetch failed for episode %s: %s", episode.id, exc)
        return None
    ads = ad_ranges_from_chapters_json(data, duration=duration)
    logger.info(
        "episode %s publisher chapters: %d ad range(s) from %s",
        episode.id,
        len(ads),
        url,
    )
    return ads


def build_chapters_json(
    episode: db.Episode,
    *,
    ad_ranges: list[dict[str, float]] | None = None,
) -> dict:
    """Podcasting 2.0 chapters JSON with Ad/Sponsor markers and content segments."""
    ads = ad_ranges if ad_ranges is not None else db.get_ad_ranges(episode)
    duration = float(episode.duration_seconds or 0.0)
    if duration <= 0 and ads:
        duration = max(float(a["end"]) for a in ads)

    ad_intervals = merge_intervals(dicts_to_intervals(ads))
    content = invert_ranges(ad_intervals, duration) if duration > 0 else []

    chapters: list[dict] = []
    events: list[tuple[float, str, float]] = []
    for c in content:
        events.append((c.start, "content", c.end))
    for a in ad_intervals:
        events.append((a.start, "ad", a.end))
    events.sort(key=lambda x: x[0])

    for start, kind, end in events:
        if kind == "ad":
            chapters.append(
                {
                    "startTime": round(start, 3),
                    "title": "Ad",
                    "endTime": round(end, 3),
                }
            )
        else:
            chapters.append(
                {
                    "startTime": round(start, 3),
                    "title": "Content",
                    "endTime": round(end, 3),
                }
            )

    if not chapters and duration > 0:
        chapters.append({"startTime": 0.0, "title": "Content", "endTime": round(duration, 3)})

    return {
        "version": "1.2.0",
        "chapters": chapters,
        "podcastGuid": episode.guid,
        "episodeTitle": episode.title,
    }
