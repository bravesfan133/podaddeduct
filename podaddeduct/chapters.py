from __future__ import annotations

import json
from pathlib import Path

from . import db
from .intervals import Interval, invert_ranges, merge_intervals


def intervals_to_dicts(intervals: list[Interval]) -> list[dict[str, float]]:
    return [{"start": round(i.start, 3), "end": round(i.end, 3)} for i in intervals]


def dicts_to_intervals(ranges: list[dict[str, float]]) -> list[Interval]:
    return [Interval(float(r["start"]), float(r["end"])) for r in ranges]


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
    # Interleave by time
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

    # If no duration / ranges, still return valid empty-ish chapters
    if not chapters and duration > 0:
        chapters.append({"startTime": 0.0, "title": "Content", "endTime": round(duration, 3)})

    return {
        "version": "1.2.0",
        "chapters": chapters,
        "podcastGuid": episode.guid,
        "episodeTitle": episode.title,
    }
