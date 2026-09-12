from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Interval:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def merge_intervals(intervals: list[Interval], gap: float = 0.35) -> list[Interval]:
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda x: x.start)
    out = [ordered[0]]
    for cur in ordered[1:]:
        prev = out[-1]
        if cur.start <= prev.end + gap:
            out[-1] = Interval(prev.start, max(prev.end, cur.end))
        else:
            out.append(cur)
    return out


def invert_ranges(ads: list[Interval], duration: float) -> list[Interval]:
    ads = merge_intervals([a for a in ads if a.end > a.start])
    content: list[Interval] = []
    cursor = 0.0
    for ad in ads:
        if ad.start > cursor:
            content.append(Interval(cursor, ad.start))
        cursor = max(cursor, ad.end)
    if cursor < duration:
        content.append(Interval(cursor, duration))
    return [c for c in content if c.duration > 0.05]
