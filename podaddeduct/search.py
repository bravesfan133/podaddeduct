"""Podcast search via Apple iTunes Search API (no key needed)."""
from __future__ import annotations

import httpx

ITUNES_URL = "https://itunes.apple.com/search"


async def search_podcasts(term: str, *, limit: int = 12) -> list[dict]:
    term = (term or "").strip()
    if not term:
        return []
    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
        resp = await client.get(
            ITUNES_URL,
            params={"media": "podcast", "term": term, "limit": max(1, min(25, limit))},
        )
        resp.raise_for_status()
        data = resp.json()
    out: list[dict] = []
    for r in data.get("results") or []:
        feed_url = r.get("feedUrl")
        if not feed_url:
            continue
        out.append(
            {
                "name": r.get("collectionName") or r.get("trackName") or "Unknown show",
                "author": r.get("artistName") or "",
                "feed_url": feed_url,
                "artwork": r.get("artworkUrl600") or r.get("artworkUrl100") or "",
                "episodes": r.get("trackCount") or 0,
            }
        )
    return out
