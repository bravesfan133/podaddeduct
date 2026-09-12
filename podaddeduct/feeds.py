from __future__ import annotations

import hashlib
import re
from email.utils import format_datetime, parsedate_to_datetime
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import feedparser
import httpx

from .config import settings
from . import db

SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str, fallback: str = "feed") -> str:
    s = SLUG_RE.sub("-", text.lower()).strip("-")
    return s[:60] or fallback


def slug_for_upstream(url: str, title: str = "") -> str:
    base = slugify(title) if title else "feed"
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
    candidate = f"{base}-{digest}"
    # Ensure uniqueness
    existing = db.get_feed_by_slug(candidate)
    if not existing:
        return candidate
    n = 2
    while db.get_feed_by_slug(f"{candidate}-{n}"):
        n += 1
    return f"{candidate}-{n}"


async def fetch_feed_bytes(url: str) -> bytes:
    async with httpx.AsyncClient(
        timeout=60.0,
        follow_redirects=True,
        headers={"User-Agent": settings.user_agent},
    ) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.content


def parse_feed(content: bytes, href: str | None = None) -> feedparser.FeedParserDict:
    return feedparser.parse(content, response_headers={"content-location": href} if href else None)


def entry_enclosure(entry: Any) -> str | None:
    if getattr(entry, "enclosures", None):
        for enc in entry.enclosures:
            href = enc.get("href") or enc.get("url")
            if href:
                return str(href)
    links = getattr(entry, "links", None) or []
    for link in links:
        if link.get("rel") == "enclosure" and link.get("href"):
            return str(link["href"])
    return None


def entry_guid(entry: Any, enclosure: str) -> str:
    return str(getattr(entry, "id", None) or getattr(entry, "guid", None) or enclosure)


def entry_pub_date(entry: Any) -> str | None:
    for key in ("published", "updated", "created"):
        val = getattr(entry, key, None)
        if val:
            return str(val)
    return None


def _xml_text(value: str | None) -> str:
    if not value:
        return ""
    # Collapse whitespace so huge HTML descriptions stay valid/small for picky apps
    cleaned = " ".join(str(value).split())
    return escape(cleaned)


def rewrite_feed_xml(
    parsed: feedparser.FeedParserDict,
    *,
    feed: db.Feed,
    episodes_by_guid: dict[str, db.Episode],
    public_base: str,
    item_limit: int | None = None,
) -> str:
    channel_title = _xml_text(feed.title or parsed.feed.get("title") or feed.slug)
    channel_desc = _xml_text(parsed.feed.get("description") or "")
    channel_link = escape(parsed.feed.get("link") or public_base)
    author = _xml_text(parsed.feed.get("author") or parsed.feed.get("publisher") or channel_title)
    language = escape(parsed.feed.get("language") or "en-us")
    image = ""
    image_href = ""
    if parsed.feed.get("image", {}).get("href"):
        image_href = parsed.feed.image.href
    elif parsed.feed.get("itunes_image", {}).get("href"):
        image_href = parsed.feed.itunes_image.href
    if image_href:
        href = escape(image_href)
        image = f"<itunes:image href=\"{href}\"/>\n    <image><url>{href}</url></image>"

    limit = item_limit if item_limit is not None else db.runtime_int("feed_item_limit", minimum=1, maximum=500)
    entries = list(parsed.entries[: max(1, limit)])

    # Every tracked episode is listed, newest first — exactly like a normal
    # podcast feed. Tapping an unready episode makes the player retry
    # /audio/{id} until the clean file is ready (strict: upstream bytes
    # with ads are never served). Entries with no local row yet are
    # skipped rather than linked upstream.
    items: list[str] = []
    for entry in entries:
        enclosure = entry_enclosure(entry)
        if not enclosure:
            continue
        guid = entry_guid(entry, enclosure)
        ep = episodes_by_guid.get(guid)
        if ep is None:
            continue
        title = _xml_text(getattr(entry, "title", None) or ep.title or "Episode")
        desc = _xml_text(getattr(entry, "summary", None) or getattr(entry, "description", None) or "")
        pub = entry_pub_date(entry) or ""
        pub_out = pub
        try:
            pub_out = format_datetime(parsedate_to_datetime(pub))
        except Exception:
            pass

        # Every listed episode is playable, so the enclosure always points
        # at our server — upstream bytes are never referenced.
        enc_url = f"{public_base.rstrip('/')}/audio/{ep.id}"

        length = "0"
        mime = "audio/mpeg"
        if getattr(entry, "enclosures", None):
            enc0 = entry.enclosures[0]
            length = str(enc0.get("length") or "0")
            mime = str(enc0.get("type") or mime)
        served = db.served_audio_path(ep)
        if served:
            try:
                length = str(served.stat().st_size)
            except OSError:
                pass

        duration_tag = ""
        itunes_duration = getattr(entry, "itunes_duration", None)
        if ep.duration_seconds:
            secs = int(ep.duration_seconds)
            duration_tag = f"<itunes:duration>{secs}</itunes:duration>"
        elif itunes_duration:
            duration_tag = f"<itunes:duration>{escape(str(itunes_duration))}</itunes:duration>"

        items.append(
            f"""
    <item>
      <title>{title}</title>
      <description>{desc}</description>
      <itunes:title>{title}</itunes:title>
      <itunes:summary>{desc}</itunes:summary>
      <itunes:author>{author}</itunes:author>
      <itunes:explicit>false</itunes:explicit>
      {duration_tag}
      <guid isPermaLink="false">{escape(guid)}</guid>
      <pubDate>{escape(pub_out)}</pubDate>
      <enclosure url="{escape(enc_url)}" length="{escape(length)}" type="{escape(mime)}" />
    </item>"""
        )

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"
  xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd"
  xmlns:content="http://purl.org/rss/1.0/modules/content/"
  xmlns:podcast="https://podcastindex.org/namespace/1.0">
  <channel>
    <title>{channel_title}</title>
    <link>{channel_link}</link>
    <language>{language}</language>
    <description>{channel_desc}</description>
    <itunes:author>{author}</itunes:author>
    <itunes:summary>{channel_desc}</itunes:summary>
    <itunes:explicit>false</itunes:explicit>
    {image}
    {''.join(items)}
  </channel>
</rss>
"""
