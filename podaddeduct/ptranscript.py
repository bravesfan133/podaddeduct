"""Publisher-provided transcripts (Podcasting 2.0 `podcast:transcript` tag).

When a show links a timestamped transcript (VTT/SRT/JSON) in its RSS, using
it is strictly better than transcribing: free, instant, often human-fixed.
Plain-text/HTML transcripts have no timestamps and are skipped.
Apple/Spotify auto-transcripts are NOT publicly downloadable — only the
publisher-linked files count here.
"""
from __future__ import annotations

import json
import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx

logger = logging.getLogger("podaddeduct.ptranscript")

PC20_NS = "https://podcastindex.org/namespace/1.0"
MAX_DOWNLOAD_BYTES = 5 * 1024 * 1024

# Preferred first: JSON with timestamps, then VTT, then SRT.
TYPE_RANK = {
    "application/json": 0,
    "text/vtt": 1,
    "application/x-subrip": 2,
}


def _localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def parse_transcript_tags(raw: bytes) -> dict[str, list[dict]]:
    """Map item key -> [{url, type, language}] from raw feed bytes.

    Keyed by BOTH guid text and enclosure URL so sync can match regardless
    of which identifier feedparser normalized. Robust to feedparser's
    namespace handling because we parse the XML ourselves.
    """
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return {}
    out: dict[str, list[dict]] = {}
    for item in root.iter("item"):
        keys: set[str] = set()
        for child in item:
            name = _localname(child.tag).lower()
            if name == "guid" and (child.text or "").strip():
                keys.add(child.text.strip())
            if name == "enclosure" and child.get("url"):
                keys.add(child.get("url", "").strip())
        links: list[dict] = []
        for child in item:
            if _localname(child.tag).lower() != "transcript":
                continue
            url = (child.get("url") or "").strip()
            if not url:
                continue
            links.append({
                "url": url,
                "type": (child.get("type") or "").strip().lower(),
                "language": (child.get("language") or "").strip().lower(),
            })
        if links:
            for k in keys:
                if k:
                    out.setdefault(k, []).extend(links)
    return out


def pick_transcript(links: list[dict]) -> dict | None:
    """Best timestamped transcript link, or None."""
    cands = [l for l in links if l.get("type") in TYPE_RANK]
    if not cands:
        return None

    def sort_key(l: dict) -> tuple[int, int]:
        lang = l.get("language") or ""
        lang_pen = 0 if (not lang or lang.startswith("en")) else 1
        return (TYPE_RANK[l["type"]], lang_pen)

    return sorted(cands, key=sort_key)[0]


_TS_VTT = re.compile(
    r"(?:(\d+):)?([0-5]?\d):([0-5]\d)[.,](\d{3})\s*-->\s*"
    r"(?:(\d+):)?([0-5]?\d):([0-5]\d)[.,](\d{3})"
)


def _ts_to_seconds(groups: tuple, off: int) -> tuple[float, float]:
    def conv(h: str | None, m: str, s: str, ms: str) -> float:
        return (int(h or 0) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0)

    g = groups
    return conv(g[off], g[off + 1], g[off + 2], g[off + 3]), \
        conv(g[off + 4], g[off + 5], g[off + 6], g[off + 7])


def parse_cues(text: str) -> list[dict]:
    """Parse VTT or SRT cue blocks into [{start, end, text}]."""
    cues: list[dict] = []
    # Split on blank lines; cue header (WEBVTT/NOTE) and numeric SRT ids
    # simply fail the timestamp regex and are skipped.
    for block in re.split(r"\r?\n\s*\r?\n", text):
        lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
        if not lines:
            continue
        m = None
        text_lines: list[str] = []
        for i, ln in enumerate(lines):
            m = _TS_VTT.search(ln)
            if m:
                text_lines = lines[i + 1:]
                break
        if not m:
            continue
        start, end = _ts_to_seconds(m.groups(), 0)
        body = " ".join(t for t in text_lines if not t.startswith("<"))
        body = re.sub(r"<[^>]+>", "", body).strip()
        if body and end > start:
            cues.append({"start": start, "end": end, "text": body})
    return cues


def _pick(d: dict, *keys: str) -> object:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def parse_json_transcript(data: object) -> list[dict]:
    """Best-effort parse of timestamped JSON transcript shapes.

    Handles segment lists like {"segments": [{start, end, text}, ...]} and
    common key variants (startTime/start_seconds/from, body/content).
    """
    items: object = data
    if isinstance(data, dict):
        for wrapper in ("segments", "lines", "cues", "entries", "transcript"):
            if isinstance(data.get(wrapper), list):
                items = data[wrapper]
                break
    if not isinstance(items, list):
        return []
    out: list[dict] = []
    for s in items:
        if not isinstance(s, dict):
            continue
        try:
            start = float(_pick(s, "start", "startTime", "start_seconds", "from", "begin") or 0)  # type: ignore[arg-type]
            end = float(_pick(s, "end", "endTime", "end_seconds", "to") or 0)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        text = str(_pick(s, "text", "body", "content", "caption") or "").strip()
        if text and end > start:
            out.append({"start": start, "end": end, "text": text})
    return out


def cues_to_sentences(cues: list[dict], *, max_span: float = 20.0) -> list[dict]:
    """Merge short cues into sentence-ish spans on end punctuation."""
    sentences: list[dict] = []
    buf: list[dict] = []

    def flush() -> None:
        if not buf:
            return
        text = " ".join(w["text"] for w in buf).strip()
        if text:
            sentences.append({"start": buf[0]["start"], "end": buf[-1]["end"], "text": text})
        buf.clear()

    for c in cues:
        buf.append(c)
        span = c["end"] - buf[0]["start"]
        if c["text"].rstrip().endswith((".", "?", "!")) or span >= max_span:
            flush()
    flush()
    return sentences


def coverage(spans: list[dict], duration: float) -> float:
    """Fraction of episode duration covered by spans (union)."""
    if duration <= 0 or not spans:
        return 0.0
    ivs = sorted((s["start"], s["end"]) for s in spans)
    total = 0.0
    cur_s, cur_e = ivs[0]
    for s, e in ivs[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    total += cur_e - cur_s
    return max(0.0, min(1.0, total / duration))


def validate(sentences: list[dict], duration: float) -> bool:
    """Accept only transcripts plausibly spanning this episode."""
    if not sentences or duration <= 0:
        return False
    if coverage(sentences, duration) < 0.5:
        return False
    last_end = max(s["end"] for s in sentences)
    if not (0.9 * duration <= last_end <= 1.1 * duration):
        return False
    return True


def fetch_transcript(url: str) -> tuple[str, bytes]:
    """Download a transcript file. Returns (mime_hint, body)."""
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        with client.stream("GET", url) as resp:
            resp.raise_for_status()
            ctype = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
            chunks: list[bytes] = []
            size = 0
            for chunk in resp.iter_bytes(64 * 1024):
                chunks.append(chunk)
                size += len(chunk)
                if size > MAX_DOWNLOAD_BYTES:
                    raise ValueError(f"transcript too large (>{MAX_DOWNLOAD_BYTES // 1024 // 1024}MB)")
            return ctype, b"".join(chunks)


def build_sentences(mime: str, declared_type: str, body: bytes) -> list[dict]:
    """Parse a transcript body to sentences, by declared type then sniffing."""
    kind = (declared_type or "").lower()
    if "json" not in kind and "vtt" not in kind and "subrip" not in kind and "srt" not in kind:
        # Fall back to content sniffing.
        if mime == "application/json" or body.lstrip()[:1] in (b"{", b"["):
            kind = "application/json"
        elif b"-->" in body:
            kind = "text/vtt"
        else:
            return []
    if "json" in kind:
        try:
            return cues_to_sentences(
                [{"start": s["start"], "end": s["end"], "text": s["text"]}
                 for s in parse_json_transcript(json.loads(body.decode("utf-8", "replace")))]
            )
        except (json.JSONDecodeError, UnicodeError):
            return []
    text = body.decode("utf-8", "replace")
    return cues_to_sentences(parse_cues(text))


def try_publisher_transcript(episode: object, duration: float) -> dict | None:
    """Fetch + validate a publisher transcript. None = fall through to STT.

    Never raises — any failure just means transcribing the normal way.
    """
    url = getattr(episode, "transcript_url", None)
    if not url:
        return None
    ep_id = getattr(episode, "id", "?")
    try:
        mime, body = fetch_transcript(str(url))
        sentences = build_sentences(mime, str(getattr(episode, "transcript_type", "") or ""), body)
        if not validate(sentences, duration):
            logger.info("episode %s publisher transcript failed validation (%d spans), falling back to STT",
                        ep_id, len(sentences))
            return None
        logger.info("episode %s using publisher transcript (%d sentences), skipping STT", ep_id, len(sentences))
        return {"sentences": sentences, "source": "publisher", "url": str(url)}
    except Exception as exc:
        logger.warning("episode %s publisher transcript failed (%s), falling back to STT", ep_id, exc)
        return None
