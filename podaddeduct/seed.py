from __future__ import annotations

import json
import logging
import re
import time

import httpx
import numpy as np

from .config import settings
from .intervals import Interval, merge_intervals
from .stt import format_timestamped_transcript

logger = logging.getLogger("podaddeduct.seed")

_JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]")

GEMINI_DEFAULT_MODEL = "gemini-3.6-flash"
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_MAX_TRIES = 3
GEMINI_MAX_OUTPUT_TOKENS = 8192

# Obvious sponsor-read cues. Contiguous hits become heuristic ad ranges.
_HEURISTIC_RE = re.compile(
    r"(?i)\b("
    r"brought to you by|"
    r"this episode is (brought to you|sponsored)|"
    r"sponsored by|"
    r"our sponsor|"
    r"today'?s sponsor|"
    r"promo code|"
    r"use (the )?code|"
    r"discount code|"
    r"visit \w[\w.-]*\.(com|net|org|io)|"
    r"go to \w[\w.-]*\.(com|net|org|io)|"
    r"head to \w[\w.-]*\.(com|net|org|io)|"
    r"support (for )?this (show|podcast) comes from|"
    r"paid for by|"
    r"ad break|"
    r"commercial break|"
    r"we'?ll be right back|"
    r"after (these|this) (messages?|break|word from)"
    r")\b"
)

SYSTEM_PROMPT = """You mark podcast advertisements for cutting.
Given a timestamped transcript ([start-end] text per line), return ONLY a JSON array of
objects {"start": <seconds>, "end": <seconds>} for every ad segment:
- Dynamic ad inserts / mid-rolls / pre-rolls / post-rolls
- Host-read sponsor reads ("brought to you by", "this episode is sponsored by", promo codes, etc.)
- Network promo blocks that are clearly ads

Do NOT mark show content, banter about the topic, or brief brand mentions that are not ads.
Merge contiguous ad sentences into one range. Use the transcript timestamps.
Return [] if there are no ads. No markdown, no commentary — JSON array only."""


def resolve_gemini_api_key() -> str | None:
    """Prefer UI-saved key, then GEMINI_API_KEY / GOOGLE_API_KEY env."""
    from .secrets import get_gemini_api_key

    return get_gemini_api_key()


def gemini_model() -> str:
    from . import db

    return (db.runtime_str("gemini_model") or settings.gemini_model or GEMINI_DEFAULT_MODEL).strip()


def _extract_json_array(text: str) -> list[dict]:
    text = (text or "").strip()
    if not text:
        return []
    if "```" in text:
        m = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", text)
        if m:
            text = m.group(1).strip()
        else:
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
    data = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                last_brace = text[:end].rfind("}")
                if last_brace > start:
                    data = json.loads(text[start : last_brace + 1] + "]")
                else:
                    raise
        elif start != -1:
            last_brace = text.rfind("}")
            if last_brace > start:
                data = json.loads(text[start : last_brace + 1] + "]")
            else:
                raise
        else:
            raise
    if not isinstance(data, list):
        raise ValueError("expected JSON array")
    out: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if "start" not in item or "end" not in item:
            continue
        try:
            start = float(item["start"])
            end = float(item["end"])
        except (TypeError, ValueError):
            continue
        if end > start:
            out.append({"start": start, "end": end})
    return out


def _gemini_generate(api_key: str, model: str, user_content: str) -> str:
    """One-shot Gemini generateContent call with retries on transient errors."""
    url = f"{GEMINI_API_BASE}/models/{model}:generateContent"
    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": user_content}]}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": GEMINI_MAX_OUTPUT_TOKENS,
            "responseMimeType": "application/json",
        },
    }
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }
    last_err = ""
    with httpx.Client(timeout=120.0) as client:
        for attempt in range(1, GEMINI_MAX_TRIES + 1):
            try:
                resp = client.post(url, headers=headers, json=payload)
            except httpx.HTTPError as exc:
                last_err = f"{type(exc).__name__}: {exc}"
                resp = None
            else:
                if resp.status_code < 400:
                    data = resp.json() if isinstance(resp.json(), dict) else {}
                    candidates = data.get("candidates") or []
                    if not candidates:
                        raise RuntimeError("Gemini returned no candidates")
                    content = (candidates[0].get("content") or {})
                    parts = content.get("parts") or []
                    # Filter out thinking process parts (thought: True) returned by Gemini 3.x Flash
                    texts = [
                        str(p.get("text") or "")
                        for p in parts
                        if isinstance(p, dict) and not p.get("thought")
                    ]
                    if not texts and parts:
                        texts = [str(parts[-1].get("text") or "")]
                    return "\n".join(t for t in texts if t).strip()
                last_err = f"{resp.status_code} {resp.text[:300]}"
                if resp.status_code not in (408, 425, 429, 500, 502, 503, 504):
                    raise httpx.HTTPStatusError(
                        last_err, request=resp.request, response=resp
                    )
            if attempt < GEMINI_MAX_TRIES:
                wait = 2 * 2 ** (attempt - 1)
                try:
                    if resp is not None:
                        wait = min(60.0, float(resp.headers.get("retry-after", "") or wait))
                except (TypeError, ValueError):
                    pass
                logger.warning("Gemini attempt %d failed, retry in %.0fs: %s", attempt, wait, last_err)
                time.sleep(wait)
    raise RuntimeError(f"Gemini failed after {GEMINI_MAX_TRIES} tries: {last_err}")


def test_gemini_connection(model: str | None = None) -> dict:
    """Send one tiny request to prove key + model work. Never raises."""
    try:
        api_key = resolve_gemini_api_key()
        if not api_key:
            return {"ok": False, "error": "No Gemini API key saved yet."}
        use_model = (model or "").strip() or gemini_model()
        start = time.monotonic()
        raw = _gemini_generate(api_key, use_model, "Return exactly: []")
        parsed = _extract_json_array(raw)
        ms = int((time.monotonic() - start) * 1000)
        return {
            "ok": True,
            "provider": "gemini",
            "model": use_model,
            "ms": ms,
            "ranges": len(parsed),
        }
    except Exception as exc:
        logger.warning("Ad-detection test failed: %s", exc)
        return {"ok": False, "error": str(exc)[-300:]}


def heuristic_ads(transcript: dict) -> list[Interval]:
    """Mark obvious sponsor-read sentences from the transcript (no LLM)."""
    sentences = transcript.get("sentences") or []
    if not isinstance(sentences, list):
        return []
    hits: list[Interval] = []
    for s in sentences:
        if not isinstance(s, dict):
            continue
        text = str(s.get("text") or "")
        if not text or not _HEURISTIC_RE.search(text):
            continue
        try:
            start = float(s["start"])
            end = float(s["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end > start:
            hits.append(Interval(start, end))
    return merge_intervals(hits, gap=8.0)


def _sentence_overlaps(start: float, end: float, covered: list[Interval]) -> bool:
    mid = (start + end) / 2.0
    for c in covered:
        if c.start <= mid <= c.end:
            return True
        overlap = min(end, c.end) - max(start, c.start)
        if overlap > 0 and overlap >= 0.5 * (end - start):
            return True
    return False


def leftover_transcript(transcript: dict, covered: list[Interval]) -> dict:
    """Transcript with sentences already covered by heuristic/publisher ads removed."""
    sentences = transcript.get("sentences") or []
    if not covered or not isinstance(sentences, list):
        return transcript
    kept = []
    for s in sentences:
        if not isinstance(s, dict):
            continue
        try:
            start = float(s["start"])
            end = float(s["end"])
        except (KeyError, TypeError, ValueError):
            kept.append(s)
            continue
        if not _sentence_overlaps(start, end, covered):
            kept.append(s)
    out = dict(transcript)
    out["sentences"] = kept
    return out


def find_ads_with_gemini(transcript: dict, progress_cb=None) -> list[Interval]:
    """One-shot Gemini call on the full leftover transcript. Empty list on soft failure."""
    api_key = resolve_gemini_api_key()
    if not api_key:
        logger.info("No Gemini API key; skipping LLM ad detection")
        return []

    body = format_timestamped_transcript(transcript)
    if not body.strip():
        return []

    model = gemini_model()
    user = (
        "Return JSON array of ad ranges for this transcript.\n\n"
        f"{body}"
    )
    if progress_cb is not None:
        progress_cb(0, 1)
    try:
        raw = _gemini_generate(api_key, model, user)
        parsed = _extract_json_array(raw)
        logger.info("gemini %s -> %d ranges", model, len(parsed))
    except Exception as exc:
        # Soft failure: heuristics still apply; never abort the episode.
        logger.warning("Gemini ad detection failed (using heuristics only): %s", exc)
        if progress_cb is not None:
            progress_cb(1, 1)
        return []
    if progress_cb is not None:
        progress_cb(1, 1)
    return merge_intervals(
        [Interval(float(r["start"]), float(r["end"])) for r in parsed],
        gap=2.0,
    )


def find_ads_with_zen(transcript: dict, progress_cb=None) -> list[Interval]:
    """Heuristics first, then Gemini on leftovers. Name kept for call sites."""
    heur = heuristic_ads(transcript)
    leftover = leftover_transcript(transcript, heur)
    if not (leftover.get("sentences") or []):
        logger.info("heuristic ads covered the transcript (%d ranges); skipping LLM", len(heur))
        return heur
    llm = find_ads_with_gemini(leftover, progress_cb=progress_cb)
    return union_ranges(heur, llm)


def snap_to_silence(
    ranges: list[Interval],
    pcm: np.ndarray,
    sr: int,
    *,
    window: float = 1.5,
    hop_ms: float = 20.0,
) -> list[Interval]:
    """Snap each range edge to a nearby RMS valley so skips aren't mid-word."""
    if not ranges or len(pcm) == 0 or sr <= 0:
        return ranges
    hop = max(1, int(sr * hop_ms / 1000.0))
    frame = max(hop, int(sr * 0.04))
    n = 1 + max(0, (len(pcm) - frame) // hop)
    if n <= 1:
        return ranges
    rms = np.empty(n, dtype=np.float32)
    for i in range(n):
        start = i * hop
        chunk = pcm[start : start + frame]
        rms[i] = float(np.sqrt(np.mean(chunk * chunk) + 1e-12))
    times = (np.arange(n) * hop + frame / 2) / float(sr)

    def nearest_valley(t: float, prefer: str) -> float:
        lo = max(0.0, t - window)
        hi = t + window
        mask = (times >= lo) & (times <= hi)
        if not np.any(mask):
            return t
        idx = np.where(mask)[0]
        local = rms[idx]
        # Prefer valleys before the start / after the end so we don't cut mid-word.
        if prefer == "before":
            before = idx[times[idx] <= t]
            if len(before):
                local_b = rms[before]
                return float(times[before[int(np.argmin(local_b))]])
        elif prefer == "after":
            after = idx[times[idx] >= t]
            if len(after):
                local_a = rms[after]
                return float(times[after[int(np.argmin(local_a))]])
        best_local = int(np.argmin(local))
        return float(times[idx[best_local]])

    snapped: list[Interval] = []
    duration = len(pcm) / float(sr)
    from . import db as _db

    for r in ranges:
        start = nearest_valley(r.start, "before")
        end = nearest_valley(r.end, "after")
        start = max(0.0, min(start, duration))
        end = max(0.0, min(end, duration))
        if end - start >= _db.runtime_float("min_ad_seconds", minimum=1.0) * 0.5:
            snapped.append(Interval(start, end))
    return merge_intervals(snapped, gap=1.5)


def union_ranges(*groups: list[Interval]) -> list[Interval]:
    merged: list[Interval] = []
    for g in groups:
        merged.extend(g)
    return merge_intervals(merged, gap=2.0)


def filter_min_duration(ranges: list[Interval], min_seconds: float | None = None) -> list[Interval]:
    from . import db as _db

    floor = min_seconds if min_seconds is not None else _db.runtime_float("min_ad_seconds", minimum=1.0)
    return [r for r in ranges if r.duration >= floor]
