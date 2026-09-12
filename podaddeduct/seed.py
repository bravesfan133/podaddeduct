from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

import httpx
import numpy as np

from .config import settings
from .intervals import Interval, merge_intervals
from .stt import chunk_transcript_lines

logger = logging.getLogger("podaddeduct.seed")

_JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]")

# Chat/completions free models on Zen — not Responses-only Muse Spark.
CURATED_ZEN_MODELS = [
    "big-pickle",
    "mimo-v2.5-free",
    "nemotron-3.5-lightning-free",
    "ling-3.0-flash-fin-free",
]

CHAT_MAX_TRIES = 4
CHAT_MAX_TOKENS = 400

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


def resolve_zen_api_key() -> str | None:
    """Prefer UI-saved key, then env, then OpenCode auth.json."""
    from .secrets import get_zen_api_key

    stored = get_zen_api_key()
    if stored:
        return stored
    key = (settings.zen_api_key or "").strip()
    if key:
        return key
    auth_path = Path(settings.zen_auth_path).expanduser()
    if not auth_path.exists():
        return None
    try:
        data = json.loads(auth_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    for provider in ("opencode", "opencode-go"):
        entry = data.get(provider)
        if isinstance(entry, dict) and entry.get("key"):
            return str(entry["key"]).strip()
    return None


SYSTEM_PROMPT = """You mark podcast advertisements for cutting.
Given a timestamped transcript ([start-end] text per line), return ONLY a JSON array of
objects {"start": <seconds>, "end": <seconds>} for every ad segment:
- Dynamic ad inserts / mid-rolls / pre-rolls / post-rolls
- Host-read sponsor reads ("brought to you by", "this episode is sponsored by", promo codes, etc.)
- Network promo blocks that are clearly ads

Do NOT mark show content, banter about the topic, or brief brand mentions that are not ads.
Merge contiguous ad sentences into one range. Use the transcript timestamps.
Return [] if there are no ads. No markdown, no commentary — JSON array only."""


def _extract_json_array(text: str) -> list[dict]:
    text = (text or "").strip()
    if not text:
        return []
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_ARRAY_RE.search(text)
        if not m:
            raise
        data = json.loads(m.group(0))
    if not isinstance(data, list):
        raise ValueError("expected JSON array")
    out: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if "start" not in item or "end" not in item:
            continue
        start = float(item["start"])
        end = float(item["end"])
        if end > start:
            out.append({"start": start, "end": end})
    return out


def ad_models_to_try() -> list[str]:
    """Primary + fallback Zen chat models."""
    from . import db

    ordered: list[str] = []
    for key in ("zen_model", "zen_fallback_model"):
        mid = db.runtime_str(key).strip()
        if mid and mid not in ordered:
            # Skip Responses-only Muse Spark — that path is gone.
            if "muse-spark" in mid and "contributor" in mid:
                logger.warning("Ignoring Responses-only model %s; using chat free models", mid)
                continue
            ordered.append(mid)
    if not ordered:
        ordered = list(CURATED_ZEN_MODELS[:2])
    return ordered


def _zen_base_url() -> str:
    from . import db

    return (db.runtime_str("zen_base_url") or settings.zen_base_url).rstrip("/")


def _chat_completions(api_key: str, model: str, user_content: str) -> str:
    """Single OpenAI-compatible chat/completions call with retries."""
    url = _zen_base_url() + "/chat/completions"
    payload = {
        "model": model,
        "temperature": 0.1,
        "max_tokens": CHAT_MAX_TOKENS,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_err = ""
    with httpx.Client(timeout=180.0) as client:
        for attempt in range(1, CHAT_MAX_TRIES + 1):
            try:
                resp = client.post(url, headers=headers, json=payload)
            except httpx.HTTPError as exc:
                last_err = f"{type(exc).__name__}: {exc}"
                resp = None
            else:
                if resp.status_code < 400:
                    data = resp.json() if isinstance(resp.json(), dict) else {}
                    choices = data.get("choices") or []
                    if not choices:
                        raise RuntimeError("Zen returned no choices")
                    msg = choices[0].get("message") or {}
                    return str(msg.get("content") or "")
                last_err = f"{resp.status_code} {resp.text[:300]}"
                body = resp.text[:500]
                if "MissingSessionID" in body or "only be used in OpenCode" in body:
                    raise RuntimeError(
                        f"Model {model} needs an OpenCode session (not a headless API). "
                        "Pick a chat/completions free model like big-pickle or mimo-v2.5-free "
                        "in Settings → Ad detection."
                    )
                if resp.status_code not in (408, 425, 429, 500, 502, 503, 504):
                    raise httpx.HTTPStatusError(
                        last_err, request=resp.request, response=resp
                    )
            if attempt < CHAT_MAX_TRIES:
                wait = 5 * 2 ** (attempt - 1)
                try:
                    if resp is not None:
                        wait = min(120.0, float(resp.headers.get("retry-after", "") or wait))
                except (TypeError, ValueError):
                    pass
                logger.warning("Zen chat attempt %d failed, retry in %.0fs: %s", attempt, wait, last_err)
                time.sleep(wait)
    raise RuntimeError(f"Zen chat failed after {CHAT_MAX_TRIES} tries: {last_err}")


def _live_model_ids(base: str, api_key: str, *, path: str = "/models") -> list[str] | None:
    """Fetch model ids from an OpenAI-style /models endpoint. None on failure."""
    try:
        with httpx.Client(timeout=20.0) as client:
            resp = client.get(
                base.rstrip("/") + path,
                headers={"Authorization": f"Bearer {api_key}"},
            )
            resp.raise_for_status()
            data = resp.json()
        ids: list[str] = []
        items = data.get("data") if isinstance(data, dict) else None
        if isinstance(items, list):
            for item in items:
                mid = item.get("id") if isinstance(item, dict) else None
                if mid and mid not in ids:
                    ids.append(str(mid))
        return sorted(ids)[:200] or None
    except Exception as exc:
        logger.warning("Live /models failed for %s, using curated list: %s", base, exc)
        return None


def fetch_zen_models() -> dict:
    """List Zen model ids. Falls back to curated free chat models."""
    api_key = resolve_zen_api_key()
    base = _zen_base_url()
    if not api_key or not base:
        return {"models": list(CURATED_ZEN_MODELS), "live": False}
    ids = _live_model_ids(base, api_key)
    if not ids:
        return {"models": list(CURATED_ZEN_MODELS), "live": False}
    # Prefer free/chat models first in the datalist.
    preferred = [m for m in CURATED_ZEN_MODELS if m in ids]
    rest = [m for m in ids if m not in preferred]
    return {"models": preferred + rest, "live": True}


def test_zen_connection(model: str | None = None) -> dict:
    """Send one tiny request to prove key + model work. Never raises."""
    try:
        api_key = resolve_zen_api_key()
        if not api_key:
            return {"ok": False, "error": "No API key saved yet."}
        use_model = (model or "").strip() or (ad_models_to_try() or [""])[0]
        if not use_model:
            return {"ok": False, "error": "No model selected."}
        start = time.monotonic()
        raw = _chat_completions(api_key, use_model, "Return exactly: []")
        parsed = _extract_json_array(raw)
        ms = int((time.monotonic() - start) * 1000)
        return {"ok": True, "provider": "zen", "model": use_model, "ms": ms, "ranges": len(parsed)}
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
    # Merge near-contiguous hits (host-read blocks span several sentences).
    return merge_intervals(hits, gap=8.0)


def _sentence_overlaps(start: float, end: float, covered: list[Interval]) -> bool:
    mid = (start + end) / 2.0
    for c in covered:
        if c.start <= mid <= c.end:
            return True
        # Also treat mostly-overlapping sentences as covered.
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


def find_ads_with_llm(transcript: dict, progress_cb=None) -> list[Interval]:
    """Run leftover transcript chunks through Zen chat/completions."""
    from . import db

    api_key = resolve_zen_api_key()
    if not api_key:
        raise RuntimeError(
            "No OpenCode Zen API key. Paste one in Settings → Ad detection "
            "(or set ZEN_API_KEY / OPENCODE_API_KEY). Get a key at https://opencode.ai/auth — "
            "use a free chat model like big-pickle or mimo-v2.5-free."
        )

    chunks = chunk_transcript_lines(
        transcript, max_chars=db.runtime_int("zen_chunk_chars", minimum=1000)
    )
    if not chunks:
        return []

    models = ad_models_to_try()
    if not models:
        raise RuntimeError("No Zen model selected. Pick one in Settings → Ad detection.")

    all_ranges: list[Interval] = []
    for i, chunk in enumerate(chunks):
        user = (
            f"Transcript chunk {i + 1}/{len(chunks)}. "
            "Return JSON array of ad ranges for THIS chunk only.\n\n"
            f"{chunk}"
        )
        parsed: list[dict] | None = None
        last_err: Exception | None = None
        raw = ""
        for model in models:
            try:
                raw = _chat_completions(api_key, model, user)
                parsed = _extract_json_array(raw)
                logger.info("zen %s chunk %s/%s -> %d ranges", model, i + 1, len(chunks), len(parsed))
                break
            except Exception as exc:
                last_err = exc
                logger.warning(
                    "zen model %s failed on chunk %s: %s; raw=%r",
                    model,
                    i + 1,
                    exc,
                    (raw or "")[:500],
                )
        if parsed is None:
            raise RuntimeError(f"Ad detection failed on chunk {i + 1}: {last_err}")
        for r in parsed:
            all_ranges.append(Interval(float(r["start"]), float(r["end"])))
        if progress_cb is not None:
            progress_cb(i + 1, len(chunks))

    return merge_intervals(all_ranges, gap=2.0)


def find_ads_with_zen(transcript: dict, progress_cb=None) -> list[Interval]:
    """Heuristics first, then LLM on leftover spans. Name kept for call sites."""
    heur = heuristic_ads(transcript)
    leftover = leftover_transcript(transcript, heur)
    if not (leftover.get("sentences") or []):
        logger.info("heuristic ads covered the transcript (%d ranges); skipping LLM", len(heur))
        return heur
    llm = find_ads_with_llm(leftover, progress_cb=progress_cb)
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

    floor = _db.runtime_float("min_ad_seconds", minimum=1.0) if min_seconds is None else min_seconds
    return [r for r in ranges if r.duration >= floor]
