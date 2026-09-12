from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field

import httpx
import numpy as np

from .config import settings
from .intervals import Interval, merge_intervals
from .stt import format_timestamped_transcript

logger = logging.getLogger("podaddeduct.seed")

_JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]")

GEMINI_DEFAULT_MODEL = "gemini-3.5-flash"
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_MAX_TRIES = 5
GEMINI_MAX_OUTPUT_TOKENS = 8192

# Boundary defaults.
EARLY_AD_SNAP_SECONDS = 30.0
POSTROLL_CLAMP_SECONDS = 45.0
AD_PAD_START_SECONDS = 0.3
AD_PAD_END_SECONDS = 0.6
NEARBY_AD_GAP_SECONDS = 15.0
HEURISTIC_MERGE_GAP_SECONDS = 8.0
SILENCE_SNAP_WINDOW_SECONDS = 2.0
# No real podcast ad is a 4–7s keyword sentence — reject micro-cuts.
MIN_AD_CUT_SECONDS = 15.0
# Real ads are short; a 51-minute "ad" is a model miss.
MAX_AD_CUT_SECONDS = 180.0
# Refuse to cut when marked ads would wipe most of the episode.
MAX_AD_COVERAGE = 0.35
MIN_REMAINING_SECONDS = 600.0
MIN_CONFIDENCE = 0.6

ALLOWED_AD_TYPES = {"host_read", "inserted_ad", "unknown_ad"}

GEMINI_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "ads": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "start": {"type": "STRING"},
                    "end": {"type": "STRING"},
                    "type": {"type": "STRING"},
                    "sponsor": {"type": "STRING"},
                    "confidence": {"type": "NUMBER"},
                },
                "required": ["start", "end"],
            },
        }
    },
    "required": ["ads"],
}


@dataclass
class AdDetectionResult:
    """Ranges plus whether Gemini ran / failed (for episode-page warnings)."""

    ranges: list[Interval] = field(default_factory=list)
    gemini_error: str | None = None
    gemini_ok: bool = False
    sources: list[str] = field(default_factory=list)


@dataclass
class CutGuardResult:
    """Whether ads are safe to cut, plus the filtered ranges."""

    ok: bool
    ranges: list[Interval]
    reason: str | None = None


# Tight sponsor-read cues only (hints / fallback when Gemini fails entirely).
# Do NOT include bare "go to …com" / "paid for by" — those fire on news talk.
_HEURISTIC_RE = re.compile(
    r"(?i)\b("
    r"brought to you by|"
    r"this episode is (brought to you|sponsored)|"
    r"sponsored by|"
    r"today'?s sponsor|"
    r"thanks to our sponsor|"
    r"word from our sponsor|"
    r"promo code|"
    r"use (the )?code [A-Z0-9]{3,}|"
    r"discount code|"
    r"enter code [A-Z0-9]{3,}"
    r")\b"
)

SYSTEM_PROMPT = """You are an advertisement detection classifier for podcast transcripts.

Your job is to identify the exact timestamp ranges containing commercial advertisements.

You will receive a timestamped transcript of a podcast episode.

## What counts as an advertisement

Mark a segment as an advertisement ONLY when it is a paid commercial break:

* Host-read sponsor advertisements with a pitch AND a call to action (URL, promo code, "shop", "try", "subscribe to the product")
* Dynamically inserted / pre-recorded advertisements
* Pre-roll, mid-roll, and post-roll commercial spots
* Consecutive sponsors in the SAME commercial break (Sponsor A then B then C with no show content between) — return ONE short range covering that break only

A host-read must look like a read: brand/product + CTA. Typical length is 30 seconds to 3 minutes.

## What does NOT count as an advertisement

Do NOT mark:

* News, politics, or commentary ABOUT advertisements, campaigns, or companies
* Ordinary discussion of products, brands, websites, books, or people
* Unpaid recommendations or casual mentions
* Podcast housekeeping, intros, outros, "subscribe to this show"
* Mentions of "paid for by", campaign ads, or political advertising in a news story
* Long stretches of show content after a midroll — NEVER extend an ad through the rest of the episode

## Critical length rules

* Each ad range should be SHORT: typically 30 seconds to 3 minutes.
* NEVER return a range longer than about 3 minutes unless it is clearly one continuous commercial break of that length.
* NEVER mark from a midroll to the end of the episode.
* NEVER merge unrelated ad breaks across normal show content into one range.
* If several sponsors play back-to-back with no podcast content between them, one continuous short break is fine.
* If normal conversation resumes, that break ENDS. Later ads are separate ranges.

## Determining ad boundaries

Use surrounding context. START is the earliest transcript timestamp that belongs to the commercial. END is the last timestamp that belongs to it — before normal conversation resumes.

You MUST use timestamps provided in the transcript. Never invent timestamps.

## Output

Return ONLY valid JSON (no Markdown, no explanation):

{
"ads": [
{
"start": "HH:MM:SS",
"end": "HH:MM:SS",
"type": "host_read",
"sponsor": "Sponsor Name",
"confidence": 0.97
}
]
}

Allowed type values: host_read, inserted_ad, unknown_ad.
If no advertisements: {"ads": []}.
Confidence 0.0–1.0. Prefer omitting uncertain segments (confidence would be below 0.6).

Silently verify: every range is a real commercial; news about ads is excluded; no range covers most of the episode; timestamps exist in the transcript."""


def resolve_gemini_api_key() -> str | None:
    """Prefer UI-saved key, then GEMINI_API_KEY / GOOGLE_API_KEY env."""
    from .secrets import get_gemini_api_key

    return get_gemini_api_key()


def is_gemini_model(model: str | None) -> bool:
    m = (model or "").strip().lower()
    return m.startswith("gemini-") or m.startswith("models/gemini")


def gemini_model() -> str:
    from . import db

    return (db.runtime_str("gemini_model") or settings.gemini_model or GEMINI_DEFAULT_MODEL).strip()


def _parse_timestamp(val) -> float | None:
    """Parse HH:MM:SS, MM:SS, or numeric seconds into float seconds."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    text = str(val).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    parts = text.split(":")
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return None
    if len(nums) == 3:
        h, m, s = nums
        return h * 3600.0 + m * 60.0 + s
    if len(nums) == 2:
        m, s = nums
        return m * 60.0 + s
    if len(nums) == 1:
        return nums[0]
    return None


def _strip_code_fences(text: str) -> str:
    text = (text or "").strip()
    if "```" not in text:
        return text
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if m:
        return m.group(1).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_ads_payload(text: str) -> list[dict]:
    """Parse model output into [{start, end, ...}] using seconds floats.

    Accepts either {"ads": [...]} (preferred) or a bare JSON array of ranges.
    """
    text = _strip_code_fences(text)
    if not text:
        return []
    data = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None

    if data is None:
        # Prefer array salvage (handles truncated tails), then object.
        arr_start = text.find("[")
        arr_end = text.rfind("]")
        if arr_start != -1 and arr_end > arr_start:
            try:
                data = json.loads(text[arr_start : arr_end + 1])
            except json.JSONDecodeError:
                # Truncated array: salvage complete objects before the cut.
                last_brace = text[:arr_end].rfind("}") if arr_end != -1 else text.rfind("}")
                if last_brace > arr_start:
                    try:
                        data = json.loads(text[arr_start : last_brace + 1] + "]")
                    except json.JSONDecodeError:
                        data = None
        if data is None:
            obj_start = text.find("{")
            obj_end = text.rfind("}")
            if obj_start != -1 and obj_end > obj_start:
                data = json.loads(text[obj_start : obj_end + 1])
            else:
                raise ValueError("could not parse ad detection JSON")

    items: list = []
    if isinstance(data, dict):
        if "ads" in data:
            ads = data.get("ads")
            if not isinstance(ads, list):
                raise ValueError("ads must be an array")
            items = ads
        elif "start" in data and "end" in data:
            items = [data]
        else:
            raise ValueError("expected JSON object with ads array")
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError("expected JSON object or array")

    out: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if "start" not in item or "end" not in item:
            continue
        start = _parse_timestamp(item.get("start"))
        end = _parse_timestamp(item.get("end"))
        if start is None or end is None or end <= start:
            continue
        ad_type = str(item.get("type") or "unknown_ad").strip() or "unknown_ad"
        if ad_type not in ALLOWED_AD_TYPES:
            ad_type = "unknown_ad"
        sponsor = str(item.get("sponsor") or "unknown").strip() or "unknown"
        try:
            confidence = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        out.append(
            {
                "start": start,
                "end": end,
                "type": ad_type,
                "sponsor": sponsor,
                "confidence": confidence,
            }
        )
    return out


def _extract_json_array(text: str) -> list[dict]:
    """Compatibility wrapper — returns [{start, end}] from model output."""
    return [{"start": r["start"], "end": r["end"]} for r in _extract_ads_payload(text)]


def _gemini_generation_config(model: str) -> dict:
    """Build generationConfig: JSON schema + no thinking tokens on Gemini 3.x."""
    cfg: dict = {
        "temperature": 0.2,
        "maxOutputTokens": GEMINI_MAX_OUTPUT_TOKENS,
        "responseMimeType": "application/json",
        "responseSchema": GEMINI_RESPONSE_SCHEMA,
    }
    # Thinking tokens eat the output budget on long transcripts; disable them.
    if "gemini-3" in (model or "").lower():
        cfg["thinkingConfig"] = {"thinkingBudget": 0}
    return cfg


def _gemini_generate(api_key: str, model: str, user_content: str) -> str:
    """One-shot Gemini generateContent call with retries on transient errors."""
    url = f"{GEMINI_API_BASE}/models/{model}:generateContent"
    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": user_content}]}],
        "generationConfig": _gemini_generation_config(model),
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
                # Quota exhaustion (daily cap / billing limit) cannot be fixed by sleeping a few seconds.
                # Fail immediately instead of wasting time and burning repeated calls.
                if resp.status_code == 429 and any(
                    k in resp.text.lower() for k in ("quota", "resource_exhausted")
                ):
                    raise RuntimeError(f"Gemini quota exceeded: {last_err}")
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
    """Prove Gemini ad detection works with the configured key/model."""
    use_model = (model or "").strip() or gemini_model()
    if not is_gemini_model(use_model):
        use_model = GEMINI_DEFAULT_MODEL
    try:
        api_key = resolve_gemini_api_key()
        if not api_key:
            return {
                "ok": False,
                "provider": "gemini",
                "model": use_model,
                "error": "No Gemini API key. Paste one in Settings → Ad detection.",
            }
        start = time.monotonic()
        raw = _gemini_generate(api_key, use_model, 'Return exactly: {"ads": []}')
        parsed = _extract_ads_payload(raw)
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
        return {"ok": False, "provider": "gemini", "model": use_model, "error": str(exc)[-300:]}


def heuristic_ads(transcript: dict) -> list[Interval]:
    """Mark obvious sponsor-read sentences (tight cues only; no LLM)."""
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
            # Pad slightly but do NOT merge across long gaps into mega-ranges.
            hits.append(Interval(max(0.0, start - 1.0), end + 2.0))
    return merge_intervals(hits, gap=HEURISTIC_MERGE_GAP_SECONDS)


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
    """Transcript with sentences already covered by ads removed."""
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


def snap_to_transcript_times(ranges: list[Interval], transcript: dict) -> list[Interval]:
    """Snap start/end to nearest sentence boundaries from the transcript."""
    sentences = transcript.get("sentences") or []
    if not ranges or not isinstance(sentences, list):
        return ranges
    edges: list[float] = []
    for s in sentences:
        if not isinstance(s, dict):
            continue
        try:
            edges.append(float(s["start"]))
            edges.append(float(s["end"]))
        except (KeyError, TypeError, ValueError):
            continue
    if not edges:
        return ranges
    edges = sorted(set(edges))

    def nearest(t: float) -> float:
        return min(edges, key=lambda e: abs(e - t))

    out: list[Interval] = []
    for r in ranges:
        start = nearest(r.start)
        end = nearest(r.end)
        if end <= start:
            # Keep original if snap collapsed the range.
            start, end = r.start, r.end
        if end > start:
            out.append(Interval(start, end))
    return out


def filter_max_duration(
    ranges: list[Interval],
    *,
    duration: float = 0.0,
    max_seconds: float = MAX_AD_CUT_SECONDS,
) -> list[Interval]:
    """Drop or clip ranges that are impossibly long for a real ad break.

    Preroll (starts at 0) and postroll (ends at duration) may be slightly longer
    but still capped. A 51-minute midroll is always dropped.
    """
    out: list[Interval] = []
    for r in ranges:
        if r.duration <= max_seconds:
            out.append(r)
            continue
        is_preroll = r.start <= 1.0
        is_postroll = duration > 0 and r.end >= duration - 1.0
        if is_preroll:
            out.append(Interval(r.start, min(r.end, r.start + max_seconds)))
        elif is_postroll:
            out.append(Interval(max(r.start, r.end - max_seconds), r.end))
        else:
            logger.warning(
                "dropping oversized ad range %.1f–%.1f (%.0fs > max %.0fs)",
                r.start,
                r.end,
                r.duration,
                max_seconds,
            )
    return out


def evaluate_cut_guards(
    ranges: list[Interval],
    duration: float,
    *,
    max_coverage: float = MAX_AD_COVERAGE,
    min_remaining: float = MIN_REMAINING_SECONDS,
) -> CutGuardResult:
    """Refuse to cut when ads would wipe most of the episode."""
    if duration <= 0 or not ranges:
        return CutGuardResult(ok=True, ranges=ranges)
    ad_time = sum(r.duration for r in ranges)
    remaining = duration - ad_time
    coverage = ad_time / duration
    if coverage > max_coverage and remaining < min_remaining:
        return CutGuardResult(
            ok=False,
            ranges=ranges,
            reason=(
                f"Ad detection marked {coverage:.0%} of this episode "
                f"({ad_time / 60:.0f} min of ads in a {duration / 60:.0f} min show). "
                "Original kept — not cutting. Re-check or fix marks by hand."
            ),
        )
    if remaining < 60.0 and duration >= 300.0:
        return CutGuardResult(
            ok=False,
            ranges=ranges,
            reason=(
                f"Ad detection would leave only {remaining:.0f}s of a "
                f"{duration / 60:.0f} min episode. Original kept — not cutting."
            ),
        )
    return CutGuardResult(ok=True, ranges=ranges)


def find_ads_with_gemini(transcript: dict, progress_cb=None) -> AdDetectionResult:
    """One-shot Gemini ad detection on the full transcript (exactly one API call)."""
    model = gemini_model()
    if not is_gemini_model(model):
        model = GEMINI_DEFAULT_MODEL

    api_key = resolve_gemini_api_key()
    if not api_key:
        return AdDetectionResult(
            ranges=[],
            gemini_error="No Gemini API key. Paste one in Settings → Ad detection.",
            gemini_ok=False,
        )

    body = format_timestamped_transcript(transcript)
    if not body.strip():
        return AdDetectionResult(ranges=[], gemini_error=None, gemini_ok=True, sources=["gemini"])

    user = "TIMESTAMPED TRANSCRIPT:\n\n" + body
    if progress_cb is not None:
        progress_cb(0, 1)

    try:
        raw = _gemini_generate(api_key, model, user)
        parsed = _extract_ads_payload(raw)
        logger.info("gemini %s -> %d ranges", model, len(parsed))
    except Exception as primary_exc:
        last_err = str(primary_exc)[-300:]
        logger.warning("Gemini ad detection failed: %s", primary_exc)
        if progress_cb is not None:
            progress_cb(1, 1)
        return AdDetectionResult(ranges=[], gemini_error=last_err, gemini_ok=False)

    if progress_cb is not None:
        progress_cb(1, 1)

    # Drop low-confidence hits when the model provided confidence.
    kept = [r for r in parsed if float(r.get("confidence") or 1.0) >= MIN_CONFIDENCE]
    ranges = merge_intervals(
        [Interval(float(r["start"]), float(r["end"])) for r in kept],
        gap=NEARBY_AD_GAP_SECONDS,
    )
    ranges = snap_to_transcript_times(ranges, transcript)
    return AdDetectionResult(
        ranges=ranges,
        gemini_error=None,
        gemini_ok=True,
        sources=["gemini"],
    )


def find_ads_with_zen(transcript: dict, progress_cb=None) -> AdDetectionResult:
    """Gemini on full transcript (one request). Name kept for call sites.

    Does NOT union with heuristic mega-ranges — that wiped news shows.
    Heuristics are only a last-resort fallback when Gemini fails entirely.
    """
    llm = find_ads_with_gemini(transcript, progress_cb=progress_cb)
    if llm.gemini_ok:
        ranges = filter_min_duration(llm.ranges, min_seconds=MIN_AD_CUT_SECONDS)
        return AdDetectionResult(
            ranges=ranges,
            gemini_error=llm.gemini_error,
            gemini_ok=True,
            sources=llm.sources or ["gemini"],
        )
    # Gemini failed — keep tight heuristic hits only (already short gaps).
    heur = heuristic_ads(transcript)
    heur = filter_min_duration(heur, min_seconds=MIN_AD_CUT_SECONDS)
    return AdDetectionResult(
        ranges=heur,
        gemini_error=llm.gemini_error,
        gemini_ok=False,
        sources=["heuristic"] if heur else [],
    )


def pad_and_clamp_ads(
    ranges: list[Interval],
    duration: float,
    *,
    pad_start: float = AD_PAD_START_SECONDS,
    pad_end: float = AD_PAD_END_SECONDS,
    early_snap: float = EARLY_AD_SNAP_SECONDS,
    postroll_clamp: float = POSTROLL_CLAMP_SECONDS,
) -> list[Interval]:
    """Pad ad edges, snap early pre-rolls to 0, and clamp late post-rolls to duration."""
    if not ranges or duration <= 0:
        return ranges
    out: list[Interval] = []
    for r in ranges:
        start = max(0.0, r.start - pad_start)
        end = min(duration, r.end + pad_end)
        # Pre-roll: if detection starts in the first N seconds, snap to 0 so
        # untranscribed cold-open audio before the first word doesn't leak.
        if start <= early_snap:
            start = 0.0
        # Post-roll: only clamp on episodes long enough that the threshold is
        # meaningful (avoids wiping short fixtures / bumper clips).
        if duration >= postroll_clamp and end >= duration - postroll_clamp:
            end = duration
        if end > start:
            out.append(Interval(start, end))
    return merge_intervals(out, gap=NEARBY_AD_GAP_SECONDS)


def snap_to_silence(
    ranges: list[Interval],
    pcm: np.ndarray,
    sr: int,
    *,
    window: float = SILENCE_SNAP_WINDOW_SECONDS,
    hop_ms: float = 20.0,
) -> list[Interval]:
    """Snap each range edge to a nearby RMS valley so skips aren't mid-word.

    Expects `pcm` to be the FULL episode only when already loaded. Prefer
    `snap_edges_windowed` which decodes only ±window around each edge.
    """
    if not ranges or len(pcm) == 0 or sr <= 0:
        return ranges
    duration = len(pcm) / float(sr)
    ranges = pad_and_clamp_ads(ranges, duration)
    if window <= 0:
        return ranges
    hop = max(1, int(sr * hop_ms / 1000.0))
    frame = max(hop, int(sr * 0.04))
    n = 1 + max(0, (len(pcm) - frame) // hop)
    if n <= 1:
        return ranges
    # Vectorized RMS over frames (still one full pass — use snap_edges_windowed to avoid).
    starts = np.arange(n) * hop
    rms = np.empty(n, dtype=np.float32)
    for i, start in enumerate(starts):
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
        if prefer == "before":
            before = idx[times[idx] <= t]
            if len(before):
                return float(times[before[int(np.argmin(rms[before]))]])
        elif prefer == "after":
            after = idx[times[idx] >= t]
            if len(after):
                return float(times[after[int(np.argmin(rms[after]))]])
        return float(times[idx[int(np.argmin(local))]])

    snapped: list[Interval] = []
    for r in ranges:
        start = nearest_valley(r.start, "before")
        end = nearest_valley(r.end, "after")
        if r.start == 0.0:
            start = 0.0
        if r.end >= duration - 0.05:
            end = duration
        start = max(0.0, min(start, duration))
        end = max(0.0, min(end, duration))
        if end - start >= 1.0:
            snapped.append(Interval(start, end))
    return merge_intervals(snapped, gap=NEARBY_AD_GAP_SECONDS)


def snap_edges_windowed(
    ranges: list[Interval],
    audio_path,
    duration: float,
    *,
    window: float = SILENCE_SNAP_WINDOW_SECONDS,
    sr: int = 11025,
) -> list[Interval]:
    """Snap ad edges using only ±window audio snippets (no full-file PCM)."""
    if not ranges or duration <= 0:
        return ranges
    ranges = pad_and_clamp_ads(ranges, duration)
    if window <= 0:
        return ranges
    from .decode import load_pcm_window

    def valley_near(t: float, prefer: str) -> float:
        lo = max(0.0, t - window)
        hi = min(duration, t + window)
        if hi - lo < 0.05:
            return t
        try:
            pcm, actual_sr = load_pcm_window(audio_path, lo, hi, target_sr=sr)
        except Exception as exc:
            logger.warning("windowed snap decode failed at %.1f: %s", t, exc)
            return t
        if len(pcm) == 0 or actual_sr <= 0:
            return t
        hop = max(1, int(actual_sr * 0.02))
        frame = max(hop, int(actual_sr * 0.04))
        n = 1 + max(0, (len(pcm) - frame) // hop)
        if n <= 1:
            return t
        rms = np.empty(n, dtype=np.float32)
        for i in range(n):
            chunk = pcm[i * hop : i * hop + frame]
            rms[i] = float(np.sqrt(np.mean(chunk * chunk) + 1e-12))
        times = lo + (np.arange(n) * hop + frame / 2) / float(actual_sr)
        if prefer == "before":
            mask = times <= t
            if np.any(mask):
                idx = np.where(mask)[0]
                return float(times[idx[int(np.argmin(rms[idx]))]])
        elif prefer == "after":
            mask = times >= t
            if np.any(mask):
                idx = np.where(mask)[0]
                return float(times[idx[int(np.argmin(rms[idx]))]])
        return float(times[int(np.argmin(rms))])

    snapped: list[Interval] = []
    for r in ranges:
        start = 0.0 if r.start == 0.0 else valley_near(r.start, "before")
        end = duration if r.end >= duration - 0.05 else valley_near(r.end, "after")
        start = max(0.0, min(start, duration))
        end = max(0.0, min(end, duration))
        if end - start >= 1.0:
            snapped.append(Interval(start, end))
    return merge_intervals(snapped, gap=NEARBY_AD_GAP_SECONDS)


def union_ranges(*groups: list[Interval]) -> list[Interval]:
    merged: list[Interval] = []
    for g in groups:
        merged.extend(g)
    return merge_intervals(merged, gap=NEARBY_AD_GAP_SECONDS)


def effective_min_ad_seconds(override: float | None = None) -> float:
    """Floor for kept ad cuts — never below MIN_AD_CUT_SECONDS."""
    from . import db as _db

    if override is not None:
        return max(MIN_AD_CUT_SECONDS, float(override))
    return max(MIN_AD_CUT_SECONDS, _db.runtime_float("min_ad_seconds", minimum=1.0))


def filter_min_duration(ranges: list[Interval], min_seconds: float | None = None) -> list[Interval]:
    """Drop ranges shorter than min_seconds.

    When min_seconds is omitted, use the configured floor (at least MIN_AD_CUT_SECONDS).
    An explicit min_seconds is honored as-is (used by tests and callers that already
    applied effective_min_ad_seconds).
    """
    if min_seconds is None:
        floor = effective_min_ad_seconds()
    else:
        floor = float(min_seconds)
    return [r for r in ranges if r.duration >= floor]
