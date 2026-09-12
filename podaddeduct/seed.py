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
GEMINI_FALLBACK_MODEL = "gemini-2.0-flash"
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_MAX_TRIES = 5
GEMINI_MAX_OUTPUT_TOKENS = 8192

# MinusPod-style boundary defaults.
EARLY_AD_SNAP_SECONDS = 30.0
POSTROLL_CLAMP_SECONDS = 45.0
AD_PAD_START_SECONDS = 0.3
AD_PAD_END_SECONDS = 0.6
NEARBY_AD_GAP_SECONDS = 15.0
HEURISTIC_MERGE_GAP_SECONDS = 35.0
SILENCE_SNAP_WINDOW_SECONDS = 2.0
# No real podcast ad is a 4–7s keyword sentence — reject micro-cuts.
MIN_AD_CUT_SECONDS = 15.0

GEMINI_RESPONSE_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "start": {"type": "NUMBER"},
            "end": {"type": "NUMBER"},
        },
        "required": ["start", "end"],
    },
}


@dataclass
class AdDetectionResult:
    """Ranges plus whether Gemini ran / failed (for episode-page warnings)."""

    ranges: list[Interval] = field(default_factory=list)
    gemini_error: str | None = None
    gemini_ok: bool = False

# Obvious sponsor-read cues (MinusPod AD_START_PHRASES + sports disclaimers).
_HEURISTIC_RE = re.compile(
    r"(?i)\b("
    r"brought to you by|"
    r"this episode is (brought to you|sponsored)|"
    r"sponsored by|"
    r"presented by|"
    r"our sponsor|"
    r"today'?s sponsor|"
    r"thanks to our sponsor|"
    r"thank our sponsor|"
    r"word from our sponsor|"
    r"a word from|"
    r"support comes from|"
    r"supported by|"
    r"let'?s take a (quick )?break|"
    r"take a quick break|"
    r"take a moment|"
    r"we'?ll be right back|"
    r"we'?ll be right back after|"
    r"after (these|this) (messages?|break|word from)|"
    r"ad break|"
    r"commercial break|"
    r"before we get (back )?into|"
    r"first let me tell you|"
    r"i want to tell you about|"
    r"let me tell you about|"
    r"promo code|"
    r"use (the )?code|"
    r"discount code|"
    r"enter code|"
    r"at checkout|"
    r"special offer|"
    r"visit \w[\w.-]*\.(com|net|org|io)|"
    r"go to \w[\w.-]*\.(com|net|org|io)|"
    r"head to \w[\w.-]*\.(com|net|org|io)|"
    r"support (for )?this (show|podcast) comes from|"
    r"paid for by|"
    r"fanduel|"
    r"draftkings|"
    r"betmgm|"
    r"caesars|"
    r"gambling problem|"
    r"1[\s-]?800[\s-]?gambler|"
    r"must be 21|"
    r"terms and conditions apply"
    r")\b"
)

SYSTEM_PROMPT = """You mark podcast advertisements for cutting.
Given a timestamped transcript ([start-end] text per line), return ONLY a JSON array of
objects {"start": <seconds>, "end": <seconds>} for every ad segment.

DETECTION VECTORS (what to look for):
1. SEMANTIC SHIFT:
   Watch for sudden departures from the main episode topic into universal commercial themes (e.g.
   health, athletic greens, meal delivery, therapy, VPNs, web hosting, hiring, mattresses,
   insurance, sports betting, financial apps, security).
2. DISCLAIMERS & PROMO CODES:
   Flag phrases like "Go to [URL] and use code...", "visit [URL]", "promo code", "discount code",
   "at checkout", "Thanks to our sponsor...", "Support for this podcast comes from...",
   or legal disclaimers like "Must be 21 or older", "gambling problem call 1-800-GAMBLER",
   "terms and conditions apply".
3. DYNAMIC & PLATFORM INSERTS:
   Pre-rolls, mid-rolls, post-rolls, and network promos/station IDs (Acast, Megaphone, iHeart,
   Spotify for Podcasters, Wondery) that sound like radio commercials.

BOUNDARY RULES (critical — leftover tails are failures):
- AD START: Include the transition INTO the ad ("let's take a break", "a word from our sponsors",
  "brought to you by", "before we get into it", "first let me tell you about"), not just the pitch.
- AD END: The ad ends when SHOW CONTENT resumes, NOT when the pitch ends. Wait for:
  - Topic change back to episode content
  - Host says "anyway", "alright", "all right", "so", "back to the show" and changes subject
  - AFTER the final URL / promo code mention (they often repeat it)
- POST-ROLLS: Mark from the first promotional word through the LAST promotional / disclaimer
  word (extend through the end of the transcript if the episode ends on ads).
- Merge contiguous / near-contiguous ad sentences into one range (gaps under ~15 seconds of filler).
- Use the transcript timestamps exactly.

WHAT NOT TO MARK:
- Show content, banter about the topic, or brief brand mentions that are not ads
- A guest discussing their own work in an interview
- The host organically mentioning their own other shows / Patreon mid-conversation (not a produced promo block)

STRUCTURED OUTPUT:
Return [] if there are no ads. No markdown explanation, no commentary — JSON array only."""


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


def _gemini_generation_config(model: str) -> dict:
    """Build generationConfig: JSON schema + no thinking tokens on Gemini 3.x."""
    cfg: dict = {
        # Gemini 3.x docs recommend leaving temperature at default (1.0).
        "temperature": 1.0,
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


OPENCODE_DEFAULT_MODELS = [
    "opencode/kimi-k2.5",
    "opencode/minimax-m2.5",
    "opencode/mimo-v2.5-free",
    "opencode/nemotron-3.5-lightning-free",
]


def get_opencode_bin() -> str | None:
    """Find opencode CLI binary in PATH or standard user install locations."""
    import os
    import shutil

    b = shutil.which("opencode")
    if b:
        return b
    home_bin = os.path.expanduser("~/.opencode/bin/opencode")
    if os.path.isfile(home_bin) and os.access(home_bin, os.X_OK):
        return home_bin
    return None


def is_opencode_available() -> bool:
    return get_opencode_bin() is not None


def opencode_generate(prompt: str, model: str | None = None) -> str:
    """Invoke opencode CLI runner to generate JSON response."""
    import subprocess

    bin_path = get_opencode_bin()
    if not bin_path:
        raise RuntimeError("opencode CLI not found in PATH")

    models_to_try = [model.strip()] if (model or "").strip() else list(OPENCODE_DEFAULT_MODELS)
    last_err = ""

    for m in models_to_try:
        cmd = [bin_path, "run", prompt, "-m", m, "--pure"]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        except subprocess.TimeoutExpired:
            last_err = f"opencode timed out after 180s on {m}"
            logger.warning(last_err)
            continue
        if proc.returncode == 0:
            return proc.stdout
        err_msg = (proc.stderr or proc.stdout or "").strip()
        last_err = f"{m} failed ({proc.returncode}): {err_msg[:200]}"
        if any(w in err_msg.lower() for w in ("insufficient balance", "creditserror", "unauthorized")):
            logger.warning("opencode %s credits unavailable; trying next model", m)
            continue
        logger.warning("opencode attempt failed: %s", last_err)

    raise RuntimeError(f"opencode failed: {last_err}")


def find_ads_with_opencode(transcript: dict, model: str | None = None) -> AdDetectionResult:
    """Run ad detection via opencode CLI on full transcript."""
    body = format_timestamped_transcript(transcript)
    if not body.strip():
        return AdDetectionResult(ranges=[], gemini_error=None, gemini_ok=True)

    user = (
        "Return JSON array of ad ranges for this podcast transcript. "
        "Timestamps are in seconds [start-end].\n\n"
        f"{body}"
    )
    prompt = SYSTEM_PROMPT + "\n\n" + user
    try:
        raw = opencode_generate(prompt, model=model)
        parsed = _extract_json_array(raw)
        logger.info("opencode ad detection -> %d ranges", len(parsed))
        ranges = merge_intervals(
            [Interval(float(r["start"]), float(r["end"])) for r in parsed],
            gap=NEARBY_AD_GAP_SECONDS,
        )
        return AdDetectionResult(ranges=ranges, gemini_error=None, gemini_ok=True)
    except Exception as exc:
        logger.warning("opencode ad detection failed: %s", exc)
        return AdDetectionResult(ranges=[], gemini_error=str(exc)[-300:], gemini_ok=False)


def test_gemini_connection(model: str | None = None) -> dict:
    """Send one tiny request to prove key + model work. Never raises."""
    try:
        use_model = (model or "").strip() or gemini_model()
        if use_model.startswith("opencode") or (not resolve_gemini_api_key() and is_opencode_available()):
            start = time.monotonic()
            raw = opencode_generate("Return strictly JSON: []", model=use_model if use_model.startswith("opencode") else None)
            parsed = _extract_json_array(raw)
            ms = int((time.monotonic() - start) * 1000)
            return {
                "ok": True,
                "provider": "opencode",
                "model": use_model,
                "ms": ms,
                "ranges": len(parsed),
            }
        api_key = resolve_gemini_api_key()
        if not api_key:
            return {"ok": False, "error": "No Gemini API key saved yet."}
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


def find_ads_with_gemini(transcript: dict, progress_cb=None) -> AdDetectionResult:
    """One-shot AI ad detection on full transcript using Gemini or OpenCode CLI."""
    model = gemini_model()
    if model.startswith("opencode"):
        if progress_cb is not None:
            progress_cb(0, 1)
        res = find_ads_with_opencode(transcript, model=model)
        if progress_cb is not None:
            progress_cb(1, 1)
        return res

    api_key = resolve_gemini_api_key()
    if not api_key:
        logger.info("No Gemini API key; skipping LLM ad detection")
        return AdDetectionResult(ranges=[], gemini_error=None, gemini_ok=False)

    body = format_timestamped_transcript(transcript)
    if not body.strip():
        return AdDetectionResult(ranges=[], gemini_error=None, gemini_ok=True)

    user = (
        "Return JSON array of ad ranges for this podcast transcript. "
        "Timestamps are in seconds [start-end].\n\n"
        f"{body}"
    )
    last_err: str | None = None
    parsed: list[dict] = []

    if progress_cb is not None:
        progress_cb(0, 1)

    try:
        raw = _gemini_generate(api_key, model, user)
        parsed = _extract_json_array(raw)
        logger.info("gemini %s -> %d ranges", model, len(parsed))
    except Exception as primary_exc:
        fallback = GEMINI_FALLBACK_MODEL
        err_text = str(primary_exc)
        if model != fallback and ("503" in err_text or "429" in err_text or "high demand" in err_text.lower() or "quota" in err_text.lower()):
            logger.warning(
                "Gemini %s failed (%s); trying fallback %s",
                model,
                primary_exc,
                fallback,
            )
            try:
                raw = _gemini_generate(api_key, fallback, user)
                parsed = _extract_json_array(raw)
                logger.info("gemini fallback %s -> %d ranges", fallback, len(parsed))
            except Exception as fallback_exc:
                last_err = str(fallback_exc)[-300:]
                logger.warning("Gemini fallback %s failed: %s", fallback, fallback_exc)
        else:
            last_err = err_text[-300:]
            logger.warning("Gemini ad detection failed: %s", primary_exc)

        # If Gemini quota exceeded or failed completely, fall back to OpenCode CLI if enabled
        from . import db as _db

        if not parsed and (settings.opencode_fallback or _db.runtime_bool("opencode_fallback")) and is_opencode_available():
            logger.warning("Gemini failed (%s); attempting OpenCode CLI backup", last_err)
            opencode_res = find_ads_with_opencode(transcript)
            if opencode_res.gemini_ok:
                if progress_cb is not None:
                    progress_cb(1, 1)
                return opencode_res

    if progress_cb is not None:
        progress_cb(1, 1)

    if last_err and not parsed:
        return AdDetectionResult(ranges=[], gemini_error=last_err, gemini_ok=False)

    ranges = merge_intervals(
        [Interval(float(r["start"]), float(r["end"])) for r in parsed],
        gap=NEARBY_AD_GAP_SECONDS,
    )
    return AdDetectionResult(ranges=ranges, gemini_error=last_err, gemini_ok=True)


def find_ads_with_zen(transcript: dict, progress_cb=None) -> AdDetectionResult:
    """Heuristics + Gemini on full transcript (MinusPod-style). Name kept for call sites."""
    heur = heuristic_ads(transcript)
    llm = find_ads_with_gemini(transcript, progress_cb=progress_cb)
    # When Gemini failed entirely, do not keep isolated heuristic micro-cuts —
    # they look like "ads removed" while leaving real ad breaks intact.
    merged = union_ranges(heur, llm.ranges)
    merged = filter_min_duration(merged, min_seconds=MIN_AD_CUT_SECONDS)
    return AdDetectionResult(
        ranges=merged,
        gemini_error=llm.gemini_error,
        gemini_ok=llm.gemini_ok,
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
    """Snap each range edge to a nearby RMS valley so skips aren't mid-word."""
    if not ranges or len(pcm) == 0 or sr <= 0:
        return ranges
    duration = len(pcm) / float(sr)
    # Pad / clamp first so silence snapping searches around the true ad edges.
    ranges = pad_and_clamp_ads(ranges, duration)
    if window <= 0:
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

    for r in ranges:
        start = nearest_valley(r.start, "before")
        end = nearest_valley(r.end, "after")
        # Re-apply early/post-roll clamps after silence snap so valleys don't undo them.
        if r.start == 0.0:
            start = 0.0
        if r.end >= duration - 0.05:
            end = duration
        start = max(0.0, min(start, duration))
        end = max(0.0, min(end, duration))
        # Keep any non-empty snap; MIN_AD_CUT_SECONDS is enforced later by
        # filter_min_duration so short fixtures / mid-snap shrinkage still work.
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
