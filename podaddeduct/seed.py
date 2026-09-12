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

GEMINI_DEFAULT_MODEL = "opencode/deepseek-v4-flash-free"
GEMINI_FALLBACK_MODEL = "opencode/nemotron-3-ultra-free"
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

SYSTEM_PROMPT = """You are an advertisement detection classifier for podcast transcripts.

Your job is to identify the exact timestamp ranges containing commercial advertisements.

You will receive a timestamped transcript of a podcast episode.

## What counts as an advertisement

Mark a segment as an advertisement when it is clearly commercial or sponsored, including:

* Host-read sponsor advertisements
* Dynamically inserted advertisements
* Pre-recorded advertisements
* Pre-roll, mid-roll, and post-roll ads
* Sponsor messages
* Paid endorsements
* Promo-code or special-URL reads
* Free-trial offers
* Discount offers
* Calls to purchase, subscribe, download, register, sign up, or visit a sponsor
* Promotional descriptions of a product or service when clearly connected to sponsorship
* Multiple sponsors presented consecutively during the same commercial break

Host-read advertisements are especially important. They may sound conversational and may use the same host voice and tone as the rest of the episode.

Common host-read patterns include:

* "This episode is brought to you by..."
* "Today's sponsor is..."
* "Thanks to ___ for sponsoring..."
* "I've been using..."
* "You guys know I love..."
* "Go to..."
* "Use code..."
* "Get X% off..."
* "Try it free..."
* "That's..."
* "Terms apply."
* A personal story that transitions into promoting a sponsor

Do not require these exact phrases. Determine whether the content is commercial from its meaning and context.

## What does NOT count as an advertisement

Do NOT mark ordinary podcast conversation as advertising merely because a brand, product, company, website, book, movie, service, or person is mentioned.

Do NOT mark:

* Normal discussion of products or companies
* Unpaid recommendations
* News or commentary about a company
* Products relevant to the podcast topic
* Casual mentions of something the host uses
* Listener questions involving products
* Podcast housekeeping
* Episode introductions
* Normal calls to follow or subscribe to the podcast itself
* Requests to rate or review the podcast

Only classify these as advertisements when there is clear evidence that the segment is sponsored or commercially promotional.

## Podcast self-promotion

Do not normally classify promotion of the current podcast itself as an advertisement.

Examples that are NOT ads:

* "Subscribe to the show."
* "Leave us a review."
* "Follow us on Instagram."
* "Check out last week's episode."

However, promotion for another commercial product, paid subscription, event, network service, course, merchandise, or separate show MAY be advertising if it functions as a commercial break.

## Detecting inserted ads

Inserted advertisements may be obvious because:

* The speaker changes
* Audio/transcription style changes suddenly
* The topic changes abruptly
* A commercial message appears without a host introduction
* Several unrelated commercial messages appear consecutively
* Normal podcast conversation resumes abruptly afterward

Treat these as advertisements even if the transcript does not explicitly contain the word "sponsor."

## Determining ad boundaries

Boundary accuracy is important.

Use surrounding context before and after the advertisement to determine where normal podcast content ends and resumes.

The START timestamp should be the earliest supplied timestamp that belongs to the commercial break.

Include a host's transition into the advertisement when the transition is clearly part of the sponsor message.

Example:

Normal discussion
→ "Before we continue, I want to tell you about..."
→ sponsor message

The advertisement begins at "Before we continue..."

The END timestamp should be the final supplied timestamp belonging to the advertisement.

Do not include normal conversation after the commercial has finished.

Example:

"...visit example.com and use code SHOW for 20% off."
→ "Okay, back to what we were talking about..."

The advertisement ends before "Okay, back to what we were talking about."

## Multiple advertisements

If several advertisements occur consecutively with no meaningful podcast content between them, treat the entire sequence as ONE advertisement break.

Example:

Sponsor A
→ Sponsor B
→ Sponsor C
→ podcast resumes

Return one continuous ad range covering all three.

If normal podcast conversation occurs between sponsors, return separate ad ranges.

## Timestamp rules

You MUST use timestamps provided in the transcript.

Never invent timestamps.

Never estimate timestamps based on word count or speaking speed.

If the exact transition occurs between two supplied timestamps, use the closest supplied transcript timestamp that correctly contains the beginning or end of the advertisement.

Do not extend an advertisement simply because you are uncertain.

## Ambiguous segments

Use the complete surrounding context when deciding whether something is an advertisement.

Host-read advertisements can intentionally sound like normal conversation.

Look for combinations of evidence such as:

* Sponsor acknowledgement
* Product benefits
* Personal testimonial
* Promotional language
* Discount
* Promo code
* Special URL
* Price
* Trial offer
* Purchase instructions
* Call to action

A conversational tone alone is NOT evidence that something is normal podcast content.

When uncertain, assign a lower confidence rather than inventing certainty.

## Output

Return ONLY valid JSON.

Do not return Markdown.

Do not explain your answer.

Do not include text before or after the JSON.

Use exactly this structure:

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

Allowed `type` values:

* `host_read`
* `inserted_ad`
* `unknown_ad`

If the sponsor cannot be determined:

"sponsor": "unknown"

Confidence must be a number between 0.0 and 1.0.

If no advertisements are present, return:

{
"ads": []
}

Before producing the JSON, silently verify that:

1. Every detected segment is genuinely commercial.
2. Host-read ads have not been missed because they sound conversational.
3. Ordinary product discussion has not been incorrectly classified as advertising.
4. Each start and end timestamp exists in the supplied transcript.
5. Consecutive ads have been combined appropriately.
6. Normal podcast content is excluded from the detected ranges."""


def resolve_gemini_api_key() -> str | None:
    """Prefer UI-saved key, then GEMINI_API_KEY / GOOGLE_API_KEY env."""
    from .secrets import get_gemini_api_key

    return get_gemini_api_key()


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
    "opencode/deepseek-v4-flash-free",
    "opencode/nemotron-3-ultra-free",
    "opencode/nemotron-3.5-lightning-free",
    "opencode/mimo-v2.5-free",
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


def _models_to_try(model: str | None = None) -> list[str]:
    models_to_try: list[str] = []
    if (model or "").strip():
        models_to_try.append(model.strip())
    for m in OPENCODE_DEFAULT_MODELS:
        if m not in models_to_try:
            models_to_try.append(m)
    return models_to_try


def opencode_generate(prompt: str, model: str | None = None) -> str:
    """Invoke opencode CLI runner to generate JSON response."""
    import subprocess

    bin_path = get_opencode_bin()
    if not bin_path:
        raise RuntimeError("opencode CLI not found in PATH")

    models_to_try = _models_to_try(model)
    last_err = ""

    for m in models_to_try:
        cmd = [bin_path, "run", prompt, "-m", m, "--pure"]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            last_err = f"opencode timed out after 300s on {m}"
            logger.warning(last_err)
            continue
        if proc.returncode == 0:
            logger.info("opencode succeeded with model %s", m)
            return proc.stdout
        err_msg = (proc.stderr or proc.stdout or "").strip()
        last_err = f"{m} failed ({proc.returncode}): {err_msg[:200]}"
        logger.warning("opencode attempt failed: %s; trying next model", last_err)

    raise RuntimeError(f"opencode failed: {last_err}")


def find_ads_with_opencode(transcript: dict, model: str | None = None) -> AdDetectionResult:
    """Run ad detection via opencode CLI on full transcript."""
    body = format_timestamped_transcript(transcript)
    if not body.strip():
        return AdDetectionResult(ranges=[], gemini_error=None, gemini_ok=True)

    prompt = SYSTEM_PROMPT + "\n\nTIMESTAMPED TRANSCRIPT:\n\n" + body
    try:
        raw = opencode_generate(prompt, model=model)
        parsed = _extract_ads_payload(raw)
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
        if use_model.startswith("opencode") or is_opencode_available():
            start = time.monotonic()
            raw = opencode_generate(
                'Return ONLY valid JSON: {"ads": []}',
                model=use_model if use_model.startswith("opencode") else None,
            )
            parsed = _extract_ads_payload(raw)
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
            return {"ok": False, "error": "No OpenCode CLI or Gemini API key available."}
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
    """One-shot AI ad detection on full transcript using OpenCode (default) or Gemini."""
    model = gemini_model()

    # OpenCode is the default path — model IDs start with "opencode".
    if model.startswith("opencode") or (not resolve_gemini_api_key() and is_opencode_available()):
        if progress_cb is not None:
            progress_cb(0, 1)
        res = find_ads_with_opencode(
            transcript,
            model=model if model.startswith("opencode") else None,
        )
        if progress_cb is not None:
            progress_cb(1, 1)
        return res

    api_key = resolve_gemini_api_key()
    if not api_key:
        if is_opencode_available():
            if progress_cb is not None:
                progress_cb(0, 1)
            res = find_ads_with_opencode(transcript)
            if progress_cb is not None:
                progress_cb(1, 1)
            return res
        logger.info("No Gemini API key / OpenCode CLI; skipping LLM ad detection")
        return AdDetectionResult(ranges=[], gemini_error=None, gemini_ok=False)

    body = format_timestamped_transcript(transcript)
    if not body.strip():
        return AdDetectionResult(ranges=[], gemini_error=None, gemini_ok=True)

    user = "TIMESTAMPED TRANSCRIPT:\n\n" + body
    last_err: str | None = None
    parsed: list[dict] = []

    if progress_cb is not None:
        progress_cb(0, 1)

    try:
        raw = _gemini_generate(api_key, model, user)
        parsed = _extract_ads_payload(raw)
        logger.info("gemini %s -> %d ranges", model, len(parsed))
    except Exception as primary_exc:
        err_text = str(primary_exc)
        last_err = err_text[-300:]
        logger.warning("Gemini ad detection failed: %s", primary_exc)

        if is_opencode_available():
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
