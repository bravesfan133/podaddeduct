from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import httpx
import numpy as np

from .config import settings
from .intervals import Interval, merge_intervals
from .stt import chunk_transcript_lines

logger = logging.getLogger("podaddeduct.seed")

_JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]")


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


SYSTEM_PROMPT = """You mark podcast advertisements for skip chapters.
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
    # Strip common fences
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


def _extract_response_text(data: dict) -> str:
    """Pull assistant text from an OpenAI-style Responses API payload."""
    direct = data.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct
    parts: list[str] = []
    for item in data.get("output") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") and item.get("type") not in {"message", "output_text"}:
            # Still walk content on message-like items
            pass
        for block in item.get("content") or []:
            if not isinstance(block, dict):
                continue
            text = block.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
            elif isinstance(text, dict) and text.get("value"):
                parts.append(str(text["value"]))
    return "\n".join(parts).strip()


def _zen_via_opencode_cli(user_content: str, *, model: str | None = None) -> str:
    """Session-authenticated path for contributor-free models (MissingSessionID on raw API)."""
    import shutil
    import subprocess

    bin_name = settings.opencode_bin
    resolved = shutil.which(bin_name) or bin_name
    # Common install location when not on PATH for the server process
    if not Path(resolved).exists() and not shutil.which(bin_name):
        home_bin = Path.home() / ".opencode" / "bin" / "opencode"
        if home_bin.exists():
            resolved = str(home_bin)
    model_id = model or settings.opencode_model
    prompt = f"{SYSTEM_PROMPT}\n\n{user_content}"
    cmd = [
        resolved,
        "run",
        "-m",
        model_id,
        "--format",
        "json",
        "--pure",
        prompt,
    ]
    logger.info("Zen via OpenCode CLI: %s %s", resolved, model_id)
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(
            f"opencode run failed ({proc.returncode}): {(proc.stderr or proc.stdout)[-1500:]}"
        )
    texts: list[str] = []
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "text":
            part = ev.get("part") or {}
            t = part.get("text")
            if isinstance(t, str) and t:
                texts.append(t)
    return "\n".join(texts).strip()


def _zen_responses(api_key: str, model: str, user_content: str) -> str:
    """Call Zen Responses API (chat/completions returns 500 for muse-spark free)."""
    from . import db

    url = db.runtime_str("zen_base_url").rstrip("/") + "/responses"
    # Prefer messages-shaped input so the system prompt stays separate.
    payload = {
        "model": model,
        "stream": False,
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=180.0) as client:
        resp = client.post(url, headers=headers, json=payload)
        if resp.status_code >= 400:
            # Retry once with the simple string input form from Zen docs.
            if resp.status_code in {400, 422, 500}:
                simple = {
                    "model": model,
                    "stream": False,
                    "input": f"{SYSTEM_PROMPT}\n\n{user_content}",
                }
                resp2 = client.post(url, headers=headers, json=simple)
                if resp2.status_code < 400:
                    data = resp2.json()
                    return _extract_response_text(data if isinstance(data, dict) else {})
                resp = resp2
            body = resp.text[:500]
            if "MissingSessionID" in body or "only be used in OpenCode" in body:
                logger.warning("Zen raw API MissingSessionID — falling back to opencode CLI")
                return _zen_via_opencode_cli(user_content)
            raise httpx.HTTPStatusError(
                f"{resp.status_code} {body}",
                request=resp.request,
                response=resp,
            )
        data = resp.json()
    return _extract_response_text(data if isinstance(data, dict) else {})


def _zen_chat_completions(api_key: str, model: str, user_content: str) -> str:
    """Chat Completions path for free models that are not Responses-only."""
    from . import db

    url = db.runtime_str("zen_base_url").rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=180.0) as client:
        resp = client.post(url, headers=headers, json=payload)
        if resp.status_code >= 400:
            body = resp.text[:500]
            if "MissingSessionID" in body or "only be used in OpenCode" in body:
                logger.warning("Zen chat MissingSessionID — falling back to opencode CLI (%s)", model)
                return _zen_via_opencode_cli(user_content, model=f"opencode/{model}")
            raise httpx.HTTPStatusError(
                f"{resp.status_code} {body}",
                request=resp.request,
                response=resp,
            )
        data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    return str(msg.get("content") or "")


def _zen_call(api_key: str, model: str, user_content: str) -> str:
    """Route muse-spark* to Responses; other free models to chat/completions."""
    if "muse-spark" in model:
        return _zen_responses(api_key, model, user_content)
    return _zen_chat_completions(api_key, model, user_content)


CURATED_ZEN_MODELS = [
    "muse-spark-1.3-contributor-free",
    "deepseek-v4-flash-free",
]


def fetch_zen_models() -> dict:
    """List model ids from the Zen API using the resolved key.

    Returns {"models": [...], "live": True} or {"models": curated, "live": False}.
    Never raises — the UI must work before any key is configured.
    """
    from . import db

    api_key = resolve_zen_api_key()
    base = (db.runtime_str("zen_base_url") or "").rstrip("/")
    if not api_key or not base:
        return {"models": list(CURATED_ZEN_MODELS), "live": False}
    try:
        with httpx.Client(timeout=20.0) as client:
            resp = client.get(
                base + "/models",
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
        if not ids:
            return {"models": list(CURATED_ZEN_MODELS), "live": False}
        return {"models": sorted(ids)[:200], "live": True}
    except Exception as exc:
        logger.warning("Zen /models failed, using curated list: %s", exc)
        return {"models": list(CURATED_ZEN_MODELS), "live": False}


def test_zen_connection(model: str | None = None) -> dict:
    """Send one tiny request to prove key + model work. Never raises."""
    from . import db

    try:
        api_key = resolve_zen_api_key()
        if not api_key:
            return {"ok": False, "error": "No API key saved yet."}
        use_model = (model or "").strip() or (db.models_to_try() + [""])[0]
        if not use_model:
            return {"ok": False, "error": "No model selected."}
        import time

        start = time.monotonic()
        raw = _zen_call(api_key, use_model, "Return exactly: []")
        parsed = _extract_json_array(raw)
        ms = int((time.monotonic() - start) * 1000)
        return {"ok": True, "model": use_model, "ms": ms, "ranges": len(parsed)}
    except Exception as exc:
        logger.warning("Zen test failed: %s", exc)
        return {"ok": False, "error": str(exc)[-300:]}


def find_ads_with_zen(transcript: dict) -> list[Interval]:
    """Call Zen free models on transcript chunks; merge all ad ranges."""
    from . import db

    api_key = resolve_zen_api_key()
    if not api_key:
        raise RuntimeError(
            "No OpenCode Zen API key. Paste one on the home page (or set ZEN_API_KEY / OPENCODE_API_KEY). "
            "Get a key at https://opencode.ai/auth — default free model is muse-spark-1.3-contributor-free "
            "(Muse Spark 1.3 Free) via POST /zen/v1/responses, with OpenCode CLI fallback."
        )

    chunks = chunk_transcript_lines(transcript, max_chars=db.runtime_int("zen_chunk_chars", minimum=1000))
    if not chunks:
        return []

    models = db.models_to_try()

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
                raw = _zen_call(api_key, model, user)
                parsed = _extract_json_array(raw)
                logger.info(
                    "Zen %s chunk %s/%s -> %d ranges",
                    model,
                    i + 1,
                    len(chunks),
                    len(parsed),
                )
                break
            except Exception as exc:
                last_err = exc
                logger.warning(
                    "Zen model %s failed on chunk %s: %s; raw=%r",
                    model,
                    i + 1,
                    exc,
                    (raw or "")[:500],
                )
        if parsed is None:
            raise RuntimeError(f"Zen failed on chunk {i + 1}: {last_err}")
        for r in parsed:
            all_ranges.append(Interval(float(r["start"]), float(r["end"])))

    return merge_intervals(all_ranges, gap=2.0)


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
    # Frame RMS
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
        # Prefer quieter frames; if tie, prefer direction
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
