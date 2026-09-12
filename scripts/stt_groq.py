"""Transcribe with Groq's Whisper API (cloud, fastest backend).

Same CLI contract as stt_sidecar.py (Parakeet/Mac) and stt_faster.py:
    python stt_groq.py <audio> <dest.json> [model]

Writes {"sentences": [{"start": s, "end": e, "text": t}, ...], "source": "groq"}.
Model is a Groq Whisper id (default whisper-large-v3-turbo).
API key comes from the GROQ_API_KEY environment variable (injected by the
app from Settings → Server, so no .env needed).

Groq caps uploads at 25MB (free tier), so long episodes are compressed to
16kHz mono and split into overlapping chunks, transcribed independently,
then merged with timestamps offset back. Overlap duplicates are dropped.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import NoReturn

import httpx

API_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
DEFAULT_MODEL = "whisper-large-v3-turbo"
# 48kbps mono ≈ 6KB/s → 20MB holds ~55min; overlap absorbs cut edges.
CHUNK_TARGET_MB = 20
CHUNK_BITRATE_KBPS = 48
OVERLAP_S = 15.0
MAX_TRIES = 5


def fail(msg: str) -> NoReturn:
    print(f"stt_groq: {msg}", file=sys.stderr)
    raise SystemExit(1)


def probe_duration(audio: Path) -> float:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(audio)],
        capture_output=True, text=True, check=False,
    )
    try:
        return max(0.0, float((proc.stdout or "").strip()))
    except ValueError:
        return 0.0


def plan_chunks(duration_s: float, target_mb: float = CHUNK_TARGET_MB,
                overlap_s: float = OVERLAP_S) -> list[tuple[float, float]]:
    """Split duration into (start, length) windows fitting the upload cap."""
    if duration_s <= 0:
        return [(0.0, 0.0)]
    bytes_per_s = CHUNK_BITRATE_KBPS * 1000 / 8
    window = max(60.0, target_mb * 1024 * 1024 / bytes_per_s)
    chunks: list[tuple[float, float]] = []
    start = 0.0
    while start < duration_s:
        chunks.append((start, min(window, duration_s - start)))
        if start + window >= duration_s:
            break
        start += window - overlap_s
    return chunks


def split_chunk(src: Path, start: float, length: float, dest: Path) -> None:
    proc = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{length:.3f}",
         "-i", str(src), "-ar", "16000", "-ac", "1", "-b:a", f"{CHUNK_BITRATE_KBPS}k",
         str(dest)],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0 or not dest.exists():
        fail(f"ffmpeg split failed: {(proc.stderr or '')[-500:]}")


def transcribe_chunk(client: httpx.Client, key: str, path: Path, model: str) -> list[dict]:
    """POST one chunk, return raw segments. Retries rate limits/server errors."""
    last_err = ""
    retry_after: float | None = None
    for attempt in range(1, MAX_TRIES + 1):
        retry_after = None
        try:
            with path.open("rb") as f:
                resp = client.post(
                    API_URL,
                    headers={"Authorization": f"Bearer {key}"},
                    files={"file": (path.name, f, "audio/mpeg")},
                    data={"model": model, "response_format": "verbose_json",
                          "timestamp_granularities[]": "segment", "temperature": "0"},
                )
            if resp.status_code == 200:
                data = resp.json()
                segs = data.get("segments") or []
                out = []
                for s in segs:
                    if not isinstance(s, dict):
                        continue
                    try:
                        start = float(s.get("start", 0))
                        end = float(s.get("end", start))
                    except (TypeError, ValueError):
                        continue
                    text = str(s.get("text") or "").strip()
                    if text and end > start:
                        out.append({"start": start, "end": end, "text": text})
                return out
            last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
            try:
                retry_after = float(resp.headers.get("retry-after", ""))
            except (TypeError, ValueError):
                retry_after = None
            if resp.status_code not in (408, 425, 429, 500, 502, 503, 504):
                fail(f"Groq rejected chunk {path.name}: {last_err}")
        except httpx.HTTPError as exc:
            last_err = f"{type(exc).__name__}: {exc}"
        if attempt < MAX_TRIES:
            wait = retry_after if retry_after and retry_after > 0 else min(120.0, 5 * 2 ** attempt)
            print(f"stt_groq: chunk {path.name} attempt {attempt} failed, retry in {wait:.0f}s: {last_err}")
            time.sleep(wait)
    fail(f"Groq gave up on chunk {path.name} after {MAX_TRIES} tries: {last_err}")


def merge_segments(chunked: list[tuple[float, list[dict]]]) -> list[dict]:
    """Offset chunk segments to episode time, dropping overlap duplicates.

    chunked: [(chunk_start, segments)]. For chunks after the first, segments
    ending inside the overlap window are duplicates of the previous chunk.
    """
    out: list[dict] = []
    for i, (offset, segs) in enumerate(chunked):
        for s in segs:
            try:
                start = float(s.get("start", 0)) + offset
                end = float(s.get("end", start)) + offset
            except (TypeError, ValueError):
                continue
            text = str(s.get("text") or "").strip()
            if not text or end <= start:
                continue
            if i > 0 and (end - offset) <= OVERLAP_S + 0.5:
                continue
            out.append({"start": start, "end": end, "text": text})
    out.sort(key=lambda s: (s["start"], s["end"]))
    return out


def main() -> None:
    if len(sys.argv) < 3:
        print("usage: stt_groq.py <audio> <dest.json> [model]", file=sys.stderr)
        raise SystemExit(2)
    audio = Path(sys.argv[1])
    dest = Path(sys.argv[2])
    model = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_MODEL
    key = (os.environ.get("GROQ_API_KEY") or "").strip()
    if not key:
        fail("GROQ_API_KEY not set (paste a Groq key in Settings → Server).")
    if not audio.exists():
        fail(f"audio not found: {audio}")

    duration = probe_duration(audio)
    if duration <= 0:
        fail("could not read audio duration (is ffmpeg/ffprobe installed?).")
    chunks = plan_chunks(duration)
    print(f"stt_groq: {duration / 60:.1f}min audio → {len(chunks)} chunk(s), model={model}")

    results: list[tuple[float, list[dict]]] = []
    with httpx.Client(timeout=600.0) as client:
        with tempfile.TemporaryDirectory(prefix="groq-chunks-") as tmp:
            for i, (start, length) in enumerate(chunks):
                cpath = Path(tmp) / f"chunk-{i:03d}.mp3"
                split_chunk(audio, start, length, cpath)
                segs = transcribe_chunk(client, key, cpath, model)
                print(f"stt_groq: chunk {i + 1}/{len(chunks)} → {len(segs)} segments")
                results.append((start, segs))

    sentences = merge_segments(results)
    if not sentences:
        fail("Groq returned no transcript segments.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json.dumps({"sentences": sentences, "source": "groq", "model": model}, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"stt_groq: {len(sentences)} sentences -> {dest}")


if __name__ == "__main__":
    main()
