from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

from .config import settings

logger = logging.getLogger("podaddeduct.stt")


def transcript_path_for(episode_id: int) -> Path:
    return settings.data_dir / "transcripts" / f"{episode_id}.json"


def load_transcript(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def format_timestamped_transcript(transcript: dict, *, max_chars: int | None = None) -> str:
    """One sentence per line: [start-end] text"""
    lines: list[str] = []
    for s in transcript.get("sentences") or []:
        text = (s.get("text") or "").strip()
        if not text:
            continue
        start = float(s["start"])
        end = float(s["end"])
        lines.append(f"[{start:.1f}-{end:.1f}] {text}")
    body = "\n".join(lines)
    if max_chars is not None and len(body) > max_chars:
        return body[:max_chars]
    return body


def chunk_transcript_lines(transcript: dict, *, max_chars: int = 12000) -> list[str]:
    """Split timestamped lines into chunks that fit the LLM context."""
    lines: list[str] = []
    for s in transcript.get("sentences") or []:
        text = (s.get("text") or "").strip()
        if not text:
            continue
        lines.append(f"[{float(s['start']):.1f}-{float(s['end']):.1f}] {text}")
    if not lines:
        return []
    chunks: list[str] = []
    buf: list[str] = []
    size = 0
    for line in lines:
        add = len(line) + 1
        if buf and size + add > max_chars:
            chunks.append("\n".join(buf))
            buf = [line]
            size = add
        else:
            buf.append(line)
            size += add
    if buf:
        chunks.append("\n".join(buf))
    return chunks


def _audio_for_stt(audio_path: Path) -> Path:
    """Parakeet/ffmpeg need a recognizable extension; we store files as .bin."""
    audio_path = Path(audio_path)
    if audio_path.suffix.lower() in {".mp3", ".m4a", ".wav", ".flac", ".ogg"}:
        return audio_path
    link = audio_path.with_suffix(".mp3")
    try:
        if link.is_symlink() or link.exists():
            if link.resolve() == audio_path.resolve():
                return link
            link.unlink()
        link.symlink_to(audio_path.name)
    except OSError:
        return audio_path
    return link


def _resolve_tool(ref: str) -> Path | None:
    """Resolve the transcription interpreter: full/relative path first,
    then PATH lookup for bare names like `python`. None if unresolvable."""
    ref = (ref or "").strip()
    if not ref:
        return None
    p = Path(ref)
    if p.exists():
        return p
    import shutil

    found = shutil.which(ref)
    return Path(found) if found else None


def backend_status() -> dict:
    """Self-check for the transcription pipeline. Cheap: no model load."""
    from . import db

    tool_ref = db.runtime_str("stt_python")
    script = Path(db.runtime_str("stt_sidecar"))
    tool = _resolve_tool(tool_ref)
    import shutil

    ffmpeg = shutil.which("ffmpeg")
    try:
        __import__("faster_whisper")
        faster_whisper = True
    except ImportError:
        faster_whisper = False
    ok = tool is not None and script.exists() and ffmpeg is not None
    return {
        "ok": ok,
        "tool_ref": tool_ref,
        "tool_resolved": str(tool) if tool else None,
        "script": str(script),
        "script_exists": script.exists(),
        "ffmpeg": ffmpeg,
        "faster_whisper_installed": faster_whisper,
        "model": db.runtime_str("stt_model"),
    }


def transcribe_audio(audio_path: Path, dest: Path, *, force: bool = False) -> dict:
    """Run the configured STT sidecar (or return cached transcript)."""
    if dest.exists() and not force:
        cached = load_transcript(dest)
        if cached and cached.get("sentences") is not None:
            logger.info("using cached transcript %s", dest)
            return cached

    from . import db

    python = _resolve_tool(db.runtime_str("stt_python"))
    script = Path(db.runtime_str("stt_sidecar"))
    model = db.runtime_str("stt_model")
    if python is None:
        raise RuntimeError(
            "Transcription tool not found. "
            "Pick a transcription backend on the settings page (Server)."
        )
    if not script.exists():
        raise RuntimeError(f"Transcription script missing: {script}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    stt_audio = _audio_for_stt(audio_path)
    cmd = [str(python), str(script), str(stt_audio), str(dest), model]
    logger.info("STT sidecar: %s", " ".join(cmd))
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"STT failed ({proc.returncode}): {proc.stderr[-2000:] or proc.stdout[-2000:]}"
        )
    if proc.stdout:
        logger.info("STT: %s", proc.stdout.strip()[-500:])
    data = load_transcript(dest)
    if not data:
        raise RuntimeError(f"STT produced no transcript at {dest}")
    return data
