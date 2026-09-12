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


def format_hms(seconds: float) -> str:
    """Format seconds as HH:MM:SS (floor to whole seconds)."""
    total = max(0, int(float(seconds)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_timestamped_transcript(transcript: dict, *, max_chars: int | None = None) -> str:
    """One sentence per line: [HH:MM:SS - HH:MM:SS] text"""
    lines: list[str] = []
    for s in transcript.get("sentences") or []:
        text = (s.get("text") or "").strip()
        if not text:
            continue
        start = float(s["start"])
        end = float(s["end"])
        lines.append(f"[{format_hms(start)} - {format_hms(end)}] {text}")
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
        start = float(s["start"])
        end = float(s["end"])
        lines.append(f"[{format_hms(start)} - {format_hms(end)}] {text}")
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
    ok = tool is not None and script.exists() and ffmpeg is not None
    return {
        "ok": ok,
        "tool_ref": tool_ref,
        "tool_resolved": str(tool) if tool else None,
        "script": str(script),
        "script_exists": script.exists(),
        "ffmpeg": ffmpeg,
        "faster_whisper_installed": False,
        "model": db.runtime_str("stt_model"),
    }


def transcribe_audio(
    audio_path: Path,
    dest: Path,
    *,
    force: bool = False,
    progress_cb=None,
) -> dict:
    """Run the configured STT sidecar (or return cached transcript).

    progress_cb(stage: str, detail: str = "") is called for each `STAGE …`
    line the sidecar prints (e.g. encoding / waiting_groq / done).
    """
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
    # Sidecars needing cloud keys (stt_groq.py) get them via env, resolved
    # the same way as the UI: secrets.json first, then process env.
    import os

    from .secrets import get_groq_api_key

    env = dict(os.environ)
    groq_key = get_groq_api_key()
    if groq_key and "GROQ_API_KEY" not in env:
        env["GROQ_API_KEY"] = groq_key
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        bufsize=1,
    )
    out_lines: list[str] = []
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.rstrip("\n")
        out_lines.append(line)
        if not line.startswith("STAGE "):
            continue
        parts = line.split()
        # STAGE encoding 1/3  |  STAGE waiting_groq 1/3  |  STAGE done
        if len(parts) < 2:
            continue
        stage = parts[1]
        detail = parts[2] if len(parts) > 2 else ""
        if progress_cb is not None:
            try:
                progress_cb(stage, detail)
            except Exception:
                logger.exception("STT progress_cb failed")
    rc = proc.wait()
    combined = "\n".join(out_lines)
    if rc != 0:
        raise RuntimeError(f"STT failed ({rc}): {combined[-2000:]}")
    if combined:
        logger.info("STT: %s", combined.strip()[-500:])
    data = load_transcript(dest)
    if not data:
        raise RuntimeError(f"STT produced no transcript at {dest}")
    return data
