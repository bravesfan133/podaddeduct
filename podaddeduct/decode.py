from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np

logger = logging.getLogger("podaddeduct.decode")


def find_ffprobe() -> str | None:
    return shutil.which("ffprobe")


def find_ffmpeg() -> str | None:
    return shutil.which("ffmpeg")


def probe_duration(path: str | Path) -> float:
    """Return media duration in seconds via ffprobe (no full decode)."""
    path = Path(path)
    ffprobe = find_ffprobe()
    if not ffprobe or not path.exists():
        return 0.0
    proc = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return max(0.0, float((proc.stdout or "").strip()))
    except ValueError:
        return 0.0


def vaapi_available() -> bool:
    """True when /dev/dri render node + vainfo look usable (N100 iGPU passthrough)."""
    import os

    render = Path("/dev/dri/renderD128")
    if not render.exists():
        return False
    vainfo = shutil.which("vainfo")
    if not vainfo:
        return False
    env = dict(os.environ)
    env.setdefault("LIBVA_DRIVER_NAME", "iHD")
    proc = subprocess.run(
        [vainfo, "--display", "drm", "--device", str(render)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0 and ("VAProfile" in out or "iHD" in out or "Intel" in out)


def _decode_via_miniaudio(path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    import miniaudio

    decoded = miniaudio.decode_file(str(path), nchannels=1, sample_rate=target_sr)
    samples = np.frombuffer(decoded.samples, dtype=np.int16).astype(np.float32)
    if decoded.nchannels > 1:
        samples = samples.reshape(-1, decoded.nchannels).mean(axis=1)
    peak = float(np.max(np.abs(samples)) or 1.0)
    samples = samples / peak
    return samples, int(decoded.sample_rate)


def _ffmpeg_pcm_cmd(path: Path, target_sr: int, *, start: float | None = None, length: float | None = None) -> list[str]:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found on PATH (needed to decode this audio format)")
    cmd = [ffmpeg, "-v", "error"]
    # VAAPI hwaccel for containers with video/cover; harmless no-op for pure audio
    # when the driver rejects it (ffmpeg falls back). Only attempt when available.
    if vaapi_available():
        cmd.extend(["-hwaccel", "vaapi", "-hwaccel_device", "/dev/dri/renderD128"])
    if start is not None and start > 0:
        cmd.extend(["-ss", f"{start:.3f}"])
    cmd.extend(["-i", str(path)])
    if length is not None and length > 0:
        cmd.extend(["-t", f"{length:.3f}"])
    cmd.extend(
        [
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ac",
            "1",
            "-ar",
            str(target_sr),
            "-",
        ]
    )
    return cmd


def _decode_via_ffmpeg(
    path: Path,
    target_sr: int,
    *,
    start: float | None = None,
    length: float | None = None,
) -> tuple[np.ndarray, int]:
    """Decode (optional window) to mono float32 PCM via ffmpeg."""
    cmd = _ffmpeg_pcm_cmd(path, target_sr, start=start, length=length)
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0 or not proc.stdout:
        # Retry without VAAPI if hwaccel failed.
        if "-hwaccel" in cmd:
            cmd = [c for c in cmd if c not in ("-hwaccel", "vaapi", "-hwaccel_device", "/dev/dri/renderD128")]
            # Rebuild cleanly without hwaccel
            cmd = [find_ffmpeg() or "ffmpeg", "-v", "error"]
            if start is not None and start > 0:
                cmd.extend(["-ss", f"{start:.3f}"])
            cmd.extend(["-i", str(path)])
            if length is not None and length > 0:
                cmd.extend(["-t", f"{length:.3f}"])
            cmd.extend(
                ["-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(target_sr), "-"]
            )
            proc = subprocess.run(cmd, capture_output=True, check=False)
        if proc.returncode != 0 or not proc.stdout:
            err = (proc.stderr or b"").decode("utf-8", errors="replace")[-500:]
            raise RuntimeError(f"ffmpeg decode failed: {err}")
    samples = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    peak = float(np.max(np.abs(samples)) or 1.0)
    samples = samples / peak
    return samples, target_sr


def decode_audio_file(path: str | Path, target_sr: int = 11025) -> tuple[np.ndarray, int]:
    """Decode audio to mono float32 PCM. Tries miniaudio, then ffmpeg for m4a/AAC."""
    path = Path(path)
    try:
        return _decode_via_miniaudio(path, target_sr)
    except Exception as exc:
        logger.info("miniaudio decode failed for %s (%s); trying ffmpeg", path.name, exc)
        return _decode_via_ffmpeg(path, target_sr)


def load_mono_pcm(path: str | Path, target_sr: int = 11025) -> tuple[np.ndarray, int]:
    return decode_audio_file(path, target_sr=target_sr)


def load_pcm_window(
    path: str | Path,
    start: float,
    end: float,
    target_sr: int = 11025,
) -> tuple[np.ndarray, int]:
    """Decode only [start, end) seconds — for silence snap around ad edges."""
    path = Path(path)
    length = max(0.05, end - start)
    return _decode_via_ffmpeg(path, target_sr, start=max(0.0, start), length=length)
