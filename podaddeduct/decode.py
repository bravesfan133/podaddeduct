from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np

logger = logging.getLogger("podaddeduct.decode")


def _decode_via_miniaudio(path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    import miniaudio

    decoded = miniaudio.decode_file(str(path), nchannels=1, sample_rate=target_sr)
    samples = np.frombuffer(decoded.samples, dtype=np.int16).astype(np.float32)
    if decoded.nchannels > 1:
        samples = samples.reshape(-1, decoded.nchannels).mean(axis=1)
    peak = float(np.max(np.abs(samples)) or 1.0)
    samples = samples / peak
    return samples, int(decoded.sample_rate)


def _decode_via_ffmpeg(path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    """Decode any container (m4a/AAC/etc.) to mono float32 PCM via ffmpeg."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found on PATH (needed to decode this audio format)")
    cmd = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        str(path),
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
