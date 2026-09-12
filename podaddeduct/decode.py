from __future__ import annotations

from pathlib import Path

import miniaudio
import numpy as np


def decode_audio_file(path: str | Path, target_sr: int = 11025) -> tuple[np.ndarray, int]:
    """Decode audio to mono float32 PCM via miniaudio (no ffmpeg)."""
    path = Path(path)
    decoded = miniaudio.decode_file(str(path), nchannels=1, sample_rate=target_sr)
    samples = np.frombuffer(decoded.samples, dtype=np.int16).astype(np.float32)
    if decoded.nchannels > 1:
        samples = samples.reshape(-1, decoded.nchannels).mean(axis=1)
    peak = np.max(np.abs(samples)) or 1.0
    samples = samples / peak
    return samples, int(decoded.sample_rate)


def load_mono_pcm(path: str | Path, target_sr: int = 11025) -> tuple[np.ndarray, int]:
    return decode_audio_file(path, target_sr=target_sr)
