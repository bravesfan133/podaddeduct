from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from .config import settings
from .intervals import Interval, invert_ranges, merge_intervals

logger = logging.getLogger("podaddeduct.cut")


def clean_path_for(episode_id: int) -> Path:
    return settings.audio_dir / f"{episode_id}.clean.mp3"


def find_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install with: brew install ffmpeg"
        )
    return path


def build_atrim_filter(content: list[Interval]) -> str:
    """Build ffmpeg -filter_complex graph that keeps only content intervals."""
    if not content:
        raise ValueError("no content intervals to keep")
    parts: list[str] = []
    labels: list[str] = []
    for i, span in enumerate(content):
        label = f"a{i}"
        # atrim end is exclusive-ish; use end time directly
        parts.append(
            f"[0:a]atrim=start={span.start:.3f}:end={span.end:.3f},asetpts=PTS-STARTPTS[{label}]"
        )
        labels.append(f"[{label}]")
    n = len(content)
    if n == 1:
        parts.append(f"{labels[0]}anull[outa]")
    else:
        parts.append(f"{''.join(labels)}concat=n={n}:v=0:a=1[outa]")
    return ";".join(parts)


def build_ffmpeg_cmd(
    ffmpeg: str,
    src: Path,
    dest: Path,
    content: list[Interval],
    *,
    bitrate: str = "128k",
) -> list[str]:
    filt = build_atrim_filter(content)
    return [
        ffmpeg,
        "-y",
        "-i",
        str(src),
        "-filter_complex",
        filt,
        "-map",
        "[outa]",
        "-c:a",
        "libmp3lame",
        "-b:a",
        bitrate,
        str(dest),
    ]


def cut_ads(
    src: Path,
    ads: list[Interval],
    dest: Path,
    *,
    duration: float,
) -> Path:
    """
    Write cleaned MP3 with ad ranges removed.
    `duration` is the source timeline length in seconds.
    """
    src = Path(src)
    dest = Path(dest)
    if not src.exists():
        raise FileNotFoundError(src)
    ads = merge_intervals([a for a in ads if a.end > a.start], gap=0.25)
    content = invert_ranges(ads, duration)
    if not content:
        raise RuntimeError("cutting would remove entire episode")

    ffmpeg = find_ffmpeg()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.stem + ".partial" + dest.suffix)
    cmd = build_ffmpeg_cmd(ffmpeg, src, tmp, content)
    logger.info("ffmpeg cut %s -> %s (%d content spans)", src.name, dest.name, len(content))
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}): {(proc.stderr or '')[-2000:]}")
    tmp.replace(dest)
    return dest
