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
    # -map_metadata 0 keeps ID3 tags (title/artist) from the source.
    # Cover art is often a video/attached_pic stream; map it if present.
    # Cap threads so N100 E-cores aren't saturated during re-encode.
    return [
        ffmpeg,
        "-y",
        "-threads",
        "1",
        "-i",
        str(src),
        "-filter_complex",
        filt,
        "-map",
        "[outa]",
        "-map",
        "0:v?",
        "-c:a",
        "libmp3lame",
        "-b:a",
        bitrate,
        "-c:v",
        "copy",
        "-map_metadata",
        "0",
        "-id3v2_version",
        "3",
        str(dest),
    ]


def _probe_audio_codec(path: Path) -> str:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return ""
    proc = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return ((getattr(proc, "stdout", None) or "")).strip().lower()


def _try_stream_copy_concat(
    ffmpeg: str,
    src: Path,
    dest: Path,
    content: list[Interval],
) -> bool:
    """Prefer -c copy concat when source audio is already MP3 (no lame)."""
    if _probe_audio_codec(src) not in {"mp3", "mp3float"}:
        return False
    import tempfile

    try:
        with tempfile.TemporaryDirectory(prefix="podcut_") as td:
            tdir = Path(td)
            list_path = tdir / "concat.txt"
            lines: list[str] = []
            for i, span in enumerate(content):
                part = tdir / f"part{i}.mp3"
                dur = max(0.01, span.end - span.start)
                cmd = [
                    ffmpeg,
                    "-y",
                    "-threads",
                    "1",
                    "-ss",
                    f"{span.start:.3f}",
                    "-t",
                    f"{dur:.3f}",
                    "-i",
                    str(src),
                    "-c",
                    "copy",
                    "-map",
                    "0:a:0",
                    str(part),
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
                if proc.returncode != 0 or not part.exists() or part.stat().st_size < 64:
                    return False
                esc = str(part).replace("'", r"'\''")
                lines.append(f"file '{esc}'")
            list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            out_tmp = dest.with_name(dest.stem + ".partial" + dest.suffix)
            concat_cmd = [
                ffmpeg,
                "-y",
                "-threads",
                "1",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_path),
                "-c",
                "copy",
                "-id3v2_version",
                "3",
                str(out_tmp),
            ]
            proc = subprocess.run(concat_cmd, capture_output=True, text=True, check=False)
            if proc.returncode != 0 or not out_tmp.exists():
                try:
                    out_tmp.unlink(missing_ok=True)
                except OSError:
                    pass
                return False
            out_tmp.replace(dest)
            return True
    except OSError:
        return False


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
    if _try_stream_copy_concat(ffmpeg, src, dest, content):
        logger.info("ffmpeg copy-concat %s -> %s (%d spans)", src.name, dest.name, len(content))
        return dest
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
