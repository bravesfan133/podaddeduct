from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from podaddeduct.cut import build_atrim_filter, build_ffmpeg_cmd, cut_ads
from podaddeduct.intervals import Interval


def test_build_atrim_filter_single():
    filt = build_atrim_filter([Interval(10.0, 20.0)])
    assert "atrim=start=10.000:end=20.000" in filt
    assert "[outa]" in filt


def test_build_atrim_filter_multi():
    filt = build_atrim_filter([Interval(0, 5), Interval(10, 15), Interval(20, 30)])
    assert "concat=n=3:v=0:a=1[outa]" in filt
    assert filt.count("atrim=") == 3


def test_build_ffmpeg_cmd_shape():
    cmd = build_ffmpeg_cmd(
        "/usr/bin/ffmpeg",
        Path("in.bin"),
        Path("out.clean.mp3"),
        [Interval(0, 10), Interval(20, 30)],
    )
    assert cmd[0] == "/usr/bin/ffmpeg"
    assert "-threads" in cmd and "1" in cmd
    assert "-filter_complex" in cmd
    assert "-c:a" in cmd and "libmp3lame" in cmd
    assert "-map_metadata" in cmd
    assert "-id3v2_version" in cmd
    assert str(Path("out.clean.mp3")) in cmd


def test_cut_ads_mocked(tmp_path: Path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"fake")
    dest = tmp_path / "out.clean.mp3"

    def fake_run(cmd, capture_output, text, check):
        # last arg is output path (tmp partial)
        out = Path(cmd[-1])
        out.write_bytes(b"mp3data")

        class R:
            returncode = 0
            stderr = ""

        return R()

    with (
        patch("podaddeduct.cut.find_ffmpeg", return_value="ffmpeg"),
        patch("podaddeduct.cut.subprocess.run", side_effect=fake_run),
    ):
        path = cut_ads(src, [Interval(5, 10)], dest, duration=30.0)
    assert path == dest
    assert dest.read_bytes() == b"mp3data"


def test_cut_ads_rejects_empty_content(tmp_path: Path):
    src = tmp_path / "full.bin"
    src.write_bytes(b"x")
    with pytest.raises(RuntimeError, match="entire episode"):
        cut_ads(src, [Interval(0, 100)], tmp_path / "y.mp3", duration=100.0)
