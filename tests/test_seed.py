from __future__ import annotations

from unittest.mock import patch

import numpy as np

from podaddeduct.intervals import Interval
from podaddeduct.seed import (
    _extract_json_array,
    filter_min_duration,
    snap_to_silence,
    union_ranges,
)
from podaddeduct.stt import chunk_transcript_lines, format_timestamped_transcript


FIXTURE_TRANSCRIPT = {
    "model": "test",
    "text": "hello ads goodbye",
    "sentences": [
        {"text": "Welcome to the show.", "start": 0.0, "end": 4.0},
        {"text": "This episode is brought to you by Acme Insurance.", "start": 4.0, "end": 22.0},
        {"text": "Use code HAMMER for twenty percent off.", "start": 22.0, "end": 30.0},
        {"text": "Alright let's talk Braves baseball.", "start": 30.0, "end": 40.0},
        {"text": "Midroll ad music and car commercial.", "start": 400.0, "end": 445.0},
        {"text": "Back to the game.", "start": 445.0, "end": 455.0},
    ],
}


def test_format_and_chunk_transcript():
    text = format_timestamped_transcript(FIXTURE_TRANSCRIPT)
    assert "[4.0-22.0]" in text
    assert "Acme Insurance" in text
    chunks = chunk_transcript_lines(FIXTURE_TRANSCRIPT, max_chars=80)
    assert len(chunks) >= 2
    assert all(isinstance(c, str) and c for c in chunks)


def test_extract_json_array_variants():
    assert _extract_json_array('[{"start":1,"end":2}]') == [{"start": 1.0, "end": 2.0}]
    fenced = '```json\n[{"start": 10, "end": 20}]\n```'
    assert _extract_json_array(fenced)[0]["start"] == 10.0
    noisy = 'Here you go:\n[{"start":5,"end":9}]\nThanks'
    assert _extract_json_array(noisy)[0]["end"] == 9.0


def test_union_merge_and_filter():
    a = [Interval(0, 10), Interval(100, 120)]
    b = [Interval(8, 15), Interval(200, 205)]
    u = union_ranges(a, b)
    assert len(u) == 3
    assert u[0].start == 0 and u[0].end == 15
    kept = filter_min_duration(u, min_seconds=8)
    assert all(r.duration >= 8 for r in kept)


def test_snap_to_silence_moves_edges():
    sr = 11025
    t = np.arange(int(sr * 10), dtype=np.float32) / sr
    pcm = 0.4 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    pcm[int(sr * 1.8) : int(sr * 2.2)] *= 0.02
    pcm[int(sr * 7.8) : int(sr * 8.2)] *= 0.02
    ranges = [Interval(2.5, 7.5)]
    snapped = snap_to_silence(ranges, pcm, sr, window=1.0)
    assert len(snapped) == 1
    assert snapped[0].start < 2.5
    assert snapped[0].end > 7.5


def test_extract_response_text():
    from podaddeduct.seed import _extract_response_text

    assert _extract_response_text({"output_text": '[{"start":1,"end":2}]'}) == '[{"start":1,"end":2}]'
    nested = {
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": '[{"start":4,"end":30}]'}],
            }
        ]
    }
    assert _extract_response_text(nested) == '[{"start":4,"end":30}]'


def test_find_ads_with_zen_mocked():
    from podaddeduct import db as db_mod
    from podaddeduct import seed as seed_mod

    def fake_responses(api_key, model, user_content):
        if "Acme" in user_content or "brought to you" in user_content:
            return '[{"start": 4.0, "end": 30.0}]'
        if "Midroll" in user_content or "car commercial" in user_content:
            return '[{"start": 400.0, "end": 445.0}]'
        return "[]"

    real_runtime_int = db_mod.runtime_int

    def fake_runtime_int(key, **kwargs):
        if key == "zen_chunk_chars":
            return 120
        return real_runtime_int(key, **kwargs)

    with (
        patch.object(seed_mod, "resolve_zen_api_key", return_value="sk-test"),
        patch.object(seed_mod, "_zen_call", side_effect=fake_responses),
        patch.object(db_mod, "runtime_int", side_effect=fake_runtime_int),
    ):
        ads = seed_mod.find_ads_with_zen(FIXTURE_TRANSCRIPT)
    assert len(ads) >= 2
    assert ads[0].start <= 4.5
    assert any(a.start >= 390 for a in ads)
