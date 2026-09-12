from __future__ import annotations

from unittest.mock import patch

import numpy as np

from podaddeduct.intervals import Interval
from podaddeduct.seed import (
    _extract_json_array,
    filter_min_duration,
    heuristic_ads,
    leftover_transcript,
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
    assert "[00:00:04 - 00:00:22]" in text
    assert "Acme Insurance" in text
    chunks = chunk_transcript_lines(FIXTURE_TRANSCRIPT, max_chars=80)
    assert len(chunks) >= 2
    assert all(isinstance(c, str) and c for c in chunks)


def test_extract_json_array_variants():
    assert _extract_json_array('[{"start":1,"end":2}]') == [{"start": 1.0, "end": 2.0}]
    fenced = '```json\n{"ads": [{"start": "00:00:10", "end": "00:00:20"}]}\n```'
    assert _extract_json_array(fenced)[0]["start"] == 10.0
    noisy = 'Here you go:\n[{"start":5,"end":9}]\nThanks'
    assert _extract_json_array(noisy)[0]["end"] == 9.0
    empty = '{"ads": []}'
    assert _extract_json_array(empty) == []


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


def test_heuristic_ads_finds_sponsor_reads():
    ads = heuristic_ads(FIXTURE_TRANSCRIPT)
    assert len(ads) >= 1
    assert ads[0].start <= 4.5
    assert ads[0].end >= 29.0  # merged with "use code" sentence


def test_leftover_transcript_drops_covered():
    covered = [Interval(4.0, 30.0)]
    left = leftover_transcript(FIXTURE_TRANSCRIPT, covered)
    texts = [s["text"] for s in left["sentences"]]
    assert "Welcome to the show." in texts
    assert "Alright let's talk Braves baseball." in texts
    assert not any("brought to you" in t for t in texts)


def test_find_ads_with_gemini_mocked():
    from podaddeduct import seed as seed_mod

    seen_bodies = []

    def fake_oc(user_content, model=None, **kwargs):
        seen_bodies.append(user_content)
        return '{"ads": [{"start": "00:06:40", "end": "00:07:25", "type": "inserted_ad", "sponsor": "unknown", "confidence": 0.9}]}'

    with patch.object(seed_mod, "opencode_generate", side_effect=fake_oc):
        result = seed_mod.find_ads_with_zen(FIXTURE_TRANSCRIPT)
    ads = result.ranges
    # Full transcript (including heuristic-covered sponsor lines) goes to the model.
    assert seen_bodies and "brought to you" in seen_bodies[0]
    assert "car commercial" in seen_bodies[0]
    # Heuristic covers the sponsor block; model covers midroll.
    assert len(ads) >= 2
    assert ads[0].start <= 4.5
    assert any(a.start >= 390 for a in ads)
    assert result.gemini_ok


def test_pad_and_clamp_ads_pre_and_postroll():
    from podaddeduct.seed import pad_and_clamp_ads

    # Early ad snaps to 0; padding extends the end.
    early = pad_and_clamp_ads([Interval(5.0, 20.0)], duration=600.0)
    assert len(early) == 1
    assert early[0].start == 0.0
    assert early[0].end >= 20.5

    # Late ad clamps through end of episode.
    late = pad_and_clamp_ads([Interval(560.0, 580.0)], duration=600.0)
    assert len(late) == 1
    assert late[0].end == 600.0
    assert late[0].start <= 559.8

    # Short fixtures must not wipe the whole clip via post-roll clamp.
    short = pad_and_clamp_ads([Interval(2.5, 7.5)], duration=10.0)
    assert len(short) == 1
    assert short[0].end < 10.0 or short[0].start == 0.0
    assert short[0].duration < 10.0


def test_heuristic_ads_finds_sports_cues():
    transcript = {
        "sentences": [
            {"text": "Welcome back.", "start": 0.0, "end": 3.0},
            {"text": "This episode is presented by FanDuel.", "start": 3.0, "end": 12.0},
            {"text": "If you or someone you know has a gambling problem call 1-800-GAMBLER.", "start": 12.0, "end": 25.0},
            {"text": "Back to the Celtics.", "start": 25.0, "end": 30.0},
        ]
    }
    ads = heuristic_ads(transcript)
    assert len(ads) >= 1
    assert ads[0].start <= 3.5
    assert ads[0].end >= 24.0  # merged across the FanDuel + helpline block


def test_filter_rejects_micro_cuts():
    from podaddeduct.seed import MIN_AD_CUT_SECONDS, filter_min_duration

    ranges = [
        Interval(0.0, 6.0),
        Interval(100.0, 104.0),
        Interval(200.0, 260.0),
    ]
    kept = filter_min_duration(ranges, min_seconds=MIN_AD_CUT_SECONDS)
    assert len(kept) == 1
    assert kept[0].start == 200.0


def test_find_ads_progress_callback():
    from podaddeduct import seed as seed_mod

    seen = []

    def fake_oc(user_content, model=None, **kwargs):
        return '{"ads": []}'

    # Transcript with no heuristic hits so the model runs once.
    plain = {
        "sentences": [
            {"text": "Baseball talk one.", "start": 0.0, "end": 10.0},
            {"text": "Baseball talk two longer filler text here.", "start": 10.0, "end": 20.0},
            {"text": "Baseball talk three even more filler.", "start": 20.0, "end": 30.0},
        ]
    }

    with patch.object(seed_mod, "opencode_generate", side_effect=fake_oc):
        seed_mod.find_ads_with_zen(plain, progress_cb=lambda d, t: seen.append((d, t)))
    assert seen, "callback never fired"
    assert seen[-1][0] == seen[-1][1]


def test_default_gemini_model():
    from podaddeduct.config import Settings

    default = Settings.model_fields["gemini_model"].default
    assert default == "opencode/nemotron-3-ultra-free"
