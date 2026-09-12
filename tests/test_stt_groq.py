from __future__ import annotations

import importlib.util
from pathlib import Path

import httpx


def _load():
    spec = importlib.util.spec_from_file_location(
        "stt_groq", Path(__file__).parent.parent / "scripts" / "stt_groq.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_plan_chunks_single_short_file():
    mod = _load()
    assert mod.plan_chunks(0) == [(0.0, 0.0)]
    assert mod.plan_chunks(600) == [(0.0, 600)]


def test_plan_chunks_overlap_and_cover_duration():
    mod = _load()
    chunks = mod.plan_chunks(12600.0)  # 3.5h
    assert chunks[0][0] == 0.0
    assert len(chunks) > 1
    for (s0, l0), (s1, _l1) in zip(chunks, chunks[1:]):
        # consecutive chunks overlap by ~OVERLAP_S
        assert abs((s1 - s0) - (l0 - mod.OVERLAP_S)) < 1.0
    last_start, last_len = chunks[-1]
    assert abs(last_start + last_len - 12600.0) < 1.0


def test_merge_segments_offsets_and_drops_overlap_dupes():
    mod = _load()
    out = mod.merge_segments([
        (0.0, [{"start": 0, "end": 10, "text": "hello"},
               {"start": 10, "end": 20, "text": "world"}]),
        (15.0, [{"start": 2, "end": 8, "text": "WORLD"},   # dup of chunk 0
                {"start": 12, "end": 20, "text": "next bit"}]),
    ])
    assert [(s["start"], s["end"]) for s in out] == [(0, 10), (10, 20), (27, 35)]
    assert [s["text"] for s in out] == ["hello", "world", "next bit"]


def test_merge_segments_skips_empties_and_bad_ranges():
    mod = _load()
    out = mod.merge_segments([
        (0.0, [{"start": 5, "end": 5, "text": "zero"},
               {"start": 1, "end": 2, "text": "  "},
               {"start": 1, "end": 2, "text": "ok"}]),
    ])
    assert out == [{"start": 1.0, "end": 2.0, "text": "ok"}]


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_transcribe_chunk_success(tmp_path):
    mod = _load()
    p = tmp_path / "c.mp3"
    p.write_bytes(b"fake")
    client = _client(lambda request: httpx.Response(200, json={
        "segments": [{"start": 0.5, "end": 3.0, "text": " hi "},
                     {"nope": True},
                     {"start": 3.0, "end": 4.0, "text": ""}],
    }))
    segs = mod.transcribe_chunk(client, "k", p, "whisper-large-v3-turbo")
    assert segs == [{"start": 0.5, "end": 3.0, "text": "hi"}]


def test_transcribe_chunk_retries_then_succeeds(tmp_path, monkeypatch):
    mod = _load()
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    p = tmp_path / "c.mp3"
    p.write_bytes(b"fake")
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, text="slow down", headers={"retry-after": "1"})
        return httpx.Response(200, json={"segments": [{"start": 0, "end": 1, "text": "yo"}]})

    segs = mod.transcribe_chunk(_client(handler), "k", p, "m")
    assert segs == [{"start": 0, "end": 1, "text": "yo"}]
    assert calls["n"] == 2


def test_transcribe_chunk_fatal_error_exits(tmp_path):
    mod = _load()
    import pytest

    p = tmp_path / "c.mp3"
    p.write_bytes(b"fake")
    client = _client(lambda request: httpx.Response(401, text="bad key"))
    with pytest.raises(SystemExit):
        mod.transcribe_chunk(client, "k", p, "m")


def test_transcribe_chunk_gives_up_after_max_tries(tmp_path, monkeypatch):
    mod = _load()
    import pytest

    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    p = tmp_path / "c.mp3"
    p.write_bytes(b"fake")
    client = _client(lambda request: httpx.Response(503, text="down"))
    with pytest.raises(SystemExit):
        mod.transcribe_chunk(client, "k", p, "m")
