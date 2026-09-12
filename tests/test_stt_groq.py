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


def test_main_skips_ffmpeg_when_under_24mb(tmp_path, monkeypatch, capsys):
    """Files under the upload cap POST as-is — no ffmpeg split."""
    mod = _load()
    audio = tmp_path / "ep.mp3"
    audio.write_bytes(b"x" * 1024)
    dest = tmp_path / "out.json"
    calls = {"ffmpeg": 0, "posts": 0}

    def fake_split(*_a, **_k):
        calls["ffmpeg"] += 1
        raise AssertionError("split_chunk must not run for small files")

    monkeypatch.setattr(mod, "split_chunk", fake_split)
    monkeypatch.setattr(mod, "probe_duration", lambda *_: 600.0)

    def fake_transcribe(client, key, path, model):
        calls["posts"] += 1
        assert path == audio
        return [{"start": 0.0, "end": 1.0, "text": "hi"}]

    monkeypatch.setattr(mod, "transcribe_chunk", fake_transcribe)
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(mod.sys, "argv", ["stt_groq.py", str(audio), str(dest), "whisper-large-v3-turbo"])
    mod.main()
    out = capsys.readouterr().out
    assert calls["ffmpeg"] == 0
    assert calls["posts"] == 1
    assert "STAGE waiting_groq 1/1" in out
    assert "STAGE done" in out
    assert "no ffmpeg" in out
    data = __import__("json").loads(dest.read_text())
    assert data["sentences"][0]["text"] == "hi"


def test_main_encodes_then_waits_when_over_24mb(tmp_path, monkeypatch, capsys):
    mod = _load()
    audio = tmp_path / "big.mp3"
    audio.write_bytes(b"x" * (mod.UPLOAD_AS_IS_MAX_BYTES + 1))
    dest = tmp_path / "out.json"
    stages: list[str] = []
    real_stage = mod.stage

    def tracking_stage(name, detail=""):
        stages.append(f"{name} {detail}".strip())
        real_stage(name, detail)

    def fake_split(src, start, length, dest_path):
        dest_path.write_bytes(b"chunk")

    monkeypatch.setattr(mod, "stage", tracking_stage)
    monkeypatch.setattr(mod, "split_chunk", fake_split)
    monkeypatch.setattr(mod, "probe_duration", lambda *_: 100.0)
    monkeypatch.setattr(mod, "plan_chunks", lambda *_a, **_k: [(0.0, 100.0)])
    monkeypatch.setattr(
        mod,
        "transcribe_chunk",
        lambda *_a, **_k: [{"start": 0.0, "end": 2.0, "text": "yo"}],
    )
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(mod.sys, "argv", ["stt_groq.py", str(audio), str(dest)])
    mod.main()
    assert stages[0].startswith("encoding")
    assert any(s.startswith("waiting_groq") for s in stages)
    assert stages[-1] == "done"
    assert "STAGE encoding 1/1" in capsys.readouterr().out


def test_split_chunk_uses_threads_1(tmp_path, monkeypatch):
    mod = _load()
    seen: list[list[str]] = []

    def fake_run(cmd, **_k):
        seen.append(list(cmd))
        dest = Path(cmd[-1])
        dest.write_bytes(b"ok")

        class R:
            returncode = 0
            stderr = ""

        return R()

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    dest = tmp_path / "c.mp3"
    mod.split_chunk(tmp_path / "src.mp3", 0.0, 30.0, dest)
    assert "-threads" in seen[0]
    assert seen[0][seen[0].index("-threads") + 1] == "1"
