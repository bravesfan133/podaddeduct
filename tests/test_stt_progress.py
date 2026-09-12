from __future__ import annotations

import json
from pathlib import Path

from podaddeduct import db
from podaddeduct.config import settings
from podaddeduct import stt as stt_mod


def _setup(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    (tmp_path / "audio").mkdir(exist_ok=True)
    db.init_db()
    db.set_global_settings(
        {
            "stt_python": "python3",
            "stt_sidecar": str(tmp_path / "fake_stt.py"),
            "stt_model": "whisper-large-v3-turbo",
        }
    )
    return tmp_path


def test_transcribe_audio_streams_stage_lines(tmp_path, monkeypatch):
    """Queue updates mid-run: STAGE lines fire progress_cb before sidecar exits."""
    _setup(tmp_path, monkeypatch)
    script = tmp_path / "fake_stt.py"
    script.write_text(
        "\n".join(
            [
                "import sys, time, json",
                "from pathlib import Path",
                "print('STAGE encoding 1/2', flush=True)",
                "time.sleep(0.05)",
                "print('STAGE waiting_groq 1/2', flush=True)",
                "time.sleep(0.05)",
                "print('STAGE done', flush=True)",
                "dest = Path(sys.argv[2])",
                "dest.write_text(json.dumps({'sentences': [{'start': 0, 'end': 1, 'text': 'hi'}]}))",
            ]
        ),
        encoding="utf-8",
    )
    audio = tmp_path / "audio" / "ep.mp3"
    audio.write_bytes(b"fake")
    dest = tmp_path / "t.json"
    seen: list[tuple[str, str]] = []

    def cb(stage, detail=""):
        seen.append((stage, detail))

    data = stt_mod.transcribe_audio(audio, dest, force=True, progress_cb=cb)
    assert data["sentences"][0]["text"] == "hi"
    assert ("encoding", "1/2") in seen
    assert ("waiting_groq", "1/2") in seen
    assert ("done", "") in seen
    # encoding must arrive before waiting_groq (live updates, not after exit)
    assert seen.index(("encoding", "1/2")) < seen.index(("waiting_groq", "1/2"))
