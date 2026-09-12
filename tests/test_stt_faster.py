from __future__ import annotations

import importlib.util
from pathlib import Path


def _load():
    spec = importlib.util.spec_from_file_location(
        "stt_faster", Path(__file__).parent.parent / "scripts" / "stt_faster.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_words_to_sentences_splits_on_punctuation():
    mod = _load()
    words = [
        {"start": 0.0, "end": 0.4, "word": "Hello"},
        {"start": 0.5, "end": 0.9, "word": "world."},
        {"start": 1.0, "end": 1.4, "word": "Buy"},
        {"start": 1.5, "end": 1.9, "word": "stuff!"},
        {"start": 2.0, "end": 2.5, "word": "bye"},
    ]
    out = mod.words_to_sentences(words)
    # "Hello world." is only 2 words, so the >=3-word guard merges it forward.
    assert [(s["start"], s["end"]) for s in out] == [(0.0, 1.9), (2.0, 2.5)]
    assert out[0]["text"] == "Hello world. Buy stuff!"
    assert out[1]["text"] == "bye"


def test_words_to_sentences_skips_empties():
    mod = _load()
    assert mod.words_to_sentences([]) == []
    out = mod.words_to_sentences([{"start": 0.0, "end": 0.5, "word": "  "}])
    assert out == []
