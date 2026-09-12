from __future__ import annotations

from fastapi.testclient import TestClient

from podaddeduct import db
from podaddeduct.config import settings
from podaddeduct.ptranscript import (
    build_sentences,
    coverage,
    cues_to_sentences,
    parse_cues,
    parse_json_transcript,
    parse_transcript_tags,
    pick_transcript,
    validate,
)

RAW_FEED = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:podcast="https://podcastindex.org/namespace/1.0">
<channel><title>T</title>
<item><title>E1</title><guid isPermaLink="false">g1</guid>
<enclosure url="https://example.com/e1.mp3" length="1" type="audio/mpeg"/>
<podcast:transcript url="https://example.com/e1.txt" type="text/plain"/>
<podcast:transcript url="https://example.com/e1.vtt" type="text/vtt"/>
</item>
<item><title>E2</title><guid isPermaLink="false">g2</guid>
<enclosure url="https://example.com/e2.mp3" length="1" type="audio/mpeg"/>
<podcast:transcript url="https://example.com/e2.json" type="application/json" language="es"/>
<podcast:transcript url="https://example.com/e2.srt" type="application/x-subrip"/>
</item>
<item><title>No transcripts here</title><guid isPermaLink="false">g3</guid>
<enclosure url="https://example.com/e3.mp3" length="1" type="audio/mpeg"/>
</item>
</channel></rss>"""

VTT = """WEBVTT

00:00.000 --> 00:02.500
Hello world this is a test

NOTE a comment

00:02.500 --> 00:05.000
of the <b>emergency</b> broadcast system.
"""

SRT = """1
00:00:01,000 --> 00:00:03,500
First line here today

2
00:00:03,500 --> 00:00:06,000
Second line is done.
"""


def _setup(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    (tmp_path / "audio").mkdir(exist_ok=True)
    db.init_db()
    return tmp_path


# --- tag extraction / picking ---

def test_parse_transcript_tags_keyed_by_guid_and_enclosure():
    m = parse_transcript_tags(RAW_FEED)
    assert set(l["url"] for l in m["g1"]) == {"https://example.com/e1.txt", "https://example.com/e1.vtt"}
    # enclosure URL works as a key too
    assert m["https://example.com/e2.mp3"]
    assert "g3" not in m


def test_parse_transcript_tags_broken_xml():
    assert parse_transcript_tags(b"not xml at all {{{") == {}


def test_pick_prefers_timestamped_and_english():
    links = [
        {"url": "a.txt", "type": "text/plain", "language": ""},
        {"url": "b.srt", "type": "application/x-subrip", "language": ""},
        {"url": "c.vtt", "type": "text/vtt", "language": ""},
    ]
    assert pick_transcript(links)["url"] == "c.vtt"
    assert pick_transcript([links[0]]) is None
    assert pick_transcript([]) is None
    es_json = {"url": "e.json", "type": "application/json", "language": "es"}
    en_vtt = {"url": "e.vtt", "type": "text/vtt", "language": ""}
    # json outranks vtt even in spanish (type first), english wins ties
    assert pick_transcript([en_vtt, es_json])["url"] == "e.json"


# --- cue / json parsing ---

def test_parse_vtt_skips_header_notes_and_tags():
    cues = parse_cues(VTT)
    assert [(c["start"], c["end"]) for c in cues] == [(0.0, 2.5), (2.5, 5.0)]
    assert cues[1]["text"] == "of the emergency broadcast system."


def test_parse_srt_comma_decimals():
    cues = parse_cues(SRT)
    assert [(c["start"], c["end"]) for c in cues] == [(1.0, 3.5), (3.5, 6.0)]
    assert cues[0]["text"] == "First line here today"


def test_parse_json_shapes():
    a = parse_json_transcript({"segments": [{"start": 0, "end": 2, "text": "hi"}]})
    assert a == [{"start": 0.0, "end": 2.0, "text": "hi"}]
    b = parse_json_transcript({"lines": [{"startTime": 1, "endTime": 2.5, "body": "yo"}]})
    assert b == [{"start": 1.0, "end": 2.5, "text": "yo"}]
    assert parse_json_transcript({"segments": "nope"}) == []
    assert parse_json_transcript([{"start": 0, "end": 1, "text": "x"}]) == [
        {"start": 0.0, "end": 1.0, "text": "x"}]
    assert parse_json_transcript({"other": 1}) == []


def test_cues_to_sentences_merges_short_cues():
    cues = [
        {"start": 0.0, "end": 1.0, "text": "Hello"},
        {"start": 1.0, "end": 2.0, "text": "world."},
        {"start": 2.0, "end": 3.0, "text": "Next one here."},
    ]
    out = cues_to_sentences(cues)
    assert [(s["start"], s["end"]) for s in out] == [(0.0, 2.0), (2.0, 3.0)]


def test_build_sentences_sniffs_json_and_cues():
    assert build_sentences("", "application/json", b'{"segments": []}') == []
    out = build_sentences("", "", b'{"segments": [{"start": 0, "end": 60, "text": "Hello world."}]}')
    assert len(out) == 1 and out[0]["text"] == "Hello world."
    out = build_sentences("text/html", "text/html", b"<p>no timestamps here</p>")
    assert out == []


# --- validation ---

def test_validate_accepts_covering_transcript():
    spans = [{"start": 0.0, "end": 1800.0, "text": "a"}]
    assert validate(spans, 3600.0) is False  # only 50%... boundary
    spans = [{"start": 0.0, "end": 2000.0, "text": "a"}]
    assert validate(spans, 3600.0) is False  # last end too early
    spans = [{"start": 0.0, "end": 3500.0, "text": "a"}]
    assert validate(spans, 3600.0) is True


def test_validate_rejects_garbage():
    assert validate([], 100.0) is False
    assert validate([{"start": 0, "end": 10, "text": "x"}], 0) is False
    assert validate([{"start": 0, "end": 5, "text": "x"}], 3600.0) is False


def test_coverage_union_not_sum():
    spans = [{"start": 0, "end": 60, "text": "a"}, {"start": 30, "end": 90, "text": "b"}]
    assert abs(coverage(spans, 180.0) - 0.5) < 1e-9


# --- db + secrets + routes ---

def test_transcript_columns_persist(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    feed = db.create_feed(slug="t1", upstream_url="https://example.com/rss", title="T")
    ep = db.upsert_episode(feed.id, guid="g1", title="E1", enclosure_url="https://e/1.mp3",
                           pub_date=None, transcript_url="https://e/1.vtt",
                           transcript_type="text/vtt")
    got = db.get_episode(ep.id)
    assert got.transcript_url == "https://e/1.vtt"
    assert got.transcript_type == "text/vtt"
    eps = db.upsert_episodes_batch(feed.id, [
        {"guid": "g1", "title": "E1", "enclosure_url": "https://e/1.mp3", "pub_date": None,
         "transcript_url": "https://e/1.vtt", "transcript_type": "text/vtt"},
        {"guid": "g2", "title": "E2", "enclosure_url": "https://e/2.mp3", "pub_date": None},
    ])
    by_guid = {e.guid: e for e in eps}
    assert by_guid["g1"].transcript_url == "https://e/1.vtt"
    assert by_guid["g2"].transcript_url is None


def test_groq_secrets_roundtrip(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    import os

    from podaddeduct.secrets import get_groq_api_key, groq_key_status, set_groq_api_key

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert get_groq_api_key() is None
    assert groq_key_status() == {"configured": False, "source": None}
    set_groq_api_key("gsk-test-123")
    assert get_groq_api_key() == "gsk-test-123"
    assert groq_key_status()["configured"] is True
    set_groq_api_key(None)
    assert get_groq_api_key() is None
    monkeypatch.setenv("GROQ_API_KEY", "gsk-from-env")
    assert get_groq_api_key() == "gsk-from-env"


def test_groq_key_route(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct.app import app
    from podaddeduct.secrets import get_groq_api_key

    with TestClient(app) as client:
        r = client.post("/settings/groq-key", data={"groq_api_key": "gsk-xyz"},
                        follow_redirects=False)
        assert r.status_code == 303
        assert r.headers["location"] == "/settings?saved=groq"
        assert get_groq_api_key() == "gsk-xyz"
        r = client.post("/settings/groq-key", data={"groq_api_key": "", "clear": "1"},
                        follow_redirects=False)
        assert get_groq_api_key() is None


def test_process_module_exposes_transcribe_audio():
    # Regression: a refactor once dropped this import → NameError at runtime.
    # Extended after transcript_path_for suffered the same fate.
    from podaddeduct import process

    assert callable(process.transcribe_audio)
    assert callable(process.transcript_path_for)
