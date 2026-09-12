from __future__ import annotations

from unittest.mock import patch

import pytest

from podaddeduct import db
from podaddeduct.config import settings


def _setup(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    (tmp_path / "audio").mkdir(exist_ok=True)
    db.init_db()
    return tmp_path


class _FakeResp:
    def __init__(self, status_code=200, payload=None, text="", headers=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text
        self.headers = headers or {}
        self.request = None

    def json(self):
        return self._payload


class _FakeClient:
    """Stand-in for httpx.Client recording posts, scripted responses."""

    instances: list = []

    def __init__(self, script, *args, **kwargs):
        self.script = list(script)
        self.posts = []
        _FakeClient.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def post(self, url, **kwargs):
        self.posts.append((url, kwargs))
        item = self.script.pop(0) if self.script else _FakeResp(status_code=500, text="empty")
        if isinstance(item, Exception):
            raise item
        return item


def _ok_gemini(text):
    return _FakeResp(
        200,
        {"candidates": [{"content": {"parts": [{"text": text}]}}]},
    )


def test_gemini_generate_success(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct import seed as seed_mod

    _FakeClient.instances.clear()
    fake = _FakeClient([_ok_gemini('{"ads": [{"start": "00:00:01", "end": "00:00:02"}]}')])
    with patch.object(seed_mod.httpx, "Client", return_value=fake):
        out = seed_mod._gemini_generate("k", "gemini-3.5-flash", "hi")
    assert "ads" in out
    url, kwargs = fake.posts[0]
    assert "gemini-3.5-flash:generateContent" in url
    assert kwargs["headers"]["x-goog-api-key"] == "k"
    gen = kwargs["json"]["generationConfig"]
    assert gen["responseMimeType"] == "application/json"
    assert gen["maxOutputTokens"] == 8192
    assert gen["temperature"] == 1.0
    assert gen["thinkingConfig"] == {"thinkingBudget": 0}
    assert gen["responseSchema"]["type"] == "OBJECT"
    assert "ads" in gen["responseSchema"]["properties"]


def test_gemini_2x_payload_omits_thinking(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct import seed as seed_mod

    fake = _FakeClient([_ok_gemini('{"ads": []}')])
    with patch.object(seed_mod.httpx, "Client", return_value=fake):
        seed_mod._gemini_generate("k", "gemini-2.5-flash", "hi")
    gen = fake.posts[0][1]["json"]["generationConfig"]
    assert "thinkingConfig" not in gen
    assert gen["responseSchema"]["type"] == "OBJECT"


def test_gemini_generate_filters_thinking_parts(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct import seed as seed_mod

    # Gemini 3.x Flash returns thought: True on reasoning part
    resp = _FakeResp(
        200,
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": 'Thinking process with unescaped "quotes" and [brackets]', "thought": True},
                            {"text": '{"ads": [{"start": "00:00:10", "end": "00:00:45"}]}'},
                        ]
                    }
                }
            ]
        },
    )
    fake = _FakeClient([resp])
    with patch.object(seed_mod.httpx, "Client", return_value=fake):
        out = seed_mod._gemini_generate("k", "gemini-3.5-flash", "hi")
    assert seed_mod._extract_json_array(out) == [{"start": 10.0, "end": 45.0}]


def test_extract_json_array_edge_cases():
    from podaddeduct import seed as seed_mod

    # Markdown wrapped
    md = '```json\n{"ads": [{"start": "00:00:01", "end": "00:00:02"}]}\n```'
    assert seed_mod._extract_json_array(md) == [{"start": 1.0, "end": 2.0}]

    # Bare array still accepted
    surrounded = 'Found ads:\n[{"start": 5.0, "end": 10.0}]\nDone.'
    assert seed_mod._extract_json_array(surrounded) == [{"start": 5.0, "end": 10.0}]

    # Truncated JSON array salvages valid objects
    truncated = '[{"start": 1.0, "end": 2.0}, {"start": 3.0, "end":'
    assert seed_mod._extract_json_array(truncated) == [{"start": 1.0, "end": 2.0}]

    # HH:MM:SS timestamps
    hms = '{"ads": [{"start": "00:01:30", "end": "00:02:00", "type": "host_read", "sponsor": "Acme", "confidence": 0.9}]}'
    parsed = seed_mod._extract_ads_payload(hms)
    assert parsed[0]["start"] == 90.0
    assert parsed[0]["end"] == 120.0
    assert parsed[0]["sponsor"] == "Acme"


def test_gemini_model_default(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct import seed as seed_mod

    assert seed_mod.gemini_model() == "opencode/deepseek-v4-flash"
    db.set_global_settings({"gemini_model": "opencode/deepseek-v4-flash"})
    assert seed_mod.gemini_model() == "opencode/deepseek-v4-flash"


def _ok_serve_text(text):
    return _FakeResp(200, {"parts": [{"type": "text", "text": text}]})


class _ServeClient:
    """httpx.Client stand-in for opencode serve (GET/POST/DELETE)."""

    def __init__(self, ads_json='{"ads": []}'):
        self.ads_json = ads_json
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        return _FakeResp(200, {"healthy": True, "version": "test"})

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        if url.rstrip("/").endswith("/session"):
            return _FakeResp(200, {"id": "ses_test"})
        if "/message" in url:
            return _ok_serve_text(self.ads_json)
        return _FakeResp(404, text="unexpected post")

    def delete(self, url, **kwargs):
        self.calls.append(("DELETE", url, kwargs))
        return _FakeResp(200, True)

    def put(self, url, **kwargs):
        self.calls.append(("PUT", url, kwargs))
        return _FakeResp(200, {"ok": True})


def test_opencode_serve_posts_deepseek_v4_flash(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct import seed as seed_mod

    fake = _ServeClient('{"ads": [{"start": "00:00:10", "end": "00:00:40"}]}')
    transcript = {
        "sentences": [
            {"text": "Welcome back to the show.", "start": 0.0, "end": 4.0},
            {"text": "Midroll car commercial here.", "start": 10.0, "end": 40.0},
        ]
    }

    def no_subproc(*a, **k):
        raise AssertionError(f"must not spawn subprocess: {a}")

    with (
        patch.object(seed_mod.httpx, "Client", return_value=fake),
        patch("subprocess.run", side_effect=no_subproc),
        patch("subprocess.Popen", side_effect=no_subproc),
    ):
        result = seed_mod.find_ads_with_gemini(transcript)
    assert result.gemini_ok
    assert len(result.ranges) == 1
    posts = [c for c in fake.calls if c[0] == "POST"]
    msg = [c for c in posts if "/message" in c[1]]
    sess = [c for c in posts if c[1].rstrip("/").endswith("/session")]
    assert sess and msg
    body = msg[0][2]["json"]
    assert body["model"]["providerID"] == "opencode"
    assert body["model"]["modelID"] == "deepseek-v4-flash"
    assert "car commercial" in body["parts"][0]["text"]
    assert "car commercial" not in body["system"]
    assert "advertisement detection" in body["system"].lower()
    assert body["tools"] == {"bash": False, "edit": False, "write": False, "read": False}
    assert not any("opencode.ai" in c[1] for c in fake.calls)
    assert not any("zen/v1" in c[1] for c in fake.calls)
    assert not any("opencode run" in str(c) for c in fake.calls)
    assert any(c[0] == "DELETE" and "/session/" in c[1] for c in fake.calls)


def test_opencode_generate_surfaces_assistant_error(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct import seed as seed_mod

    class _ErrClient(_ServeClient):
        def post(self, url, **kwargs):
            self.calls.append(("POST", url, kwargs))
            if url.rstrip("/").endswith("/session"):
                return _FakeResp(200, {"id": "ses_err"})
            if "/message" in url:
                return _FakeResp(
                    200,
                    {
                        "info": {
                            "role": "assistant",
                            "error": {
                                "name": "APIError",
                                "data": {"message": "Insufficient balance."},
                            },
                        },
                        "parts": [],
                    },
                )
            return _FakeResp(404, text="unexpected post")

    fake = _ErrClient()
    with patch.object(seed_mod.httpx, "Client", return_value=fake):
        with pytest.raises(RuntimeError, match="Insufficient balance"):
            seed_mod.opencode_generate("hi")


def test_find_ads_uses_opencode(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct import seed as seed_mod

    transcript = {"sentences": [{"text": "Welcome back to the show.", "start": 0.0, "end": 4.0},
                                {"text": "Midroll car commercial here.", "start": 4.0, "end": 30.0}]}
    calls = []

    def fake_oc(user_content, model=None, **kwargs):
        calls.append(model)
        return '{"ads": [{"start": "00:00:04", "end": "00:00:30"}]}'

    with patch.object(seed_mod, "opencode_generate", side_effect=fake_oc):
        result = seed_mod.find_ads_with_zen(transcript)
    assert len(result.ranges) == 1
    assert result.gemini_ok
    assert calls and "deepseek-v4-flash" in (calls[0] or "")


def test_find_ads_without_key_uses_heuristics_only(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct import seed as seed_mod

    transcript = {
        "sentences": [
            {"text": "This episode is brought to you by Acme.", "start": 4.0, "end": 30.0},
            {"text": "Back to the show.", "start": 30.0, "end": 40.0},
        ]
    }
    with (
        patch.object(seed_mod, "resolve_gemini_api_key", return_value=None),
        patch.object(seed_mod, "opencode_generate", side_effect=RuntimeError("serve down")),
        patch.object(seed_mod, "gemini_model", return_value="gemini-2.5-flash"),
    ):
        result = seed_mod.find_ads_with_zen(transcript)
    assert len(result.ranges) >= 1
    assert result.ranges[0].start <= 4.5
    assert result.gemini_ok is False


def test_find_ads_gemini_soft_fails_to_heuristics(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct import seed as seed_mod

    transcript = {
        "sentences": [
            {"text": "This episode is brought to you by Acme.", "start": 4.0, "end": 30.0},
            {"text": "Plain talk.", "start": 40.0, "end": 50.0},
        ]
    }

    def boom(*a, **k):
        raise RuntimeError("network down")

    with (
        patch.object(seed_mod, "resolve_gemini_api_key", return_value="AIza-test"),
        patch.object(seed_mod, "gemini_model", return_value="gemini-2.5-flash"),
        patch.object(seed_mod, "opencode_generate", side_effect=RuntimeError("serve down")),
        patch.object(seed_mod, "_gemini_generate", side_effect=boom),
    ):
        result = seed_mod.find_ads_with_zen(transcript)
    assert len(result.ranges) >= 1  # heuristics still applied; never raises
    assert result.gemini_ok is False
    assert result.gemini_error and "network down" in result.gemini_error


def test_micro_cuts_filtered_when_gemini_fails(tmp_path, monkeypatch):
    """Isolated 4–7s keyword sentences must not become 'ads removed'."""
    _setup(tmp_path, monkeypatch)
    from podaddeduct import seed as seed_mod

    transcript = {
        "sentences": [
            {"text": "Welcome to Talkin Baseball.", "start": 0.0, "end": 5.0},
            {"text": "Let's take a quick break.", "start": 67.0, "end": 73.0},
            {"text": "Show content resumes here for a while.", "start": 73.0, "end": 500.0},
            {"text": "Use code FOO at checkout.", "start": 525.0, "end": 529.0},
            {"text": "More baseball talk.", "start": 529.0, "end": 1400.0},
            {"text": "Visit example.com for details.", "start": 1436.0, "end": 1441.0},
            {"text": "Back to the show again.", "start": 1441.0, "end": 4900.0},
            {"text": "Call 1-800-GAMBLER if you have a gambling problem.", "start": 4909.0, "end": 4916.0},
            {"text": "Thanks for listening.", "start": 4916.0, "end": 4920.0},
        ]
    }

    def boom(*a, **k):
        raise RuntimeError("429 rate limited")

    with (
        patch.object(seed_mod, "resolve_gemini_api_key", return_value="AIza-test"),
        patch.object(seed_mod, "gemini_model", return_value="gemini-2.5-flash"),
        patch.object(seed_mod, "opencode_generate", side_effect=RuntimeError("serve down")),
        patch.object(seed_mod, "_gemini_generate", side_effect=boom),
    ):
        result = seed_mod.find_ads_with_zen(transcript)
    assert result.gemini_ok is False
    assert result.gemini_error
    assert all(r.duration >= seed_mod.MIN_AD_CUT_SECONDS for r in result.ranges)
    # Four isolated cue sentences must not survive as micro-cuts.
    assert not any(r.duration < 15 for r in result.ranges)


def test_full_transcript_opencode_ad_detection(tmp_path, monkeypatch):
    """Full transcript is sent in one request to save API quota and preserve context."""
    _setup(tmp_path, monkeypatch)
    import json

    from podaddeduct import seed as seed_mod

    sentences = []
    t = 0.0
    for i in range(200):
        text = f"Baseball content sentence number {i} with enough filler words to grow."
        sentences.append({"text": text, "start": t, "end": t + 8.0})
        t += 8.0
    sentences[5] = {"text": "This midroll is a car commercial pitch.", "start": 40.0, "end": 100.0}
    sentences[150] = {"text": "Post show sponsor read continues here.", "start": 1200.0, "end": 1280.0}
    transcript = {"sentences": sentences}

    bodies = []

    def fake_oc(user_content, model=None, **kwargs):
        bodies.append(user_content)
        return json.dumps({
            "ads": [
                {"start": "00:00:40", "end": "00:01:40", "type": "inserted_ad", "sponsor": "unknown", "confidence": 0.9},
                {"start": "00:20:00", "end": "00:21:20", "type": "host_read", "sponsor": "unknown", "confidence": 0.9},
            ]
        })

    seen_progress = []
    with patch.object(seed_mod, "opencode_generate", side_effect=fake_oc):
        result = seed_mod.find_ads_with_gemini(
            transcript, progress_cb=lambda d, tot: seen_progress.append((d, tot))
        )
    assert result.gemini_ok
    assert len(bodies) == 1, "expected exactly 1 request per episode"
    assert "car commercial pitch" in bodies[0]
    assert "sponsor read continues" in bodies[0]
    assert len(result.ranges) == 2
    assert result.ranges[0].start == 40.0
    assert result.ranges[1].end == 1280.0
    assert seen_progress and seen_progress[-1][0] == seen_progress[-1][1]


def test_gemini_quota_exceeded_fails_fast_without_retries(tmp_path, monkeypatch):
    """HTTP 429 quota exhaustion must fail immediately without wasting retry sleep loops."""
    _setup(tmp_path, monkeypatch)
    from podaddeduct import seed as seed_mod

    quota_resp = _FakeResp(429, {}, text="You exceeded your current quota, please check your plan and billing details.")
    fake = _FakeClient([quota_resp])
    with patch.object(seed_mod.httpx, "Client", return_value=fake):
        with pytest.raises(RuntimeError) as exc_info:
            seed_mod._gemini_generate("k", "gemini-3.5-flash", "hi")
        assert "Gemini quota exceeded" in str(exc_info.value)
    # Must have only posted ONCE (no retry loop on hard quota error)
    assert len(fake.posts) == 1


def test_gemini_retries_429_then_succeeds(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct import seed as seed_mod

    fake = _FakeClient([
        _FakeResp(429, text="slow down", headers={"retry-after": "0"}),
        _ok_gemini('{"ads": []}'),
    ])
    sleeps = []
    with (
        patch.object(seed_mod.httpx, "Client", return_value=fake),
        patch("time.sleep", side_effect=lambda s: sleeps.append(s)),
    ):
        out = seed_mod._gemini_generate("k", "m", "hi")
    assert '{"ads": []}' in out
    assert len(fake.posts) == 2
    assert sleeps, "expected a backoff sleep between attempts"


def test_gemini_fatal_401_raises(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct import seed as seed_mod

    fake = _FakeClient([_FakeResp(401, text="bad key")])
    with patch.object(seed_mod.httpx, "Client", return_value=fake):
        with pytest.raises(Exception, match="401"):
            seed_mod._gemini_generate("k", "m", "hi")
    assert len(fake.posts) == 1


def test_settings_save_accepts_gemini_model(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from fastapi.testclient import TestClient

    from podaddeduct.app import app

    with TestClient(app) as client:
        r = client.post(
            "/settings/global",
            data={"settings_form": "1", "gemini_model": "opencode/deepseek-v4-flash"},
            follow_redirects=False,
        )
        assert r.status_code == 303
    assert db.runtime_str("gemini_model") == "opencode/deepseek-v4-flash"


def test_opencode_direct_routing(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct import seed as seed_mod

    transcript = {
        "sentences": [
            {"text": "Welcome to the show.", "start": 0.0, "end": 10.0},
            {"text": "Ad here.", "start": 10.0, "end": 40.0},
        ]
    }
    db.set_global_settings({"gemini_model": "opencode/deepseek-v4-flash"})

    with patch.object(
        seed_mod,
        "opencode_generate",
        return_value='{"ads": [{"start": "00:00:10", "end": "00:00:40"}]}',
    ) as mock_gen:
        res = seed_mod.find_ads_with_gemini(transcript)
    assert res.gemini_ok
    assert len(res.ranges) == 1
    assert res.ranges[0].start == 10.0
    mock_gen.assert_called_once()
    assert mock_gen.call_args[1]["model"] == "opencode/deepseek-v4-flash"


def test_gemini_quota_falls_back_to_opencode(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct import seed as seed_mod

    transcript = {
        "sentences": [
            {"text": "Welcome to the show.", "start": 0.0, "end": 10.0},
            {"text": "Ad break content.", "start": 10.0, "end": 40.0},
        ]
    }

    def boom(*a, **k):
        raise RuntimeError("Gemini quota exceeded: 429 quota")

    with (
        patch.object(seed_mod, "resolve_gemini_api_key", return_value="AIza-test"),
        patch.object(seed_mod, "gemini_model", return_value="gemini-2.5-flash"),
        patch.object(seed_mod, "_gemini_generate", side_effect=boom),
        patch.object(
            seed_mod,
            "opencode_generate",
            return_value='{"ads": [{"start": "00:00:10", "end": "00:00:40"}]}',
        ) as mock_gen,
    ):
        res = seed_mod.find_ads_with_gemini(transcript)
    assert res.gemini_ok
    assert len(res.ranges) == 1
    assert res.ranges[0].start == 10.0
    mock_gen.assert_called_once()


def test_reprocess_all_show_endpoint(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from fastapi.testclient import TestClient

    from podaddeduct.app import app

    feed = db.create_feed(slug="show-a", upstream_url="https://example.com/rss", title="A")
    ep1 = db.upsert_episode(feed.id, guid="g1", title="E1", enclosure_url="https://x/1.mp3", pub_date=None)
    ep2 = db.upsert_episode(feed.id, guid="g2", title="E2", enclosure_url="https://x/2.mp3", pub_date=None)
    db.update_episode(ep1.id, status="ready")
    db.update_episode(ep2.id, status="ready")

    queued = []

    async def fake_enqueue(episode_id, *, reseed=False, priority=False):
        queued.append((episode_id, reseed))
        return True

    with (
        patch("podaddeduct.app.enqueue_episode", side_effect=fake_enqueue),
        TestClient(app) as client,
    ):
        r = client.post(f"/shows/{feed.slug}/reprocess-all", follow_redirects=False)
        assert r.status_code == 303
        assert "reprocess=1" in r.headers["location"]
    assert sorted(queued) == sorted([(ep1.id, True), (ep2.id, True)])


def test_parse_timestamp_helpers():
    from podaddeduct.seed import _parse_timestamp
    from podaddeduct.stt import format_hms

    assert format_hms(0) == "00:00:00"
    assert format_hms(65) == "00:01:05"
    assert format_hms(3661) == "01:01:01"
    assert _parse_timestamp("00:01:30") == 90.0
    assert _parse_timestamp("1:30") == 90.0
    assert _parse_timestamp(12.5) == 12.5
    assert _parse_timestamp("bad") is None


def test_connection_test_uses_opencode_even_when_gemini_key_exists(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct import seed as seed_mod

    with (
        patch.object(seed_mod, "opencode_generate", return_value='{"ads": []}'),
        patch.object(seed_mod, "resolve_gemini_api_key", return_value="AIza-must-not-use"),
        patch.object(seed_mod, "_gemini_generate", side_effect=AssertionError("must not call Gemini")),
    ):
        out = seed_mod.test_gemini_connection()
    assert out["ok"] is True
    assert out["provider"] == "opencode"


def test_connection_test_serve_down_does_not_use_gemini_key(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct import opencode_server as oc
    from podaddeduct import seed as seed_mod

    with (
        patch.object(oc, "serve_health", return_value={"ok": False, "error": "OpenCode server is not running."}),
        patch.object(oc, "ensure_opencode_serve", return_value={"ok": False, "error": "OpenCode server is not running."}),
        patch.object(seed_mod, "resolve_gemini_api_key", return_value="AIza-test"),
        patch.object(seed_mod, "_gemini_generate", side_effect=AssertionError("must not call Gemini")),
    ):
        out = seed_mod.test_gemini_connection()
    assert out["ok"] is False
    assert out["provider"] == "opencode"
    assert "not running" in out["error"]
