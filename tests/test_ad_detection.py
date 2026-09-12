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
    fake = _FakeClient([_ok_gemini('[{"start": 1, "end": 2}]')])
    with patch.object(seed_mod.httpx, "Client", return_value=fake):
        out = seed_mod._gemini_generate("k", "gemini-3.6-flash", "hi")
    assert out == '[{"start": 1, "end": 2}]'
    url, kwargs = fake.posts[0]
    assert "gemini-3.6-flash:generateContent" in url
    assert kwargs["headers"]["x-goog-api-key"] == "k"
    gen = kwargs["json"]["generationConfig"]
    assert gen["responseMimeType"] == "application/json"
    assert gen["maxOutputTokens"] == 8192
    assert gen["temperature"] == 1.0
    assert gen["thinkingConfig"] == {"thinkingBudget": 0}
    assert gen["responseSchema"]["type"] == "ARRAY"
    assert "start" in gen["responseSchema"]["items"]["properties"]


def test_gemini_2x_payload_omits_thinking(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct import seed as seed_mod

    fake = _FakeClient([_ok_gemini("[]")])
    with patch.object(seed_mod.httpx, "Client", return_value=fake):
        seed_mod._gemini_generate("k", "gemini-2.5-flash", "hi")
    gen = fake.posts[0][1]["json"]["generationConfig"]
    assert "thinkingConfig" not in gen
    assert gen["responseSchema"]["type"] == "ARRAY"


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
                            {"text": '[{"start": 10.5, "end": 45.0}]'},
                        ]
                    }
                }
            ]
        },
    )
    fake = _FakeClient([resp])
    with patch.object(seed_mod.httpx, "Client", return_value=fake):
        out = seed_mod._gemini_generate("k", "gemini-3.6-flash", "hi")
    assert out == '[{"start": 10.5, "end": 45.0}]'
    assert seed_mod._extract_json_array(out) == [{"start": 10.5, "end": 45.0}]


def test_extract_json_array_edge_cases():
    from podaddeduct import seed as seed_mod

    # Markdown wrapped
    md = "```json\n[{\"start\": 1.0, \"end\": 2.5}]\n```"
    assert seed_mod._extract_json_array(md) == [{"start": 1.0, "end": 2.5}]

    # Explanatory text surrounding array
    surrounded = "Found ads:\n[{\"start\": 5.0, \"end\": 10.0}]\nDone."
    assert seed_mod._extract_json_array(surrounded) == [{"start": 5.0, "end": 10.0}]

    # Truncated JSON array salvages valid objects
    truncated = '[{"start": 1.0, "end": 2.0}, {"start": 3.0, "end":'
    assert seed_mod._extract_json_array(truncated) == [{"start": 1.0, "end": 2.0}]


def test_gemini_retries_429_then_succeeds(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct import seed as seed_mod

    fake = _FakeClient([
        _FakeResp(429, text="slow down", headers={"retry-after": "0"}),
        _ok_gemini("[]"),
    ])
    sleeps = []
    with (
        patch.object(seed_mod.httpx, "Client", return_value=fake),
        patch("time.sleep", side_effect=lambda s: sleeps.append(s)),
    ):
        out = seed_mod._gemini_generate("k", "m", "hi")
    assert out == "[]"
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


def test_gemini_model_default(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct import seed as seed_mod

    assert seed_mod.gemini_model() == "gemini-3.5-flash"
    db.set_global_settings({"gemini_model": "gemini-2.5-flash"})
    assert seed_mod.gemini_model() == "gemini-2.5-flash"


def test_gemini_503_falls_back_to_flash(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct import seed as seed_mod

    calls = []

    def fake_gemini(api_key, model, user_content):
        calls.append(model)
        if model == seed_mod.GEMINI_DEFAULT_MODEL:
            raise RuntimeError("503 This model is currently experiencing high demand.")
        return '[{"start": 10.0, "end": 40.0}]'

    transcript = {
        "sentences": [
            {"text": "Welcome back to the show.", "start": 0.0, "end": 4.0},
            {"text": "Midroll car commercial here.", "start": 10.0, "end": 40.0},
        ]
    }
    with (
        patch.object(seed_mod, "resolve_gemini_api_key", return_value="AIza-test"),
        patch.object(seed_mod, "_gemini_generate", side_effect=fake_gemini),
    ):
        result = seed_mod.find_ads_with_gemini(transcript)
    assert result.gemini_ok
    assert len(result.ranges) == 1
    assert calls == [seed_mod.GEMINI_DEFAULT_MODEL, seed_mod.GEMINI_FALLBACK_MODEL]
    assert result.ranges[0].start == 10.0


def test_find_ads_uses_gemini(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct import seed as seed_mod

    # No heuristic hits so Gemini runs on the full transcript.
    transcript = {"sentences": [{"text": "Welcome back to the show.", "start": 0.0, "end": 4.0},
                                {"text": "Midroll car commercial here.", "start": 4.0, "end": 30.0}]}
    calls = []

    def fake_gemini(api_key, model, user_content):
        calls.append(model)
        return '[{"start": 4.0, "end": 30.0}]'

    with (
        patch.object(seed_mod, "resolve_gemini_api_key", return_value="AIza-test"),
        patch.object(seed_mod, "_gemini_generate", side_effect=fake_gemini),
    ):
        result = seed_mod.find_ads_with_zen(transcript)
    assert len(result.ranges) == 1
    assert result.gemini_ok
    assert calls and calls[0] == "gemini-3.5-flash"


def test_find_ads_without_key_uses_heuristics_only(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct import seed as seed_mod

    transcript = {
        "sentences": [
            {"text": "This episode is brought to you by Acme.", "start": 4.0, "end": 30.0},
            {"text": "Back to the show.", "start": 30.0, "end": 40.0},
        ]
    }
    with patch.object(seed_mod, "resolve_gemini_api_key", return_value=None):
        result = seed_mod.find_ads_with_zen(transcript)
    assert len(result.ranges) >= 1
    assert result.ranges[0].start <= 4.5
    assert result.gemini_ok is False
    assert result.gemini_error is None


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
        patch.object(seed_mod, "_gemini_generate", side_effect=boom),
    ):
        result = seed_mod.find_ads_with_zen(transcript)
    assert result.gemini_ok is False
    assert result.gemini_error
    assert all(r.duration >= seed_mod.MIN_AD_CUT_SECONDS for r in result.ranges)
    # Four isolated cue sentences must not survive as micro-cuts.
    assert not any(r.duration < 15 for r in result.ranges)


def test_full_transcript_gemini_ad_detection(tmp_path, monkeypatch):
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

    def fake_gemini(api_key, model, user_content):
        bodies.append(user_content)
        return json.dumps([
            {"start": 40.0, "end": 100.0},
            {"start": 1200.0, "end": 1280.0},
        ])

    seen_progress = []
    with (
        patch.object(seed_mod, "resolve_gemini_api_key", return_value="AIza-test"),
        patch.object(seed_mod, "_gemini_generate", side_effect=fake_gemini),
    ):
        result = seed_mod.find_ads_with_gemini(
            transcript, progress_cb=lambda d, tot: seen_progress.append((d, tot))
        )
    assert result.gemini_ok
    assert len(bodies) == 1, "expected exactly 1 Gemini request per episode"
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


def test_settings_save_accepts_gemini_model(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from fastapi.testclient import TestClient

    from podaddeduct.app import app

    with TestClient(app) as client:
        r = client.post(
            "/settings/global",
            data={"settings_form": "1", "gemini_model": "gemini-3.5-flash"},
            follow_redirects=False,
        )
        assert r.status_code == 303
    assert db.runtime_str("gemini_model") == "gemini-3.5-flash"


def test_opencode_direct_routing(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct import seed as seed_mod

    transcript = {
        "sentences": [
            {"text": "Welcome to the show.", "start": 0.0, "end": 10.0},
            {"text": "Ad here.", "start": 10.0, "end": 40.0},
        ]
    }
    db.set_global_settings({"gemini_model": "opencode/kimi-k2.5"})

    with patch.object(seed_mod, "opencode_generate", return_value='[{"start": 10.0, "end": 40.0}]') as mock_gen:
        res = seed_mod.find_ads_with_gemini(transcript)
    assert res.gemini_ok
    assert len(res.ranges) == 1
    assert res.ranges[0].start == 10.0
    mock_gen.assert_called_once()
    assert mock_gen.call_args[1]["model"] == "opencode/kimi-k2.5"


def test_gemini_quota_falls_back_to_opencode_when_enabled(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct import seed as seed_mod

    transcript = {
        "sentences": [
            {"text": "Welcome to the show.", "start": 0.0, "end": 10.0},
            {"text": "Ad break content.", "start": 10.0, "end": 40.0},
        ]
    }
    db.set_global_settings({"opencode_fallback": "1"})

    def boom(*a, **k):
        raise RuntimeError("Gemini quota exceeded: 429 quota")

    with (
        patch.object(seed_mod, "resolve_gemini_api_key", return_value="AIza-test"),
        patch.object(seed_mod, "_gemini_generate", side_effect=boom),
        patch.object(seed_mod, "is_opencode_available", return_value=True),
        patch.object(seed_mod, "opencode_generate", return_value='[{"start": 10.0, "end": 40.0}]') as mock_gen,
    ):
        res = seed_mod.find_ads_with_gemini(transcript)
    assert res.gemini_ok
    assert len(res.ranges) == 1
    assert res.ranges[0].start == 10.0
    mock_gen.assert_called_once()
