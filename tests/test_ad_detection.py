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
    assert kwargs["json"]["generationConfig"]["responseMimeType"] == "application/json"
    assert kwargs["json"]["generationConfig"]["maxOutputTokens"] == 8192


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

    assert seed_mod.gemini_model() == "gemini-3.6-flash"
    db.set_global_settings({"gemini_model": "gemini-2.0-flash"})
    assert seed_mod.gemini_model() == "gemini-2.0-flash"


def test_find_ads_uses_gemini(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct import seed as seed_mod

    # No heuristic hits so Gemini runs on leftovers.
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
        ads = seed_mod.find_ads_with_zen(transcript)
    assert len(ads) == 1
    assert calls and calls[0] == "gemini-3.6-flash"


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
        ads = seed_mod.find_ads_with_zen(transcript)
    assert len(ads) >= 1
    assert ads[0].start <= 4.5


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
        ads = seed_mod.find_ads_with_zen(transcript)
    assert len(ads) >= 1  # heuristics still applied; never raises


def test_settings_save_accepts_gemini_model(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from fastapi.testclient import TestClient

    from podaddeduct.app import app

    with TestClient(app) as client:
        r = client.post(
            "/settings/global",
            data={"settings_form": "1", "gemini_model": "gemini-3.6-flash"},
            follow_redirects=False,
        )
        assert r.status_code == 303
    assert db.runtime_str("gemini_model") == "gemini-3.6-flash"
