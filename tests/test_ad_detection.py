from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from podaddeduct import db
from podaddeduct.config import settings
from podaddeduct.intervals import Interval


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
    assert gen["temperature"] == 0.2
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


def test_gemini_generate_filters_thinking_parts(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct import seed as seed_mod

    resp = _FakeResp(
        200,
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": 'Thinking process with unescaped "quotes"', "thought": True},
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


def test_gemini_model_default(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct import seed as seed_mod

    assert seed_mod.gemini_model() == "gemini-3.5-flash"
    db.set_global_settings({"gemini_model": "gemini-2.5-flash"})
    assert seed_mod.gemini_model() == "gemini-2.5-flash"


def test_find_ads_uses_gemini_once(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct import seed as seed_mod

    transcript = {
        "sentences": [
            {"text": "Welcome.", "start": 0.0, "end": 4.0},
            {"text": "Baseball talk.", "start": 30.0, "end": 40.0},
            {"text": "Car commercial midroll.", "start": 400.0, "end": 445.0},
            {"text": "Back to the game.", "start": 445.0, "end": 455.0},
        ]
    }
    calls = []

    def fake_gen(api_key, model, user_content):
        calls.append(user_content)
        return '{"ads": [{"start": "00:06:40", "end": "00:07:25", "type": "inserted_ad", "sponsor": "x", "confidence": 0.9}]}'

    with (
        patch.object(seed_mod, "resolve_gemini_api_key", return_value="AIza-test"),
        patch.object(seed_mod, "_gemini_generate", side_effect=fake_gen),
    ):
        result = seed_mod.find_ads_with_zen(transcript)
    assert len(calls) == 1
    assert "Car commercial" in calls[0]
    assert result.gemini_ok
    assert any(a.start >= 390 for a in result.ranges)


def test_find_ads_heuristic_only_when_gemini_fails(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct import seed as seed_mod

    transcript = {
        "sentences": [
            {"text": "Welcome.", "start": 0.0, "end": 3.0},
            {"text": "This episode is brought to you by Acme.", "start": 4.0, "end": 22.0},
            {"text": "Use code HAMMER for twenty percent off.", "start": 22.0, "end": 30.0},
            {"text": "Back to baseball.", "start": 30.0, "end": 40.0},
        ]
    }
    with (
        patch.object(seed_mod, "resolve_gemini_api_key", return_value="AIza-test"),
        patch.object(seed_mod, "_gemini_generate", side_effect=RuntimeError("quota")),
    ):
        result = seed_mod.find_ads_with_zen(transcript)
    assert not result.gemini_ok
    assert result.ranges
    assert "heuristic" in (result.sources or [])


def test_filter_max_duration_drops_51_minute_midroll():
    from podaddeduct.seed import filter_max_duration

    duration = 62 * 60.0
    ads = [Interval(0.0, 60.0), Interval(9 * 60 + 3, 60 * 60 + 7)]
    kept = filter_max_duration(ads, duration=duration)
    assert len(kept) == 1
    assert kept[0].end <= 60.0 + 1.0
    assert all(r.duration <= 180.0 or r.start <= 1.0 for r in kept)


def test_cut_guards_refuse_episode_wipe():
    from podaddeduct.seed import evaluate_cut_guards

    duration = 62 * 60.0
    ads = [Interval(0.0, 60.0), Interval(9 * 60 + 3, 60 * 60 + 7)]
    # Even after filter_max, a huge remaining coverage must refuse.
    huge = [Interval(60.0, duration - 60.0)]
    guard = evaluate_cut_guards(huge, duration)
    assert not guard.ok
    assert "not cutting" in (guard.reason or "").lower() or "original kept" in (guard.reason or "").lower()


def test_connection_test_uses_gemini(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct import seed as seed_mod

    with (
        patch.object(seed_mod, "resolve_gemini_api_key", return_value="AIza-test"),
        patch.object(seed_mod, "_gemini_generate", return_value='{"ads": []}'),
    ):
        out = seed_mod.test_gemini_connection()
    assert out["ok"] is True
    assert out["provider"] == "gemini"


def test_settings_save_gemini_model(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct.app import app

    with TestClient(app) as client:
        r = client.post(
            "/settings/global",
            data={"settings_form": "1", "gemini_model": "gemini-3.5-flash"},
            follow_redirects=False,
        )
        assert r.status_code == 303
    assert db.runtime_str("gemini_model") == "gemini-3.5-flash"
