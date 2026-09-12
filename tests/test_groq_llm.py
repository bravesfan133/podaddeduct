from __future__ import annotations

from unittest.mock import patch

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


def _ok_choices(text):
    return _FakeResp(200, {"choices": [{"message": {"content": text}}]})


def test_zen_chat_success(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct import seed as seed_mod

    _FakeClient.instances.clear()
    fake = _FakeClient([_ok_choices('[{"start": 1, "end": 2}]')])
    with patch.object(seed_mod.httpx, "Client", return_value=fake):
        out = seed_mod._chat_completions("k", "big-pickle", "hi")
    assert out == '[{"start": 1, "end": 2}]'
    url, kwargs = fake.posts[0]
    assert url.endswith("/chat/completions")
    assert kwargs["headers"]["Authorization"] == "Bearer k"
    assert kwargs["json"]["model"] == "big-pickle"
    assert kwargs["json"]["max_tokens"] == seed_mod.CHAT_MAX_TOKENS


def test_zen_chat_retries_429_then_succeeds(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct import seed as seed_mod

    fake = _FakeClient([
        _FakeResp(429, text="slow down", headers={"retry-after": "0"}),
        _ok_choices("[]"),
    ])
    sleeps = []
    with (
        patch.object(seed_mod.httpx, "Client", return_value=fake),
        patch("time.sleep", side_effect=lambda s: sleeps.append(s)),
    ):
        out = seed_mod._chat_completions("k", "m", "hi")
    assert out == "[]"
    assert len(fake.posts) == 2
    assert sleeps, "expected a backoff sleep between attempts"


def test_zen_chat_fatal_401_raises(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    import pytest

    from podaddeduct import seed as seed_mod

    fake = _FakeClient([_FakeResp(401, text="bad key")])
    with patch.object(seed_mod.httpx, "Client", return_value=fake):
        with pytest.raises(Exception, match="401"):
            seed_mod._chat_completions("k", "m", "hi")
    assert len(fake.posts) == 1


def test_zen_chat_session_error_is_actionable(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    import pytest

    from podaddeduct import seed as seed_mod

    fake = _FakeClient([_FakeResp(400, text='{"error":"MissingSessionID only be used in OpenCode"}')])
    with patch.object(seed_mod.httpx, "Client", return_value=fake):
        with pytest.raises(RuntimeError, match="big-pickle"):
            seed_mod._chat_completions("k", "muse-spark-1.3-contributor-free", "hi")


def test_ad_models_default_to_zen_free(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct import seed as seed_mod

    assert seed_mod.ad_models_to_try() == ["big-pickle", "mimo-v2.5-free"]


def test_ad_models_skip_muse_spark_contributor(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct import seed as seed_mod

    db.set_global_settings({
        "zen_model": "muse-spark-1.3-contributor-free",
        "zen_fallback_model": "big-pickle",
    })
    assert seed_mod.ad_models_to_try() == ["big-pickle"]


def test_find_ads_uses_zen_models(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from podaddeduct import seed as seed_mod

    # No heuristic hits — forces LLM path.
    transcript = {"sentences": [{"text": "Talking about baseball all day.", "start": 4.0, "end": 30.0}]}
    calls = []

    def fake_chat(api_key, model, user_content):
        calls.append(model)
        return '[{"start": 4.0, "end": 30.0}]'

    with (
        patch.object(seed_mod, "resolve_zen_api_key", return_value="sk-test"),
        patch.object(seed_mod, "_chat_completions", side_effect=fake_chat),
    ):
        ads = seed_mod.find_ads_with_zen(transcript)
    assert len(ads) == 1
    assert calls and calls[0] == "big-pickle"


def test_find_ads_missing_key_errors_helpfully(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    import pytest

    from podaddeduct import seed as seed_mod

    with patch.object(seed_mod, "resolve_zen_api_key", return_value=None):
        with pytest.raises(RuntimeError, match="Zen API key"):
            seed_mod.find_ads_with_llm({"sentences": [{"text": "hello", "start": 0, "end": 1}]})


def test_settings_save_accepts_zen_models(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from fastapi.testclient import TestClient

    from podaddeduct.app import app

    with TestClient(app) as client:
        r = client.post(
            "/settings/global",
            data={"settings_form": "1", "zen_model": "big-pickle",
                  "zen_fallback_model": "mimo-v2.5-free"},
            follow_redirects=False,
        )
        assert r.status_code == 303
    assert db.runtime_str("zen_model") == "big-pickle"
    assert db.runtime_str("zen_fallback_model") == "mimo-v2.5-free"
