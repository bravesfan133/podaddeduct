from __future__ import annotations

import json
import os
from pathlib import Path

from .config import settings
from . import db as _db

_SECRETS_NAME = "secrets.json"


def secrets_path() -> Path:
    return Path(settings.data_dir) / _SECRETS_NAME


def load_secrets() -> dict:
    path = secrets_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_secrets(updates: dict) -> dict:
    path = secrets_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    current = load_secrets()
    for key, value in updates.items():
        if value is None:
            current.pop(key, None)
        else:
            current[key] = value
    path.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return current


def get_gemini_api_key() -> str | None:
    """UI-saved key first, then GEMINI_API_KEY / GOOGLE_API_KEY env."""
    stored = (load_secrets().get("gemini_api_key") or "").strip()
    if stored:
        return stored
    env = (settings.gemini_api_key or "").strip()
    if env:
        return env
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        val = (os.environ.get(name) or "").strip()
        if val:
            return val
    return None


def set_gemini_api_key(key: str | None) -> None:
    key = (key or "").strip()
    save_secrets({"gemini_api_key": key or None})


def gemini_key_status() -> dict:
    """Safe status for UI — never returns the raw key."""
    stored = (load_secrets().get("gemini_api_key") or "").strip()
    env = (settings.gemini_api_key or "").strip()
    if not env:
        for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            env = (os.environ.get(name) or "").strip()
            if env:
                break
    source = None
    hint = ""
    if stored:
        source = "ui"
        hint = stored[:6] + "…" + stored[-4:] if len(stored) > 12 else "••••"
    elif env:
        source = "env"
        hint = env[:6] + "…" + env[-4:] if len(env) > 12 else "••••"
    return {
        "configured": bool(stored or env),
        "source": source,
        "hint": hint,
        "model": _db.runtime_str("gemini_model") or "opencode/deepseek-v4-flash",
    }


def get_zen_api_key() -> str | None:
    """UI-saved Zen key first, then ZEN_API_KEY / OPENCODE_API_KEY env."""
    stored = (load_secrets().get("zen_api_key") or "").strip()
    if stored:
        return stored
    env = (settings.zen_api_key or "").strip()
    if env:
        return env
    for name in ("ZEN_API_KEY", "OPENCODE_API_KEY"):
        val = (os.environ.get(name) or "").strip()
        if val:
            return val
    return None


def set_zen_api_key(key: str | None) -> None:
    key = (key or "").strip()
    save_secrets({"zen_api_key": key or None})


def zen_key_status() -> dict:
    stored = (load_secrets().get("zen_api_key") or "").strip()
    env = (settings.zen_api_key or "").strip()
    if not env:
        for name in ("ZEN_API_KEY", "OPENCODE_API_KEY"):
            env = (os.environ.get(name) or "").strip()
            if env:
                break
    source = None
    hint = ""
    if stored:
        source = "ui"
        hint = stored[:6] + "…" + stored[-4:] if len(stored) > 12 else "••••"
    elif env:
        source = "env"
        hint = env[:6] + "…" + env[-4:] if len(env) > 12 else "••••"
    return {"configured": bool(stored or env), "source": source, "hint": hint}


def get_groq_api_key() -> str | None:
    """UI-saved key first, then GROQ_API_KEY env. Used only for optional cloud Whisper STT."""
    stored = (load_secrets().get("groq_api_key") or "").strip()
    if stored:
        return stored
    env = (os.environ.get("GROQ_API_KEY") or "").strip()
    return env or None


def set_groq_api_key(key: str | None) -> None:
    key = (key or "").strip()
    save_secrets({"groq_api_key": key or None})


def groq_key_status() -> dict:
    stored = bool((load_secrets().get("groq_api_key") or "").strip())
    env = bool((os.environ.get("GROQ_API_KEY") or "").strip())
    source = "ui" if stored else ("env" if env else None)
    return {"configured": bool(stored or env), "source": source}


def get_app_password() -> str:
    """Effective family password: UI value wins ("" disables), else env."""
    data = load_secrets()
    if "app_password" in data:
        return str(data.get("app_password") or "")
    return str(settings.app_password or "")


def password_source() -> str | None:
    data = load_secrets()
    if "app_password" in data:
        return "ui" if data.get("app_password") else None
    if settings.app_password:
        return "env"
    return None


def set_app_password(value: str) -> None:
    save_secrets({"app_password": value})
