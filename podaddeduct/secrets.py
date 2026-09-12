from __future__ import annotations

import json
from pathlib import Path

from .config import settings

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


def get_zen_api_key() -> str | None:
    stored = (load_secrets().get("zen_api_key") or "").strip()
    return stored or None


def set_zen_api_key(key: str | None) -> None:
    key = (key or "").strip()
    save_secrets({"zen_api_key": key or None})


def zen_key_status() -> dict:
    """Safe status for UI — never returns the raw key."""
    stored = get_zen_api_key()
    env = (settings.zen_api_key or "").strip()
    auth_fallback = False
    if not stored and not env:
        auth_path = Path(settings.zen_auth_path).expanduser()
        if auth_path.exists():
            try:
                data = json.loads(auth_path.read_text(encoding="utf-8"))
                for provider in ("opencode", "opencode-go"):
                    entry = data.get(provider)
                    if isinstance(entry, dict) and entry.get("key"):
                        auth_fallback = True
                        break
            except (OSError, json.JSONDecodeError):
                pass
    source = None
    hint = ""
    if stored:
        source = "ui"
        hint = stored[:6] + "…" + stored[-4:] if len(stored) > 12 else "••••"
    elif env:
        source = "env"
        hint = env[:6] + "…" + env[-4:] if len(env) > 12 else "••••"
    elif auth_fallback:
        source = "opencode-auth"
        hint = "from OpenCode login"
    return {
        "configured": bool(stored or env or auth_fallback),
        "source": source,
        "hint": hint,
        "model": settings.zen_model,
    }
