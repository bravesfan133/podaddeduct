"""Long-lived `opencode serve` helper. Never runs `opencode run`."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from urllib.parse import urlparse

import httpx

from .config import settings

logger = logging.getLogger("podaddeduct.opencode")

_proc: subprocess.Popen | None = None


def opencode_server_url() -> str:
    from . import db

    return (db.runtime_str("opencode_server_url") or settings.opencode_server_url or "http://127.0.0.1:4096").rstrip("/")


def get_opencode_bin() -> str | None:
    b = shutil.which("opencode")
    if b:
        return b
    home_bin = os.path.expanduser("~/.opencode/bin/opencode")
    if os.path.isfile(home_bin) and os.access(home_bin, os.X_OK):
        return home_bin
    return None


def serve_health(url: str | None = None, timeout: float = 2.0) -> dict:
    base = (url or opencode_server_url()).rstrip("/")
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.get(f"{base}/global/health")
        if r.status_code >= 400:
            return {"ok": False, "url": base, "error": f"{r.status_code} {r.text[:200]}"}
        data = r.json() if r.content else {}
        return {"ok": True, "url": base, "version": (data or {}).get("version")}
    except Exception as exc:
        return {"ok": False, "url": base, "error": str(exc)[-200:]}


def push_zen_auth(url: str | None = None) -> None:
    from .secrets import get_zen_api_key

    key = get_zen_api_key()
    if not key:
        return
    base = (url or opencode_server_url()).rstrip("/")
    with httpx.Client(timeout=10.0) as client:
        r = client.put(f"{base}/auth/opencode", json={"type": "api", "key": key})
        if r.status_code >= 400:
            raise RuntimeError(f"OpenCode auth failed: {r.status_code} {r.text[:200]}")


def _port_from_url(url: str) -> int:
    port = urlparse(url).port
    return int(port or 4096)


def ensure_opencode_serve() -> dict:
    """Start a long-lived `opencode serve` if health is down. Not `opencode run`."""
    global _proc
    url = opencode_server_url()
    health = serve_health(url)
    if health.get("ok"):
        try:
            push_zen_auth(url)
        except Exception as exc:
            logger.warning("OpenCode Zen auth push failed: %s", exc)
        return health

    bin_path = get_opencode_bin()
    if not bin_path:
        health["error"] = "OpenCode CLI not found; cannot start opencode serve."
        return health

    if _proc is not None and _proc.poll() is None:
        time.sleep(0.4)
        health = serve_health(url)
        if health.get("ok"):
            return health

    port = _port_from_url(url)
    logger.info("Starting opencode serve on 127.0.0.1:%s", port)
    _proc = subprocess.Popen(
        [bin_path, "serve", "--hostname", "127.0.0.1", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    deadline = time.monotonic() + 12.0
    while time.monotonic() < deadline:
        health = serve_health(url)
        if health.get("ok"):
            try:
                push_zen_auth(url)
            except Exception as exc:
                logger.warning("OpenCode Zen auth push failed: %s", exc)
            return health
        if _proc.poll() is not None:
            health["error"] = f"opencode serve exited ({_proc.returncode})"
            return health
        time.sleep(0.25)
    health["error"] = health.get("error") or "opencode serve did not become healthy"
    return health
