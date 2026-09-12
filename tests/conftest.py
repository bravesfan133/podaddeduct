"""Pytest defaults: never start a real `opencode serve` daemon."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _skip_opencode_daemon(monkeypatch):
    healthy = {"ok": True, "url": "http://127.0.0.1:4096", "version": "test"}
    monkeypatch.setattr(
        "podaddeduct.opencode_server.ensure_opencode_serve",
        lambda: healthy,
    )
    monkeypatch.setattr(
        "podaddeduct.opencode_server.serve_health",
        lambda url=None, timeout=2.0: healthy,
    )
