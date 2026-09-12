from __future__ import annotations

import socket
from pathlib import Path
from urllib.parse import urlparse

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def detect_lan_base(port: int = 8080) -> str:
    """Best-effort LAN URL so phones on Wi‑Fi can reach this Mac."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        if ip and not ip.startswith("127."):
            return f"http://{ip}:{port}"
    except OSError:
        pass
    return f"http://127.0.0.1:{port}"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Empty = auto-detect LAN IP (do not use 127.0.0.1 for phone clients)
    public_base_url: str = ""
    process_recent: int = 3
    feed_item_limit: int = 40
    min_ad_seconds: float = 8.0
    data_dir: Path = Path("./data")
    host: str = "0.0.0.0"
    port: int = 8080
    user_agent: str = "Podcasts/4025.610.1 CFNetwork/1.0 Darwin/24.0.0"

    # --- Storage: small cache, not an archive (2-3 GB default) ---
    max_cache_gb: float = 3.0
    keep_last_n: int = 5
    delete_after_days: int = 14
    # Delete the big original download after a clean file is cut.
    # Re-cut later just re-downloads + reuses saved ad marks (no extra AI cost).
    delete_original_after_cut: bool = True
    # Background RSS check interval (metadata only, kilobytes — no audio).
    poll_minutes: int = 30
    # When a new episode appears, prepare the latest 1 automatically.
    auto_prepare_latest: bool = True
    # Optional shared password when exposed via tunnel (empty = no login locally).
    app_password: str = ""

    # Drea-style seed: local Parakeet + OpenCode Zen
    seed_on_first_download: bool = True
    stt_python: str = "./.venv-stt/bin/python"
    stt_sidecar: str = "./scripts/stt_sidecar.py"
    stt_model: str = "mlx-community/parakeet-tdt-0.6b-v3"
    zen_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("ZEN_API_KEY", "OPENCODE_API_KEY", "zen_api_key"),
    )
    zen_base_url: str = "https://opencode.ai/zen/v1"
    # Muse Spark 1.3 Free (API id). Responses-only — chat/completions returns 500.
    zen_model: str = "muse-spark-1.3-contributor-free"
    zen_fallback_model: str = "deepseek-v4-flash-free"
    zen_auth_path: str = "~/.local/share/opencode/auth.json"
    zen_chunk_chars: int = 12000
    # When /responses returns MissingSessionID for free models, call OpenCode CLI.
    opencode_bin: str = "opencode"
    opencode_model: str = "opencode/muse-spark-1.3-contributor-free"
    silence_snap_window: float = 1.5

    def zen_models_to_try(self) -> list[str]:
        ordered: list[str] = []
        for mid in [self.zen_model, self.zen_fallback_model]:
            mid = mid.strip()
            if mid and mid not in ordered:
                ordered.append(mid)
        return ordered

    @property
    def audio_dir(self) -> Path:
        return self.data_dir / "audio"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "podaddeduct.db"

    @property
    def transcripts_dir(self) -> Path:
        return self.data_dir / "transcripts"

    def resolved_public_base(self) -> str:
        raw = (self.public_base_url or "").strip().rstrip("/")
        if not raw:
            return detect_lan_base(self.port)
        host = urlparse(raw).hostname or ""
        if host in {"127.0.0.1", "localhost"}:
            return detect_lan_base(self.port)
        return raw


settings = Settings()
