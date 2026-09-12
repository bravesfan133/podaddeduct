from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import httpx

from .config import settings

logger = logging.getLogger("podaddeduct.download")

# Log download progress at most this often (bytes between log lines).
PROGRESS_EVERY = 25 * 1024 * 1024


def complete_marker_for(dest: Path) -> Path:
    """Sidecar proving dest downloaded fully. A bare .bin without its
    marker is a leftover partial from a killed/failed attempt and must
    never be trusted (it would poison transcription)."""
    return dest.with_name(dest.name + ".complete")


async def download_file(
    url: str,
    dest: Path,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> Path:
    """Stream url to dest atomically. progress(done_bytes, total_bytes).

    Writes to a temp file and renames only on success; any failure deletes
    the partial so the next attempt starts clean. Marker file proves
    completeness for future runs.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    marker = complete_marker_for(dest)
    try:
        try:
            marker.unlink()
        except OSError:
            pass
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(300.0, connect=30.0, read=60.0),
            follow_redirects=True,
            headers={"User-Agent": settings.user_agent},
        ) as client:
            async with client.stream("GET", url) as resp:
                resp.raise_for_status()
                try:
                    total = int(resp.headers.get("content-length") or 0)
                except ValueError:
                    total = 0
                done = 0
                next_log = PROGRESS_EVERY
                with tmp.open("wb") as f:
                    async for chunk in resp.aiter_bytes(1024 * 256):
                        f.write(chunk)
                        done += len(chunk)
                        if progress:
                            progress(done, total)
                        if done >= next_log:
                            if total:
                                logger.info("download %s: %d/%d MB", dest.name, done // 1024**2, total // 1024**2)
                            else:
                                logger.info("download %s: %d MB (size unknown)", dest.name, done // 1024**2)
                            next_log += PROGRESS_EVERY
        tmp.replace(dest)
        marker.touch()
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return dest
