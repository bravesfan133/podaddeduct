from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .config import settings

_lock = threading.Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Feed:
    id: int
    slug: str
    upstream_url: str
    title: str
    created_at: str
    artwork_url: str = ""
    settings_json: str = "{}"


@dataclass
class Episode:
    id: int
    feed_id: int
    guid: str
    title: str
    enclosure_url: str
    pub_date: str | None
    duration_seconds: float | None
    status: str
    audio_path: str | None
    ad_ranges_json: str
    content_fp_path: str | None
    clean_audio_path: str | None
    error: str | None
    updated_at: str
    size_bytes: int = 0
    last_served_at: str | None = None


SCHEMA = """
CREATE TABLE IF NOT EXISTS feeds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    slug TEXT NOT NULL UNIQUE,
    upstream_url TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    feed_id INTEGER NOT NULL REFERENCES feeds(id),
    guid TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    enclosure_url TEXT NOT NULL,
    pub_date TEXT,
    duration_seconds REAL,
    status TEXT NOT NULL DEFAULT 'pending',
    audio_path TEXT,
    ad_ranges_json TEXT NOT NULL DEFAULT '[]',
    content_fp_path TEXT,
    clean_audio_path TEXT,
    error TEXT,
    updated_at TEXT NOT NULL,
    UNIQUE(feed_id, guid)
);

CREATE INDEX IF NOT EXISTS idx_episodes_feed ON episodes(feed_id);
CREATE INDEX IF NOT EXISTS idx_episodes_status ON episodes(status);

CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL DEFAULT ''
);
"""


def _migrate(conn: sqlite3.Connection) -> None:
    feed_cols = {row[1] for row in conn.execute("PRAGMA table_info(feeds)").fetchall()}
    if "artwork_url" not in feed_cols:
        conn.execute("ALTER TABLE feeds ADD COLUMN artwork_url TEXT NOT NULL DEFAULT ''")
    if "settings_json" not in feed_cols:
        conn.execute("ALTER TABLE feeds ADD COLUMN settings_json TEXT NOT NULL DEFAULT '{}'")
    ep_cols = {row[1] for row in conn.execute("PRAGMA table_info(episodes)").fetchall()}
    if "clean_audio_path" not in ep_cols:
        conn.execute("ALTER TABLE episodes ADD COLUMN clean_audio_path TEXT")
    if "size_bytes" not in ep_cols:
        conn.execute("ALTER TABLE episodes ADD COLUMN size_bytes INTEGER NOT NULL DEFAULT 0")
    if "last_served_at" not in ep_cols:
        conn.execute("ALTER TABLE episodes ADD COLUMN last_served_at TEXT")


def init_db() -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.audio_dir.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        conn.executescript(SCHEMA)
        _migrate(conn)
        # Stale worker kill (e.g. server restart mid-STT) leaves 'processing' forever —
        # reset to pending so next feed view / worker run retries.
        conn.execute("UPDATE episodes SET status = 'pending', error = NULL WHERE status = 'working' OR status = 'processing'")


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(settings.db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        with _lock:
            yield conn
            conn.commit()
    finally:
        conn.close()


def _feed_from_row(row: sqlite3.Row) -> Feed:
    keys = set(row.keys())
    return Feed(
        id=row["id"],
        slug=row["slug"],
        upstream_url=row["upstream_url"],
        title=row["title"],
        created_at=row["created_at"],
        artwork_url=row["artwork_url"] if "artwork_url" in keys else "",
        settings_json=row["settings_json"] if "settings_json" in keys else "{}",
    )


def _episode_from_row(row: sqlite3.Row) -> Episode:
    keys = set(row.keys())
    return Episode(
        id=row["id"],
        feed_id=row["feed_id"],
        guid=row["guid"],
        title=row["title"],
        enclosure_url=row["enclosure_url"],
        pub_date=row["pub_date"],
        duration_seconds=row["duration_seconds"],
        status=row["status"],
        audio_path=row["audio_path"],
        ad_ranges_json=row["ad_ranges_json"] or "[]",
        content_fp_path=row["content_fp_path"],
        clean_audio_path=row["clean_audio_path"] if "clean_audio_path" in keys else None,
        error=row["error"],
        updated_at=row["updated_at"],
        size_bytes=int(row["size_bytes"] or 0) if "size_bytes" in keys else 0,
        last_served_at=row["last_served_at"] if "last_served_at" in keys else None,
    )


def get_feed_by_slug(slug: str) -> Feed | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM feeds WHERE slug = ?", (slug,)).fetchone()
    return _feed_from_row(row) if row else None


def get_feed_by_upstream(url: str) -> Feed | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM feeds WHERE upstream_url = ?", (url,)).fetchone()
    return _feed_from_row(row) if row else None


def get_feed(feed_id: int) -> Feed | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM feeds WHERE id = ?", (feed_id,)).fetchone()
    return _feed_from_row(row) if row else None


def list_feeds() -> list[Feed]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM feeds ORDER BY id DESC").fetchall()
    return [_feed_from_row(r) for r in rows]


def create_feed(slug: str, upstream_url: str, title: str = "") -> Feed:
    now = _utc_now()
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO feeds (slug, upstream_url, title, created_at) VALUES (?, ?, ?, ?)",
            (slug, upstream_url, title, now),
        )
        feed_id = int(cur.lastrowid)
    feed = get_feed(feed_id)
    assert feed is not None
    return feed


def update_feed_title(feed_id: int, title: str) -> None:
    with connect() as conn:
        conn.execute("UPDATE feeds SET title = ? WHERE id = ?", (title, feed_id))


def get_episode(episode_id: int) -> Episode | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM episodes WHERE id = ?", (episode_id,)).fetchone()
    return _episode_from_row(row) if row else None


def get_episode_by_guid(feed_id: int, guid: str) -> Episode | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM episodes WHERE feed_id = ? AND guid = ?",
            (feed_id, guid),
        ).fetchone()
    return _episode_from_row(row) if row else None


def list_episodes(feed_id: int) -> list[Episode]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM episodes WHERE feed_id = ? ORDER BY id DESC",
            (feed_id,),
        ).fetchall()
    return [_episode_from_row(r) for r in rows]


def upsert_episode(
    feed_id: int,
    guid: str,
    title: str,
    enclosure_url: str,
    pub_date: str | None,
) -> Episode:
    existing = get_episode_by_guid(feed_id, guid)
    now = _utc_now()
    if existing:
        enclosure_changed = existing.enclosure_url != enclosure_url
        with connect() as conn:
            if enclosure_changed:
                # New file for same GUID — re-download + re-detect
                conn.execute(
                    """
                    UPDATE episodes
                    SET title = ?, enclosure_url = ?, pub_date = ?, updated_at = ?,
                        audio_path = NULL, status = 'pending', error = NULL,
                        ad_ranges_json = '[]', clean_audio_path = NULL
                    WHERE id = ?
                    """,
                    (title, enclosure_url, pub_date, now, existing.id),
                )
            else:
                conn.execute(
                    """
                    UPDATE episodes
                    SET title = ?, enclosure_url = ?, pub_date = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (title, enclosure_url, pub_date, now, existing.id),
                )
        ep = get_episode(existing.id)
        assert ep is not None
        return ep

    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO episodes (
                feed_id, guid, title, enclosure_url, pub_date,
                status, ad_ranges_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, 'pending', '[]', ?)
            """,
            (feed_id, guid, title, enclosure_url, pub_date, now),
        )
        episode_id = int(cur.lastrowid)
    ep = get_episode(episode_id)
    assert ep is not None
    return ep


def update_episode(episode_id: int, **fields: Any) -> Episode:
    if not fields:
        ep = get_episode(episode_id)
        assert ep is not None
        return ep
    fields = {**fields, "updated_at": _utc_now()}
    cols = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [episode_id]
    with connect() as conn:
        conn.execute(f"UPDATE episodes SET {cols} WHERE id = ?", values)
    ep = get_episode(episode_id)
    assert ep is not None
    return ep


def get_ad_ranges(episode: Episode) -> list[dict[str, float]]:
    try:
        data = json.loads(episode.ad_ranges_json or "[]")
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return data


def set_ad_ranges(episode_id: int, ranges: list[dict[str, float]], status: str = "ready") -> Episode:
    return update_episode(
        episode_id,
        ad_ranges_json=json.dumps(ranges),
        status=status,
        error=None,
    )


def content_fp_path_for(episode_id: int) -> Path:
    """Legacy (pre-cut-only): old rows may still have content_fp_path set. Unused."""
    return settings.data_dir / "fingerprints" / f"{episode_id}.json"


def served_audio_path(episode: Episode) -> Path | None:
    """Prefer cleaned MP3 when present."""
    if episode.clean_audio_path and Path(episode.clean_audio_path).exists():
        return Path(episode.clean_audio_path)
    if episode.audio_path and Path(episode.audio_path).exists():
        return Path(episode.audio_path)
    return None


def has_clean_audio(episode: Episode) -> bool:
    return bool(episode.clean_audio_path and Path(episode.clean_audio_path).exists())


# --- Show settings (per-feed) ---

DEFAULT_FEED_SETTINGS: dict[str, Any] = {
    "auto_download": True,  # prepare latest episode automatically
    "keep_last": 5,  # how many recent cleaned episodes to keep
    "mode": "cut",  # "cut" = remove ads from file, "chapters" = mark only, ~0 storage
}


def get_feed_settings(feed: Feed) -> dict[str, Any]:
    merged = dict(DEFAULT_FEED_SETTINGS)
    try:
        data = json.loads(feed.settings_json or "{}")
    except json.JSONDecodeError:
        data = {}
    if isinstance(data, dict):
        for k, v in data.items():
            if k in merged:
                merged[k] = v
    return merged


def update_feed_settings(feed_id: int, updates: dict[str, Any]) -> Feed:
    feed = get_feed(feed_id)
    assert feed is not None
    current = get_feed_settings(feed)
    for k, v in updates.items():
        if k in DEFAULT_FEED_SETTINGS:
            current[k] = v
    with connect() as conn:
        conn.execute(
            "UPDATE feeds SET settings_json = ? WHERE id = ?",
            (json.dumps(current), feed_id),
        )
    updated = get_feed(feed_id)
    assert updated is not None
    return updated


def update_feed_artwork(feed_id: int, artwork_url: str) -> None:
    with connect() as conn:
        conn.execute("UPDATE feeds SET artwork_url = ? WHERE id = ?", (artwork_url, feed_id))


def delete_feed(feed_id: int) -> None:
    """Remove a show and all its files from the cache."""
    eps = list_episodes(feed_id)
    for ep in eps:
        for p in (ep.audio_path, ep.clean_audio_path):
            if p:
                try:
                    Path(p).unlink(missing_ok=True)
                except OSError:
                    pass
        try:
            from .stt import transcript_path_for

            transcript_path_for(ep.id).unlink(missing_ok=True)
        except OSError:
            pass
        try:
            from .cut import clean_path_for

            clean_path_for(ep.id).unlink(missing_ok=True)
        except OSError:
            pass
    with connect() as conn:
        conn.execute("DELETE FROM episodes WHERE feed_id = ?", (feed_id,))
        conn.execute("DELETE FROM feeds WHERE id = ?", (feed_id,))


# --- Global settings (tiny kv store) ---

GLOBAL_DEFAULTS: dict[str, str] = {
    "max_cache_gb": "3.0",
    "keep_last_n": "5",
    "delete_after_days": "14",
    "poll_minutes": "30",
}


def get_global_settings() -> dict[str, str]:
    merged = dict(GLOBAL_DEFAULTS)
    with connect() as conn:
        try:
            rows = conn.execute("SELECT key, value FROM kv").fetchall()
        except Exception:
            return merged
    for r in rows:
        if r["key"] in merged:
            merged[r["key"]] = r["value"]
    return merged


def set_global_settings(updates: dict[str, str]) -> dict[str, str]:
    with connect() as conn:
        for k, v in updates.items():
            if k in GLOBAL_DEFAULTS:
                conn.execute(
                    "INSERT INTO kv (key, value) VALUES (?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (k, str(v)),
                )
    return get_global_settings()


# --- Storage ---

def storage_stats() -> dict[str, Any]:
    total = 0
    per_feed: dict[int, int] = {}
    with connect() as conn:
        rows = conn.execute(
            "SELECT feed_id, clean_audio_path, audio_path, size_bytes FROM episodes"
        ).fetchall()
    for r in rows:
        size = int(r["size_bytes"] or 0)
        if size <= 0:
            # Fall back to on-disk size for rows written before size tracking.
            for p in (r["clean_audio_path"], r["audio_path"]):
                if p:
                    try:
                        size = Path(p).stat().st_size
                        break
                    except OSError:
                        continue
        total += size
        per_feed[r["feed_id"]] = per_feed.get(r["feed_id"], 0) + size
    try:
        limit_gb = float(get_global_settings().get("max_cache_gb") or 3.0)
    except ValueError:
        limit_gb = 3.0
    return {"bytes": total, "limit_bytes": int(limit_gb * 1024**3), "per_feed": per_feed}


def touch_served(episode_id: int) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE episodes SET last_served_at = ?, updated_at = ? WHERE id = ?",
            (_utc_now(), _utc_now(), episode_id),
        )
