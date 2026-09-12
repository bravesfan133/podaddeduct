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
    description: str = ""
    author: str = ""


@dataclass
class Episode:
    id: int
    feed_id: int
    guid: str
    title: str
    enclosure_url: str
    pub_date: str | None
    pub_ts: float = 0.0
    transcript_url: str | None = None
    transcript_type: str | None = None
    chapters_url: str | None = None
    duration_seconds: float | None = None
    status: str = "pending"
    audio_path: str | None = None
    ad_ranges_json: str = "[]"
    content_fp_path: str | None = None
    clean_audio_path: str | None = None
    error: str | None = None
    updated_at: str = ""
    size_bytes: int = 0
    last_served_at: str | None = None
    description: str = ""


def pub_ts_for(pub_date: str | None) -> float:
    """Sortable publish timestamp. 0.0 when the date is missing/unparseable."""
    if not pub_date:
        return 0.0
    try:
        from email.utils import parsedate_to_datetime

        dt = parsedate_to_datetime(str(pub_date))
    except (TypeError, ValueError, OverflowError):
        return 0.0
    try:
        return dt.timestamp()
    except (ValueError, OverflowError, OSError):
        return 0.0


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
    pub_ts REAL NOT NULL DEFAULT 0,
    transcript_url TEXT,
    transcript_type TEXT,
    chapters_url TEXT,
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
    if "description" not in feed_cols:
        conn.execute("ALTER TABLE feeds ADD COLUMN description TEXT NOT NULL DEFAULT ''")
    if "author" not in feed_cols:
        conn.execute("ALTER TABLE feeds ADD COLUMN author TEXT NOT NULL DEFAULT ''")
    ep_cols = {row[1] for row in conn.execute("PRAGMA table_info(episodes)").fetchall()}
    if "clean_audio_path" not in ep_cols:
        conn.execute("ALTER TABLE episodes ADD COLUMN clean_audio_path TEXT")
    if "size_bytes" not in ep_cols:
        conn.execute("ALTER TABLE episodes ADD COLUMN size_bytes INTEGER NOT NULL DEFAULT 0")
    if "last_served_at" not in ep_cols:
        conn.execute("ALTER TABLE episodes ADD COLUMN last_served_at TEXT")
    if "pub_ts" not in ep_cols:
        conn.execute("ALTER TABLE episodes ADD COLUMN pub_ts REAL NOT NULL DEFAULT 0")
    if "transcript_url" not in ep_cols:
        conn.execute("ALTER TABLE episodes ADD COLUMN transcript_url TEXT")
    if "transcript_type" not in ep_cols:
        conn.execute("ALTER TABLE episodes ADD COLUMN transcript_type TEXT")
    if "chapters_url" not in ep_cols:
        conn.execute("ALTER TABLE episodes ADD COLUMN chapters_url TEXT")
    if "description" not in ep_cols:
        conn.execute("ALTER TABLE episodes ADD COLUMN description TEXT NOT NULL DEFAULT ''")
    # Backfill sortable timestamps for rows written before pub_ts existed.
    try:
        stale = conn.execute(
            "SELECT id, pub_date FROM episodes WHERE pub_ts = 0 AND pub_date IS NOT NULL AND pub_date != ''"
        ).fetchall()
    except Exception:
        stale = []
    for row in stale:
        ts = pub_ts_for(row["pub_date"])
        if ts > 0:
            conn.execute("UPDATE episodes SET pub_ts = ? WHERE id = ?", (ts, row["id"]))
    _migrate_gemini_detector(conn)


def _migrate_gemini_detector(conn: sqlite3.Connection) -> None:
    """Map leftover OpenCode / Zen model IDs to gemini-3.5-flash."""
    flag = conn.execute(
        "SELECT value FROM kv WHERE key = ?", ("_migrated_gemini_detector_v2",)
    ).fetchone()
    if flag:
        return
    row = conn.execute("SELECT value FROM kv WHERE key = ?", ("gemini_model",)).fetchone()
    val = ((row["value"] if row else "") or "").strip()
    legacy = (
        not val
        or val.startswith("opencode/")
        or "nemotron" in val.lower()
        or "deepseek" in val.lower()
        or val in {"gemini-2.0-flash", "gemini-2.5-flash"}
    )
    if legacy:
        conn.execute(
            "INSERT INTO kv (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            ("gemini_model", "gemini-3.5-flash"),
        )
    for dead in ("opencode_server_url", "opencode_fallback"):
        conn.execute("DELETE FROM kv WHERE key = ?", (dead,))
    conn.execute(
        "INSERT INTO kv (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        ("_migrated_gemini_detector_v2", "1"),
    )


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
        description=row["description"] if "description" in keys else "",
        author=row["author"] if "author" in keys else "",
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
        pub_ts=float(row["pub_ts"] or 0.0) if "pub_ts" in keys else 0.0,
        transcript_url=row["transcript_url"] if "transcript_url" in keys else None,
        transcript_type=row["transcript_type"] if "transcript_type" in keys else None,
        chapters_url=row["chapters_url"] if "chapters_url" in keys else None,
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
        description=row["description"] if "description" in keys else "",
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


def update_feed_channel(feed_id: int, *, description: str | None = None, author: str | None = None) -> None:
    """Persist channel-level metadata used by the custom player feed."""
    updates: list[str] = []
    values: list[Any] = []
    if description is not None:
        updates.append("description = ?")
        values.append(description)
    if author is not None:
        updates.append("author = ?")
        values.append(author)
    if not updates:
        return
    values.append(feed_id)
    with connect() as conn:
        conn.execute(f"UPDATE feeds SET {', '.join(updates)} WHERE id = ?", values)


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
    # Newest first by actual publish date. (Row id order is oldest-last:
    # the newest upstream item is inserted first, so id order alone
    # shows oldest first — hence pub_ts, not id.)
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM episodes WHERE feed_id = ? ORDER BY pub_ts DESC, id ASC",
            (feed_id,),
        ).fetchall()
    return [_episode_from_row(r) for r in rows]


def count_episodes(feed_id: int, *, status_filter: str | None = None) -> int:
    """Count episodes for a show. status_filter: ready | pending | all/None."""
    where, args = _episode_filter_sql(feed_id, status_filter)
    with connect() as conn:
        row = conn.execute(f"SELECT COUNT(*) AS n FROM episodes WHERE {where}", args).fetchone()
    return int(row["n"] if row else 0)


def list_episodes_page(
    feed_id: int,
    *,
    offset: int = 0,
    limit: int = 50,
    status_filter: str | None = None,
) -> list[Episode]:
    """Paginated newest-first episode list for the show page."""
    where, args = _episode_filter_sql(feed_id, status_filter)
    limit = max(1, min(200, int(limit)))
    offset = max(0, int(offset))
    with connect() as conn:
        rows = conn.execute(
            f"SELECT * FROM episodes WHERE {where} ORDER BY pub_ts DESC, id ASC LIMIT ? OFFSET ?",
            (*args, limit, offset),
        ).fetchall()
    return [_episode_from_row(r) for r in rows]


def _episode_filter_sql(feed_id: int, status_filter: str | None) -> tuple[str, tuple]:
    filt = (status_filter or "all").strip().lower()
    if filt == "ready":
        return "feed_id = ? AND status IN ('ready', 'manual')", (feed_id,)
    if filt == "pending":
        return "feed_id = ? AND status NOT IN ('ready', 'manual')", (feed_id,)
    return "feed_id = ?", (feed_id,)


def list_ready_episodes(feed_id: int) -> list[Episode]:
    """Episodes that already have a processed/clean file on disk (or status)."""
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM episodes
            WHERE feed_id = ?
              AND (status IN ('ready', 'manual') OR clean_audio_path IS NOT NULL)
            ORDER BY pub_ts DESC, id ASC
            """,
            (feed_id,),
        ).fetchall()
    return [_episode_from_row(r) for r in rows]


def upsert_episode(
    feed_id: int,
    guid: str,
    title: str,
    enclosure_url: str,
    pub_date: str | None,
    transcript_url: str | None = None,
    transcript_type: str | None = None,
    chapters_url: str | None = None,
) -> Episode:
    existing = get_episode_by_guid(feed_id, guid)
    now = _utc_now()
    ts = pub_ts_for(pub_date)
    if existing:
        enclosure_changed = existing.enclosure_url != enclosure_url
        with connect() as conn:
            if enclosure_changed:
                # New file for same GUID — re-download + re-detect
                conn.execute(
                    """
                    UPDATE episodes
                    SET title = ?, enclosure_url = ?, pub_date = ?, pub_ts = ?, updated_at = ?,
                        transcript_url = ?, transcript_type = ?, chapters_url = ?,
                        audio_path = NULL, status = 'pending', error = NULL,
                        ad_ranges_json = '[]', clean_audio_path = NULL
                    WHERE id = ?
                    """,
                    (title, enclosure_url, pub_date, ts, now, transcript_url, transcript_type,
                     chapters_url, existing.id),
                )
            else:
                conn.execute(
                    """
                    UPDATE episodes
                    SET title = ?, enclosure_url = ?, pub_date = ?, pub_ts = ?, updated_at = ?,
                        transcript_url = ?, transcript_type = ?, chapters_url = ?
                    WHERE id = ?
                    """,
                    (title, enclosure_url, pub_date, ts, now, transcript_url, transcript_type,
                     chapters_url, existing.id),
                )
        ep = get_episode(existing.id)
        assert ep is not None
        return ep

    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO episodes (
                feed_id, guid, title, enclosure_url, pub_date, pub_ts,
                transcript_url, transcript_type, chapters_url,
                status, ad_ranges_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', '[]', ?)
            """,
            (feed_id, guid, title, enclosure_url, pub_date, ts, transcript_url, transcript_type,
             chapters_url, now),
        )
        episode_id = int(cur.lastrowid)
    ep = get_episode(episode_id)
    assert ep is not None
    return ep


def clip_description(value: str | None, limit: int = 2000) -> str:
    """Collapse whitespace and truncate long show notes for SQLite storage."""
    if not value:
        return ""
    cleaned = " ".join(str(value).split())
    return cleaned[:limit] if len(cleaned) > limit else cleaned


def upsert_episodes_batch(feed_id: int, items: list[dict]) -> list[Episode]:
    """Upsert many episodes in a single transaction, preserving input order.

    A big show (1600+ episodes) upserted row-by-row costs one open/commit/
    fsync per row — ~50s on slow disks, on EVERY feed view. One transaction
    with a single commit: well under a second. Same per-row semantics as
    upsert_episode (enclosure change resets processing state).
    Items: dicts with guid/title/enclosure_url/pub_date keys
    (plus optional transcript_url/transcript_type/chapters_url/description).
    """
    now = _utc_now()
    rows = [
        (
            str(it["guid"]),
            str(it.get("title") or "Episode"),
            str(it["enclosure_url"]),
            it.get("pub_date"),
            pub_ts_for(it.get("pub_date")),
            it.get("transcript_url"),
            it.get("transcript_type"),
            it.get("chapters_url"),
            clip_description(it.get("description")),
        )
        for it in items
        if it.get("guid") and it.get("enclosure_url")
    ]
    if not rows:
        return []
    with connect() as conn:
        conn.executemany(
            """
            INSERT INTO episodes (
                feed_id, guid, title, enclosure_url, pub_date, pub_ts,
                transcript_url, transcript_type, chapters_url, description,
                status, ad_ranges_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', '[]', ?)
            ON CONFLICT(feed_id, guid) DO UPDATE SET
                title = excluded.title,
                enclosure_url = excluded.enclosure_url,
                pub_date = excluded.pub_date,
                pub_ts = excluded.pub_ts,
                transcript_url = excluded.transcript_url,
                transcript_type = excluded.transcript_type,
                chapters_url = excluded.chapters_url,
                description = excluded.description,
                updated_at = excluded.updated_at,
                audio_path = CASE
                    WHEN episodes.enclosure_url != excluded.enclosure_url THEN NULL
                    ELSE episodes.audio_path END,
                clean_audio_path = CASE
                    WHEN episodes.enclosure_url != excluded.enclosure_url THEN NULL
                    ELSE episodes.clean_audio_path END,
                ad_ranges_json = CASE
                    WHEN episodes.enclosure_url != excluded.enclosure_url THEN '[]'
                    ELSE episodes.ad_ranges_json END,
                status = CASE
                    WHEN episodes.enclosure_url != excluded.enclosure_url THEN 'pending'
                    ELSE episodes.status END,
                error = CASE
                    WHEN episodes.enclosure_url != excluded.enclosure_url THEN NULL
                    ELSE episodes.error END
            """,
            [
                (feed_id, guid, title, enc, pub, ts, turl, ttype, curl, desc, now)
                for (guid, title, enc, pub, ts, turl, ttype, curl, desc) in rows
            ],
        )
        sel = conn.execute(
            f"SELECT * FROM episodes WHERE feed_id = ? AND guid IN ({','.join('?' * len(rows))})",
            [feed_id] + [r[0] for r in rows],
        ).fetchall()
    by_guid = {r["guid"]: _episode_from_row(r) for r in sel}
    return [by_guid[r[0]] for r in rows if r[0] in by_guid]


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
    # None = inherit the global keep_last_n. New shows always inherit;
    # clearing the field on the show page reverts to inherit.
    "keep_last": None,  # how many recent cleaned episodes to keep
}


def global_keep_last() -> int:
    try:
        return max(1, int(get_global_settings().get("keep_last_n") or 5))
    except ValueError:
        return 5


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
    if merged.get("keep_last") is None:
        merged["keep_last"] = global_keep_last()
    return merged


def feed_keep_last_raw(feed: Feed) -> Any:
    """The show's own override, or None when inheriting the global value."""
    try:
        data = json.loads(feed.settings_json or "{}")
    except json.JSONDecodeError:
        return None
    return data.get("keep_last") if isinstance(data, dict) else None


def update_feed_settings(feed_id: int, updates: dict[str, Any]) -> Feed:
    feed = get_feed(feed_id)
    assert feed is not None
    # Start from the raw stored JSON (not the resolved view) so an
    # inherited keep_last=None stays inherited when other keys change.
    try:
        raw = json.loads(feed.settings_json or "{}")
    except json.JSONDecodeError:
        raw = {}
    if not isinstance(raw, dict):
        raw = {}
    # Persist the raw overrides (keep_last=None stays inherited) rather
    # than the resolved view, so inherit sticks when other keys change.
    stored: dict[str, Any] = {k: v for k, v in raw.items() if k in DEFAULT_FEED_SETTINGS}
    for k, v in updates.items():
        if k in DEFAULT_FEED_SETTINGS:
            stored[k] = v
    with connect() as conn:
        conn.execute(
            "UPDATE feeds SET settings_json = ? WHERE id = ?",
            (json.dumps(stored), feed_id),
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
#
# Precedence for every runtime setting: UI value (kv) > environment (.env)
# > code default (config.Settings). Empty kv value ("") means "inherit".
# Only HOST/PORT/DATA_DIR stay env-only (they bind before the app runs).

GLOBAL_DEFAULTS: dict[str, str] = {
    "max_cache_gb": "3.0",
    "keep_last_n": "5",
    "delete_after_days": "14",
    "poll_minutes": "30",
    # Server / sharing
    "public_base_url": "",
    # Processing
    "process_recent": "",
    "feed_item_limit": "",
    "min_ad_seconds": "",
    "silence_snap_window": "",
    "delete_original_after_cut": "",
    "auto_prepare_latest": "",
    # Ad detection (Gemini)
    "gemini_model": "",
    # Transcription backend
    "stt_python": "",
    "stt_sidecar": "",
    "stt_model": "",
}

# kv key -> Settings attribute used when kv is empty (env-then-default).
_RUNTIME_ATTRS: dict[str, str] = {
    "max_cache_gb": "max_cache_gb",
    "keep_last_n": "keep_last_n",
    "delete_after_days": "delete_after_days",
    "poll_minutes": "poll_minutes",
    "public_base_url": "public_base_url",
    "process_recent": "process_recent",
    "feed_item_limit": "feed_item_limit",
    "min_ad_seconds": "min_ad_seconds",
    "silence_snap_window": "silence_snap_window",
    "delete_original_after_cut": "delete_original_after_cut",
    "auto_prepare_latest": "auto_prepare_latest",
    "gemini_model": "gemini_model",
    "stt_python": "stt_python",
    "stt_sidecar": "stt_sidecar",
    "stt_model": "stt_model",
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


def clear_global_settings(keys: list[str]) -> dict[str, str]:
    """Reset keys to inherit (env-then-default) by removing kv overrides."""
    with connect() as conn:
        for k in keys:
            if k in GLOBAL_DEFAULTS:
                conn.execute("DELETE FROM kv WHERE key = ?", (k,))
    return get_global_settings()


def _env_fallback(key: str) -> str:
    attr = _RUNTIME_ATTRS.get(key)
    if not attr:
        return ""
    val = getattr(settings, attr, "")
    return "" if val is None else str(val)


def runtime_str(key: str) -> str:
    """Effective string value: UI (kv) wins, then env, then code default."""
    if key in GLOBAL_DEFAULTS:
        with connect() as conn:
            row = conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        if row and (row["value"] or "").strip():
            return str(row["value"]).strip()
    return _env_fallback(key).strip()


def runtime_int(key: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
    try:
        val = int(float(runtime_str(key)))
    except ValueError:
        val = 0
    if minimum is not None:
        val = max(minimum, val)
    if maximum is not None:
        val = min(maximum, val)
    return val


def runtime_float(key: str, *, minimum: float | None = None, maximum: float | None = None) -> float:
    try:
        val = float(runtime_str(key))
    except ValueError:
        val = 0.0
    if minimum is not None:
        val = max(minimum, val)
    if maximum is not None:
        val = min(maximum, val)
    return val


def runtime_bool(key: str) -> bool:
    raw = runtime_str(key).lower()
    if raw in {"1", "true", "on", "yes"}:
        return True
    if raw in {"0", "false", "off", "no"}:
        return False
    attr = _RUNTIME_ATTRS.get(key)
    return bool(getattr(settings, attr, False))


def models_to_try() -> list[str]:
    """Effective Gemini model list (single primary for now)."""
    mid = runtime_str("gemini_model").strip()
    return [mid] if mid else []


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
