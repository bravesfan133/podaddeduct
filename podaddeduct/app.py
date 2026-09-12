from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import shutil
from pathlib import Path

import httpx
from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
    Response,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import db
from podaddeduct import __version__ as APP_VERSION
from .chapters import build_chapters_json
from .config import settings
from .feeds import (
    entry_enclosure,
    entry_guid,
    entry_pub_date,
    fetch_feed_bytes,
    parse_feed,
    rewrite_feed_xml,
    slug_for_upstream,
)
from .process import enqueue_episode, ensure_worker, friendly_error
from .secrets import get_app_password, password_source, set_app_password, set_zen_api_key, zen_key_status

logger = logging.getLogger("podaddeduct")
logging.basicConfig(level=logging.INFO)

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    await ensure_worker()
    global _poll_task
    _poll_task = asyncio.create_task(_poll_loop())
    try:
        from .stt import backend_status

        st = backend_status()
        if st["ok"]:
            logger.info("STT backend OK: tool=%s script=%s model=%s ffmpeg=%s",
                        st["tool_resolved"], st["script"], st["model"], st["ffmpeg"])
        else:
            logger.warning("STT backend BROKEN: %s (see /api/health)", st)
    except Exception:
        logger.exception("STT self-check failed")
    yield
    if _poll_task:
        _poll_task.cancel()


app = FastAPI(title="podaddeduct", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

_poll_task: asyncio.Task | None = None


def configured_public_base() -> str:
    """Admin-configured base URL from the UI (kv) or env. "" = auto-detect."""
    raw = (db.runtime_str("public_base_url") or "").strip().rstrip("/")
    if not raw:
        return ""
    from urllib.parse import urlparse as _urlparse

    host = _urlparse(raw).hostname or ""
    if host in {"127.0.0.1", "localhost"} or not host:
        return ""
    return raw


def public_base(request: Request) -> str:
    """URL phones must use. Configured value wins; else the Host header the
    client actually hit (how tunnels work with no config); else LAN detect."""
    conf = configured_public_base()
    if conf:
        return conf
    host = request.headers.get("host") or ""
    if host and not host.startswith("127.0.0.1") and not host.startswith("localhost"):
        scheme = request.headers.get("x-forwarded-proto") or request.url.scheme or "http"
        return f"{scheme}://{host}".rstrip("/")
    return settings.resolved_public_base()


def _feed_artwork(parsed) -> str:
    try:
        if parsed.feed.get("image", {}).get("href"):
            return str(parsed.feed.image.href)
        if parsed.feed.get("itunes_image", {}).get("href"):
            return str(parsed.feed.itunes_image.href)
    except Exception:
        pass
    return ""


# Rendered-feed cache: podcast apps poll aggressively and big upstream
# feeds are megabytes to parse. 60s TTL per (show, base URL) keeps the N100
# responsive. In-flight cleaning still progresses; slightly stale lengths
# self-correct on the next poll.
_feed_cache: dict[tuple[str, str], tuple[float, bytes]] = {}
_FEED_TTL = 60.0


async def render_feed_response(feed: db.Feed, request: Request) -> Response:
    import time

    base = public_base(request)
    cache_key = (feed.slug, base)
    hit = _feed_cache.get(cache_key)
    if hit and time.monotonic() - hit[0] < _FEED_TTL:
        return Response(
            content=hit[1],
            media_type="application/rss+xml; charset=utf-8",
            headers={"Cache-Control": "no-cache"},
        )
    # Feed refreshes are cheap metadata syncs. Never bulk-queue here —
    # podcast apps refresh often and would flood the worker.
    try:
        raw = await fetch_feed_bytes(feed.upstream_url)
    except Exception as exc:
        if hit:
            return Response(
                content=hit[1],
                media_type="application/rss+xml; charset=utf-8",
                headers={"Cache-Control": "no-cache"},
            )
        raise HTTPException(502, f"Couldn't reach the publisher's feed: {exc}") from exc
    try:
        await sync_feed_episodes(feed, queue_recent=False, autodl=True, raw=raw)
    except Exception:
        logger.exception("Feed sync failed")
    feed = db.get_feed(feed.id) or feed
    parsed = parse_feed(raw, href=feed.upstream_url)
    episodes = {e.guid: e for e in db.list_episodes(feed.id)}
    xml = rewrite_feed_xml(
        parsed,
        feed=feed,
        episodes_by_guid=episodes,
        public_base=base,
    )
    content = xml.encode("utf-8")
    _feed_cache[cache_key] = (time.monotonic(), content)
    return Response(
        content=content,
        media_type="application/rss+xml; charset=utf-8",
        headers={"Cache-Control": "no-cache"},
    )


async def sync_feed_episodes(
    feed: db.Feed, *, queue_recent: bool = False, autodl: bool = False,
    queue_count: int | None = None, raw: bytes | None = None,
) -> list[db.Episode]:
    if raw is None:
        raw = await fetch_feed_bytes(feed.upstream_url)
    parsed = parse_feed(raw, href=feed.upstream_url)
    title = parsed.feed.get("title") or feed.title or feed.slug
    if title != feed.title:
        db.update_feed_title(feed.id, title)
        feed = db.get_feed(feed.id) or feed
    art = _feed_artwork(parsed)
    if art and art != (feed.artwork_url or ""):
        db.update_feed_artwork(feed.id, art)

    episodes = db.upsert_episodes_batch(
        feed.id,
        [
            {
                "guid": entry_guid(entry, enclosure),
                "title": str(getattr(entry, "title", None) or "Episode"),
                "enclosure_url": enclosure,
                "pub_date": entry_pub_date(entry),
            }
            for entry in parsed.entries
            if (enclosure := entry_enclosure(entry))
        ],
    )

    fs = db.get_feed_settings(feed)
    auto_on = fs.get("auto_download", True) and db.runtime_bool("auto_prepare_latest")
    should_queue = queue_recent or (autodl and auto_on)
    if should_queue and episodes:
        # Only the newest unprepared episode(s) — never the whole back-catalog.
        # Older episodes wait for an explicit Prepare tap (web UI or player).
        if queue_count is None:
            queue_count = 1 if (autodl and not queue_recent) else db.runtime_int("process_recent", minimum=1, maximum=20)
        target = episodes[: max(1, queue_count)]
        for ep in target:
            fresh = db.get_episode(ep.id)
            if not fresh:
                continue
            if fresh.status in {"pending", "error"} and not db.served_audio_path(fresh):
                await enqueue_episode(fresh.id)
    return episodes


async def get_or_create_feed(upstream_url: str, artwork: str = "") -> db.Feed:
    existing = db.get_feed_by_upstream(upstream_url)
    if existing:
        if artwork and not existing.artwork_url:
            db.update_feed_artwork(existing.id, artwork)
            existing = db.get_feed(existing.id) or existing
        return existing
    title = ""
    art = artwork
    try:
        raw = await fetch_feed_bytes(upstream_url)
        parsed = parse_feed(raw, href=upstream_url)
        title = str(parsed.feed.get("title") or "")
        art = art or _feed_artwork(parsed)
    except Exception as exc:
        logger.warning("Could not prefetch feed title: %s", exc)
    slug = slug_for_upstream(upstream_url, title)
    feed = db.create_feed(slug=slug, upstream_url=upstream_url, title=title)
    if art:
        db.update_feed_artwork(feed.id, art)
        feed = db.get_feed(feed.id) or feed
    # Prepare the latest episode right away so the first play is fast.
    # Just one: the rest wait for an explicit Prepare tap (disk stays small).
    try:
        await sync_feed_episodes(feed, queue_recent=True, queue_count=1)
    except Exception:
        logger.exception("Initial sync failed for %s", upstream_url)
    return feed


async def _poll_loop() -> None:
    """Background RSS check: metadata only (kilobytes). Audio downloads only
    for the newest unprepared episode per show with auto-download on."""
    await asyncio.sleep(10)
    while True:
        try:
            g = db.get_global_settings()
            try:
                interval = max(5, int(g.get("poll_minutes") or 30)) * 60
            except ValueError:
                interval = 30 * 60
            for feed in db.list_feeds():
                try:
                    await sync_feed_episodes(feed, queue_recent=False, autodl=True)
                except Exception:
                    logger.exception("Poll failed for %s", feed.slug)
            try:
                from .retain import run_janitor

                run_janitor()
            except Exception:
                logger.exception("Janitor failed")
        except Exception:
            logger.exception("Poll loop error")
        await asyncio.sleep(interval)


# --- Auth (family password for direct/LAN access; Cloudflare Access sits in front remotely) ---
#
# Default-deny: every route checks _authed except the explicit public
# allowlist (player feed XML + audio + login page). Podcast apps can't log
# in, so /feeds/*.xml and /audio/* stay open by design.

_login_attempts: dict[str, list[float]] = {}


def _login_allowed(ip: str) -> bool:
    import time

    now = time.monotonic()
    hits = [t for t in _login_attempts.get(ip, []) if now - t < 600]
    _login_attempts[ip] = hits
    return len(hits) < 5


def _login_failed(ip: str) -> None:
    import time

    _login_attempts.setdefault(ip, []).append(time.monotonic())


def _cookie_secure(request: Request) -> bool:
    """Secure cookie over https (tunnel) without breaking LAN-http login."""
    if request.url.scheme == "https":
        return True
    return (request.headers.get("x-forwarded-proto") or "").split(",")[0].strip() == "https"


def _authed(request: Request) -> bool:
    if not get_app_password():
        return True
    tok = request.cookies.get("podaddeduct_auth", "")
    want = hashlib.sha256(get_app_password().encode()).hexdigest()
    return hmac.compare_digest(tok, want)


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "login.html", {"error": ""})


@app.post("/login")
async def login(request: Request, password: str = Form("")) -> RedirectResponse:
    ip = request.client.host if request.client else "?"
    if not _login_allowed(ip):
        return templates.TemplateResponse(
            request, "login.html",
            {"error": "Too many tries — wait a few minutes."}, status_code=429,
        )
    if get_app_password() and hmac.compare_digest(password, get_app_password()):
        _login_attempts.pop(ip, None)
        resp = RedirectResponse("/", status_code=303)
        resp.set_cookie(
            "podaddeduct_auth",
            hashlib.sha256(get_app_password().encode()).hexdigest(),
            httponly=True,
            samesite="lax",
            secure=_cookie_secure(request),
        )
        return resp
    _login_failed(ip)
    return templates.TemplateResponse(request, "login.html", {"error": "Wrong password."})


# --- Pages ---

@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    if not _authed(request):
        return RedirectResponse("/login", status_code=303)
    feeds = db.list_feeds()
    cards = []
    for f in feeds:
        eps = db.list_episodes(f.id)
        ready = sum(1 for e in eps if e.status in {"ready", "manual"} and db.served_audio_path(e))
        working = sum(1 for e in eps if e.status == "working")
        cards.append(
            {
                "feed": f,
                "settings": db.get_feed_settings(f),
                "total": len(eps),
                "ready": ready,
                "working": working,
                "latest": eps[0] if eps else None,
            }
        )
    stats = db.storage_stats()
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "public_base": public_base(request),
            "cards": cards,
            "storage": stats,
            "storage_pct": round(100 * stats["bytes"] / max(1, stats["limit_bytes"]), 1),
        },
    )


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request) -> HTMLResponse:
    if not _authed(request):
        return RedirectResponse("/login", status_code=303)
    stats = db.storage_stats()
    eff = {
        "public_base_url": configured_public_base(),
        "process_recent": db.runtime_int("process_recent", minimum=1, maximum=20),
        "feed_item_limit": db.runtime_int("feed_item_limit", minimum=1, maximum=500),
        "min_ad_seconds": db.runtime_float("min_ad_seconds", minimum=1.0, maximum=300.0),
        "silence_snap_window": db.runtime_float("silence_snap_window", minimum=0.0, maximum=10.0),
        "delete_original_after_cut": db.runtime_bool("delete_original_after_cut"),
        "auto_prepare_latest": db.runtime_bool("auto_prepare_latest"),
        "zen_model": db.runtime_str("zen_model"),
        "zen_fallback_model": db.runtime_str("zen_fallback_model"),
        "zen_base_url": db.runtime_str("zen_base_url"),
        "zen_chunk_chars": db.runtime_int("zen_chunk_chars", minimum=1000),
        "stt_python": db.runtime_str("stt_python"),
        "stt_sidecar": db.runtime_str("stt_sidecar"),
        "stt_model": db.runtime_str("stt_model"),
    }
    return templates.TemplateResponse(
        request,
        "settings.html",
        {
            "public_base": public_base(request),
            "storage": stats,
            "storage_pct": round(100 * stats["bytes"] / max(1, stats["limit_bytes"]), 1),
            "global_settings": db.get_global_settings(),
            "effective": eff,
            "zen": zen_key_status(),
            "password_set": bool(get_app_password()),
            "password_source": password_source(),
            "app_version": APP_VERSION,
            "saved": request.query_params.get("saved"),
            "settings_error": request.query_params.get("err"),
        },
    )


@app.get("/api/search")
async def api_search(request: Request, q: str = "") -> JSONResponse:
    if not _authed(request):
        raise HTTPException(401, "Sign in first.")
    from .search import search_podcasts

    try:
        results = await search_podcasts(q)
    except Exception as exc:
        raise HTTPException(502, f"Search failed: {exc}") from exc
    return JSONResponse({"results": results})


@app.get("/api/storage")
async def api_storage(request: Request) -> JSONResponse:
    if not _authed(request):
        raise HTTPException(401, "Sign in first.")
    stats = db.storage_stats()
    return JSONResponse(stats)


@app.get("/api/health")
async def api_health(request: Request) -> JSONResponse:
    if not _authed(request):
        raise HTTPException(401, "Sign in first.")
    from .process import queue_depth, worker_state
    from .stt import backend_status

    stt = backend_status()
    try:
        usage = shutil.disk_usage(settings.data_dir)
        disk = {"total": usage.total, "free": usage.free}
    except OSError:
        disk = {"total": None, "free": None}
    return JSONResponse({
        "ok": bool(stt["ok"]),
        "stt": stt,
        "queue_depth": queue_depth(),
        "worker": worker_state(),
        "disk": disk,
    })


@app.get("/api/episodes/{episode_id}/status")
async def api_episode_status(episode_id: int, request: Request) -> JSONResponse:
    if not _authed(request):
        raise HTTPException(401, "Sign in first.")
    ep = db.get_episode(episode_id)
    if not ep:
        raise HTTPException(404, "Episode not found")
    ranges = db.get_ad_ranges(ep)
    saved = round(sum(max(0.0, r["end"] - r["start"]) for r in ranges), 1) if ranges else 0.0
    from .process import describe_job, queue_position, worker_state

    st = worker_state()
    job_text = ""
    queue_pos = queue_position(episode_id)
    cur = st.get("current") or {}
    if cur.get("episode_id") == episode_id:
        job_text = describe_job(cur)
        queue_pos = None
    elif queue_pos is not None:
        job_text = f"#{queue_pos} in line"
    return JSONResponse(
        {
            "status": ep.status,
            "has_audio": bool(db.served_audio_path(ep)),
            "has_clean": bool(ep.clean_audio_path and Path(ep.clean_audio_path).exists()),
            "ads": len(ranges),
            "saved_seconds": saved,
            "error": ep.error,
            "job_text": job_text,
            "queue_position": queue_pos,
        }
    )


@app.post("/settings/zen-key")
async def save_zen_key(
    request: Request,
    zen_api_key: str = Form(""),
    clear: str = Form(""),
) -> RedirectResponse:
    if not _authed(request):
        return RedirectResponse("/login", status_code=303)
    if clear:
        set_zen_api_key(None)
    else:
        set_zen_api_key(zen_api_key)
    ref = request.headers.get("referer") or "/"
    if "/episodes/" in ref:
        sep = "&" if "?" in ref else "?"
        return RedirectResponse(f"{ref}{sep}key_saved=1", status_code=303)
    from urllib.parse import urlparse as _urlparse2

    if _urlparse2(ref).path.startswith("/settings"):
        return RedirectResponse("/settings?saved=zen", status_code=303)
    return RedirectResponse("/?saved=zen", status_code=303)


@app.get("/api/zen-status")
async def api_zen_status(request: Request) -> JSONResponse:
    if not _authed(request):
        raise HTTPException(401, "Sign in first.")
    return JSONResponse(zen_key_status())


@app.post("/feeds")
async def add_feed(request: Request) -> RedirectResponse:
    if not _authed(request):
        return RedirectResponse("/login", status_code=303)
    form = await request.form()
    raw = str(form.get("upstream_url") or form.get("feed_url") or form.get("url") or "").strip()
    artwork = str(form.get("artwork") or "").strip()
    if not raw or not raw.startswith(("http://", "https://")):
        raise HTTPException(400, "That doesn't look like a podcast link.")
    feed = await get_or_create_feed(raw, artwork=artwork)
    return RedirectResponse(f"/shows/{feed.slug}", status_code=303)


@app.post("/settings/global")
async def save_global_settings(request: Request) -> RedirectResponse:
    if not _authed(request):
        return RedirectResponse("/login", status_code=303)
    form = await request.form()
    updates: dict[str, str] = {}
    errors: list[str] = []

    def num(key: str, *, minimum: float, maximum: float) -> None:
        raw = form.get(key)
        if raw in (None, ""):
            return
        try:
            val = float(str(raw))
        except ValueError:
            errors.append(f"{key} must be a number.")
            return
        if not (minimum <= val <= maximum):
            errors.append(f"{key} must be between {minimum:g} and {maximum:g}.")
            return
        updates[key] = str(raw)

    for k in ("max_cache_gb", "keep_last_n", "delete_after_days", "poll_minutes",
              "process_recent", "feed_item_limit", "min_ad_seconds",
              "silence_snap_window", "zen_chunk_chars"):
        num(k, minimum=0.1 if k in {"max_cache_gb", "min_ad_seconds", "silence_snap_window"} else 1,
            maximum=500 if k == "feed_item_limit" else (10**6 if k == "zen_chunk_chars" else 1440 if k == "poll_minutes" else 365 if k == "delete_after_days" else 50))

    for k in ("zen_model", "zen_fallback_model", "zen_base_url",
              "stt_python", "stt_sidecar", "stt_model"):
        raw = form.get(k)
        if raw not in (None, ""):
            updates[k] = str(raw).strip()

    raw_base = str(form.get("public_base_url") or "").strip().rstrip("/")
    if raw_base:
        import ipaddress as _ipaddress

        from urllib.parse import urlparse as _urlparse

        parts = _urlparse(raw_base if "://" in raw_base else f"https://{raw_base}")
        host = (parts.hostname or "").strip().lower()
        # Overcast fetches from its own servers: only a public https domain
        # works. Literal IPs, LAN names, and plain http are rejected with
        # a plain-language error instead of failing mysteriously later.
        is_ip = True
        try:
            _ipaddress.ip_address(host)
        except ValueError:
            is_ip = False
        if (
            not host
            or host in {"localhost"}
            or host.endswith(".local")
            or is_ip
            or (parts.scheme and parts.scheme != "https")
        ):
            errors.append("Public address must be a public https:// domain (for Overcast). LAN IPs don't work there.")
        else:
            updates["public_base_url"] = f"https://{host}" + (
                f":{parts.port}" if parts.port not in (None, 443) else ""
            )
    elif form.get("public_base_url") == "":
        updates["public_base_url"] = ""

    for k in ("auto_prepare_latest", "delete_original_after_cut"):
        if k in form:
            updates[k] = "true" if str(form.get(k)).lower() in {"1", "on", "true"} else "false"
        elif form.get("settings_form"):
            updates[k] = "false"

    if updates:
        db.set_global_settings(updates)
    from urllib.parse import quote as _quote

    dest = "/settings?saved=settings"
    if errors:
        dest += "&err=" + _quote("; ".join(errors))
    return RedirectResponse(dest, status_code=303)


@app.post("/settings/password")
async def save_app_password(request: Request) -> RedirectResponse:
    if not _authed(request):
        return RedirectResponse("/login", status_code=303)
    form = await request.form()
    current = str(form.get("current") or "")
    new = str(form.get("new") or "")
    confirm = str(form.get("confirm") or "")
    existing = get_app_password()
    from urllib.parse import quote as _quote2

    if existing and not hmac.compare_digest(current, existing):
        return RedirectResponse("/settings?err=" + _quote2("Current password didn't match."), status_code=303)
    if new != confirm:
        return RedirectResponse("/settings?err=" + _quote2("New passwords didn't match."), status_code=303)
    set_app_password(new)
    if not new:
        resp = RedirectResponse("/settings?saved=password-off", status_code=303)
        resp.delete_cookie("podaddeduct_auth")
        return resp
    return RedirectResponse("/settings?saved=password", status_code=303)


@app.get("/api/zen-models")
async def api_zen_models(request: Request) -> JSONResponse:
    if not _authed(request):
        raise HTTPException(401, "Sign in first.")
    from .seed import fetch_zen_models

    return JSONResponse(fetch_zen_models())


@app.post("/api/zen-test")
async def api_zen_test(request: Request) -> JSONResponse:
    if not _authed(request):
        raise HTTPException(401, "Sign in first.")
    try:
        body = await request.json()
    except Exception:
        body = {}
    model = str((body or {}).get("model") or "")
    from .seed import test_zen_connection

    return JSONResponse(test_zen_connection(model or None))


@app.get("/feeds/{slug}.xml")
async def feed_xml(slug: str, request: Request) -> Response:
    feed = db.get_feed_by_slug(slug)
    if not feed:
        raise HTTPException(404, "Show not found")
    return await render_feed_response(feed, request)


def _show_page(feed: db.Feed, request: Request) -> HTMLResponse:
    episodes = db.list_episodes(feed.id)
    rows = []
    for e in episodes:
        ranges = db.get_ad_ranges(e)
        saved = round(sum(max(0.0, r["end"] - r["start"]) for r in ranges), 1) if ranges else 0.0
        rows.append({"ep": e, "ads": len(ranges), "saved": saved})
    return templates.TemplateResponse(
        request,
        "feed.html",
        {
            "feed": feed,
            "feed_settings": db.get_feed_settings(feed),
            "keep_last_raw": db.feed_keep_last_raw(feed),
            "keep_last_global": db.global_keep_last(),
            "episodes": episodes,
            "rows": rows,
            "public_base": public_base(request),
            "player_url": f"{public_base(request)}/feeds/{feed.slug}.xml",
        },
    )


@app.get("/shows/{slug}", response_class=HTMLResponse)
async def show_page(slug: str, request: Request) -> HTMLResponse:
    if not _authed(request):
        return RedirectResponse("/login", status_code=303)  # type: ignore[return-value]
    feed = db.get_feed_by_slug(slug)
    if not feed:
        raise HTTPException(404, "Show not found")
    return _show_page(feed, request)


@app.get("/feeds/{slug}", response_class=HTMLResponse)
async def feed_status(slug: str, request: Request) -> HTMLResponse:
    if not _authed(request):
        return RedirectResponse("/login", status_code=303)  # type: ignore[return-value]
    feed = db.get_feed_by_slug(slug)
    if not feed:
        raise HTTPException(404, "Show not found")
    return _show_page(feed, request)


@app.post("/shows/{slug}/settings")
async def save_show_settings(slug: str, request: Request) -> RedirectResponse:
    if not _authed(request):
        return RedirectResponse("/login", status_code=303)
    feed = db.get_feed_by_slug(slug)
    if not feed:
        raise HTTPException(404, "Show not found")
    form = await request.form()
    updates: dict = {}
    if "auto_download" in form:
        updates["auto_download"] = str(form.get("auto_download")).lower() in {"1", "on", "true"}
    elif form.get("settings_form"):
        updates["auto_download"] = False
    if form.get("keep_last") in (None, ""):
        # Empty = inherit the global number.
        updates["keep_last"] = None
    else:
        try:
            updates["keep_last"] = max(1, min(50, int(str(form.get("keep_last")))))
        except ValueError:
            pass
    if form.get("mode") in ("cut", "chapters"):
        updates["mode"] = str(form.get("mode"))
    if updates:
        db.update_feed_settings(feed.id, updates)
        from .retain import run_janitor

        try:
            run_janitor()
        except Exception:
            pass
    return RedirectResponse(f"/shows/{slug}?saved=1", status_code=303)


@app.post("/shows/{slug}/prepare")
async def prepare_next(slug: str, request: Request) -> RedirectResponse:
    """Queue the next few unprepared episodes, newest first (couch taps)."""
    if not _authed(request):
        return RedirectResponse("/login", status_code=303)
    feed = db.get_feed_by_slug(slug)
    if not feed:
        raise HTTPException(404, "Show not found")
    form = await request.form()
    try:
        n = max(1, min(20, int(str(form.get("n") or 5))))
    except ValueError:
        n = 5
    queued = 0
    for ep in db.list_episodes(feed.id):
        if queued >= n:
            break
        fresh = db.get_episode(ep.id)
        if not fresh or db.served_audio_path(fresh):
            continue
        if fresh.status not in {"pending", "error"}:
            continue
        if await enqueue_episode(ep.id):
            queued += 1
    return RedirectResponse(f"/shows/{slug}?queued={queued}", status_code=303)


@app.post("/shows/{slug}/refresh")
async def refresh_show(slug: str, request: Request) -> RedirectResponse:
    if not _authed(request):
        return RedirectResponse("/login", status_code=303)
    feed = db.get_feed_by_slug(slug)
    if not feed:
        raise HTTPException(404, "Show not found")
    try:
        await sync_feed_episodes(feed, queue_recent=True)
    except Exception as exc:
        raise HTTPException(502, f"Couldn't reach the publisher's feed: {exc}") from exc
    return RedirectResponse(f"/shows/{slug}", status_code=303)


@app.post("/shows/{slug}/delete")
async def delete_show(slug: str, request: Request) -> RedirectResponse:
    if not _authed(request):
        return RedirectResponse("/login", status_code=303)
    feed = db.get_feed_by_slug(slug)
    if not feed:
        raise HTTPException(404, "Show not found")
    db.delete_feed(feed.id)
    return RedirectResponse("/", status_code=303)


@app.get("/episodes/{episode_id}")
async def episode_page(episode_id: int, request: Request) -> HTMLResponse:
    if not _authed(request):
        return RedirectResponse("/login", status_code=303)  # type: ignore[return-value]
    ep = db.get_episode(episode_id)
    if not ep:
        raise HTTPException(404, "Episode not found")
    feed = db.get_feed(ep.feed_id)
    ranges = db.get_ad_ranges(ep)
    saved = round(sum(max(0.0, r["end"] - r["start"]) for r in ranges), 1)
    return templates.TemplateResponse(
        request,
        "episode.html",
        {
            "episode": ep,
            "feed": feed,
            "ranges": ranges,
            "ranges_text": "\n".join(f"{r['start']}-{r['end']}" for r in ranges),
            "saved_seconds": saved,
            "public_base": public_base(request),
            "zen": zen_key_status(),
            "zen_model": db.runtime_str("zen_model"),
            "key_saved": request.query_params.get("key_saved"),
            "has_clean": db.has_clean_audio(ep),
            "has_audio": bool(db.served_audio_path(ep)),
            "friendly_error": friendly_error(ep.error),
        },
    )


@app.post("/episodes/{episode_id}/recheck")
async def recheck_episode(episode_id: int, request: Request) -> RedirectResponse:
    if not _authed(request):
        return RedirectResponse("/login", status_code=303)
    ep = db.get_episode(episode_id)
    if not ep:
        raise HTTPException(404, "Episode not found")
    await enqueue_episode(episode_id, reseed=True)
    return RedirectResponse(f"/episodes/{episode_id}", status_code=303)


@app.post("/episodes/{episode_id}/reseed")
async def reseed_episode(episode_id: int, request: Request) -> RedirectResponse:
    return await recheck_episode(episode_id, request)


@app.post("/episodes/{episode_id}/ranges")
async def save_ranges(
    episode_id: int,
    request: Request,
    ranges_text: str = Form(""),
) -> RedirectResponse:
    if not _authed(request):
        return RedirectResponse("/login", status_code=303)
    from .process import apply_manual_ranges

    ep = db.get_episode(episode_id)
    if not ep:
        raise HTTPException(404, "Episode not found")
    parsed: list[dict[str, float]] = []
    for line in ranges_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.replace(",", " ")
        if "-" in line:
            a, b = line.split("-", 1)
        else:
            parts = line.split()
            if len(parts) != 2:
                raise HTTPException(400, f"Couldn't read this line: {line} (use 12.5-34.0)")
            a, b = parts
        try:
            start, end = float(a), float(b)
        except ValueError:
            raise HTTPException(400, f"Couldn't read this line: {line} (use 12.5-34.0)") from None
        if end <= start:
            raise HTTPException(400, f"End must be after start: {line}")
        parsed.append({"start": start, "end": end})
    try:
        apply_manual_ranges(episode_id, parsed)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return RedirectResponse(f"/episodes/{episode_id}", status_code=303)


@app.api_route("/audio/{episode_id}", methods=["GET", "HEAD"])
async def audio(episode_id: int, request: Request) -> Response:
    """Serve cleaned audio. Strict: unprocessed bytes never leave this server.

    - HEAD: report upstream headers only, queue nothing (feed refreshes).
    - GET with a clean file: serve it (206 range support via FileResponse).
    - GET with a finished original and nothing to cut (no ads found, or the
      show is in chapters mode): serve the original.
    - GET otherwise: queue at the front and answer 503 + Retry-After so the
      player keeps retrying until the clean file is ready. Never redirects
      to the publisher's with-ads file.
    """
    ep = db.get_episode(episode_id)
    if not ep:
        raise HTTPException(404, "Episode not found")

    if request.method == "HEAD":
        try:
            async with httpx.AsyncClient(
                timeout=30.0, follow_redirects=True,
                headers={"User-Agent": settings.user_agent},
            ) as client:
                resp = await client.head(ep.enclosure_url)
                headers = {}
                for k in ("content-type", "content-length", "accept-ranges"):
                    if resp.headers.get(k):
                        headers[k] = resp.headers[k]
                return Response(status_code=200, headers=headers)
        except Exception:
            return Response(status_code=200, headers={"content-type": "audio/mpeg"})

    has_clean = bool(ep.clean_audio_path and Path(ep.clean_audio_path).exists())
    served = db.served_audio_path(ep)
    finished = ep.status in {"ready", "manual"}
    if served and (has_clean or finished):
        db.touch_served(episode_id)
        try:
            size = served.stat().st_size
            if size > 0:
                db.update_episode(episode_id, size_bytes=size)
        except OSError:
            pass
        return FileResponse(served, media_type="audio/mpeg", filename=f"{episode_id}.mp3")

    # Not clean yet: jump the queue and tell the player to retry.
    # It shows "downloading" meanwhile and picks up the clean file
    # on a later attempt — ads never play.
    await enqueue_episode(episode_id, priority=True)
    return Response(
        status_code=503,
        headers={"Retry-After": "60", "Cache-Control": "no-store"},
        content="Episode is being cleaned — retry shortly.",
        media_type="text/plain",
    )


@app.get("/chapters/{episode_id}.json")
async def chapters_json(episode_id: int, request: Request) -> JSONResponse:
    if not _authed(request):
        raise HTTPException(401, "Sign in first.")
    ep = db.get_episode(episode_id)
    if not ep:
        raise HTTPException(404, "Episode not found")
    return JSONResponse(build_chapters_json(ep))


@app.get("/export.opml")
async def export_opml(request: Request) -> PlainTextResponse:
    if not _authed(request):
        return RedirectResponse("/login", status_code=303)  # type: ignore[return-value]
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<opml version="1.0"><head><title>podaddeduct</title></head><body>',
    ]
    for f in db.list_feeds():
        title = (f.title or f.slug).replace('"', "")
        lines.append(
            f'<outline type="rss" text="{title}" title="{title}" xmlUrl="{f.upstream_url}"/>'
        )
    lines.append("</body></opml>")
    return PlainTextResponse("\n".join(lines), media_type="text/xml")


@app.post("/import-opml")
async def import_opml(request: Request, file: UploadFile) -> RedirectResponse:
    if not _authed(request):
        return RedirectResponse("/login", status_code=303)
    import re
    import xml.etree.ElementTree as ET

    body = (await file.read()).decode("utf-8", "ignore")
    urls: list[str] = []
    try:
        root = ET.fromstring(body)
        for el in root.iter("outline"):
            u = el.get("xmlUrl") or el.get("url") or ""
            if u.startswith(("http://", "https://")):
                urls.append(u)
    except ET.ParseError:
        urls = re.findall(r'https?://[^\s"\'<>]+', body)
    for u in urls[:100]:
        try:
            await get_or_create_feed(u)
        except Exception:
            logger.exception("OPML import failed for %s", u)
    return RedirectResponse("/", status_code=303)
