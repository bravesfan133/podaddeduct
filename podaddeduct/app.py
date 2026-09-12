from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
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
from .process import enqueue_episode, ensure_worker
from .secrets import set_zen_api_key, zen_key_status

logger = logging.getLogger("podaddeduct")
logging.basicConfig(level=logging.INFO)

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    await ensure_worker()
    global _poll_task
    _poll_task = asyncio.create_task(_poll_loop())
    yield
    if _poll_task:
        _poll_task.cancel()


app = FastAPI(title="podaddeduct", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

_poll_task: asyncio.Task | None = None


def public_base(request: Request) -> str:
    """URL phones must use. Prefer the Host header the client actually hit."""
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


async def render_feed_response(feed: db.Feed, request: Request) -> Response:
    # Feed refreshes are cheap metadata syncs. Never bulk-queue here —
    # podcast apps refresh often and would flood the worker.
    try:
        await sync_feed_episodes(feed, queue_recent=False, autodl=True)
    except Exception:
        logger.exception("Feed sync failed")
    feed = db.get_feed(feed.id) or feed
    try:
        raw = await fetch_feed_bytes(feed.upstream_url)
    except Exception as exc:
        raise HTTPException(502, f"Couldn't reach the publisher's feed: {exc}") from exc
    parsed = parse_feed(raw, href=feed.upstream_url)
    episodes = {e.guid: e for e in db.list_episodes(feed.id)}
    xml = rewrite_feed_xml(
        parsed,
        feed=feed,
        episodes_by_guid=episodes,
        public_base=public_base(request),
    )
    return Response(
        content=xml,
        media_type="application/rss+xml; charset=utf-8",
        headers={"Cache-Control": "no-cache"},
    )


async def sync_feed_episodes(
    feed: db.Feed, *, queue_recent: bool = False, autodl: bool = False
) -> list[db.Episode]:
    raw = await fetch_feed_bytes(feed.upstream_url)
    parsed = parse_feed(raw, href=feed.upstream_url)
    title = parsed.feed.get("title") or feed.title or feed.slug
    if title != feed.title:
        db.update_feed_title(feed.id, title)
        feed = db.get_feed(feed.id) or feed
    art = _feed_artwork(parsed)
    if art and art != (feed.artwork_url or ""):
        db.update_feed_artwork(feed.id, art)

    episodes: list[db.Episode] = []
    for entry in parsed.entries:
        enclosure = entry_enclosure(entry)
        if not enclosure:
            continue
        guid = entry_guid(entry, enclosure)
        ep = db.upsert_episode(
            feed.id,
            guid=guid,
            title=str(getattr(entry, "title", None) or "Episode"),
            enclosure_url=enclosure,
            pub_date=entry_pub_date(entry),
        )
        episodes.append(ep)

    fs = db.get_feed_settings(feed)
    auto_on = fs.get("auto_download", True) and settings.auto_prepare_latest
    should_queue = queue_recent or (autodl and auto_on)
    if should_queue and episodes:
        # Only the newest unprepared episode — never the whole back-catalog.
        target = episodes[:1] if autodl and not queue_recent else episodes[: settings.process_recent]
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
    try:
        await sync_feed_episodes(feed, queue_recent=True)
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


# --- Auth (optional shared password for tunnel exposure) ---

def _authed(request: Request) -> bool:
    if not settings.app_password:
        return True
    tok = request.cookies.get("podaddeduct_auth", "")
    want = hashlib.sha256(settings.app_password.encode()).hexdigest()
    return hmac.compare_digest(tok, want)


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "login.html", {"error": ""})


@app.post("/login")
async def login(request: Request, password: str = Form("")) -> RedirectResponse:
    if settings.app_password and hmac.compare_digest(password, settings.app_password):
        resp = RedirectResponse("/", status_code=303)
        resp.set_cookie(
            "podaddeduct_auth",
            hashlib.sha256(settings.app_password.encode()).hexdigest(),
            httponly=True,
            samesite="lax",
        )
        return resp
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
            "global_settings": db.get_global_settings(),
            "zen": zen_key_status(),
            "zen_model": settings.zen_model,
            "saved": request.query_params.get("saved"),
        },
    )


@app.get("/api/search")
async def api_search(q: str = "") -> JSONResponse:
    from .search import search_podcasts

    try:
        results = await search_podcasts(q)
    except Exception as exc:
        raise HTTPException(502, f"Search failed: {exc}") from exc
    return JSONResponse({"results": results})


@app.get("/api/storage")
async def api_storage() -> JSONResponse:
    stats = db.storage_stats()
    return JSONResponse(stats)


@app.get("/api/episodes/{episode_id}/status")
async def api_episode_status(episode_id: int) -> JSONResponse:
    ep = db.get_episode(episode_id)
    if not ep:
        raise HTTPException(404, "Episode not found")
    ranges = db.get_ad_ranges(ep)
    saved = round(sum(max(0.0, r["end"] - r["start"]) for r in ranges), 1) if ranges else 0.0
    return JSONResponse(
        {
            "status": ep.status,
            "has_audio": bool(db.served_audio_path(ep)),
            "has_clean": bool(ep.clean_audio_path and Path(ep.clean_audio_path).exists()),
            "ads": len(ranges),
            "saved_seconds": saved,
            "error": ep.error,
        }
    )


@app.post("/settings/zen-key")
async def save_zen_key(
    request: Request,
    zen_api_key: str = Form(""),
    clear: str = Form(""),
) -> RedirectResponse:
    if clear:
        set_zen_api_key(None)
    else:
        set_zen_api_key(zen_api_key)
    ref = request.headers.get("referer") or "/"
    if "/episodes/" in ref:
        sep = "&" if "?" in ref else "?"
        return RedirectResponse(f"{ref}{sep}key_saved=1", status_code=303)
    return RedirectResponse("/?saved=zen", status_code=303)


@app.get("/api/zen-status")
async def api_zen_status() -> JSONResponse:
    return JSONResponse(zen_key_status())


@app.post("/feeds")
async def add_feed(request: Request) -> RedirectResponse:
    form = await request.form()
    raw = str(form.get("upstream_url") or form.get("feed_url") or form.get("url") or "").strip()
    artwork = str(form.get("artwork") or "").strip()
    if not raw or not raw.startswith(("http://", "https://")):
        raise HTTPException(400, "That doesn't look like a podcast link.")
    feed = await get_or_create_feed(raw, artwork=artwork)
    return RedirectResponse(f"/shows/{feed.slug}", status_code=303)


@app.post("/settings/global")
async def save_global_settings(request: Request) -> RedirectResponse:
    form = await request.form()
    updates: dict[str, str] = {}
    for k in ("max_cache_gb", "keep_last_n", "delete_after_days", "poll_minutes"):
        if form.get(k) not in (None, ""):
            updates[k] = str(form.get(k))
    if updates:
        db.set_global_settings(updates)
    return RedirectResponse("/?saved=settings", status_code=303)


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
    feed = db.get_feed_by_slug(slug)
    if not feed:
        raise HTTPException(404, "Show not found")
    return _show_page(feed, request)


@app.post("/shows/{slug}/settings")
async def save_show_settings(slug: str, request: Request) -> RedirectResponse:
    feed = db.get_feed_by_slug(slug)
    if not feed:
        raise HTTPException(404, "Show not found")
    form = await request.form()
    updates: dict = {}
    if "auto_download" in form:
        updates["auto_download"] = str(form.get("auto_download")).lower() in {"1", "on", "true"}
    elif form.get("settings_form"):
        updates["auto_download"] = False
    if form.get("keep_last") not in (None, ""):
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


@app.post("/shows/{slug}/refresh")
async def refresh_show(slug: str) -> RedirectResponse:
    feed = db.get_feed_by_slug(slug)
    if not feed:
        raise HTTPException(404, "Show not found")
    try:
        await sync_feed_episodes(feed, queue_recent=True)
    except Exception as exc:
        raise HTTPException(502, f"Couldn't reach the publisher's feed: {exc}") from exc
    return RedirectResponse(f"/shows/{slug}", status_code=303)


@app.post("/shows/{slug}/delete")
async def delete_show(slug: str) -> RedirectResponse:
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
            "zen_model": settings.zen_model,
            "key_saved": request.query_params.get("key_saved"),
            "has_clean": db.has_clean_audio(ep),
            "has_audio": bool(db.served_audio_path(ep)),
        },
    )


@app.post("/episodes/{episode_id}/recheck")
async def recheck_episode(episode_id: int) -> RedirectResponse:
    ep = db.get_episode(episode_id)
    if not ep:
        raise HTTPException(404, "Episode not found")
    await enqueue_episode(episode_id, reseed=True)
    return RedirectResponse(f"/episodes/{episode_id}", status_code=303)


@app.post("/episodes/{episode_id}/reseed")
async def reseed_episode(episode_id: int) -> RedirectResponse:
    return await recheck_episode(episode_id)


@app.post("/episodes/{episode_id}/ranges")
async def save_ranges(
    episode_id: int,
    ranges_text: str = Form(""),
) -> RedirectResponse:
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
    """Serve the cleaned file. Never block a podcast app.

    - HEAD: report upstream headers only, queue nothing (feed refreshes).
    - GET with clean file: serve it (206 range support via FileResponse).
    - GET while working / not started: queue in background and redirect to
      the publisher's original file so playback starts instantly (with ads
      this once); the next refresh gets the clean file.
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

    served = db.served_audio_path(ep)
    if served:
        db.touch_served(episode_id)
        try:
            size = served.stat().st_size
            if size > 0:
                db.update_episode(episode_id, size_bytes=size)
        except OSError:
            pass
        return FileResponse(served, media_type="audio/mpeg", filename=f"{episode_id}.mp3")

    # Nothing ready: work in background, play original meanwhile.
    await enqueue_episode(episode_id)
    return RedirectResponse(ep.enclosure_url, status_code=302)


@app.get("/chapters/{episode_id}.json")
async def chapters_json(episode_id: int) -> JSONResponse:
    ep = db.get_episode(episode_id)
    if not ep:
        raise HTTPException(404, "Episode not found")
    return JSONResponse(build_chapters_json(ep))


@app.get("/export.opml")
async def export_opml() -> PlainTextResponse:
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
async def import_opml(file: UploadFile) -> RedirectResponse:
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
