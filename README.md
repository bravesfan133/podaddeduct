# podaddeduct

Your podcasts, without the ads. Download → find sponsor reads → cut → clean RSS for your phone.

## Use it (2 minutes)

```bash
cd ~/Projects/podaddeduct
cp -n .env.example .env 2>/dev/null || true
./run.sh
```

Open `http://127.0.0.1:8080/`:

1. **Search** for a show by name, press Add. Ad detection uses local `opencode serve` (`opencode/deepseek-v4-flash`). Paste your Zen key in Settings. Optional Gemini is under Settings → Ad detection → Gemini.
2. **Copy its link** → iPhone Podcasts → Library → **…** → **Follow a Show by URL** → paste. (Phone + computer on the same Wi-Fi.)
3. **Play.** New episodes prepare automatically; first play waits until the clean file is ready (never streams the with-ads original). Afterwards it's instant.

## How it stays small

podaddeduct is a **cache, not an archive** (default 3 GB):

- Originals are deleted after the clean copy is cut.
- Each show keeps only its latest 5 episodes; anything older than 14 days goes.
- Re-requesting a deleted episode re-downloads and re-cuts from saved marks in seconds — no extra AI cost.
- Tune it in **Settings**: max storage, keep-latest, delete-after, check-for-new interval.
- Per show: automatic prepare on/off and keep-latest.

Podcast-app refresh checks (`HEAD`) never start work. First play of an unprepared episode returns **503** until cleaning finishes — turn on auto-prepare so that wait is rare.

## Ad detection

Cheapest path first:

1. **Publisher chapters** with Ad/Sponsor titles → cut immediately (no AI).
2. **Publisher transcript** in the RSS (free) → else local STT (Parakeet on Mac / faster-whisper on Linux) → else optional Groq Whisper.
3. Cheap **sponsor-read heuristics**, then **OpenCode serve** (`opencode/deepseek-v4-flash`) in one shot.
4. Snap to silence, cut with ffmpeg (ID3 tags and cover art preserved).

Everything is configured in **Settings** — no config files needed:

- **Ad detection:** paste a Zen key once. Test pings `opencode serve` (`GET /global/health`) then a tiny session message. Gemini key is optional and only used if you set a `gemini-*` model.
- **Server:** transcription backend (Mac Parakeet / Linux faster-whisper / optional Groq Whisper), Groq key for cloud STT only, public address for Overcast, family password.
- **Processing:** how many episodes to prepare, shortest ad to cut.

The app starts `opencode serve --hostname 127.0.0.1 --port 4096` if the CLI is on PATH. You can also start it yourself. If serve is down, obvious sponsor-read phrases ("sponsored by", promo codes, etc.) are still cut via heuristics. Optional live check:

```bash
./scripts/test_zen_ad_detection.sh
```

## Home server (Docker)

```bash
git clone git@github.com:bravesfan133/podaddeduct.git
cd podaddeduct
docker compose up -d --build
```

Data (DB, audio, transcripts) lives in the `podaddeduct-data` volume. The app serves port **7887**: point your Cloudflare Tunnel hostname at `http://localhost:7887` (tunnel on the same machine) and set that `https://…` URL as the public address in Settings.

The image includes the OpenCode CLI and starts `opencode serve` in the same container (no second Compose service). Paste the Zen key in Settings, or set `ZEN_API_KEY`. If serve already runs on the host instead, set `OPENCODE_SERVER_URL` (e.g. `http://172.17.0.1:4096`).

## Outside home Wi-Fi / Overcast

Overcast and Pocket Casts fetch from their servers, so they need a public `https://` address:

1. **Cloudflare Tunnel** (free, no VPS, no port forwarding): route a hostname (e.g. `podcasts.example.com`) to `http://localhost:7887`, then set that `https://…` URL as the public address in **Settings → Server**.
2. **VPS**: only if this machine can't stay awake; same Docker image runs anywhere.

### Locking it down (tunnel is public internet)

- Set the **family password** in Settings → Server. Podcast apps can't log in, so `/feeds/*.xml` and `/audio/*` stay open by design — everything else requires the password.
- Optional second layer: Cloudflare Access with Google login on a second application. Scope it to the UI only; the player needs its own application with path destinations `…/feeds/*` and `…/audio/*` plus a Bypass/Everyone policy. **Never put a login (Access or otherwise) in front of `/feeds/*` or `/audio/*` — Overcast will silently fail.**
- On your home Wi-Fi the app is reachable directly (bypassing Cloudflare), so the family password matters there too.

## Move shows between apps

OPML import on the home page, backup via `/export.opml`.

## Tests

```bash
.venv/bin/python -m pytest -q
```
