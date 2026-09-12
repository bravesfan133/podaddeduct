# podaddeduct

Your podcasts, without the ads. Download → find sponsor reads → cut → clean RSS for your phone.

## Use it (2 minutes)

```bash
cd ~/Projects/podaddeduct
cp -n .env.example .env 2>/dev/null || true
./run.sh
```

Open `http://127.0.0.1:8080/`:

1. **Search** for a show by name, press Add. Ad detection uses **Gemini** (`gemini-3.5-flash` by default) on a full transcript. Paste your Gemini key in Settings. Transcription defaults to Groq Whisper on Docker / N100 hosts (Parakeet on Mac).
2. **Copy its link** → iPhone Podcasts → Library → **…** → **Follow a Show by URL** → paste. (Phone + computer on the same Wi-Fi.)
3. **Play.** New episodes prepare automatically; first clean takes a few minutes (scan + cut). Afterwards it's instant.

## How it stays small

podaddeduct is a **cache, not an archive** (default 3 GB):

- Originals are deleted after the clean copy is cut.
- Each show keeps only its latest 5 episodes; anything older than 14 days goes.
- Re-requesting a deleted episode re-downloads and re-cuts from saved marks in seconds — no extra AI cost.
- Tune it in **Settings**: max storage, keep-latest, delete-after, check-for-new interval.
- Per show: automatic prepare on/off and keep-latest.

Podcast-app refresh checks (`HEAD`) never start work. The custom RSS lists **Ready** episodes only. Unprepared episodes stay off the player feed until cleaning finishes.

## Ad detection

Cheapest path first:

1. **Publisher chapters** with Ad/Sponsor titles → cut immediately (no AI).
2. **Publisher transcript** in the RSS (free) → else Groq Whisper (Docker) / Parakeet (Mac). Files under 24 MB upload as-is; larger ones get a single-thread ffmpeg compress first.
3. **One Gemini `generateContent` call** on the full transcript (host-reads + inserts). Temperature 0.2; server-side max span (~3 min midrolls) and coverage guards refuse unsafe cuts.
4. Snap edges near transcript times / short silence windows, cut with ffmpeg (`-threads 1`; stream-copy when the source is already MP3).

Everything is configured in **Settings**:

- **Ad detection:** Gemini API key + model (`gemini-3.5-flash`). Test talks to Gemini.
- **Server:** transcription backend (Groq recommended on N100), Groq key, public address for Overcast, family password. Health shows ffmpeg + VAAPI yes/no.
- **Processing:** how many episodes to prepare, shortest ad to cut.

## Home server (Docker / Intel N100)

```bash
git clone git@github.com:bravesfan133/podaddeduct.git
cd podaddeduct
docker compose up -d --build
```

Data (DB, audio, transcripts) lives in the `podaddeduct-data` volume. The app serves port **7887**: point your Cloudflare Tunnel hostname at `http://localhost:7887` (tunnel on the same machine) and set that `https://…` URL as the public address in Settings.

Compose passes through the N100 iGPU (`/dev/dri`, `LIBVA_DRIVER_NAME=iHD`) so ffmpeg can use **VAAPI decode** when the codec allows. Device passthrough is enough (the container runs as root). Groq/Gemini wait on the network; residual local CPU is ffmpeg only when a file exceeds Groq’s upload cap (and Quick Sync does not encode MP3). Idle process list should be uvicorn only. If `/dev/dri` is missing on the host, Compose will fail on `devices` and VAAPI stays off.

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
