# podaddeduct

Your podcasts, without the ads. Download → transcribe → find sponsor reads → cut → clean RSS for your phone.

## Use it (2 minutes)

```bash
cd ~/Projects/podaddeduct
cp -n .env.example .env 2>/dev/null || true
./run.sh
```

Open `http://127.0.0.1:8080/`:

1. **Search** for a show by name, press Add.
2. **Copy its link** → iPhone Podcasts → Library → **…** → **Follow a Show by URL** → paste. (Phone + computer on the same Wi-Fi.)
3. **Play.** A new episode takes a few minutes to clean once; afterwards it's instant.

## How it stays small

podaddeduct is a **cache, not an archive** (default 3 GB):

- Originals are deleted after the clean copy is cut.
- Each show keeps only its latest 5 episodes; anything older than 14 days goes.
- Re-requesting a deleted episode re-downloads and re-cuts from saved marks in seconds — no extra AI cost.
- Tune it in **Settings**: max storage, keep-latest, delete-after, check-for-new interval.
- Per show: automatic preparation on/off, keep-latest, and **chapters-only mode** (marks ads, stores ~nothing).

While an episode is still being cleaned, the app plays the publisher's original so playback never blocks. Podcast-app refresh checks (`HEAD`) never start work.

## Ad detection

Transcription + OpenCode Zen to spot sponsor reads, snapped to silence, cut with ffmpeg. Everything is configured in **Settings** (gear icon, top right) — no config files needed:

- **Ad detection card:** paste the API key once, pick the model (listed from your key), Test button proves it works.
- **Server card:** transcription backend shortcut (Mac Parakeet / Linux faster-whisper / Groq cloud), Groq API key, public address for Overcast, family password.
- **Processing card:** how many episodes to prepare, shortest ad to cut, auto-prepare on/off.

Transcripts come from the fastest available source, automatically: publisher-provided file in the RSS feed when one exists (free, instant) → Groq Whisper API (~1–2 min, needs free key from console.groq.com) → local faster-whisper on CPU.

- **Mac dev:** local Parakeet transcription (Apple Silicon sidecar):

```bash
python3.12 -m venv .venv-stt
.venv-stt/bin/pip install parakeet-mlx
```

- **Linux / home server:** `faster-whisper` on CPU (`STT_MODEL=base` default, `tiny` if the CPU is slow). The Docker image includes it.

## Home server (Docker)

```bash
git clone git@github.com:bravesfan133/podaddeduct.git
cd podaddeduct
docker compose up -d --build
```

Data (DB, audio, transcripts) lives in the `podaddeduct-data` volume. The app serves port **7887**: point your Cloudflare Tunnel hostname at `http://localhost:7887` (tunnel on the same machine) and set that `https://…` URL as the public address in Settings.

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
