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
- Tune it on the home page: max storage, keep-latest, delete-after, check-for-new interval.
- Per show: automatic preparation on/off, keep-latest, and **chapters-only mode** (marks ads, stores ~nothing).

While an episode is still being cleaned, the app plays the publisher's original so playback never blocks. Podcast-app refresh checks (`HEAD`) never start work.

## Ad detection

Transcription + OpenCode Zen to spot sponsor reads, snapped to silence, cut with ffmpeg. Paste the key once on the home page (or set `OPENCODE_API_KEY`).

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
cp -n .env.example .env
docker compose up -d --build
```

Data (DB, audio, transcripts) lives in the `podaddeduct-data` volume. Point your Cloudflare Tunnel hostname at `http://<server-lan-ip>:8080` and set that `https://…` URL as `PUBLIC_BASE_URL`.

## Outside home Wi-Fi / Overcast

Overcast and Pocket Casts fetch from their servers, so a LAN address won't work there. Options, easiest first:

1. **Cloudflare Tunnel** (free, no VPS): exposes this machine as `https://…`, set it as `PUBLIC_BASE_URL`.
2. **VPS**: only if this machine can't stay awake; same Docker image runs anywhere.

## Move shows between apps

OPML import on the home page, backup via `/export.opml`.

## Tests

```bash
.venv/bin/python -m pytest -q
```
