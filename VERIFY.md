# Local verification results (2026-09-10)

## mac-smoke — PASS
- UI http://127.0.0.1:8080/ → 200, shows Hammer Territory
- Feed → 200, valid XML, 40 items
- Audio /audio/1 → 206 Partial Content, ID3/mpeg
- Chapters /chapters/1.json → 200 (includes prior manual Ad ranges)

## phone-lan readiness — PASS (endpoints)
- PUBLIC_BASE_URL set to http://192.168.0.93:8080 (tunnel URL removed)
- Enclosures in feed now point at LAN host
- lan_ui / lan_feed / lan_audio all 200/206 from this Mac

Subscribe on phone (same Wi‑Fi), Apple Podcasts → Library → … → Follow a Show by URL:

http://192.168.0.93:8080/feeds/hammer-territory-an-atlanta-braves-podcast-c793e38a.xml

## hold-homeserver — HELD
- No copy to homeserver
- No Tailscale cutover
- .env is LAN-only
