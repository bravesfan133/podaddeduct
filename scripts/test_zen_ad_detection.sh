#!/bin/sh
# Exercise local `opencode serve` the same way podaddeduct does.
# Start serve first (or let the app start it), then:
#
#   ./scripts/test_zen_ad_detection.sh
#
# Optional:
#   URL=http://127.0.0.1:4096 MODEL=deepseek-v4-flash ./scripts/test_zen_ad_detection.sh

set -eu

URL="${URL:-http://127.0.0.1:4096}"
MODEL="${MODEL:-deepseek-v4-flash}"

echo "GET $URL/global/health"
if ! curl -sS -f "$URL/global/health"; then
  echo
  echo "OpenCode serve is not running at $URL" >&2
  exit 1
fi
echo
echo

SID=$(curl -sS -X POST "$URL/session" -H "Content-Type: application/json" \
  -d '{"title":"podaddeduct ads test"}' | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('id') or (d.get('info') or {}).get('id') or '')")
if [ -z "$SID" ]; then
  echo "Could not create session" >&2
  exit 1
fi
echo "session $SID"
echo "POST $URL/session/$SID/message  model=opencode/$MODEL"
echo

MSG=/tmp/podaddeduct-zen-msg.json
curl -sS -X POST "$URL/session/$SID/message" \
  -H "Content-Type: application/json" \
  -o "$MSG" \
  -d "$(cat <<EOF
{
  "model": {"providerID": "opencode", "modelID": "$MODEL"},
  "system": "Identify advertisements in the transcript. Return JSON only: {\"ads\":[{\"start\":\"HH:MM:SS\",\"end\":\"HH:MM:SS\",\"type\":\"host_read\",\"sponsor\":\"Name\",\"confidence\":0.9}]}. If none, {\"ads\":[]}.",
  "tools": {"bash": false, "edit": false, "write": false, "read": false},
  "parts": [{"type": "text", "text": "[00:00:00 - 00:00:08] Welcome to the show.\\n[00:00:10 - 00:00:40] This episode is brought to you by Acme. Use code SAVE20 at acme.com.\\n[00:00:42 - 00:01:00] Back to baseball."}]
}
EOF
)"

python3 - "$MSG" <<'PY'
import json, sys
from pathlib import Path
data = json.loads(Path(sys.argv[1]).read_text())
info = data.get("info") if isinstance(data.get("info"), dict) else {}
err = info.get("error")
if err:
    print("--- error ---")
    print(json.dumps(err, indent=2) if not isinstance(err, str) else err)
    sys.exit(1)
parts = data.get("parts") or []
text = "\n".join(p.get("text") or "" for p in parts if isinstance(p, dict) and p.get("type") == "text")
print("--- text ---")
print(text)
print("--- ads ---")
try:
    ads = json.loads(text)
except json.JSONDecodeError:
    start, end = text.find("{"), text.rfind("}")
    ads = json.loads(text[start:end+1]) if start != -1 and end > start else {"_raw": text}
print(json.dumps(ads, indent=2))
PY

curl -sS -o /dev/null -X DELETE "$URL/session/$SID" || true
