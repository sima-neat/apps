#!/bin/bash
# Streams a chat completion through the Studio's same-origin proxy
# (POST /v1/chat/completions on the web UI port).
#
# Usage:
#   ./chat.sh [host:port] ["prompt"]
# The model defaults to the chat/VLM model currently loaded (looked up through
# the Studio's catalog); set CHAT_MODEL=<name> to pick one explicitly.

if [ -n "$1" ] && [[ "$1" == *:* ]]; then
  HOST="$1"
  shift
else
  HOST=${MODALIX_HOST:-127.0.0.1:5000}
fi
PROMPT="${1:-Explain time and space in two sentences.}"

MODEL="${CHAT_MODEL:-}"
if [ -z "$MODEL" ]; then
  MODEL=$(curl -k -s "https://${HOST}/models/catalog" | python3 -c '
import json, sys
try:
    cat = json.load(sys.stdin).get("catalog", [])
except Exception:
    cat = []
loaded = [m["name"] for m in cat if m.get("loaded") and m.get("type", "chat") != "asr"]
print(loaded[0] if loaded else "")')
fi
if [ -z "$MODEL" ]; then
  echo "No chat model is loaded; load one in the Studio or set CHAT_MODEL=<name>." >&2
  exit 1
fi

BODY=$(python3 -c 'import json,sys; print(json.dumps({"model": sys.argv[1], "messages": [{"role": "user", "content": sys.argv[2]}], "stream": True}))' "$MODEL" "$PROMPT")

curl -N -k -sS -X POST "https://${HOST}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "${BODY}"
echo
