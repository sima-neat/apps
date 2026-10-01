#!/bin/bash
set -u
# Synthesize text through the Studio's OpenAI-compatible endpoint
# (POST /v1/audio/speech on the web UI port) and save the WAV.
#
# Usage:
#   ./speech.sh [host:port | base URL] "text to synthesize" [output_file] [model] [voice] [language] [speed]
# Examples:
#   ./speech.sh "Hello from Neat GenAI Studio."
#   ./speech.sh 10.0.0.5:5000 "Guten Morgen." morgen.wav supertonic F2 de 1.2
#
# model: default | supertonic | piper-plus | piper-tts   voice: F1..F5 / M1..M5 (Supertonic)
# speed: 0.25-4.0 (clamped to the engine's range; the effective value is in X-Speed)

# Where the Studio listens: a full base URL (http://… or https://…) as the first
# argument, or host:port (scheme from STUDIO_SCHEME, default https), or
# STUDIO_URL / MODALIX_HOST. Use http:// when app.web.https is false.
if [ -n "$1" ] && [[ "$1" == http://* || "$1" == https://* ]]; then
  BASE="${1%/}"
  shift
elif [ -n "$1" ] && [[ "$1" == *:* && "$1" != *" "* ]]; then
  BASE="${STUDIO_SCHEME:-https}://$1"
  shift
else
  BASE="${STUDIO_URL:-${STUDIO_SCHEME:-https}://${MODALIX_HOST:-127.0.0.1:5000}}"
  BASE="${BASE%/}"
fi

TEXT="$1"
OUTPUT_FILE="${2:-output.wav}"
MODEL="${3:-default}"
VOICE="${4:-default}"
LANGUAGE="${5:-en}"
SPEED="${6:-1.0}"

if [ -z "$TEXT" ]; then
  echo "Usage: $0 [host:port | base URL] \"text to synthesize\" [output_file] [model] [voice] [language] [speed]"
  exit 1
fi

BODY=$(python3 -c 'import json,sys; print(json.dumps({"input": sys.argv[1], "model": sys.argv[2], "voice": sys.argv[3], "language": sys.argv[4], "speed": float(sys.argv[5])}))' \
  "$TEXT" "$MODEL" "$VOICE" "$LANGUAGE" "$SPEED")

# -D - prints the response headers (X-Engine, X-Voice, X-Speed, X-RTF, ...).
# --fail-with-body: a 4xx/5xx sets a non-zero status while still printing the
# body, so a failed check cannot report success or leave an error page saved as
# though it were audio.
# Capture curl's own status: inside `if ! curl ...` the `!` has already
# inverted it, so $? there is 0 and the failure would be reported as success.
curl -k -sS --fail-with-body -D - -X POST "${BASE}/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -o "${OUTPUT_FILE}" \
  -d "${BODY}"
status=$?
if [ "${status}" -ne 0 ]; then
  echo "❌ Speech synthesis failed (curl exit ${status}); see the response above." >&2
  rm -f "${OUTPUT_FILE}"
  exit "${status}"
fi

echo "✅ Saved synthesized speech to ${OUTPUT_FILE}"
