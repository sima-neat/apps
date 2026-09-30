#!/bin/bash
# Synthesize text through the Studio's OpenAI-compatible endpoint
# (POST /v1/audio/speech on the web UI port) and save the WAV.
#
# Usage:
#   ./speech.sh [host:port] "text to synthesize" [output_file] [model] [voice] [language] [speed]
# Examples:
#   ./speech.sh "Hello from Neat GenAI Studio."
#   ./speech.sh 10.0.0.5:5000 "Guten Morgen." morgen.wav supertonic F2 de 1.2
#
# model: default | supertonic | piper-plus | piper-tts   voice: F1..F5 / M1..M5 (Supertonic)
# speed: 0.25-4.0 (clamped to the engine's range; the effective value is in X-Speed)

if [ "$1" ] && [[ "$1" == *:* && "$1" != *" "* ]]; then
  HOST="$1"
  shift
else
  HOST=${MODALIX_HOST:-127.0.0.1:5000}
fi

TEXT="$1"
OUTPUT_FILE="${2:-output.wav}"
MODEL="${3:-default}"
VOICE="${4:-default}"
LANGUAGE="${5:-en}"
SPEED="${6:-1.0}"

if [ -z "$TEXT" ]; then
  echo "Usage: $0 [host:port] \"text to synthesize\" [output_file] [model] [voice] [language] [speed]"
  exit 1
fi

BODY=$(python3 -c 'import json,sys; print(json.dumps({"input": sys.argv[1], "model": sys.argv[2], "voice": sys.argv[3], "language": sys.argv[4], "speed": float(sys.argv[5])}))' \
  "$TEXT" "$MODEL" "$VOICE" "$LANGUAGE" "$SPEED")

# -D - prints the response headers (X-Engine, X-Voice, X-Speed, X-RTF, ...).
curl -k -sS -D - -X POST "https://${HOST}/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -o "${OUTPUT_FILE}" \
  -d "${BODY}"

echo "✅ Saved synthesized speech to ${OUTPUT_FILE}"
