#!/bin/bash
# Translate speech in any language into English text through the Studio's
# OpenAI-compatible endpoint (POST /v1/audio/translations on the web UI port;
# the Studio proxies it to the active Whisper model's translate task).
#
# Usage:
#   ./translations.sh [host:port | base URL] <audio_file> [language] [response_format] [model]
# Examples:
#   ./translations.sh recording.webm                  # auto language, json
#   ./translations.sh 10.0.0.5:5000 audio.wav de verbose_json
#   MODALIX_HOST=box:5000 ./translations.sh clip.wav auto text

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

FILE_PATH="$1"
LANGUAGE="${2:-auto}"
FORMAT="${3:-json}"
MODEL="${4:-}"

if [ -z "$FILE_PATH" ] || [ ! -f "$FILE_PATH" ]; then
  echo "Usage: $0 [host:port | base URL] <audio_file> [language] [response_format json|verbose_json|text] [model]"
  exit 1
fi

args=(-F "file=@${FILE_PATH}" -F "language=${LANGUAGE}" -F "response_format=${FORMAT}")
[ -n "$MODEL" ] && args+=(-F "model=${MODEL}")

curl --fail-with-body -k -sS -D - -X POST "${BASE}/v1/audio/translations" "${args[@]}"
echo
