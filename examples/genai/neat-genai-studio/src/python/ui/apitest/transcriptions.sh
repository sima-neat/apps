#!/bin/bash
# Transcribe an audio file through the Studio's OpenAI-compatible endpoint
# (POST /v1/audio/transcriptions on the web UI port; the Studio proxies it to
# the model server's active speech-to-text model).
#
# Usage:
#   ./transcriptions.sh [host:port] <audio_file> [language] [response_format] [model]
# Examples:
#   ./transcriptions.sh recording.webm                  # auto language, json
#   ./transcriptions.sh 10.0.0.5:5000 audio.wav de verbose_json
#   MODALIX_HOST=box:5000 ./transcriptions.sh clip.wav auto text

if [ -n "$1" ] && [[ "$1" == *:* ]]; then
  HOST="$1"
  shift
else
  HOST=${MODALIX_HOST:-127.0.0.1:5000}
fi

FILE_PATH="$1"
LANGUAGE="${2:-auto}"
FORMAT="${3:-json}"
MODEL="${4:-}"

if [ -z "$FILE_PATH" ] || [ ! -f "$FILE_PATH" ]; then
  echo "Usage: $0 [host:port] <audio_file> [language] [response_format json|verbose_json|text] [model]"
  exit 1
fi

args=(-F "file=@${FILE_PATH}" -F "language=${LANGUAGE}" -F "response_format=${FORMAT}")
[ -n "$MODEL" ] && args+=(-F "model=${MODEL}")

curl -k -sS -D - -X POST "https://${HOST}/v1/audio/transcriptions" "${args[@]}"
echo
