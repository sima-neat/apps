#!/bin/bash

# Usage:
#   ./transcribe.sh [host:port] <audio_file_path> [language]
# Examples:
#   ./transcribe.sh 10.0.0.5:8000 audio.wav es
#   ./transcribe.sh audio.wav

# Determine if the first argument is a host (contains a ':')
if [ -n "$1" ] && [[ "$1" == *:* ]]; then
  HOST="$1"
  shift
else
  HOST=${MODALIX_HOST:-192.168.2.20:5000}
fi

# Now the first positional argument is always the file
FILE_PATH="$1"
LANGUAGE="${2:-en}"

if [ -z "$FILE_PATH" ]; then
  echo "Usage: $0 [host:port] <audio_file_path> [language]"
  exit 1
fi

curl -k -X POST "https://${HOST}/v1/audio/transcriptions" \
  -H "Content-Type: multipart/form-data" \
  -F file=@${FILE_PATH} \
  -F language=${LANGUAGE}
