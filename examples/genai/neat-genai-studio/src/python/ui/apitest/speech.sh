#!/bin/bash

# Usage:
#   ./synthesize.sh [host:port] "text to synthesize" [output_file]

# Detect if the first argument is a host (contains a colon, no spaces)
if [ "$1" ] && [[ "$1" == *:* && "$1" != *" "* ]]; then
  HOST="$1"
  shift
else
  HOST=${MODALIX_HOST:-127.0.0.1:5000}
fi

# Now $1 is always the TEXT, $2 is optional output_filepath
TEXT="$1"
OUTPUT_FILE="${2:-output.wav}"

# Validate
if [ -z "$TEXT" ]; then
  echo "Usage: $0 [host:port] \"text to synthesize\" [output_file]"
  exit 1
fi

# Send request
curl -k -X POST "https://${HOST}/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -o "${OUTPUT_FILE}" \
  -d @- <<EOF
{
  "input": "${TEXT}",
  "language": "en"
}
EOF

echo "✅ Saved synthesized speech to ${OUTPUT_FILE}"
