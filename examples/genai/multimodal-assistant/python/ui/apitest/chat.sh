#!/bin/bash

# Usage:
#   ./script.sh <host:port>
# or:
#   MODALIX_HOST=1.2.3.4:8000 ./script.sh

# Get host from argument, or environment var, or default
if [ -n "$1" ]; then
  HOST="$1"
else
  HOST=${MODALIX_HOST:-192.168.2.20:5000}
fi

curl -N -k -X POST "https://${HOST}/v1/chat" \
  -H "Content-Type: application/json" \
  -d @- <<EOF
{
  "model": "llava",
  "messages": [
    {
      "role": "user",
      "content": "Explain time and space"
    }
  ],
  "stream": true
}
EOF
