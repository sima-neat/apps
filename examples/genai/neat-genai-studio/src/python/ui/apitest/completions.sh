#!/bin/bash

# Get host from environment variable or use default
HOST=${MODALIX_HOST:-127.0.0.1:5000}

curl -N -k -X POST "https://${HOST}/v1/chat/completions" \
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
  "stream": false
}
EOF
