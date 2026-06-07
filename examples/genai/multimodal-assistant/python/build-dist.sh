#!/bin/bash

set -e  # Exit on any error

# === Archive : simaai-genai-demo.tar.gz ===
echo "Creating simaai-genai-demo.tar.gz..."

COPYFILE_DISABLE=1 tar --exclude='.venv*' \
                       --exclude='__pycache__' \
                       --exclude='ffmpeg-build' \
                       --exclude='*.onnx' \
                       --exclude='*.tar.gz' \
                       --exclude='.DS_Store' \
                       --exclude='._*' \
                       --exclude='.git' \
		               --exclude='.cache' \
                       --exclude='server.log' \
                       --exclude='apitest/*.wav' \
                       --exclude='output_wavs/*.wav' \
                       --exclude='gte-small-local' \
                       --exclude='metadata.json' \
                       --exclude='install.sh' \
                       --exclude='run.sh' \
                       -czvf /tmp/simaai-genai-demo.tar.gz .

# === Archive : simaai-genai-demo.zip ===
echo "Created simaai-genai-demo.tar.gz under /tmp folder..."
