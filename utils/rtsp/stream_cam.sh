#!/bin/bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  stream_cam.sh --video-path <file> [--rtsp-host <host>] [--rtsp-port <port>] [--rtsp-path <path>] [--fps <n>]

Defaults:
  RTSP host:  ${RTSP_HOST:-${HOST_IP:-127.0.0.1}}
  RTSP port:  ${RTSP_PORT:-8555}
  RTSP path:  ${RTSP_PATH:-cam1}
  FPS:        ${FPS:-25}
  Width:      ${STREAM_WIDTH:-1280}
  Height:     ${STREAM_HEIGHT:-720}
EOF
}

VIDEO_PATH="${VIDEO_PATH:-}"
RTSP_HOST="${RTSP_HOST:-${HOST_IP:-127.0.0.1}}"
RTSP_PORT="${RTSP_PORT:-8555}"
RTSP_PATH="${RTSP_PATH:-cam1}"
FPS="${FPS:-25}"
STREAM_WIDTH="${STREAM_WIDTH:-1280}"
STREAM_HEIGHT="${STREAM_HEIGHT:-720}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --video-path) VIDEO_PATH="$2"; shift 2 ;;
    --rtsp-host) RTSP_HOST="$2"; shift 2 ;;
    --rtsp-port) RTSP_PORT="$2"; shift 2 ;;
    --rtsp-path) RTSP_PATH="$2"; shift 2 ;;
    --fps) FPS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "${VIDEO_PATH}" ]]; then
  echo "Error: --video-path is required." >&2
  usage
  exit 1
fi

RTSP_URL="rtsp://${RTSP_HOST}:${RTSP_PORT}/${RTSP_PATH}"

echo "[stream_cam] Input: ${VIDEO_PATH}"
echo "[stream_cam] Output: ${RTSP_URL}"
echo "[stream_cam] Video: ${STREAM_WIDTH}x${STREAM_HEIGHT}@${FPS}"

exec ffmpeg -re -stream_loop -1 -i "${VIDEO_PATH}" \
  -vf "fps=${FPS},scale=${STREAM_WIDTH}:${STREAM_HEIGHT}" \
  -c:v libx264 \
  -preset ultrafast \
  -profile:v baseline \
  -level 3.1 \
  -pix_fmt yuv420p \
  -b:v 2M \
  -maxrate 2M \
  -bufsize 4M \
  -g "${FPS}" \
  -keyint_min "${FPS}" \
  -rtsp_transport tcp \
  -an \
  -f rtsp "${RTSP_URL}"
