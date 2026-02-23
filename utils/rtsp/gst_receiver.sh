#!/bin/bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  gst_receiver.sh [--port <port>]

Default port:
  ${UDP_PORT:-5000}
EOF
}

PORT="${UDP_PORT:-5000}"
VIDEO_SINK="autovideosink"
TEXT_OVERLAY="true"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" && "${VIDEO_SINK}" == "autovideosink" ]]; then
  VIDEO_SINK="fakesink"
  TEXT_OVERLAY="false"
fi

echo "[gst_receiver] Listening on UDP port ${PORT}"

gst-launch-1.0 -e -v \
  udpsrc port="${PORT}" caps="application/x-rtp,media=(string)video,encoding-name=(string)H264,payload=(int)96,clock-rate=(int)90000" \
  ! rtpjitterbuffer latency=100 drop-on-latency=true \
  ! rtph264depay \
  ! h264parse \
  ! avdec_h264 \
  ! videoconvert \
  ! fpsdisplaysink video-sink="${VIDEO_SINK}" text-overlay="${TEXT_OVERLAY}" sync=false
