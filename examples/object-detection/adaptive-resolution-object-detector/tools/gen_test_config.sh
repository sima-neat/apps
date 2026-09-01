#!/usr/bin/env bash
# Generate a ready-to-run config for adaptive-resolution-object-detector.
# Fills in the placeholders the shipped src/common/config.yaml leaves blank
# (Insight host, RTSP URLs, model paths) so you can run immediately.
#
# Usage:
#   gen_test_config.sh OUT.yaml INSIGHT_HOST RTSP_HOST SRC [SRC ...]
# Examples:
#   # bare names -> rtsp://RTSP_HOST:8554/<name> (Insight mediamtx sources)
#   gen_test_config.sh /tmp/mytest.yaml 192.168.131.68 192.168.131.68 src1 src2
#   # full URLs used verbatim (your own RTSP server); RTSP_HOST then ignored
#   gen_test_config.sh /tmp/mytest.yaml 192.168.131.68 - \
#     rtsp://192.168.131.68:8555/busy rtsp://192.168.131.68:8555/simple
#
# Each SRC becomes cam-N. A SRC containing "://" is used as the URL as-is;
# otherwise it becomes rtsp://RTSP_HOST:8554/SRC.
# Tune behaviour with env vars (defaults in parentheses):
#   MIN_SCORE(0.30) FPS(0) FRAMES(0) MAX_STREAMS(8) VIDEO(true) PROFILE(true)
#   DEBUG_DIR(/tmp/adaptive_out) SAVE_EVERY(20) MODEL_DIR(models/, falling back to assets/models/)
set -euo pipefail

OUT="${1:?OUT.yaml required}"; INSIGHT_HOST="${2:?INSIGHT_HOST required}"; RTSP_HOST="${3:?RTSP_HOST required}"
shift 3
[ "$#" -ge 1 ] || { echo "need at least one SRC (e.g. src1 src2)"; exit 2; }

APPS_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
# models/ is where download_models.sh and the READMEs put packs; assets/models/
# is the older location, kept for a checkout that predates the move. Mirrors
# pipelines/pipeline-scale/pipeline.py's _MODEL_DIRS - a fresh install has no
# assets/models/ at all, so defaulting to it left every TESTING.md command that
# omits MODEL_DIR pointing at a path that does not exist.
if [ -n "${MODEL_DIR:-}" ]; then
  :
elif [ -d "$APPS_ROOT/models" ]; then
  MODEL_DIR="$APPS_ROOT/models"
else
  MODEL_DIR="$APPS_ROOT/assets/models"
fi

{
echo "model:"
echo "  path: $MODEL_DIR/${MODEL:-yolo26n-det-int8-b1.tar.gz}"
echo "  labels: $APPS_ROOT/examples/object-detection/adaptive-resolution-object-detector/src/common/coco_label.txt"
echo "input:"
echo "  tcp: true"
echo "  latency_ms: 100"
echo "inference:"
echo "  frames: ${FRAMES:-0}"
echo "  fps: ${FPS:-0}"
echo "  min_score: ${MIN_SCORE:-0.30}"
echo "  nms_iou: 0.60"
echo "  max_detections: ${MAX_DET:-50}"
echo "runtime:"
echo "  profile: ${PROFILE:-true}"
echo "  warmup_frames: 5"
echo "  config_watch_seconds: 1.0"
echo "output:"
echo "  insight:"
echo "    host: $INSIGHT_HOST"
echo "    video_port: 9000"
echo "    metadata_port: 9100"
echo "  video_enabled: ${VIDEO:-true}"
echo "  debug_dir: ${DEBUG_DIR:-/tmp/adaptive_out}"
echo "  save_every: ${SAVE_EVERY:-20}"
echo "streams:"
echo "  max_streams: ${MAX_STREAMS:-8}"
echo "  sources:"
i=1
for s in "$@"; do
  case "$s" in
    *://*) url="$s" ;;                        # full URL, use verbatim
    *)     url="rtsp://$RTSP_HOST:8554/$s" ;; # bare name -> Insight mediamtx source
  esac
  echo "    - id: cam-$i"; echo "      rtsp_url: $url"; i=$((i+1))
done
} > "$OUT"
echo "wrote $OUT ($(($#)) stream(s))"
