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
#   RES(320,640,960) HYST(15) BUDGET(12) MINPX(24) CONF_LOW(0.40) DENSITY(20)
#   MIN_SCORE(0.30) FPS(0) FRAMES(0) MAX_STREAMS(8) VIDEO(true) PROFILE(true)
#   DEBUG_DIR(/tmp/adaptive_out) SAVE_EVERY(20) MODEL_DIR(assets/models)
set -euo pipefail

OUT="${1:?OUT.yaml required}"; INSIGHT_HOST="${2:?INSIGHT_HOST required}"; RTSP_HOST="${3:?RTSP_HOST required}"
shift 3
[ "$#" -ge 1 ] || { echo "need at least one SRC (e.g. src1 src2)"; exit 2; }

APPS_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
MODEL_DIR="${MODEL_DIR:-$APPS_ROOT/assets/models}"
RES="${RES:-320,640,960}"
IFS=',' read -r -a RESA <<< "$RES"

{
echo "model:"
echo "  path: $MODEL_DIR/yolo26n-640-det-int8-mla_tess-b1.tar.gz"
echo "  labels: $APPS_ROOT/examples/object-detection/adaptive-resolution-object-detector/src/common/coco_label.txt"
echo "  tiers:"
for r in "${RESA[@]}"; do echo "    $r: $MODEL_DIR/yolo26n-${r}-det-int8-mla_tess-b1.tar.gz"; done
echo "adaptive:"
echo "  resolutions: [$(IFS=', '; echo "${RESA[*]}")]"
echo "  confidence_low: ${CONF_LOW:-0.40}"
echo "  min_object_px: ${MINPX:-24}"
echo "  hysteresis_frames: ${HYST:-15}"
echo "  density_high: ${DENSITY:-20}"
echo "  budget_units: ${BUDGET:-12}"
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
