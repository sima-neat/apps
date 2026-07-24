#!/usr/bin/env bash
# run.sh — Launch face-recognizer on the DevKit.
# Run this script FROM the DevKit or via `ssh devkit 'bash -s' < run.sh`.
#
# Usage:
#   ./run.sh [--input <rtsp-uri|video-file>] [--gallery <path>]
#            [--test] [--output <sink>]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BINARY="${DEPLOY_DIR}/face-recognizer"
CONFIG="${DEPLOY_DIR}/config.yaml"
INPUT=""
GALLERY="${DEPLOY_DIR}/gallery.bin"
OUTPUT=""
TEST_MODE=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)   INPUT="$2";   shift 2 ;;
    --gallery) GALLERY="$2"; shift 2 ;;
    --output)  OUTPUT="$2";  shift 2 ;;
    --test)    TEST_MODE=1;  shift ;;
    *)         EXTRA_ARGS+=("$1"); shift ;;
  esac
done

# Workaround for A65 processcvu memory leak (see sdk_64k_frame_ceiling memory).
export SIMA_PROCESSCVU_RUN_TARGET="${SIMA_PROCESSCVU_RUN_TARGET:-EV74}"

CMD_ARR=("${BINARY}" --config "${CONFIG}")
[[ -n "${INPUT}" ]]    && CMD_ARR+=(--input   "${INPUT}")
[[ -n "${GALLERY}" ]]  && CMD_ARR+=(--gallery "${GALLERY}")
[[ -n "${OUTPUT}" ]]   && CMD_ARR+=(--output  "${OUTPUT}")
[[ $TEST_MODE -eq 1 ]] && CMD_ARR+=(--test)
for a in "${EXTRA_ARGS[@]}"; do CMD_ARR+=("${a}"); done

echo "SIMA_PROCESSCVU_RUN_TARGET=${SIMA_PROCESSCVU_RUN_TARGET}"
echo "Running: ${CMD_ARR[*]}"
exec "${CMD_ARR[@]}"
