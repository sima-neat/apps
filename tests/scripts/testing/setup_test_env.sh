#!/usr/bin/env bash
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "[ERROR] Source this script so exported vars persist in your shell."
  echo "        source tests/scripts/testing/setup_test_env.sh"
  exit 1
fi

# set vars (defaults)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_APPS_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

apps_root="${APPS_ROOT:-${DEFAULT_APPS_ROOT}}"
models_dir="${apps_root}/assets/models"
input_dir="${apps_root}/assets/test_images"
output_dir="/tmp"
classification_image="${apps_root}/assets/test_images_classification/goldfish.jpeg"
timeout_ms="180000"
require_e2e="1"
keep_output="0"
rtsp_url="rtsp://127.0.0.1:8554/stream0"
rtsp_urls="rtsp://127.0.0.1:8554/stream0,rtsp://127.0.0.1:8554/stream1"

# export resolved values
export APPS_ROOT="${apps_root}"
export SIMANEAT_APPS_TEST_MODELS_DIR="${SIMANEAT_APPS_TEST_MODELS_DIR:-${models_dir}}"
export SIMANEAT_APPS_TEST_INPUT_DIR="${SIMANEAT_APPS_TEST_INPUT_DIR:-${input_dir}}"
export SIMANEAT_APPS_TEST_OUTPUT_DIR="${SIMANEAT_APPS_TEST_OUTPUT_DIR:-${output_dir}}"
export SIMANEAT_APPS_TEST_CLASSIFICATION_IMAGE="${SIMANEAT_APPS_TEST_CLASSIFICATION_IMAGE:-${classification_image}}"
export SIMANEAT_APPS_TEST_TIMEOUT_MS="${SIMANEAT_APPS_TEST_TIMEOUT_MS:-${timeout_ms}}"
export SIMANEAT_APPS_TEST_REQUIRE_E2E="${SIMANEAT_APPS_TEST_REQUIRE_E2E:-${require_e2e}}"
export SIMANEAT_APPS_TEST_KEEP_OUTPUT="${SIMANEAT_APPS_TEST_KEEP_OUTPUT:-${keep_output}}"
export SIMANEAT_APPS_TEST_RTSP_URL="${SIMANEAT_APPS_TEST_RTSP_URL:-${rtsp_url}}"
export SIMANEAT_APPS_TEST_RTSP_URLS="${SIMANEAT_APPS_TEST_RTSP_URLS:-${rtsp_urls}}"

echo "[INFO] test environment configured:"
echo "  APPS_ROOT=${APPS_ROOT}"
echo "  SIMANEAT_APPS_TEST_MODELS_DIR=${SIMANEAT_APPS_TEST_MODELS_DIR}"
echo "  SIMANEAT_APPS_TEST_INPUT_DIR=${SIMANEAT_APPS_TEST_INPUT_DIR}"
echo "  SIMANEAT_APPS_TEST_OUTPUT_DIR=${SIMANEAT_APPS_TEST_OUTPUT_DIR}"
echo "  SIMANEAT_APPS_TEST_CLASSIFICATION_IMAGE=${SIMANEAT_APPS_TEST_CLASSIFICATION_IMAGE}"
echo "  SIMANEAT_APPS_TEST_TIMEOUT_MS=${SIMANEAT_APPS_TEST_TIMEOUT_MS}"
echo "  SIMANEAT_APPS_TEST_REQUIRE_E2E=${SIMANEAT_APPS_TEST_REQUIRE_E2E}"
echo "  SIMANEAT_APPS_TEST_KEEP_OUTPUT=${SIMANEAT_APPS_TEST_KEEP_OUTPUT}"
echo "  SIMANEAT_APPS_TEST_RTSP_URL=${SIMANEAT_APPS_TEST_RTSP_URL}"
echo "  SIMANEAT_APPS_TEST_RTSP_URLS=${SIMANEAT_APPS_TEST_RTSP_URLS}"
if [[ -n "${SIMANEAT_APPS_TEST_OPTIVIEW_VIDEO_PORT:-}" ]]; then
  echo "  SIMANEAT_APPS_TEST_OPTIVIEW_VIDEO_PORT=${SIMANEAT_APPS_TEST_OPTIVIEW_VIDEO_PORT}"
fi
if [[ -n "${SIMANEAT_APPS_TEST_OPTIVIEW_JSON_PORT:-}" ]]; then
  echo "  SIMANEAT_APPS_TEST_OPTIVIEW_JSON_PORT=${SIMANEAT_APPS_TEST_OPTIVIEW_JSON_PORT}"
fi
