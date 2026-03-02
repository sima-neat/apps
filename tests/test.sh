#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-build}"
PYTHON_TEST_BIN="${PYTHON_TEST_BIN:-python3}"

RUN_UNIT=0
RUN_E2E=0
RUN_CPP=0
RUN_PYTHON=0
RUN_ALL=0

usage() {
  cat <<'EOF'
Usage: ./tests/test.sh [options]

Options:
  --unit              Run unit tests (C++ + Python)
  --e2e               Run e2e tests (C++ + Python)
  --cpp               Run C++ tests (unit + e2e)
  --python            Run Python tests (unit + e2e)
  --all               Run everything
  --build-dir <dir>   Build directory (default: build)
  -h, --help          Show help

Combine flags for specific subsets:
  --unit --cpp        C++ unit tests only
  --e2e --python      Python e2e tests only

Environment:
  SIMANEAT_APPS_TEST_MODELS_DIR     Model directory (default: assets/models)
  SIMANEAT_APPS_TEST_INPUT_DIR      Input images directory (default: assets/test_images)
  SIMANEAT_APPS_TEST_OUTPUT_DIR     E2E output temp root (default: /tmp)
  SIMANEAT_APPS_TEST_CLASSIFICATION_IMAGE  Classification goldfish image path
  SIMANEAT_APPS_TEST_KEEP_OUTPUT    Keep e2e output dirs (1=yes, default: 0)
  SIMANEAT_APPS_TEST_MPK            Explicit model .tar.gz path override
  SIMANEAT_APPS_TEST_RTSP_URL       Single RTSP stream URL
  SIMANEAT_APPS_TEST_RTSP_URLS      Comma-separated RTSP URLs (multistream)
  SIMANEAT_APPS_TEST_TIMEOUT_MS     Timeout in ms (default: 180000)
  SIMANEAT_APPS_TEST_REQUIRE_E2E    Set to 1 to fail instead of skip on missing deps
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --unit) RUN_UNIT=1; shift ;;
    --e2e) RUN_E2E=1; shift ;;
    --cpp) RUN_CPP=1; shift ;;
    --python) RUN_PYTHON=1; shift ;;
    --all) RUN_ALL=1; shift ;;
    --build-dir) BUILD_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

# Default: run everything
if [[ "${RUN_UNIT}" -eq 0 && "${RUN_E2E}" -eq 0 && \
      "${RUN_CPP}" -eq 0 && "${RUN_PYTHON}" -eq 0 && \
      "${RUN_ALL}" -eq 0 ]]; then
  RUN_ALL=1
fi

if [[ "${RUN_ALL}" -eq 1 ]]; then
  RUN_UNIT=1
  RUN_E2E=1
  RUN_CPP=1
  RUN_PYTHON=1
fi

# If only --unit or --e2e specified (no language filter), run both languages
if [[ "${RUN_CPP}" -eq 0 && "${RUN_PYTHON}" -eq 0 ]]; then
  RUN_CPP=1
  RUN_PYTHON=1
fi

# If only --cpp or --python specified (no test-type filter), run both types
if [[ "${RUN_UNIT}" -eq 0 && "${RUN_E2E}" -eq 0 ]]; then
  RUN_UNIT=1
  RUN_E2E=1
fi

OVERALL_RC=0
STRICT_E2E="${SIMANEAT_APPS_TEST_REQUIRE_E2E:-0}"
PYTHON_READY=1

format_env_value() {
  local value="${1:-}"
  if [[ -z "${value}" ]]; then
    echo "<unset>"
  else
    echo "${value}"
  fi
}

format_env_status() {
  local value="${1:-}"
  if [[ -z "${value}" ]]; then
    echo "unset"
  else
    echo "set"
  fi
}

preflight_e2e_env() {
  local strict="${SIMANEAT_APPS_TEST_REQUIRE_E2E:-0}"
  local models_dir_raw="${SIMANEAT_APPS_TEST_MODELS_DIR:-}"
  local input_dir_raw="${SIMANEAT_APPS_TEST_INPUT_DIR:-}"
  local timeout_raw="${SIMANEAT_APPS_TEST_TIMEOUT_MS:-180000}"
  local output_dir_raw="${SIMANEAT_APPS_TEST_OUTPUT_DIR:-}"
  local class_image_raw="${SIMANEAT_APPS_TEST_CLASSIFICATION_IMAGE:-}"
  local keep_output_raw="${SIMANEAT_APPS_TEST_KEEP_OUTPUT:-0}"
  local rtsp_url="${SIMANEAT_APPS_TEST_RTSP_URL:-}"
  local rtsp_urls="${SIMANEAT_APPS_TEST_RTSP_URLS:-}"
  local mpk="${SIMANEAT_APPS_TEST_MPK:-}"

  local models_dir="${models_dir_raw:-${ROOT_DIR}/assets/models}"
  local input_dir="${input_dir_raw:-${ROOT_DIR}/assets/test_images}"
  local output_dir="${output_dir_raw:-/tmp}"
  local class_image="${class_image_raw:-${ROOT_DIR}/assets/test_images_classification/goldfish.jpeg}"

  local model_count=0
  if [[ -d "${models_dir}" ]]; then
    model_count="$(find "${models_dir}" -maxdepth 1 -type f -name '*.tar.gz' | wc -l | tr -d ' ')"
  fi

  echo ""
  echo "  E2E environment preflight"
  echo "  $(printf '%.0s-' {1..50})"
  echo "  SIMANEAT_APPS_TEST_REQUIRE_E2E : $(format_env_status "${SIMANEAT_APPS_TEST_REQUIRE_E2E:-}") -> ${strict}"
  echo "  SIMANEAT_APPS_TEST_MODELS_DIR  : $(format_env_status "${models_dir_raw}") -> ${models_dir} (${model_count} model tar.gz files)"
  echo "  SIMANEAT_APPS_TEST_INPUT_DIR   : $(format_env_status "${input_dir_raw}") -> ${input_dir}"
  echo "  SIMANEAT_APPS_TEST_OUTPUT_DIR  : $(format_env_status "${output_dir_raw}") -> ${output_dir}"
  echo "  SIMANEAT_APPS_TEST_CLASSIFICATION_IMAGE : $(format_env_status "${class_image_raw}") -> ${class_image}"
  echo "  SIMANEAT_APPS_TEST_KEEP_OUTPUT : $(format_env_status "${SIMANEAT_APPS_TEST_KEEP_OUTPUT:-}") -> ${keep_output_raw}"
  echo "  SIMANEAT_APPS_TEST_TIMEOUT_MS  : $(format_env_status "${SIMANEAT_APPS_TEST_TIMEOUT_MS:-}") -> ${timeout_raw}"
  echo "  SIMANEAT_APPS_TEST_MPK         : $(format_env_status "${mpk}") -> $(format_env_value "${mpk}")"
  echo "  SIMANEAT_APPS_TEST_RTSP_URL    : $(format_env_status "${rtsp_url}") -> $(format_env_value "${rtsp_url}")"
  echo "  SIMANEAT_APPS_TEST_RTSP_URLS   : $(format_env_status "${rtsp_urls}") -> $(format_env_value "${rtsp_urls}")"

  local preflight_fail=0
  if [[ ! "${timeout_raw}" =~ ^[0-9]+$ ]]; then
    echo "  [FAIL] SIMANEAT_APPS_TEST_TIMEOUT_MS must be a positive integer."
    preflight_fail=1
  fi
  if [[ "${keep_output_raw}" != "0" && "${keep_output_raw}" != "1" ]]; then
    echo "  [FAIL] SIMANEAT_APPS_TEST_KEEP_OUTPUT must be 0 or 1."
    preflight_fail=1
  fi
  if [[ ! -d "${models_dir}" ]]; then
    echo "  [WARN] models directory does not exist: ${models_dir}"
    if [[ "${strict}" == "1" ]]; then
      preflight_fail=1
    fi
  fi
  if [[ ! -d "${input_dir}" ]]; then
    echo "  [WARN] input images directory does not exist: ${input_dir}"
    if [[ "${strict}" == "1" ]]; then
      preflight_fail=1
    fi
  fi
  if [[ ! -f "${class_image}" ]]; then
    echo "  [WARN] classification image does not exist: ${class_image}"
    if [[ "${strict}" == "1" ]]; then
      preflight_fail=1
    fi
  fi
  if [[ -e "${output_dir}" && ! -d "${output_dir}" ]]; then
    echo "  [WARN] output root exists but is not a directory: ${output_dir}"
    if [[ "${strict}" == "1" ]]; then
      preflight_fail=1
    fi
  elif [[ -d "${output_dir}" && ! -w "${output_dir}" ]]; then
    echo "  [WARN] output root is not writable: ${output_dir}"
    if [[ "${strict}" == "1" ]]; then
      preflight_fail=1
    fi
  fi
  if [[ -z "${rtsp_url}" && -z "${rtsp_urls}" ]]; then
    echo "  [WARN] no RTSP env configured; RTSP e2e tests will skip."
    echo "  [WARN] RTSP e2e tests: single-rtsp-object-detection-optiview, multistream-rtsp-detection-pipeline, live-rtsp-depth-estimation."
    echo "  [WARN] make sure you run RTSP source(s) in another terminal before e2e:"
    echo "         ./utils/rtsp/run_rtsp_server.sh --host-port 8555 --detach"
    echo "         ./utils/rtsp/stream_cam.sh --video-path assets/videos/neat-video-1.mp4 --rtsp-host 127.0.0.1 --rtsp-port 8555 --rtsp-path stream0"
    echo "         # optional second stream for multistream tests"
    echo "         ./utils/rtsp/stream_cam.sh --video-path assets/videos/neat-video-2.mp4 --rtsp-host 127.0.0.1 --rtsp-port 8555 --rtsp-path stream1"
    echo "         export SIMANEAT_APPS_TEST_RTSP_URL=rtsp://127.0.0.1:8555/stream0"
    echo "         export SIMANEAT_APPS_TEST_RTSP_URLS=rtsp://127.0.0.1:8555/stream0,rtsp://127.0.0.1:8555/stream1"
    if [[ "${strict}" == "1" ]]; then
      preflight_fail=1
    fi
  elif [[ -n "${rtsp_url}" && -z "${rtsp_urls}" ]]; then
    echo "  [WARN] SIMANEAT_APPS_TEST_RTSP_URLS is unset; multistream RTSP tests may skip."
    echo "  [WARN] set SIMANEAT_APPS_TEST_RTSP_URLS with 2+ URLs for full multistream coverage."
  elif [[ -z "${rtsp_url}" && -n "${rtsp_urls}" ]]; then
    echo "  [WARN] SIMANEAT_APPS_TEST_RTSP_URL is unset; single-stream RTSP tests may skip."
    echo "  [WARN] set SIMANEAT_APPS_TEST_RTSP_URL for full single-stream RTSP coverage."
  fi
  if [[ "${model_count}" == "0" && -z "${mpk}" ]]; then
    echo "  [WARN] no models discovered and SIMANEAT_APPS_TEST_MPK is unset; e2e tests may skip/fail."
    if [[ "${strict}" == "1" ]]; then
      preflight_fail=1
    fi
  fi

  return "${preflight_fail}"
}

resolve_pytest_python() {
  if "${PYTHON_TEST_BIN}" -m pytest --version >/dev/null 2>&1; then
    return 0
  fi
  for cand in "/root/pyneat/.venv/bin/python3" "/root/pyneat/bin/python3"; do
    if [[ -x "${cand}" ]] && "${cand}" -m pytest --version >/dev/null 2>&1; then
      PYTHON_TEST_BIN="${cand}"
      return 0
    fi
  done
  return 1
}

if [[ "${RUN_E2E}" -eq 1 ]]; then
  if ! preflight_e2e_env; then
    echo "  [FAIL] e2e preflight failed."
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# C++ tests via CTest
# ---------------------------------------------------------------------------
run_ctest() {
  local label="$1"
  local label_upper
  label_upper="$(echo "${label}" | tr '[:lower:]' '[:upper:]')"

  local build_path="${ROOT_DIR}/${BUILD_DIR}"
  if [[ ! -d "${build_path}" ]]; then
    echo "  [SKIP] Build directory ${build_path} not found. Run ./build.sh first."
    return 0
  fi

  echo ""
  echo "  C++ ${label_upper} tests (ctest -L ${label})"
  echo "  $(printf '%.0s-' {1..50})"

  local log_file
  log_file="$(mktemp)"
  if ! ctest --test-dir "${build_path}" -L "${label}" --output-on-failure | tee "${log_file}"; then
    OVERALL_RC=1
  fi
  if [[ "${STRICT_E2E}" == "1" && "${label}" == "e2e" ]]; then
    if rg -q '\*\*\*Skipped|Not Run' "${log_file}"; then
      echo "  [FAIL] Strict e2e mode is enabled but C++ e2e tests were skipped."
      OVERALL_RC=1
    fi
  fi
  rm -f "${log_file}"
}

if [[ "${RUN_CPP}" -eq 1 ]]; then
  if [[ "${RUN_UNIT}" -eq 1 ]]; then
    run_ctest "unit"
  fi
  if [[ "${RUN_E2E}" -eq 1 ]]; then
    run_ctest "e2e"
  fi
fi

# ---------------------------------------------------------------------------
# Python tests via pytest
# ---------------------------------------------------------------------------
run_pytest() {
  local marker="$1"
  local marker_upper
  marker_upper="$(echo "${marker}" | tr '[:lower:]' '[:upper:]')"

  echo ""
  echo "  Python ${marker_upper} tests (pytest -m ${marker})"
  echo "  $(printf '%.0s-' {1..50})"

  local log_file
  log_file="$(mktemp)"
  if ! "${PYTHON_TEST_BIN}" -m pytest -c "${ROOT_DIR}/tests/pytest.ini" -m "${marker}" --rootdir="${ROOT_DIR}" -v | tee "${log_file}"; then
    OVERALL_RC=1
  fi
  if [[ "${STRICT_E2E}" == "1" && "${marker}" == "e2e" ]]; then
    if rg -q '[0-9]+ skipped' "${log_file}"; then
      echo "  [FAIL] Strict e2e mode is enabled but Python e2e tests were skipped."
      OVERALL_RC=1
    fi
  fi
  rm -f "${log_file}"
}

if [[ "${RUN_PYTHON}" -eq 1 ]]; then
  if ! resolve_pytest_python; then
    echo ""
    echo "  [FAIL] pytest is not available for ${PYTHON_TEST_BIN}."
    echo "  Set PYTHON_TEST_BIN to a Python interpreter with pytest installed."
    OVERALL_RC=1
    PYTHON_READY=0
  fi
fi

if [[ "${RUN_PYTHON}" -eq 1 && "${PYTHON_READY}" -eq 1 ]]; then
  echo ""
  echo "  Python test interpreter: ${PYTHON_TEST_BIN}"
  if [[ "${RUN_UNIT}" -eq 1 ]]; then
    run_pytest "unit"
  fi
  if [[ "${RUN_E2E}" -eq 1 ]]; then
    run_pytest "e2e"
  fi
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
if [[ "${OVERALL_RC}" -eq 0 ]]; then
  echo "  All requested tests passed."
else
  echo "  Some tests failed."
fi

exit "${OVERALL_RC}"
