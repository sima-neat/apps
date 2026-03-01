#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-build}"
PYTHON_TEST_BIN="${PYTHON_TEST_BIN:-python3}"

RUN_UNIT=0
RUN_E2E=0
RUN_CPP=0
RUN_PYTHON=0
RUN_ALL=0

usage() {
  cat <<'EOF'
Usage: ./test.sh [options]

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
  if ! "${PYTHON_TEST_BIN}" -m pytest -m "${marker}" --rootdir="${ROOT_DIR}" -v | tee "${log_file}"; then
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
