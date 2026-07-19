#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-build}"
CONFIG_FILE=""
STRICT_CLI=0
ORIGINAL_ARGS=("$@")

RUN_UNIT=0
RUN_E2E=0
RUN_CPP=0
RUN_PYTHON=0
RUN_ALL=0

CONFIG_ENV_VARS=(
  APPS_ROOT
  PYTHON_TEST_BIN
  SIMANEAT_APPS_TEST_MODELS_DIR
  SIMANEAT_APPS_TEST_INPUT_DIR
  SIMANEAT_APPS_TEST_OUTPUT_DIR
  SIMANEAT_APPS_TEST_SCOPE_FILE
  SIMANEAT_APPS_TEST_CLASSIFICATION_IMAGE
  SIMANEAT_APPS_TEST_KEEP_OUTPUT
  SIMANEAT_APPS_TEST_WRITE_SUMMARY_LOGS
  SIMANEAT_APPS_TEST_WRITE_PROCESS_LOGS
  SIMANEAT_TEST_RTSP_H264_URL
  SIMANEAT_TEST_RTSP_H264_URLS
  SIMANEAT_TEST_RTSP_MJPEG_URL
  SIMANEAT_TEST_RTSP_MJPEG_URLS
  SIMANEAT_TEST_HTTP_MJPEG_URL
  SIMANEAT_TEST_HTTP_MJPEG_URLS
  SIMANEAT_APPS_TEST_TIMEOUT_MS
  SIMANEAT_APPS_TEST_REQUIRE_E2E
  SIMANEAT_APPS_TEST_LABELS_FILE
  SIMANEAT_APPS_TEST_INSIGHT_HOST
  SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT
  SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT
  NEAT_APPS_SKIP_MODEL_DOWNLOAD
)

PROCESS_ENV_WAS_SET=()
PROCESS_ENV_VALUE=()

capture_process_env() {
  local name
  local i=0
  for name in "${CONFIG_ENV_VARS[@]}"; do
    if [[ "${!name+x}" == "x" ]]; then
      PROCESS_ENV_WAS_SET["${i}"]=1
      PROCESS_ENV_VALUE["${i}"]="${!name}"
    else
      PROCESS_ENV_WAS_SET["${i}"]=0
    fi
    i=$((i + 1))
  done
}

restore_process_env_overrides() {
  local name
  local i=0
  for name in "${CONFIG_ENV_VARS[@]}"; do
    if [[ "${PROCESS_ENV_WAS_SET[${i}]}" == "1" ]]; then
      export "${name}=${PROCESS_ENV_VALUE[${i}]}"
    fi
    i=$((i + 1))
  done
}

capture_process_env

usage() {
  cat <<'EOF'
Usage: ./tests/test.sh [options]

Options:
  --unit              Run unit tests (C++ + Python)
  --e2e               Run e2e tests (C++ + Python)
  --cpp               Run C++ tests (unit + e2e)
  --python            Run Python tests (unit + e2e)
  --all               Run everything
  --config <file>     Load local test configuration file
  --strict            Fail if selected e2e prerequisites are missing or skipped
  --build-dir <dir>   Build directory (default: build)
  -h, --help          Show help

Config:
  tests/configs/.env.local is auto-loaded when present.
  Use --config <file> to load a different local config.
  Process environment variables override config file values.

Combine flags for specific subsets:
  --unit --cpp        C++ unit tests only
  --e2e --python      Python e2e tests only

Environment:
  SIMANEAT_APPS_TEST_MODELS_DIR     Model directory (default: models)
  SIMANEAT_APPS_TEST_INPUT_DIR      Input images directory (default: assets/datasets-test/coco)
  SIMANEAT_APPS_TEST_OUTPUT_DIR     E2E output root (default: sandbox-test)
  SIMANEAT_APPS_TEST_CLASSIFICATION_IMAGE  Classification goldfish image path
  SIMANEAT_APPS_TEST_KEEP_OUTPUT    Keep e2e output dirs (1=yes, default: 1)
  SIMANEAT_APPS_TEST_WRITE_SUMMARY_LOGS    Write summary logs (1=yes, default: 1)
  SIMANEAT_APPS_TEST_WRITE_PROCESS_LOGS    Write command/stdout/stderr logs (1=yes, default: 1)
  SIMANEAT_TEST_RTSP_H264_URL       Single RTSP H.264 stream URL
  SIMANEAT_TEST_RTSP_H264_URLS      Comma-separated RTSP H.264 URLs
  SIMANEAT_TEST_RTSP_MJPEG_URL      Single RTSP MJPEG stream URL
  SIMANEAT_TEST_RTSP_MJPEG_URLS     Comma-separated RTSP MJPEG URLs
  SIMANEAT_TEST_HTTP_MJPEG_URL      Single HTTP MJPEG stream URL
  SIMANEAT_TEST_HTTP_MJPEG_URLS     Comma-separated HTTP MJPEG URLs
  SIMANEAT_APPS_TEST_TIMEOUT_MS     Timeout in ms (default: 180000)
  SIMANEAT_APPS_TEST_REQUIRE_E2E    Backward-compatible strict e2e env flag
  SIMANEAT_APPS_TEST_SCOPE_FILE     Test scope source (default: examples)
  NEAT_APPS_SKIP_MODEL_DOWNLOAD     Skip e2e model download (1=yes, default: 0)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --unit) RUN_UNIT=1; shift ;;
    --e2e) RUN_E2E=1; shift ;;
    --cpp) RUN_CPP=1; shift ;;
    --python) RUN_PYTHON=1; shift ;;
    --all) RUN_ALL=1; shift ;;
    --config)
      if [[ $# -lt 2 ]]; then
        echo "--config requires a file path" >&2
        exit 1
      fi
      CONFIG_FILE="$2"
      shift 2
      ;;
    --strict) STRICT_CLI=1; shift ;;
    --build-dir)
      if [[ $# -lt 2 ]]; then
        echo "--build-dir requires a directory path" >&2
        exit 1
      fi
      BUILD_DIR="$2"
      shift 2
      ;;
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
PYTHON_READY=1

resolve_config_file() {
  local config_path="$1"
  if [[ "${config_path}" == /* ]]; then
    printf '%s\n' "${config_path}"
  elif [[ -f "${config_path}" ]]; then
    printf '%s\n' "${config_path}"
  else
    printf '%s\n' "${ROOT_DIR}/${config_path}"
  fi
}

load_config_file() {
  local config_path="$1"
  if [[ ! -f "${config_path}" ]]; then
    echo "Config file not found: ${config_path}" >&2
    exit 1
  fi

  set -a
  # shellcheck disable=SC1090
  source "${config_path}"
  set +a
  restore_process_env_overrides
}

resolve_test_runtime_env() {
  if [[ -z "${APPS_ROOT:-}" ]]; then
    export APPS_ROOT="${ROOT_DIR}"
  fi

  export SIMANEAT_APPS_TEST_MODELS_DIR="${SIMANEAT_APPS_TEST_MODELS_DIR:-${APPS_ROOT}/models}"
  export SIMANEAT_APPS_TEST_INPUT_DIR="${SIMANEAT_APPS_TEST_INPUT_DIR:-${APPS_ROOT}/assets/datasets-test/coco}"
  export SIMANEAT_APPS_TEST_OUTPUT_DIR="${SIMANEAT_APPS_TEST_OUTPUT_DIR:-${APPS_ROOT}/sandbox-test}"
  export SIMANEAT_APPS_TEST_SCOPE_FILE="${SIMANEAT_APPS_TEST_SCOPE_FILE:-${APPS_ROOT}/examples}"
  export SIMANEAT_APPS_TEST_CLASSIFICATION_IMAGE="${SIMANEAT_APPS_TEST_CLASSIFICATION_IMAGE:-${APPS_ROOT}/assets/datasets-test/imagenet/goldfish.jpeg}"
  export SIMANEAT_APPS_TEST_KEEP_OUTPUT="${SIMANEAT_APPS_TEST_KEEP_OUTPUT:-1}"
  export SIMANEAT_APPS_TEST_WRITE_SUMMARY_LOGS="${SIMANEAT_APPS_TEST_WRITE_SUMMARY_LOGS:-1}"
  export SIMANEAT_APPS_TEST_WRITE_PROCESS_LOGS="${SIMANEAT_APPS_TEST_WRITE_PROCESS_LOGS:-1}"
  export SIMANEAT_APPS_TEST_TIMEOUT_MS="${SIMANEAT_APPS_TEST_TIMEOUT_MS:-180000}"
  export SIMANEAT_TEST_RTSP_H264_URL="${SIMANEAT_TEST_RTSP_H264_URL:-}"
  export SIMANEAT_TEST_RTSP_H264_URLS="${SIMANEAT_TEST_RTSP_H264_URLS:-}"
  export SIMANEAT_TEST_RTSP_MJPEG_URL="${SIMANEAT_TEST_RTSP_MJPEG_URL:-}"
  export SIMANEAT_TEST_RTSP_MJPEG_URLS="${SIMANEAT_TEST_RTSP_MJPEG_URLS:-}"
  export SIMANEAT_TEST_HTTP_MJPEG_URL="${SIMANEAT_TEST_HTTP_MJPEG_URL:-}"
  export SIMANEAT_TEST_HTTP_MJPEG_URLS="${SIMANEAT_TEST_HTTP_MJPEG_URLS:-}"
  export SIMANEAT_APPS_TEST_INSIGHT_HOST="${SIMANEAT_APPS_TEST_INSIGHT_HOST:-127.0.0.1}"
  export NEAT_APPS_SKIP_MODEL_DOWNLOAD="${NEAT_APPS_SKIP_MODEL_DOWNLOAD:-0}"
}

test_invocation() {
  printf './tests/test.sh'
  local arg
  for arg in "${ORIGINAL_ARGS[@]}"; do
    printf ' %q' "${arg}"
  done
  printf '\n'
}

start_summary_log() {
  local language="$1"
  local marker="$2"
  if [[ "${SIMANEAT_APPS_TEST_WRITE_SUMMARY_LOGS:-1}" != "1" ]]; then
    mktemp
    return
  fi
  local output_root="${SIMANEAT_APPS_TEST_OUTPUT_DIR:-${ROOT_DIR}/sandbox-test}"
  local summary_dir="${output_root}/summary"
  mkdir -p "${summary_dir}"
  local summary_file="${summary_dir}/${language}-${marker}.log"
  {
    echo "command: $(test_invocation)"
    echo "cwd: ${ROOT_DIR}"
    echo "mode: ${language} ${marker}"
    echo "date_utc: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo ""
  } >"${summary_file}"
  printf '%s\n' "${summary_file}"
}

if [[ -z "${APPS_ROOT:-}" ]]; then
  export APPS_ROOT="${ROOT_DIR}"
fi

if [[ -z "${CONFIG_FILE}" && -f "${ROOT_DIR}/tests/configs/.env.local" ]]; then
  CONFIG_FILE="${ROOT_DIR}/tests/configs/.env.local"
fi

if [[ -n "${CONFIG_FILE}" ]]; then
  CONFIG_FILE="$(resolve_config_file "${CONFIG_FILE}")"
  load_config_file "${CONFIG_FILE}"
fi

resolve_test_runtime_env

if [[ "${STRICT_CLI}" -eq 1 || "${SIMANEAT_APPS_TEST_REQUIRE_E2E:-0}" == "1" ]]; then
  STRICT_MODE=1
else
  STRICT_MODE=0
fi
export SIMANEAT_APPS_TEST_REQUIRE_E2E="${STRICT_MODE}"

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

is_absolute_path() {
  local value="${1:-}"
  [[ -z "${value}" || "${value}" == /* ]]
}

SCOPE_PYTHON_BIN=""

python_executable_exists() {
  local cand="$1"
  if [[ "${cand}" == */* ]]; then
    [[ -x "${cand}" ]]
  else
    command -v "${cand}" >/dev/null 2>&1
  fi
}

resolve_scope_python() {
  if [[ -n "${SCOPE_PYTHON_BIN}" ]]; then
    return 0
  fi

  local candidates=()
  if [[ -n "${PYTHON_TEST_BIN:-}" ]]; then
    candidates+=("${PYTHON_TEST_BIN}")
  fi
  candidates+=("python3")
  candidates+=("${HOME}/pyneat/bin/python3")
  candidates+=("/media/nvme/pyneat/bin/python3")
  candidates+=("${HOME}/.pyneat/bin/python3")

  local cand
  for cand in "${candidates[@]}"; do
    if ! python_executable_exists "${cand}"; then
      continue
    fi
    if "${cand}" -c "import yaml" >/dev/null 2>&1; then
      SCOPE_PYTHON_BIN="${cand}"
      return 0
    fi
  done

  echo "  [FAIL] tests/test-scope.yaml requires a Python interpreter with PyYAML." >&2
  echo "         Install PyYAML into pyneat or set PYTHON_TEST_BIN." >&2
  return 1
}

scope_tool() {
  resolve_scope_python || return 1
  "${SCOPE_PYTHON_BIN}" "${APPS_ROOT}/tests/utils/test_scope.py" \
    --scope-file "${SIMANEAT_APPS_TEST_SCOPE_FILE}" "$@"
}

validate_test_scope() {
  echo ""
  echo "  Validating test scope"
  echo "  $(printf '%.0s-' {1..50})"
  echo "  Scope source: ${SIMANEAT_APPS_TEST_SCOPE_FILE}"
  scope_tool validate --quiet
}

export_cpp_model_files() {
  local label="$1"
  if [[ "${label}" != "e2e" ]]; then
    return 0
  fi

  local model_files
  if ! model_files="$(scope_tool model-files --kind "${label}" --language cpp)"; then
    return 1
  fi
  export SIMANEAT_APPS_TEST_MODEL_FILES="${model_files}"
}

preflight_e2e_env() {
  local strict="${STRICT_MODE}"
  local models_dir_raw="${SIMANEAT_APPS_TEST_MODELS_DIR:-}"
  local input_dir_raw="${SIMANEAT_APPS_TEST_INPUT_DIR:-}"
  local scope_file_raw="${SIMANEAT_APPS_TEST_SCOPE_FILE:-}"
  local timeout_raw="${SIMANEAT_APPS_TEST_TIMEOUT_MS:-180000}"
  local output_dir_raw="${SIMANEAT_APPS_TEST_OUTPUT_DIR:-}"
  local class_image_raw="${SIMANEAT_APPS_TEST_CLASSIFICATION_IMAGE:-}"
  local keep_output_raw="${SIMANEAT_APPS_TEST_KEEP_OUTPUT:-0}"
  local write_summary_raw="${SIMANEAT_APPS_TEST_WRITE_SUMMARY_LOGS:-1}"
  local write_process_raw="${SIMANEAT_APPS_TEST_WRITE_PROCESS_LOGS:-1}"
  local insight_host_raw="${SIMANEAT_APPS_TEST_INSIGHT_HOST:-}"
  local skip_model_download_raw="${NEAT_APPS_SKIP_MODEL_DOWNLOAD:-0}"
  local rtsp_h264_url="${SIMANEAT_TEST_RTSP_H264_URL:-}"
  local rtsp_h264_urls="${SIMANEAT_TEST_RTSP_H264_URLS:-}"
  local rtsp_mjpeg_url="${SIMANEAT_TEST_RTSP_MJPEG_URL:-}"
  local rtsp_mjpeg_urls="${SIMANEAT_TEST_RTSP_MJPEG_URLS:-}"
  local http_mjpeg_url="${SIMANEAT_TEST_HTTP_MJPEG_URL:-}"
  local http_mjpeg_urls="${SIMANEAT_TEST_HTTP_MJPEG_URLS:-}"
  local models_dir="${models_dir_raw:-${ROOT_DIR}/models}"
  local input_dir="${input_dir_raw:-${ROOT_DIR}/assets/datasets-test/coco}"
  local scope_file="${scope_file_raw:-${ROOT_DIR}/examples}"
  local output_dir="${output_dir_raw:-/tmp}"
  local class_image="${class_image_raw:-${ROOT_DIR}/assets/datasets-test/imagenet/goldfish.jpeg}"

  local model_count=0
  if [[ -d "${models_dir}" ]]; then
    model_count="$(find "${models_dir}" -maxdepth 1 -type f -name '*.tar.gz' | wc -l | tr -d ' ')"
  fi

  echo ""
  echo "  E2E environment preflight"
  echo "  $(printf '%.0s-' {1..50})"
  echo "  Strict mode                 : ${strict}"
  echo "  Test config file            : $(format_env_value "${CONFIG_FILE}")"
  echo "  SIMANEAT_APPS_TEST_MODELS_DIR  : $(format_env_status "${models_dir_raw}") -> ${models_dir} (${model_count} model tar.gz files)"
  echo "  SIMANEAT_APPS_TEST_INPUT_DIR   : $(format_env_status "${input_dir_raw}") -> ${input_dir}"
  echo "  SIMANEAT_APPS_TEST_SCOPE_FILE  : $(format_env_status "${scope_file_raw}") -> ${scope_file}"
  echo "  SIMANEAT_APPS_TEST_OUTPUT_DIR  : $(format_env_status "${output_dir_raw}") -> ${output_dir}"
  echo "  SIMANEAT_APPS_TEST_CLASSIFICATION_IMAGE : $(format_env_status "${class_image_raw}") -> ${class_image}"
  echo "  SIMANEAT_APPS_TEST_KEEP_OUTPUT : $(format_env_status "${SIMANEAT_APPS_TEST_KEEP_OUTPUT:-}") -> ${keep_output_raw}"
  echo "  SIMANEAT_APPS_TEST_WRITE_SUMMARY_LOGS : $(format_env_status "${SIMANEAT_APPS_TEST_WRITE_SUMMARY_LOGS:-}") -> ${write_summary_raw}"
  echo "  SIMANEAT_APPS_TEST_WRITE_PROCESS_LOGS : $(format_env_status "${SIMANEAT_APPS_TEST_WRITE_PROCESS_LOGS:-}") -> ${write_process_raw}"
  echo "  SIMANEAT_APPS_TEST_INSIGHT_HOST : $(format_env_status "${insight_host_raw}") -> $(format_env_value "${insight_host_raw}")"
  echo "  NEAT_APPS_SKIP_MODEL_DOWNLOAD : $(format_env_status "${NEAT_APPS_SKIP_MODEL_DOWNLOAD:-}") -> ${skip_model_download_raw}"
  echo "  SIMANEAT_APPS_TEST_TIMEOUT_MS  : $(format_env_status "${SIMANEAT_APPS_TEST_TIMEOUT_MS:-}") -> ${timeout_raw}"
  echo "  SIMANEAT_TEST_RTSP_H264_URL    : $(format_env_status "${rtsp_h264_url}") -> $(format_env_value "${rtsp_h264_url}")"
  echo "  SIMANEAT_TEST_RTSP_H264_URLS   : $(format_env_status "${rtsp_h264_urls}") -> $(format_env_value "${rtsp_h264_urls}")"
  echo "  SIMANEAT_TEST_RTSP_MJPEG_URL   : $(format_env_status "${rtsp_mjpeg_url}") -> $(format_env_value "${rtsp_mjpeg_url}")"
  echo "  SIMANEAT_TEST_RTSP_MJPEG_URLS  : $(format_env_status "${rtsp_mjpeg_urls}") -> $(format_env_value "${rtsp_mjpeg_urls}")"
  echo "  SIMANEAT_TEST_HTTP_MJPEG_URL   : $(format_env_status "${http_mjpeg_url}") -> $(format_env_value "${http_mjpeg_url}")"
  echo "  SIMANEAT_TEST_HTTP_MJPEG_URLS  : $(format_env_status "${http_mjpeg_urls}") -> $(format_env_value "${http_mjpeg_urls}")"

  local preflight_fail=0
  if [[ ! "${timeout_raw}" =~ ^[0-9]+$ ]]; then
    echo "  [FAIL] SIMANEAT_APPS_TEST_TIMEOUT_MS must be a positive integer."
    preflight_fail=1
  fi
  if [[ "${keep_output_raw}" != "0" && "${keep_output_raw}" != "1" ]]; then
    echo "  [FAIL] SIMANEAT_APPS_TEST_KEEP_OUTPUT must be 0 or 1."
    preflight_fail=1
  fi
  if [[ "${write_summary_raw}" != "0" && "${write_summary_raw}" != "1" ]]; then
    echo "  [FAIL] SIMANEAT_APPS_TEST_WRITE_SUMMARY_LOGS must be 0 or 1."
    preflight_fail=1
  fi
  if [[ "${write_process_raw}" != "0" && "${write_process_raw}" != "1" ]]; then
    echo "  [FAIL] SIMANEAT_APPS_TEST_WRITE_PROCESS_LOGS must be 0 or 1."
    preflight_fail=1
  fi
  if [[ "${skip_model_download_raw}" != "0" && "${skip_model_download_raw}" != "1" ]]; then
    echo "  [FAIL] NEAT_APPS_SKIP_MODEL_DOWNLOAD must be 0 or 1."
    preflight_fail=1
  fi
  if ! is_absolute_path "${models_dir_raw}"; then
    echo "  [FAIL] SIMANEAT_APPS_TEST_MODELS_DIR must be an absolute path."
    preflight_fail=1
  fi
  if ! is_absolute_path "${input_dir_raw}"; then
    echo "  [FAIL] SIMANEAT_APPS_TEST_INPUT_DIR must be an absolute path."
    preflight_fail=1
  fi
  if ! is_absolute_path "${scope_file_raw}"; then
    echo "  [FAIL] SIMANEAT_APPS_TEST_SCOPE_FILE must be an absolute path."
    preflight_fail=1
  fi
  if ! is_absolute_path "${output_dir_raw}"; then
    echo "  [FAIL] SIMANEAT_APPS_TEST_OUTPUT_DIR must be an absolute path."
    preflight_fail=1
  fi
  if ! is_absolute_path "${class_image_raw}"; then
    echo "  [FAIL] SIMANEAT_APPS_TEST_CLASSIFICATION_IMAGE must be an absolute path."
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
  if [[ ! -e "${scope_file}" ]]; then
    echo "  [FAIL] test scope source does not exist: ${scope_file}"
    preflight_fail=1
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
  if [[ -z "${rtsp_h264_url}" && -z "${rtsp_h264_urls}" && -z "${rtsp_mjpeg_url}" && \
        -z "${rtsp_mjpeg_urls}" && -z "${http_mjpeg_url}" && -z "${http_mjpeg_urls}" ]]; then
    echo "  [WARN] no codec stream env configured; stream e2e tests will skip."
    echo "  [WARN] stream e2e tests: single-stream-object-detector, single-stream-instance-segmenter,"
    echo "         multi-stream-object-detector, multi-stream-people-tracker."
    echo "  [WARN] make sure stream source(s) are running before e2e. If you want a quick setup, tool-mediasources on the host is one option:"
    echo "         sima-cli install gh:sima-ai/tool-mediasources"
    echo "         open preview.html"
    echo "         export SIMANEAT_TEST_RTSP_H264_URL=<rtsp-h264-url>"
    echo "         export SIMANEAT_TEST_RTSP_H264_URLS=<rtsp-h264-url-0>,<rtsp-h264-url-1>"
    echo "         export SIMANEAT_TEST_RTSP_MJPEG_URL=<rtsp-mjpeg-url>"
    echo "         export SIMANEAT_TEST_HTTP_MJPEG_URL=<http-mjpeg-url>"
    if [[ "${strict}" == "1" ]]; then
      echo "  [FAIL] strict mode requires codec stream config."
      echo "         Local: cp tests/configs/.env.example tests/configs/.env.local and set stream URLs."
      echo "         CI   : set SIMANEAT_TEST_RTSP_H264_URL, SIMANEAT_TEST_RTSP_H264_URLS,"
      echo "                SIMANEAT_TEST_RTSP_MJPEG_URL, and SIMANEAT_TEST_HTTP_MJPEG_URL as needed."
      preflight_fail=1
    fi
  else
    if [[ -z "${rtsp_h264_url}" ]]; then
      echo "  [WARN] SIMANEAT_TEST_RTSP_H264_URL is unset; single-stream RTSP H.264 tests may skip."
    fi
    if [[ -z "${rtsp_h264_urls}" ]]; then
      echo "  [WARN] SIMANEAT_TEST_RTSP_H264_URLS is unset; multistream RTSP H.264 tests may skip."
    fi
    if [[ -z "${rtsp_mjpeg_url}" ]]; then
      echo "  [WARN] SIMANEAT_TEST_RTSP_MJPEG_URL is unset; single-stream RTSP MJPEG tests may skip."
    fi
    if [[ -z "${http_mjpeg_url}" ]]; then
      echo "  [WARN] SIMANEAT_TEST_HTTP_MJPEG_URL is unset; single-stream HTTP MJPEG tests may skip."
    fi
  fi
  if [[ "${model_count}" == "0" ]]; then
    echo "  [WARN] no models discovered under SIMANEAT_APPS_TEST_MODELS_DIR; e2e tests may skip/fail."
    if [[ "${strict}" == "1" ]]; then
      preflight_fail=1
    fi
  fi

  return "${preflight_fail}"
}

ensure_e2e_models() {
  if [[ "${RUN_E2E}" -ne 1 ]]; then
    return 0
  fi
  if [[ "${NEAT_APPS_SKIP_MODEL_DOWNLOAD:-0}" == "1" ]]; then
    echo ""
    echo "  Skipping e2e model download because NEAT_APPS_SKIP_MODEL_DOWNLOAD=1."
    return 0
  fi
  local downloader="${APPS_ROOT}/scripts/download_models.sh"
  if [[ ! -f "${downloader}" ]]; then
    echo ""
    echo "  Skipping e2e model download because ${downloader} was not found."
    return 0
  fi

  local language_args=()
  if [[ "${RUN_PYTHON}" -eq 1 ]]; then
    language_args+=(--language python)
  fi
  if [[ "${RUN_CPP}" -eq 1 ]]; then
    language_args+=(--language cpp)
  fi

  echo ""
  echo "  Ensuring scoped e2e models are available"
  echo "  $(printf '%.0s-' {1..50})"
  TEST_SCOPE_PYTHON_BIN="${SCOPE_PYTHON_BIN}" \
    MODELS_DIR="${SIMANEAT_APPS_TEST_MODELS_DIR}" bash "${downloader}" \
    --scope-file "${SIMANEAT_APPS_TEST_SCOPE_FILE}" \
    --kind e2e \
    "${language_args[@]}"
}

PYTEST_CHECKED_CANDIDATES=()
PYTEST_INSTALL_TARGET=""

python_candidate_exists() {
  local cand="$1"
  if [[ "${cand}" == */* ]]; then
    [[ -x "${cand}" ]]
  else
    command -v "${cand}" >/dev/null 2>&1
  fi
}

resolve_pytest_python() {
  local cand
  local python_candidates=()
  PYTEST_CHECKED_CANDIDATES=()
  PYTEST_INSTALL_TARGET=""

  if [[ -n "${PYTHON_TEST_BIN:-}" ]]; then
    python_candidates+=("${PYTHON_TEST_BIN}")
  fi
  python_candidates+=("${HOME}/pyneat/bin/python3")
  python_candidates+=("/media/nvme/pyneat/bin/python3")
  python_candidates+=("${HOME}/.pyneat/bin/python3")
  python_candidates+=("/opt/pyneat/bin/python3")
  python_candidates+=("/opt/sima/pyneat/bin/python3")
  python_candidates+=("/opt/sima.ai/pyneat/bin/python3")
  if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    python_candidates+=("${VIRTUAL_ENV}/bin/python3")
  fi
  python_candidates+=("${APPS_ROOT}/.venv/bin/python3")
  if [[ "${APPS_ROOT}" != "${ROOT_DIR}" ]]; then
    python_candidates+=("${ROOT_DIR}/.venv/bin/python3")
  fi
  python_candidates+=("python3")

  for cand in "${python_candidates[@]}"; do
    if ! python_candidate_exists "${cand}"; then
      PYTEST_CHECKED_CANDIDATES+=("${cand} (not found)")
      continue
    fi
    if [[ -z "${PYTEST_INSTALL_TARGET}" ]]; then
      PYTEST_INSTALL_TARGET="${cand}"
    fi
    if "${cand}" -m pytest --version >/dev/null 2>&1; then
      export PYTHON_TEST_BIN="${cand}"
      return 0
    fi
    PYTEST_CHECKED_CANDIDATES+=("${cand} (missing pytest)")
  done

  return 1
}

if ! validate_test_scope; then
  echo "  [FAIL] test scope validation failed."
  exit 1
fi

if [[ "${RUN_E2E}" -eq 1 ]]; then
  ensure_e2e_models
  if ! preflight_e2e_env; then
    echo "  [FAIL] e2e preflight failed."
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# C++ tests via CTest or packaged prebuilt binaries
# ---------------------------------------------------------------------------
run_packaged_cpp_tests() {
  local label="$1"
  local label_upper
  local suffix
  local skipped_count=0
  local enabled_examples=()
  label_upper="$(echo "${label}" | tr '[:lower:]' '[:upper:]')"
  suffix="_${label}_test"

  if ! export_cpp_model_files "${label}"; then
    OVERALL_RC=1
    return 0
  fi

  mapfile -t enabled_examples < <(scope_tool enabled-examples --kind "${label}" --language cpp)

  cpp_test_bins=()
  while IFS= read -r test_bin; do
    local rel category rest packaged_example example_key
    rel="${test_bin#${ROOT_DIR}/examples/}"
    category="${rel%%/*}"
    rest="${rel#*/}"
    packaged_example="${rest%%/*}"
    example_key="${category}/${packaged_example%_cpp}"
    local enabled=0
    local scoped
    for scoped in "${enabled_examples[@]}"; do
      if [[ "${scoped}" == "${example_key}" ]]; then
        enabled=1
        break
      fi
    done
    if [[ "${enabled}" -eq 1 ]]; then
      cpp_test_bins+=("${test_bin}")
    fi
  done < <(find "${ROOT_DIR}/examples" -type f -path '*/tests/cpp/*' -name "*${suffix}" | sort)

  if [[ "${#cpp_test_bins[@]}" -eq 0 ]]; then
    echo "  [SKIP] No enabled packaged C++ ${label} tests found under ${ROOT_DIR}/examples."
    if [[ "${STRICT_MODE}" == "1" && "${label}" == "e2e" ]]; then
      echo "  [FAIL] Strict mode is enabled but no packaged C++ e2e tests were enabled."
      OVERALL_RC=1
    fi
    return 0
  fi

  echo ""
  echo "  C++ ${label_upper} tests (packaged binaries)"
  echo "  $(printf '%.0s-' {1..50})"
  local summary_file
  summary_file="$(start_summary_log "cpp" "${label}")"

  local test_bin
  for test_bin in "${cpp_test_bins[@]}"; do
    local example_name test_dir example_bin rc
    example_name="$(basename "${test_bin}")"
    example_name="${example_name%${suffix}}"
    test_dir="$(dirname "${test_bin}")"
    # Packaged layout:
    # examples/<category>/<example>[/_cpp]/tests/cpp/<example>_{unit,e2e}_test
    # examples/<category>/<example>[/_cpp]/<example>
    example_bin="${test_dir}/../../${example_name}"

    echo "  [RUN] ${test_bin#${ROOT_DIR}/}"
    echo "[RUN] ${test_bin#${ROOT_DIR}/}" >>"${summary_file}"
    if [[ ! -x "${example_bin}" ]]; then
      echo "  [ERR] example binary not found for test: ${example_bin}"
      echo "[ERR] example binary not found for test: ${example_bin}" >>"${summary_file}"
      OVERALL_RC=1
      continue
    fi

    set +e
    "${test_bin}" "${example_bin}" 2>&1 | tee -a "${summary_file}"
    rc=${PIPESTATUS[0]}
    set -e

    if [[ "${rc}" -eq 77 ]]; then
      echo "  [SKIP] ${test_bin#${ROOT_DIR}/} (exit 77)"
      skipped_count=$((skipped_count + 1))
      continue
    fi

    if [[ "${rc}" -ne 0 ]]; then
      OVERALL_RC=1
    fi
  done

  if [[ "${STRICT_MODE}" == "1" && "${label}" == "e2e" && "${skipped_count}" -gt 0 ]]; then
    echo "  [FAIL] Strict mode is enabled but packaged C++ e2e tests were skipped (${skipped_count})."
    OVERALL_RC=1
  fi
}

run_ctest() {
  local label="$1"
  local label_upper
  label_upper="$(echo "${label}" | tr '[:lower:]' '[:upper:]')"

  local build_path
  if [[ "${BUILD_DIR}" == /* ]]; then
    build_path="${BUILD_DIR}"
  else
    build_path="${ROOT_DIR}/${BUILD_DIR}"
  fi
  if [[ ! -d "${build_path}" ]]; then
    run_packaged_cpp_tests "${label}"
    return $?
  fi

  echo ""
  echo "  C++ ${label_upper} tests (ctest -L ${label})"
  echo "  $(printf '%.0s-' {1..50})"

  local log_file
  local test_regex
  if ! export_cpp_model_files "${label}"; then
    OVERALL_RC=1
    return 0
  fi

  test_regex="$(scope_tool ctest-regex --kind "${label}")" || {
    OVERALL_RC=1
    return 0
  }
  if [[ -z "${test_regex}" ]]; then
    echo "  [SKIP] No enabled C++ ${label} tests in ${SIMANEAT_APPS_TEST_SCOPE_FILE}."
    if [[ "${STRICT_MODE}" == "1" && "${label}" == "e2e" ]]; then
      echo "  [FAIL] Strict mode is enabled but no C++ e2e tests were enabled."
      OVERALL_RC=1
    fi
    return 0
  fi

  log_file="$(start_summary_log "cpp" "${label}")"
  if ! ctest --test-dir "${build_path}" -L "${label}" -R "${test_regex}" --output-on-failure | tee -a "${log_file}"; then
    OVERALL_RC=1
  fi
  if [[ "${STRICT_MODE}" == "1" && "${label}" == "e2e" ]]; then
    if rg -q '\*\*\*Skipped|Not Run' "${log_file}"; then
      echo "  [FAIL] Strict mode is enabled but C++ e2e tests were skipped."
      OVERALL_RC=1
    fi
  fi
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
  local skipped_count=0
  marker_upper="$(echo "${marker}" | tr '[:lower:]' '[:upper:]')"

  case "${marker}" in
    unit|e2e) ;;
    *)
      echo "  [FAIL] unsupported python test marker: ${marker}"
      OVERALL_RC=1
      return
      ;;
  esac

  echo ""
  echo "  Python ${marker_upper} tests (isolated pytest per example)"
  echo "  $(printf '%.0s-' {1..50})"
  local summary_file
  summary_file="$(start_summary_log "python" "${marker}")"

  python_test_files=()
  mapfile -t python_test_files < <(scope_tool python-files --kind "${marker}")

  if [[ "${#python_test_files[@]}" -eq 0 ]]; then
    echo "  [SKIP] No enabled Python ${marker} tests found in ${SIMANEAT_APPS_TEST_SCOPE_FILE}."
    if [[ "${STRICT_MODE}" == "1" && "${marker}" == "e2e" ]]; then
      echo "  [FAIL] Strict mode is enabled but no Python e2e tests were enabled."
      OVERALL_RC=1
    fi
    return 0
  fi

  local test_file
  for test_file in "${python_test_files[@]}"; do
    local log_file rc
    log_file="$(mktemp)"

    echo "  [RUN] ${test_file#${ROOT_DIR}/}"
    echo "[RUN] ${test_file#${ROOT_DIR}/}" >>"${summary_file}"
    set +e
    "${PYTHON_TEST_BIN}" -m pytest -c "${ROOT_DIR}/tests/pytest.ini" -m "${marker}" \
      --rootdir="${ROOT_DIR}" -v "${test_file}" | tee "${log_file}" | tee -a "${summary_file}"
    rc=${PIPESTATUS[0]}
    set -e

    if [[ "${rc}" -ne 0 ]]; then
      OVERALL_RC=1
    fi

    if [[ "${STRICT_MODE}" == "1" && "${marker}" == "e2e" ]] && \
       grep -Eq '[0-9]+ skipped' "${log_file}"; then
      echo "  [FAIL] Strict mode is enabled but Python e2e tests were skipped."
      OVERALL_RC=1
      skipped_count=$((skipped_count + 1))
    fi

    rm -f "${log_file}"
  done
}

if [[ "${RUN_PYTHON}" -eq 1 ]]; then
  if ! resolve_pytest_python; then
    echo ""
    echo "  [FAIL] pytest is not available for any known Python test interpreter."
    echo "  Checked:"
    for candidate_status in "${PYTEST_CHECKED_CANDIDATES[@]}"; do
      echo "    - ${candidate_status}"
    done
    if [[ -n "${PYTEST_INSTALL_TARGET}" ]]; then
      echo "  Install pytest into the selected interpreter:"
      echo "    ${PYTEST_INSTALL_TARGET} -m pip install pytest"
    else
      echo "  Set PYTHON_TEST_BIN to a Python interpreter with pip and pytest available."
    fi
    echo "  You can also set PYTHON_TEST_BIN in tests/configs/.env.local."
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
