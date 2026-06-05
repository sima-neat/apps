#!/usr/bin/env bash
set -euo pipefail

DEST_DIR="${NEAT_APPS_INSTALL_DIR:-neat-apps}"
RUNTIME_SRC="${NEAT_APPS_RUNTIME_SRC:-neat-apps-runtime}"
RUNTIME_DST="${DEST_DIR}/neat-apps-runtime"
VULCAN_ENV="${NEAT_VULCAN_ENV:-${VULCAN_ENV:-dev}}"
NEAT_CORE_INSTALL_DIR=""
NEAT_CORE_INSTALL_DIR_OWNED=0

resolve_sima_cli_bin() {
  if [[ -n "${SIMA_CLI_BIN:-}" && -x "${SIMA_CLI_BIN}" ]]; then
    printf '%s\n' "${SIMA_CLI_BIN}"
    return 0
  fi
  if command -v sima-cli >/dev/null 2>&1; then
    command -v sima-cli
    return 0
  fi
  local candidate
  for candidate in \
    /data/sima-cli/.venv/bin/sima-cli \
    "${HOME}/.local/bin/sima-cli" \
    "${HOME}/sima-cli/.venv/bin/sima-cli" \
    /opt/sima-cli/.venv/bin/sima-cli \
    /opt/bin/sima-cli; do
    if [[ -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

extract_neat_core_target() {
  local json_path="$1"
  python3 - <<'PY' "${json_path}"
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as fh:
    data = json.load(fh)

neat_core = data.get("neat_core", {})
branch = str(neat_core.get("branch", "")).strip()
version = str(neat_core.get("version", "")).strip()

if not branch or not version:
    raise SystemExit(1)

print(branch)
print(version)
PY
}

prepare_vulcan_core_install_dir() {
  NEAT_CORE_INSTALL_DIR_OWNED=0
  if [[ -z "${NEAT_APPS_CORE_INSTALL_DIR:-}" ]]; then
    NEAT_CORE_INSTALL_DIR="$(mktemp -d /tmp/neat-apps-core-install.XXXXXX)"
    NEAT_CORE_INSTALL_DIR_OWNED=1
    return 0
  fi

  NEAT_CORE_INSTALL_DIR="${NEAT_APPS_CORE_INSTALL_DIR}"
  if [[ -z "${NEAT_CORE_INSTALL_DIR}" || "${NEAT_CORE_INSTALL_DIR}" == "/" ]]; then
    echo "ERROR: unsafe NEAT_APPS_CORE_INSTALL_DIR: ${NEAT_CORE_INSTALL_DIR}" >&2
    exit 1
  fi
  rm -rf "${NEAT_CORE_INSTALL_DIR}"
  mkdir -p "${NEAT_CORE_INSTALL_DIR}"
}

cleanup_vulcan_core_install_dir() {
  if [[ "${NEAT_CORE_INSTALL_DIR_OWNED}" == "1" && -n "${NEAT_CORE_INSTALL_DIR}" ]]; then
    rm -rf "${NEAT_CORE_INSTALL_DIR}"
  fi
  NEAT_CORE_INSTALL_DIR=""
  NEAT_CORE_INSTALL_DIR_OWNED=0
}

run_sima_cli_core_install() {
  local sima_cli_bin="$1"
  shift
  local log_path
  log_path="$(mktemp /tmp/neat-apps-sima-cli-install.XXXXXX.log)"
  if ! "${sima_cli_bin}" neat install "$@" 2>&1 | tee "${log_path}"; then
    rm -f "${log_path}"
    return 1
  fi
  if grep -Fq "Installation script exited" "${log_path}"; then
    echo "ERROR: sima-cli reported a NEAT core installer failure." >&2
    rm -f "${log_path}"
    return 1
  fi
  rm -f "${log_path}"
}

if [[ ! -d "${RUNTIME_SRC}" ]]; then
  runtime_candidates=()
  while IFS= read -r candidate; do
    runtime_candidates+=("${candidate}")
  done < <(find . -mindepth 1 -maxdepth 3 -type d -name "${RUNTIME_SRC}" | sort)
  if [[ "${#runtime_candidates[@]}" -eq 1 ]]; then
    RUNTIME_SRC="${runtime_candidates[0]}"
  else
    echo "ERROR: ${RUNTIME_SRC} was not extracted from the apps package." >&2
    printf 'Found candidates:\n' >&2
    printf '  %s\n' "${runtime_candidates[@]}" >&2
    exit 1
  fi
fi

rm -rf "${DEST_DIR}"
mkdir -p "${DEST_DIR}"
mv "${RUNTIME_SRC}" "${RUNTIME_DST}"

NEAT_CORE_JSON_PATH="${RUNTIME_DST}/neat-core.json"
if [[ ! -f "${NEAT_CORE_JSON_PATH}" ]]; then
  echo "ERROR: extracted apps package is missing neat-core.json." >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 is required to parse ${NEAT_CORE_JSON_PATH}." >&2
  exit 1
fi

if ! NEAT_CORE_TARGET_OUTPUT="$(extract_neat_core_target "${NEAT_CORE_JSON_PATH}")"; then
  echo "ERROR: failed to parse NEAT core dependency from ${NEAT_CORE_JSON_PATH}." >&2
  exit 1
fi

SIMA_CLI_RESOLVED="$(resolve_sima_cli_bin)" || {
  echo "ERROR: sima-cli is required to install matching NEAT core." >&2
  exit 1
}

NEAT_CORE_BRANCH="$(printf '%s\n' "${NEAT_CORE_TARGET_OUTPUT}" | sed -n '1p')"
NEAT_CORE_VERSION="$(printf '%s\n' "${NEAT_CORE_TARGET_OUTPUT}" | sed -n '2p')"

echo
echo "Installing matching NEAT core from Vulcan:"
echo "  Environment: ${VULCAN_ENV}"
echo "  Branch     : ${NEAT_CORE_BRANCH}"
echo "  Version    : ${NEAT_CORE_VERSION}"
prepare_vulcan_core_install_dir
echo "  Scratch dir: ${NEAT_CORE_INSTALL_DIR}"
INSTALL_STATUS=0
(
  cd "${NEAT_CORE_INSTALL_DIR}"
  run_sima_cli_core_install "${SIMA_CLI_RESOLVED}" \
    --env "${VULCAN_ENV}" \
    -d . \
    -t minimal \
    "core@${NEAT_CORE_BRANCH}:${NEAT_CORE_VERSION}"
) || INSTALL_STATUS=$?
cleanup_vulcan_core_install_dir
if [[ "${INSTALL_STATUS}" -ne 0 ]]; then
  exit "${INSTALL_STATUS}"
fi

DOWNLOAD_MODELS_SCRIPT="${RUNTIME_DST}/scripts/download_models.sh"
if [[ "${NEAT_APPS_SKIP_MODEL_DOWNLOAD:-0}" == "1" ]]; then
  echo
  echo "Skipping model downloads because NEAT_APPS_SKIP_MODEL_DOWNLOAD=1."
elif [[ -x "${DOWNLOAD_MODELS_SCRIPT}" || -f "${DOWNLOAD_MODELS_SCRIPT}" ]]; then
  echo
  echo "Downloading models referenced by packaged README metadata ..."
  (
    cd "${RUNTIME_DST}"
    chmod +x scripts/download_models.sh
    SIMA_CLI_BIN="${SIMA_CLI_RESOLVED}" bash scripts/download_models.sh
  )
fi

echo
echo "Installed apps runtime under:"
echo "  ${DEST_DIR}"
