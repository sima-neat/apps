#!/usr/bin/env bash
set -euo pipefail

DEST_DIR="${NEAT_APPS_INSTALL_DIR:-prebuilt-apps}"
RUNTIME_SRC="${1:-}"
INSTALL_ROOT="$(dirname "${DEST_DIR}")"
LEGACY_RUNTIME_DIR="${INSTALL_ROOT}/neat-apps/neat-apps-runtime"
DEPS_DIR="${RUNTIME_SRC}/deps"
CORE_INSTALL_DIR="${DEPS_DIR}/core"
INSIGHT_INSTALL_DIR="${DEPS_DIR}/insight"
PACKAGE_DIR="$(pwd -P)"
CORE_METADATA_PATH=""
APPS_METADATA_PATH=""
PACKAGE_EXTRACT_DIR=""
PACKAGE_ARCHIVE=""
BACKUP_DIR="${INSTALL_ROOT}/.prebuilt-apps-backup"

if [[ -z "${RUNTIME_SRC}" || ! -d "${RUNTIME_SRC}" || -L "${RUNTIME_SRC}" ]]; then
  echo "Usage: install-vulcan-apps-package.sh <runtime-source>" >&2
  exit 1
fi

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
  python3 - "${CORE_METADATA_PATH}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    core = json.load(handle).get("neat-core")

if not isinstance(core, dict):
    raise SystemExit(1)

ref = core.get("ref")
spec = core.get("spec")
if not isinstance(ref, str) or not isinstance(spec, str):
    raise SystemExit(1)

ref = ref.strip()
spec = spec.strip()
if not ref or not spec or spec == "latest":
    raise SystemExit(1)

print(ref)
print(spec)
PY
}

cleanup_dependency_workspace() {
  rm -rf "${CORE_INSTALL_DIR}" "${INSIGHT_INSTALL_DIR}"
}

run_sima_cli_install() {
  local install_dir="$1"
  local sima_cli_bin="$2"
  shift 2
  mkdir -p "${install_dir}"
  (
    cd "${install_dir}"
    local log_path
    log_path="$(mktemp ./sima-cli-install.XXXXXX.log)"
    if ! "${sima_cli_bin}" neat install "$@" 2>&1 | tee "${log_path}"; then
      rm -f "${log_path}"
      return 1
    fi
    if grep -Fq "Installation script exited" "${log_path}"; then
      echo "ERROR: sima-cli reported a NEAT installer failure." >&2
      rm -f "${log_path}"
      return 1
    fi
    rm -f "${log_path}"
  )
}

confirm_runtime_replacement() {
  local previous_dir=""
  local response=""

  if [[ -e "${DEST_DIR}" || -L "${DEST_DIR}" ]]; then
    previous_dir="${DEST_DIR}"
  elif [[ -e "${LEGACY_RUNTIME_DIR}" || -L "${LEGACY_RUNTIME_DIR}" ]]; then
    previous_dir="${LEGACY_RUNTIME_DIR}"
  fi
  if [[ -z "${previous_dir}" ]]; then
    return 0
  fi

  echo
  echo "The existing prebuilt-apps installation will be replaced."
  echo "  Preserved models: ${DEST_DIR}/models"
  echo "  Previous files  : ${BACKUP_DIR}"
  if [[ -e "${BACKUP_DIR}" || -L "${BACKUP_DIR}" ]]; then
    echo "  Existing backup : will be replaced"
  fi

  if [[ "${NEAT_APPS_OVERWRITE:-0}" == "1" ]]; then
    return 0
  fi

  printf 'Continue? [y/N] '
  if ! IFS= read -r response || [[ ! "${response}" =~ ^[Yy]$ ]]; then
    echo "Apps installation cancelled."
    return 1
  fi
}

locate_package_staging() {
  local runtime_parent runtime_parent_abs archive_candidate

  runtime_parent="$(dirname "${RUNTIME_SRC}")"
  if [[ "${runtime_parent}" == "." ]]; then
    return 0
  fi
  runtime_parent_abs="$(cd "${runtime_parent}" && pwd -P)"
  if [[ "$(dirname "${runtime_parent_abs}")" != "${PACKAGE_DIR}" \
    || "$(basename "${runtime_parent_abs}")" != neat-apps-* ]]; then
    return 0
  fi

  PACKAGE_EXTRACT_DIR="${runtime_parent_abs}"
  archive_candidate="${PACKAGE_DIR}/$(basename "${runtime_parent_abs}").tar.gz"
  if [[ -f "${archive_candidate}" && ! -L "${archive_candidate}" ]]; then
    PACKAGE_ARCHIVE="${archive_candidate}"
  fi
}

cleanup_package_staging() {
  if [[ -n "${PACKAGE_EXTRACT_DIR}" && -d "${PACKAGE_EXTRACT_DIR}" \
    && ! -L "${PACKAGE_EXTRACT_DIR}" ]]; then
    rm -rf "${PACKAGE_EXTRACT_DIR}"
  fi
  if [[ -n "${PACKAGE_ARCHIVE}" ]]; then
    rm -f "${PACKAGE_ARCHIVE}"
  fi
  rm -f "${PACKAGE_DIR}/install_vulcan_apps_package.sh"
}

promote_apps_runtime() {
  local source_dir="$1"
  local dest_dir="$2"
  local legacy_dir="$3"
  local dest_parent stage_root staged_runtime previous_dir=""
  local models_preserved=0

  dest_parent="$(dirname "${dest_dir}")"
  if ! mkdir -p "${dest_parent}"; then
    echo "ERROR: failed to create Apps install parent: ${dest_parent}" >&2
    return 1
  fi
  if ! stage_root="$(mktemp -d "${dest_parent}/.prebuilt-apps-stage.XXXXXX")"; then
    echo "ERROR: failed to create Apps staging directory under ${dest_parent}." >&2
    return 1
  fi
  staged_runtime="${stage_root}/prebuilt-apps"
  if ! mv "${source_dir}" "${staged_runtime}"; then
    echo "ERROR: failed to stage the Apps runtime under ${dest_parent}." >&2
    rm -rf "${stage_root}"
    return 1
  fi

  if [[ -e "${dest_dir}" || -L "${dest_dir}" ]]; then
    previous_dir="${dest_dir}"
  elif [[ -e "${legacy_dir}" || -L "${legacy_dir}" ]]; then
    previous_dir="${legacy_dir}"
  fi

  if [[ -n "${previous_dir}" ]]; then
    if [[ -e "${BACKUP_DIR}" || -L "${BACKUP_DIR}" ]]; then
      if ! rm -rf "${BACKUP_DIR}"; then
        echo "ERROR: failed to replace the previous Apps backup: ${BACKUP_DIR}" >&2
        rm -rf "${stage_root}"
        return 1
      fi
    fi
    if ! mv "${previous_dir}" "${BACKUP_DIR}"; then
      echo "ERROR: failed to stage the existing Apps runtime for rollback." >&2
      rm -rf "${stage_root}"
      return 1
    fi

    if [[ -e "${BACKUP_DIR}/models" || -L "${BACKUP_DIR}/models" ]]; then
      rm -rf "${staged_runtime}/models"
      if ! mv "${BACKUP_DIR}/models" "${staged_runtime}/models"; then
        echo "ERROR: failed to preserve the existing Apps models directory." >&2
        if ! mv "${BACKUP_DIR}" "${previous_dir}"; then
          echo "ERROR: previous Apps runtime remains at ${BACKUP_DIR}." >&2
        fi
        rm -rf "${stage_root}"
        return 1
      fi
      models_preserved=1
    fi
  fi

  if ! mv "${staged_runtime}" "${dest_dir}"; then
    echo "ERROR: failed to promote the candidate Apps runtime." >&2
    if [[ "${models_preserved}" == "1" ]]; then
      if ! mv "${staged_runtime}/models" "${BACKUP_DIR}/models"; then
        echo "ERROR: preserved models remain at ${staged_runtime}/models." >&2
        return 1
      fi
    fi
    if [[ -n "${previous_dir}" ]] && ! mv "${BACKUP_DIR}" "${previous_dir}"; then
      echo "ERROR: previous Apps runtime remains at ${BACKUP_DIR}." >&2
      return 1
    fi
    rm -rf "${stage_root}"
    return 1
  fi

  rm -rf "${stage_root}"
  if [[ -n "${previous_dir}" && "${previous_dir}" != "${dest_dir}" ]]; then
    rmdir "$(dirname "${previous_dir}")" 2>/dev/null || true
  fi
}

locate_package_staging

APPS_METADATA_PATH="${RUNTIME_SRC}/deps/neat-apps.json"
if [[ ! -f "${APPS_METADATA_PATH}" || -L "${APPS_METADATA_PATH}" ]]; then
  echo "ERROR: Apps package is missing ${APPS_METADATA_PATH}." >&2
  exit 1
fi
CORE_METADATA_PATH="${RUNTIME_SRC}/deps/neat-core.json"
if [[ ! -f "${CORE_METADATA_PATH}" || -L "${CORE_METADATA_PATH}" ]]; then
  echo "ERROR: Apps package is missing ${CORE_METADATA_PATH}." >&2
  exit 1
fi
if ! CORE_TARGET="$(extract_neat_core_target)"; then
  echo "ERROR: invalid Core dependency metadata: ${CORE_METADATA_PATH}." >&2
  exit 1
fi
CORE_REF="$(printf '%s\n' "${CORE_TARGET}" | sed -n '1p')"
CORE_SPEC="$(printf '%s\n' "${CORE_TARGET}" | sed -n '2p')"

confirm_runtime_replacement || exit 1

if [[ "${NEAT_APPS_SKIP_DEPENDENCIES:-0}" == "1" ]]; then
  echo
  echo "WARNING: Skipping Core and Insight dependency installation."
else
  SIMA_CLI_RESOLVED="$(resolve_sima_cli_bin)" || {
    echo "ERROR: sima-cli is required to install Core and Insight." >&2
    exit 1
  }

  echo
  echo "Installing the Core selected by Apps:"
  echo "  Ref        : ${CORE_REF}"
  echo "  Spec       : ${CORE_SPEC}"
  trap cleanup_dependency_workspace EXIT
  cleanup_dependency_workspace
  mkdir -p "${CORE_INSTALL_DIR}" "${INSIGHT_INSTALL_DIR}"
  echo "  Scratch dir: ${CORE_INSTALL_DIR}"
  run_sima_cli_install "${CORE_INSTALL_DIR}" "${SIMA_CLI_RESOLVED}" \
    -d . \
    -t minimal \
    "core@${CORE_REF}:${CORE_SPEC}"

  echo
  echo "Installing Insight through sima-cli:"
  echo "  Scratch dir: ${INSIGHT_INSTALL_DIR}"
  run_sima_cli_install "${INSIGHT_INSTALL_DIR}" "${SIMA_CLI_RESOLVED}" \
    -d . \
    insight

  cleanup_dependency_workspace
  trap - EXIT
fi

if ! promote_apps_runtime "${RUNTIME_SRC}" "${DEST_DIR}" "${LEGACY_RUNTIME_DIR}"; then
  exit 1
fi
cleanup_package_staging
INSTALLED_DIR="$(cd "$(dirname "${DEST_DIR}")" && pwd -P)/$(basename "${DEST_DIR}")"

echo
echo "Installed apps runtime under:"
echo "  ${INSTALLED_DIR}"
if [[ -d "${BACKUP_DIR}" && ! -L "${BACKUP_DIR}" ]]; then
  echo "Previous apps runtime retained under:"
  echo "  ${BACKUP_DIR}"
fi
