#!/usr/bin/env bash
set -euo pipefail

DEST_DIR="${NEAT_APPS_INSTALL_DIR:-prebuilt-apps}"
RUNTIME_SRC="${NEAT_APPS_RUNTIME_SRC:-prebuilt-apps}"
INSTALL_ROOT="$(dirname "${DEST_DIR}")"
LEGACY_RUNTIME_DIR="${INSTALL_ROOT}/neat-apps/neat-apps-runtime"
DEPS_DIR="${INSTALL_ROOT}/deps"
DEPS_MARKER="${DEPS_DIR}/.neat-apps-installer-owned"
DEPS_MARKER_VALUE="sima-neat/apps"
CORE_INSTALL_DIR="${DEPS_DIR}/core"
INSIGHT_INSTALL_DIR="${DEPS_DIR}/insight"
DEPS_WORKSPACE_ACTIVE=0

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

dependency_workspace_is_owned() {
  [[ -d "${DEPS_DIR}" && ! -L "${DEPS_DIR}" \
    && -f "${DEPS_MARKER}" && ! -L "${DEPS_MARKER}" ]] \
    && grep -Fqx "${DEPS_MARKER_VALUE}" "${DEPS_MARKER}"
}

cleanup_dependency_workspace() {
  if [[ "${DEPS_WORKSPACE_ACTIVE}" != "1" ]]; then
    return 0
  fi
  if ! dependency_workspace_is_owned; then
    echo "ERROR: refusing to remove dependency workspace without its ownership marker: ${DEPS_DIR}" >&2
    return 1
  fi
  rm -rf "${DEPS_DIR}"
  DEPS_WORKSPACE_ACTIVE=0
}

prepare_dependency_workspace() {
  if [[ -e "${DEPS_DIR}" || -L "${DEPS_DIR}" ]]; then
    if ! dependency_workspace_is_owned; then
      echo "ERROR: refusing to replace unowned dependency workspace: ${DEPS_DIR}" >&2
      return 1
    fi
    rm -rf "${DEPS_DIR}"
  fi

  mkdir -p "${DEPS_DIR}"
  if ! printf '%s\n' "${DEPS_MARKER_VALUE}" >"${DEPS_MARKER}"; then
    rmdir "${DEPS_DIR}" 2>/dev/null || true
    return 1
  fi
  DEPS_WORKSPACE_ACTIVE=1
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

promote_apps_runtime() {
  local source_dir="$1"
  local dest_dir="$2"
  local legacy_dir="$3"
  local dest_parent stage_root staged_runtime backup_dir="" previous_dir=""
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
    if ! backup_dir="$(mktemp -d "${dest_parent}/.prebuilt-apps-backup.XXXXXX")"; then
      echo "ERROR: failed to create an Apps rollback path under ${dest_parent}." >&2
      rm -rf "${stage_root}"
      return 1
    fi
    if ! rmdir "${backup_dir}"; then
      echo "ERROR: failed to prepare the Apps rollback path: ${backup_dir}" >&2
      rm -rf "${stage_root}" "${backup_dir}"
      return 1
    fi
    if ! mv "${previous_dir}" "${backup_dir}"; then
      echo "ERROR: failed to stage the existing Apps runtime for rollback." >&2
      rm -rf "${stage_root}"
      return 1
    fi

    if [[ -e "${backup_dir}/models" || -L "${backup_dir}/models" ]]; then
      rm -rf "${staged_runtime}/models"
      if ! mv "${backup_dir}/models" "${staged_runtime}/models"; then
        echo "ERROR: failed to preserve the existing Apps models directory." >&2
        if ! mv "${backup_dir}" "${previous_dir}"; then
          echo "ERROR: previous Apps runtime remains at ${backup_dir}." >&2
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
      if ! mv "${staged_runtime}/models" "${backup_dir}/models"; then
        echo "ERROR: preserved models remain at ${staged_runtime}/models." >&2
        return 1
      fi
    fi
    if [[ -n "${backup_dir}" ]] && ! mv "${backup_dir}" "${previous_dir}"; then
      echo "ERROR: previous Apps runtime remains at ${backup_dir}." >&2
      return 1
    fi
    rm -rf "${stage_root}"
    return 1
  fi

  rm -rf "${stage_root}"
  if [[ -n "${backup_dir}" ]]; then
    rm -rf "${backup_dir}"
  fi
  if [[ -n "${previous_dir}" && "${previous_dir}" != "${dest_dir}" ]]; then
    rmdir "$(dirname "${previous_dir}")" 2>/dev/null || true
  fi
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

APPS_MANIFEST_PATH="${RUNTIME_SRC}/manifest.json"
if [[ ! -f "${APPS_MANIFEST_PATH}" ]]; then
  echo "ERROR: extracted apps package is missing manifest.json." >&2
  exit 1
fi

if [[ "${NEAT_APPS_SKIP_DEPENDENCIES:-0}" == "1" ]]; then
  echo
  echo "WARNING: Skipping Core and Insight dependency installation."
else
  SIMA_CLI_RESOLVED="$(resolve_sima_cli_bin)" || {
    echo "ERROR: sima-cli is required to install Core and Insight." >&2
    exit 1
  }

  echo
  echo "Installing the latest Core version from main:"
  trap cleanup_dependency_workspace EXIT
  prepare_dependency_workspace
  echo "  Scratch dir: ${CORE_INSTALL_DIR}"
  run_sima_cli_install "${CORE_INSTALL_DIR}" "${SIMA_CLI_RESOLVED}" \
    -d . \
    -t minimal \
    core

  echo
  echo "Installing the latest Insight version from main:"
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
INSTALLED_DIR="$(cd "$(dirname "${DEST_DIR}")" && pwd -P)/$(basename "${DEST_DIR}")"

echo
echo "Installed apps runtime under:"
echo "  ${INSTALLED_DIR}"
