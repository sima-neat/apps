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

# Prints core=skip|install|replace, insight=keep|install, and the installed
# versions they refer to. An unreadable status installs both, as before.
decide_dependency_installs() {
  local status_json=""
  if command -v neat >/dev/null 2>&1; then
    if command -v timeout >/dev/null 2>&1; then
      status_json="$(timeout 20 neat --json --offline 2>/dev/null)" || status_json=""
    else
      status_json="$(neat --json --offline 2>/dev/null)" || status_json=""
    fi
  fi
  NEAT_STATUS_JSON="${status_json}" python3 - "${CORE_REF}" "${CORE_SPEC}" <<'PY'
import json
import os
import re
import sys

ref, spec = sys.argv[1:3]
core_action, core_installed = "install", ""
insight_action, insight_installed = "install", ""


def commit_of(core):
    tag = core.get("tag") or ""
    if re.fullmatch(r"[0-9a-f]{7,40}", tag):
        return tag
    # Tag builds carry no commit in their version; their buildinfo records it.
    buildinfo_path = (core.get("provenance") or {}).get("buildinfoFile")
    if not buildinfo_path:
        return ""
    try:
        with open(buildinfo_path, encoding="utf-8") as handle:
            commit = json.load(handle).get("build", {}).get("ci_requested_git_sha") or ""
    except (OSError, ValueError, AttributeError):
        return ""
    return commit if re.fullmatch(r"[0-9a-f]{7,40}", commit) else ""


try:
    status = json.loads(os.environ["NEAT_STATUS_JSON"])
    components = status["components"]
    core = components.get("core") or {}
    provenance = core.get("provenance") or {}
    core_version = core.get("version")
    # From an SDK container the Core install also updates the paired DevKit,
    # which this status does not describe, so only a board can skip it.
    if status.get("environment", {}).get("mode") == "modalix-board" and core_version:
        installed_ref = provenance.get("vulcanRef") or ""
        commit = commit_of(core)
        core_installed = f"{installed_ref}:{commit[:12]}" if installed_ref and commit else core_version
        same_core = (
            installed_ref == ref
            and provenance.get("vulcanEnvironment") == "prod"
            and bool(commit)
            and (commit.startswith(spec) or spec.startswith(commit))
        )
        if not same_core:
            core_action = "replace"
        # The same Core with a different PyNeat is reinstalled without asking.
        elif (components.get("pyneat") or {}).get("version") == core_version:
            core_action = "skip"
    insight_version = (components.get("insight") or {}).get("version")
    if insight_version and insight_version != "unknown":
        insight_action, insight_installed = "keep", insight_version
except (KeyError, TypeError, ValueError, AttributeError):
    pass

print(f"core={core_action}")
print(f"core_installed={core_installed}")
print(f"insight={insight_action}")
print(f"insight_installed={insight_installed}")
PY
}

confirm_core_replacement() {
  local installed_core="$1"
  local response=""

  echo
  echo "These Apps were built for Core ${CORE_REF}:${CORE_SPEC}."
  echo "  Installed Core: ${installed_core}"
  if [[ "${NEAT_APPS_OVERWRITE:-0}" == "1" ]]; then
    return 0
  fi
  printf 'Replace the installed Core? [Y/n] '
  if ! IFS= read -r response; then
    echo
    return 0
  fi
  [[ ! "${response}" =~ ^[Nn] ]]
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
trap 'cleanup_dependency_workspace; cleanup_package_staging' EXIT

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

  DEPENDENCY_PLAN="$(decide_dependency_installs || true)"
  CORE_ACTION="$(printf '%s\n' "${DEPENDENCY_PLAN}" | sed -n 's/^core=//p')"
  INSTALLED_CORE="$(printf '%s\n' "${DEPENDENCY_PLAN}" | sed -n 's/^core_installed=//p')"
  INSIGHT_ACTION="$(printf '%s\n' "${DEPENDENCY_PLAN}" | sed -n 's/^insight=//p')"
  INSTALLED_INSIGHT="$(printf '%s\n' "${DEPENDENCY_PLAN}" | sed -n 's/^insight_installed=//p')"
  if [[ "${CORE_ACTION}" == "replace" ]] && ! confirm_core_replacement "${INSTALLED_CORE}"; then
    echo "WARNING: Keeping Core ${INSTALLED_CORE}; examples built for Core ${CORE_REF}:${CORE_SPEC} may fail with it."
    CORE_ACTION="keep"
  fi
  cleanup_dependency_workspace

  echo
  if [[ "${CORE_ACTION}" == "skip" ]]; then
    echo "Core ${CORE_REF}:${CORE_SPEC} is already installed; skipping it."
    echo "  To reinstall it: sima-cli neat install -t minimal core@${CORE_REF}:${CORE_SPEC}"
  elif [[ "${CORE_ACTION}" != "keep" ]]; then
    echo "Installing the Core selected by Apps:"
    echo "  Ref        : ${CORE_REF}"
    echo "  Spec       : ${CORE_SPEC}"
    echo "  Scratch dir: ${CORE_INSTALL_DIR}"
    run_sima_cli_install "${CORE_INSTALL_DIR}" "${SIMA_CLI_RESOLVED}" \
      -d . \
      -t minimal \
      "core@${CORE_REF}:${CORE_SPEC}"
  fi

  echo
  if [[ "${INSIGHT_ACTION}" == "keep" ]]; then
    echo "Insight ${INSTALLED_INSIGHT} is already installed; keeping it."
    echo "  To update it: sima-cli neat install insight"
  else
    echo "Installing Insight through sima-cli:"
    echo "  Scratch dir: ${INSIGHT_INSTALL_DIR}"
    run_sima_cli_install "${INSIGHT_INSTALL_DIR}" "${SIMA_CLI_RESOLVED}" \
      -d . \
      insight
  fi

  cleanup_dependency_workspace
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
