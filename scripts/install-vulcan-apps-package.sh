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

# Enter, end of input and NEAT_APPS_OVERWRITE=1 answer yes.
confirm_default_yes() {
  local question="$1"
  local response=""

  if [[ "${NEAT_APPS_OVERWRITE:-0}" == "1" ]]; then
    return 0
  fi
  printf '❓ %s [Y/n] ' "${question}"
  if ! IFS= read -r response; then
    echo
    return 0
  fi
  [[ ! "${response}" =~ ^[Nn] ]]
}

run_with_timeout() {
  local seconds="$1"
  shift
  if command -v timeout >/dev/null 2>&1; then
    timeout "${seconds}" "$@"
  else
    "$@"
  fi
}

# Prints the version that core@<ref>:<spec> installs, read from its published
# metadata, or nothing when the metadata is unavailable.
resolve_compatible_core_version() {
  local resolve_output
  resolve_output="$(SIMA_CLI_CHECK_FOR_UPDATE=0 run_with_timeout 60 "${SIMA_CLI_RESOLVED}" \
    neat install --json -t minimal "${CORE_PACKAGE}" </dev/null 2>/dev/null)" \
    || return 0
  RESOLVE_OUTPUT="${resolve_output}" python3 - 2>/dev/null <<'PY' || true
import json
import os
import urllib.request

# sima-cli prints status lines before the JSON object.
text = os.environ["RESOLVE_OUTPUT"]
resolved, _ = json.JSONDecoder().raw_decode(text[text.index("{"):])
with urllib.request.urlopen(resolved["metadata_url"], timeout=20) as response:
    version = json.load(response)["version"]
if isinstance(version, str):
    print(version)
PY
}

# Core's neat marks the SDK with the same file, and neat itself is missing
# until Core is installed.
detect_install_place() {
  if [[ -f "${ELXR_SDK_RELEASE_FILE:-/etc/sdk-release}" ]]; then
    echo "SDK"
  else
    echo "DevKit"
  fi
}

# Prints core=matches|differs|missing, core_version= and insight_version= from
# one `neat --json --offline` status.
summarize_neat_status() {
  NEAT_STATUS_JSON="$1" python3 - "${COMPATIBLE_CORE_VERSION}" "${CORE_REF}" <<'PY'
import json
import os
import re
import sys

compatible_version, core_ref = sys.argv[1:]
core_version = insight_version = ""
core_state = "missing"


def normalize(version):
    # A PyNeat wheel spells the "-" of a branch name as "_".
    return re.sub(r"[-_]", ".", version or "")


try:
    components = json.loads(os.environ["NEAT_STATUS_JSON"])["components"]
    core = components.get("core") or {}
    core_version = core.get("version") or ""
    pyneat_version = (components.get("pyneat") or {}).get("version")
    insight_version = (components.get("insight") or {}).get("version") or ""
    if core_version:
        # A local Core build keeps the release version, so the ref must match too.
        matches = (
            core_version == compatible_version
            and (core.get("provenance") or {}).get("vulcanRef") == core_ref
            and normalize(pyneat_version) == normalize(core_version)
        )
        core_state = "matches" if matches else "differs"
except (ValueError, KeyError, TypeError, AttributeError):
    pass

print(f"core={core_state}")
print(f"core_version={core_version}")
print(f"insight_version={'' if insight_version == 'unknown' else insight_version}")
PY
}

summary_field() {
  printf '%s\n' "$1" | sed -n "s/^$2=//p"
}

read_local_neat_status() {
  if command -v neat >/dev/null 2>&1; then
    run_with_timeout 20 neat --json --offline 2>/dev/null || true
  fi
}

# Sets DEVKIT_LINK to not-sdk, unpaired, unreachable or paired, and DEVKIT_STATUS
# to the paired DevKit's summary, read from the account Core's installer uses.
check_paired_devkit() {
  local status_json=""
  local exit_code=0

  DEVKIT_TARGET="${DEVKIT_DEPLOY_USER:-sima}@${DEVKIT_SYNC_DEVKIT_IP:-}"
  DEVKIT_STATUS=""
  if [[ -z "${DEVKIT_SYNC_DEVKIT_IP:-}" ]]; then
    DEVKIT_LINK="not-sdk"
    if [[ "${INSTALL_PLACE}" == "SDK" ]]; then
      DEVKIT_LINK="unpaired"
    fi
    return 0
  fi
  # -n keeps ssh from reading the answers meant for the prompts.
  status_json="$(run_with_timeout 30 ssh -n -T -o BatchMode=yes -o ConnectTimeout=5 \
    "${DEVKIT_TARGET}" neat --json --offline 2>/dev/null)" || exit_code=$?
  # ssh exits 255 when it cannot connect or log in, and timeout exits 124.
  if [[ "${exit_code}" == 255 || "${exit_code}" == 124 ]]; then
    DEVKIT_LINK="unreachable"
    return 0
  fi
  DEVKIT_LINK="paired"
  DEVKIT_STATUS="$(summarize_neat_status "${status_json}")"
}

# Pads only the label: printf counts an emoji's bytes, not its width.
print_status_row() {
  printf '   %s %-8s %s\n' "$1" "$2" "$3"
}

print_core_row() {
  local label="$1"
  local summary="$2"
  local version

  version="$(summary_field "${summary}" core_version)"
  case "$(summary_field "${summary}" core)" in
    matches) print_status_row "✅" "${label}" "${version}" ;;
    differs)
      # A local build or a mismatched PyNeat keeps the compatible version number.
      if [[ "${version}" == "${COMPATIBLE_CORE_VERSION}" ]]; then
        version+=" (different build)"
      fi
      print_status_row "🔄" "${label}" "${version}"
      ;;
    *) print_status_row "❌" "${label}" "not installed" ;;
  esac
}

print_dependency_status() {
  local insight_version
  insight_version="$(summary_field "${LOCAL_STATUS}" insight_version)"

  echo
  echo "🔍 Compatible Neat Library: ${COMPATIBLE_CORE_VERSION:-unknown (${CORE_PACKAGE})}"
  print_core_row "${INSTALL_PLACE}" "${LOCAL_STATUS}"
  case "${DEVKIT_LINK}" in
    unpaired) print_status_row "🔌" "DevKit" "not paired" ;;
    unreachable) print_status_row "🔌" "DevKit" "unreachable (${DEVKIT_TARGET})" ;;
    paired) print_core_row "DevKit" "${DEVKIT_STATUS}" ;;
  esac
  if [[ -n "${insight_version}" ]]; then
    print_status_row "✅" "Insight" "${insight_version}"
  else
    print_status_row "❌" "Insight" "not installed"
  fi
}

# Sets CORE_INSTALL to current, declined, sdk or all. "all" lets Core's
# installer also install on the paired DevKit; "sdk" installs in the SDK only.
decide_core_install() {
  local local_core devkit_core="" question=""

  local_core="$(summary_field "${LOCAL_STATUS}" core)"
  if [[ "${DEVKIT_LINK}" == "paired" ]]; then
    devkit_core="$(summary_field "${DEVKIT_STATUS}" core)"
  fi
  # Only a paired DevKit's Core can make a matching SDK Core worth installing.
  if [[ "${local_core}" == "matches" && "${devkit_core:-matches}" == "matches" ]]; then
    CORE_INSTALL="current"
    return 0
  fi

  CORE_INSTALL="sdk"
  case "${DEVKIT_LINK}" in
    not-sdk)
      CORE_INSTALL="all"
      question="Install Neat Library ${CORE_LABEL}?"
      ;;
    paired)
      if [[ "${devkit_core}" == "matches" ]]; then
        question="Install Neat Library ${CORE_LABEL} in the SDK?"
      else
        CORE_INSTALL="all"
        question="Install Neat Library ${CORE_LABEL} in the SDK and on the DevKit?"
      fi
      ;;
    unpaired)
      echo "⚠️  This SDK is not paired with a DevKit, so the DevKit will not get" \
        "Neat Library ${CORE_LABEL}. To pair, run on the host:" \
        "sima-cli sdk setup --devkit <devkit-ip>"
      question="Install Neat Library ${CORE_LABEL} in the SDK only?"
      ;;
    unreachable)
      echo "⚠️  The DevKit is unreachable, so it will not get Neat Library ${CORE_LABEL}." \
        "Check that it is on and connected, or run on the host: sima-cli sdk doctor network"
      question="Install Neat Library ${CORE_LABEL} in the SDK only?"
      ;;
  esac
  if ! confirm_default_yes "${question}"; then
    CORE_INSTALL="declined"
    echo "⚠️  Keeping the installed Neat Library." \
      "Examples built for Neat Library ${CORE_LABEL} may fail."
  fi
}

# Sets INSIGHT_INSTALL to current, declined or install.
decide_insight_install() {
  INSIGHT_INSTALL="current"
  if [[ -n "$(summary_field "${LOCAL_STATUS}" insight_version)" ]]; then
    return 0
  fi
  INSIGHT_INSTALL="declined"
  if confirm_default_yes "Install Insight?"; then
    INSIGHT_INSTALL="install"
  fi
}

report_installed_dependencies() {
  local where=""

  if [[ "${CORE_INSTALL}" != "current" || "${INSIGHT_INSTALL}" != "current" ]]; then
    return 0
  fi
  # The DevKit's version is unknown when it is unpaired or unreachable.
  if [[ "${DEVKIT_LINK}" == "unpaired" || "${DEVKIT_LINK}" == "unreachable" ]]; then
    where=" in the SDK"
  fi
  echo "✅ Neat Library and Insight are already installed${where}."
}

install_core() {
  case "${CORE_INSTALL}" in
    current | declined) return 0 ;;
    sdk) local -x NEAT_INSTALLER_SKIP_DEVKIT_SYNC=ON ;;
  esac
  echo
  echo "Installing Neat Library ${CORE_LABEL}:"
  run_sima_cli_install "${CORE_INSTALL_DIR}" "${SIMA_CLI_RESOLVED}" \
    -d . -t minimal "${CORE_PACKAGE}"
}

install_insight() {
  if [[ "${INSIGHT_INSTALL}" != "install" ]]; then
    return 0
  fi
  echo
  echo "Installing Insight:"
  run_sima_cli_install "${INSIGHT_INSTALL_DIR}" "${SIMA_CLI_RESOLVED}" -d . insight
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
  echo "⚠️  Skipping the Neat Library and Insight installation."
else
  SIMA_CLI_RESOLVED="$(resolve_sima_cli_bin)" || {
    echo "ERROR: sima-cli is required to install Neat Library and Insight." >&2
    exit 1
  }

  cleanup_dependency_workspace
  CORE_PACKAGE="core@${CORE_REF}:${CORE_SPEC}"
  COMPATIBLE_CORE_VERSION="$(resolve_compatible_core_version)"
  CORE_LABEL="${COMPATIBLE_CORE_VERSION:-${CORE_PACKAGE}}"
  INSTALL_PLACE="$(detect_install_place)"
  LOCAL_STATUS="$(summarize_neat_status "$(read_local_neat_status)")"
  check_paired_devkit
  print_dependency_status
  decide_core_install
  decide_insight_install
  report_installed_dependencies
  install_core
  install_insight
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
