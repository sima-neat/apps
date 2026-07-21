#!/usr/bin/env bash
set -euo pipefail

# install-neat-apps.sh
#
# Downloads a runtime apps bundle from the published apps artifact index.
#
# Usage:
#   ./install-neat-apps.sh [branch] [latest|git-short-hash]
#   ./install-neat-apps.sh /path/to/neat-apps-<branch>-<tag>.tar.gz
#
# Behavior:
# - If no branch is provided, fetch branches.json and prompt the user.
# - If no tag is provided, or the tag is "latest", fetch branch/latest.tag.
# - Download neat-apps-<branch-key>-<tag>.tar.gz from the apps download root.
# - If arg 1 points to an existing .tar.gz file, use that local archive directly.
# - Extract the archive into the current directory as ./prebuilt-apps.

BASE_URL="${NEAT_APPS_BASE_URL:-https://apps.sima-neat.com/download}"
BRANCH="${1:-}"
TAG_INPUT="${2:-latest}"
DEST_DIR="${NEAT_APPS_INSTALL_DIR:-.}"
LOCAL_ARCHIVE="${NEAT_APPS_ARCHIVE:-}"

usage() {
  cat <<'USAGE'
Usage:
  install-neat-apps.sh [branch] [latest|git-short-hash]
  install-neat-apps.sh /path/to/neat-apps-<branch>-<tag>.tar.gz

Environment:
  NEAT_APPS_BASE_URL      Base URL for apps downloads
                          default: https://apps.sima-neat.com/download
  NEAT_APPS_INSTALL_DIR   Parent directory for the extracted prebuilt-apps
                          default: current directory
  NEAT_APPS_ARCHIVE       Use an already downloaded local apps archive
  SIMA_CLI_BIN            Path to sima-cli
USAGE
}

download_text() {
  local url="$1"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "${url}"
    return 0
  fi
  if command -v wget >/dev/null 2>&1; then
    wget -qO- "${url}"
    return 0
  fi
  echo "Neither curl nor wget is installed." >&2
  return 1
}

download_file() {
  local url="$1"
  local out="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -fL "${url}" -o "${out}"
    return 0
  fi
  if command -v wget >/dev/null 2>&1; then
    wget -O "${out}" "${url}"
    return 0
  fi
  echo "Neither curl nor wget is installed." >&2
  return 1
}

branch_key_for_url() {
  printf '%s' "$1" | tr '/ ' '--'
}

resolve_sima_cli_bin() {
  if [[ -n "${SIMA_CLI_BIN:-}" && -x "${SIMA_CLI_BIN}" ]]; then
    printf '%s\n' "${SIMA_CLI_BIN}"
    return 0
  fi
  command -v sima-cli
}

if [[ "${BRANCH}" == "-h" || "${BRANCH}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ -z "${LOCAL_ARCHIVE}" && -n "${BRANCH}" && "${BRANCH}" == *.tar.gz ]]; then
  if [[ ! -f "${BRANCH}" ]]; then
    echo "Local archive not found: ${BRANCH}" >&2
    exit 1
  fi
  LOCAL_ARCHIVE="${BRANCH}"
  BRANCH=""
  TAG_INPUT="latest"
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required to parse branches.json." >&2
  exit 1
fi

if [[ -z "${BRANCH}" ]]; then
  BRANCHES_JSON="$(download_text "${BASE_URL}/branches.json")"

  mapfile -t BRANCHES < <(python3 - <<'PY' "${BRANCHES_JSON}"
import json
import sys

raw = sys.argv[1]
data = json.loads(raw)
values = data.get("branches", []) if isinstance(data, dict) else []
for item in values:
    text = str(item).strip()
    if text:
        print(text)
PY
)

  if [[ "${#BRANCHES[@]}" -eq 0 ]]; then
    echo "No branches found in ${BASE_URL}/branches.json." >&2
    exit 1
  fi

  echo "Available branches:"
  for i in "${!BRANCHES[@]}"; do
    printf "  %2d) %s\n" "$((i + 1))" "${BRANCHES[$i]}"
  done

  read -r -p "Choose branch [1-${#BRANCHES[@]}]: " choice
  if [[ ! "${choice}" =~ ^[0-9]+$ ]] || (( choice < 1 || choice > ${#BRANCHES[@]} )); then
    echo "Invalid selection: ${choice}" >&2
    exit 1
  fi
  BRANCH="${BRANCHES[$((choice - 1))]}"
fi

ARCHIVE_NAME=""
ARCHIVE_URL=""
TMP_ARCHIVE=""
TMP_DEPS_DIR=""

cleanup() {
  if [[ -n "${TMP_ARCHIVE}" ]]; then
    rm -f "${TMP_ARCHIVE}"
  fi
  if [[ -n "${TMP_DEPS_DIR}" ]]; then
    rm -rf "${TMP_DEPS_DIR}"
  fi
}
trap cleanup EXIT

if [[ -n "${LOCAL_ARCHIVE}" ]]; then
  if [[ ! -f "${LOCAL_ARCHIVE}" ]]; then
    echo "Local archive not found: ${LOCAL_ARCHIVE}" >&2
    exit 1
  fi
  ARCHIVE_NAME="$(basename "${LOCAL_ARCHIVE}")"
  echo "Using local archive:"
  echo "  ${LOCAL_ARCHIVE}"
else
  BRANCH_KEY="$(branch_key_for_url "${BRANCH}")"

  if [[ "${TAG_INPUT}" == "latest" || -z "${TAG_INPUT}" ]]; then
    TAG="$(download_text "${BASE_URL}/${BRANCH_KEY}/latest.tag" | tr -d '[:space:]')"
    if [[ -z "${TAG}" ]]; then
      echo "latest.tag is empty for branch: ${BRANCH}" >&2
      exit 1
    fi
  else
    TAG="${TAG_INPUT}"
  fi

  ARCHIVE_NAME="neat-apps-${BRANCH_KEY}-${TAG}.tar.gz"
  ARCHIVE_URL="${BASE_URL}/${ARCHIVE_NAME}"
  TMP_ARCHIVE="$(mktemp -t neat-apps.XXXXXX.tar.gz)"

  echo "Branch: ${BRANCH}"
  echo "Tag:    ${TAG}"
  echo "URL:    ${ARCHIVE_URL}"
  echo
  echo "Downloading ${ARCHIVE_NAME} ..."
  download_file "${ARCHIVE_URL}" "${TMP_ARCHIVE}"
  LOCAL_ARCHIVE="${TMP_ARCHIVE}"
fi

mkdir -p "${DEST_DIR}"
echo "Extracting into ${DEST_DIR}/ ..."
tar -xzf "${LOCAL_ARCHIVE}" -C "${DEST_DIR}"

RUNTIME_DIR="${DEST_DIR}/prebuilt-apps"
APPS_MANIFEST_PATH="${RUNTIME_DIR}/manifest.json"

if [[ ! -f "${APPS_MANIFEST_PATH}" ]]; then
  echo "ERROR: extracted apps package is missing manifest.json." >&2
  exit 1
fi

SIMA_CLI_RESOLVED="$(resolve_sima_cli_bin)" || {
  echo "ERROR: sima-cli is required to install Core and Insight." >&2
  exit 1
}
TMP_DEPS_DIR="$(mktemp -d -t neat-apps-deps.XXXXXX)"
mkdir -p "${TMP_DEPS_DIR}/core" "${TMP_DEPS_DIR}/insight"

echo
echo "Installing the latest Core version from main:"
(
  cd "${TMP_DEPS_DIR}/core"
  "${SIMA_CLI_RESOLVED}" neat install -d . -t minimal core
)

echo
echo "Installing the latest Insight version from main:"
(
  cd "${TMP_DEPS_DIR}/insight"
  "${SIMA_CLI_RESOLVED}" neat install -d . insight
)

echo
echo "Installed apps runtime under:"
echo "  ${RUNTIME_DIR}"
