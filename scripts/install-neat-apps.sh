#!/usr/bin/env bash
set -euo pipefail

# install-neat-apps.sh
#
# Downloads a runtime apps bundle from the published apps artifact index.
#
# Usage:
#   ./install-neat-apps.sh [branch] [latest|git-short-hash]
#
# Behavior:
# - If no branch is provided, fetch branches.json and prompt the user.
# - If no tag is provided, or the tag is "latest", fetch branch/latest.tag.
# - Download neat-apps-<branch-key>-<tag>.tar.gz from the apps download root.
# - Extract the archive into ./neat-apps.

BASE_URL="${NEAT_APPS_BASE_URL:-https://apps.sima-neat.com/download}"
NEAT_INSTALLER_URL="${NEAT_INSTALLER_URL:-https://tools.modalix.info/install-neat-from-a-branch.sh}"
BRANCH="${1:-}"
TAG_INPUT="${2:-latest}"
DEST_DIR="${NEAT_APPS_INSTALL_DIR:-neat-apps}"

usage() {
  cat <<'USAGE'
Usage:
  install-neat-apps.sh [branch] [latest|git-short-hash]

Environment:
  NEAT_APPS_BASE_URL      Base URL for apps downloads
                          default: https://apps.sima-neat.com/download
  NEAT_INSTALLER_URL      Hosted NEAT core installer URL
                          default: https://tools.modalix.info/install-neat-from-a-branch.sh
  NEAT_APPS_INSTALL_DIR   Destination directory for extracted files
                          default: ./neat-apps
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

if [[ "${BRANCH}" == "-h" || "${BRANCH}" == "--help" ]]; then
  usage
  exit 0
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
TMP_INSTALLER=""

cleanup() {
  rm -f "${TMP_ARCHIVE}"
  if [[ -n "${TMP_INSTALLER}" ]]; then
    rm -f "${TMP_INSTALLER}"
  fi
}
trap cleanup EXIT

echo "Branch: ${BRANCH}"
echo "Tag:    ${TAG}"
echo "URL:    ${ARCHIVE_URL}"
echo
echo "Downloading ${ARCHIVE_NAME} ..."
download_file "${ARCHIVE_URL}" "${TMP_ARCHIVE}"

mkdir -p "${DEST_DIR}"
echo "Extracting into ${DEST_DIR}/ ..."
tar -xzf "${TMP_ARCHIVE}" -C "${DEST_DIR}"

RUNTIME_DIR="${DEST_DIR}/neat-apps-runtime"
NEAT_CORE_JSON_PATH="${RUNTIME_DIR}/neat-core.json"

if [[ ! -f "${NEAT_CORE_JSON_PATH}" ]]; then
  echo "ERROR: extracted apps package is missing neat-core.json." >&2
  exit 1
fi

if ! mapfile -t NEAT_CORE_TARGET < <(extract_neat_core_target "${NEAT_CORE_JSON_PATH}"); then
  echo "ERROR: failed to parse NEAT core dependency from ${NEAT_CORE_JSON_PATH}." >&2
  exit 1
fi

NEAT_CORE_BRANCH="${NEAT_CORE_TARGET[0]}"
NEAT_CORE_VERSION="${NEAT_CORE_TARGET[1]}"

TMP_INSTALLER="$(mktemp -t install-neat-core.XXXXXX.sh)"
download_file "${NEAT_INSTALLER_URL}" "${TMP_INSTALLER}"
chmod +x "${TMP_INSTALLER}"

echo
echo "Installing matching NEAT core:"
echo "  Branch : ${NEAT_CORE_BRANCH}"
echo "  Version: ${NEAT_CORE_VERSION}"
if ! "${TMP_INSTALLER}" --minimum "${NEAT_CORE_BRANCH}" "${NEAT_CORE_VERSION}"; then
  if [[ "${NEAT_CORE_VERSION}" != "latest" ]]; then
    echo
    echo "Exact NEAT core version ${NEAT_CORE_VERSION} was not installable."
    echo "Falling back to latest for branch ${NEAT_CORE_BRANCH} ..."
    "${TMP_INSTALLER}" --minimum "${NEAT_CORE_BRANCH}" latest
  else
    exit 1
  fi
fi

echo
echo "Installed apps runtime under:"
echo "  ${DEST_DIR}"
