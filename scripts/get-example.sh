#!/usr/bin/env bash
set -euo pipefail

# Fetch one example from the apps repo archive and leave a standalone example
# directory. This intentionally avoids requiring git on target systems.
#
# Usage:
#   get-example.sh multimodal-assistant
#   get-example.sh genai/multimodal-assistant
#   get-example.sh examples/genai/multimodal-assistant

REPO_URL="${NEAT_APPS_REPO_URL:-https://github.com/sima-neat/apps.git}"
ARCHIVE_URL="${NEAT_APPS_ARCHIVE_URL:-}"
BRANCH="${NEAT_APPS_BRANCH:-main}"
DEST_DIR="${NEAT_APPS_EXAMPLE_DEST_DIR:-}"
FORCE=0

usage() {
  cat <<'USAGE'
Usage:
  get-example.sh [--branch <branch>] [--dest <dir>] [--force] <example>

Examples:
  get-example.sh multimodal-assistant
  get-example.sh genai/multimodal-assistant
  get-example.sh examples/genai/multimodal-assistant

Environment:
  NEAT_APPS_REPO_URL          Apps Git repository URL
                              default: https://github.com/sima-neat/apps.git
  NEAT_APPS_ARCHIVE_URL       Prebuilt source archive URL or local .tar.gz path
                              default: derived from NEAT_APPS_REPO_URL and branch
  NEAT_APPS_BRANCH            Git branch to fetch
                              default: main
  NEAT_APPS_EXAMPLE_DEST_DIR  Destination directory
                              default: ./<example-name>
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --branch)
      BRANCH="$2"
      shift 2
      ;;
    --dest)
      DEST_DIR="$2"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --*)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
    *)
      if [[ -n "${EXAMPLE_ARG:-}" ]]; then
        echo "Only one example can be fetched at a time." >&2
        usage
        exit 2
      fi
      EXAMPLE_ARG="$1"
      shift
      ;;
  esac
done

if [[ -z "${EXAMPLE_ARG:-}" ]]; then
  echo "Missing example name." >&2
  usage
  exit 2
fi

if ! command -v tar >/dev/null 2>&1; then
  echo "tar is required." >&2
  exit 1
fi

normalize_example_path() {
  local raw="$1"
  case "${raw}" in
    multimodal-assistant|multimodal-assist)
      printf '%s\n' "examples/genai/multimodal-assistant"
      ;;
    neat-genai-studio|genai-studio)
      printf '%s\n' "examples/genai/neat-genai-studio"
      ;;
    examples/*/*)
      printf '%s\n' "${raw}"
      ;;
    */*)
      printf '%s\n' "examples/${raw}"
      ;;
    *)
      echo "Unknown example alias: ${raw}" >&2
      echo "Use a path like genai/multimodal-assistant." >&2
      return 1
      ;;
  esac
}

download_file() {
  local url="$1"
  local out="$2"

  if [[ -f "${url}" ]]; then
    cp "${url}" "${out}"
    return 0
  fi

  if command -v curl >/dev/null 2>&1; then
    curl -fL "${url}" -o "${out}"
    return 0
  fi
  if command -v wget >/dev/null 2>&1; then
    wget -O "${out}" "${url}"
    return 0
  fi

  echo "curl or wget is required to download the apps archive." >&2
  return 1
}

repo_archive_url() {
  local repo="$1"
  local ref="$2"

  repo="${repo%.git}"

  case "${repo}" in
    https://github.com/*/*)
      printf '%s/archive/refs/heads/%s.tar.gz\n' "${repo}" "${ref}"
      ;;
    git@github.com:*/*)
      repo="${repo#git@github.com:}"
      printf 'https://github.com/%s/archive/refs/heads/%s.tar.gz\n' "${repo}" "${ref}"
      ;;
    *)
      echo "Cannot derive a GitHub archive URL from: ${repo}" >&2
      echo "Set NEAT_APPS_ARCHIVE_URL to a .tar.gz archive URL or local path." >&2
      return 1
      ;;
  esac
}

EXAMPLE_PATH="$(normalize_example_path "${EXAMPLE_ARG}")"
EXAMPLE_NAME="$(basename "${EXAMPLE_PATH}")"
DEST_DIR="${DEST_DIR:-${EXAMPLE_NAME}}"
TMP_DIR="$(mktemp -d /tmp/neat-apps-example.XXXXXX)"
TMP_ARCHIVE="${TMP_DIR}/apps.tar.gz"

cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

if [[ -e "${DEST_DIR}" ]]; then
  if [[ "${FORCE}" -ne 1 ]]; then
    echo "Destination already exists: ${DEST_DIR}" >&2
    echo "Remove it, choose --dest <dir>, or rerun with --force." >&2
    exit 1
  fi
  rm -rf "${DEST_DIR}"
fi

if [[ -z "${DEST_DIR}" || "${DEST_DIR}" == "/" ]]; then
  echo "Refusing unsafe destination: ${DEST_DIR:-<empty>}" >&2
  exit 1
fi

if [[ -z "${ARCHIVE_URL}" ]]; then
  ARCHIVE_URL="$(repo_archive_url "${REPO_URL}" "${BRANCH}")"
fi

echo "Fetching ${EXAMPLE_PATH} from ${ARCHIVE_URL} ..."
download_file "${ARCHIVE_URL}" "${TMP_ARCHIVE}"

tar -tzf "${TMP_ARCHIVE}" > "${TMP_DIR}/archive-list.txt"
ARCHIVE_ROOT="$(sed -n '1s#/.*##p;q' "${TMP_DIR}/archive-list.txt")"
if [[ -z "${ARCHIVE_ROOT}" ]]; then
  echo "Archive is empty: ${ARCHIVE_URL}" >&2
  exit 1
fi

ARCHIVE_EXAMPLE_PATH="${ARCHIVE_ROOT}/${EXAMPLE_PATH}"
if ! grep -qxF "${ARCHIVE_EXAMPLE_PATH}/" "${TMP_DIR}/archive-list.txt"; then
  echo "Example not found in archive: ${EXAMPLE_PATH}" >&2
  exit 1
fi

tar -xzf "${TMP_ARCHIVE}" -C "${TMP_DIR}" "${ARCHIVE_EXAMPLE_PATH}"
cp -a "${TMP_DIR}/${ARCHIVE_ROOT}/${EXAMPLE_PATH}" "${DEST_DIR}"

echo ""
echo "Created:"
echo "  ${DEST_DIR}"
echo ""
echo "Next:"
echo "  cd ${DEST_DIR}"
echo "  ./setup.sh"
echo "  ./run.sh"
