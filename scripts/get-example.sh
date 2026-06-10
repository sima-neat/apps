#!/usr/bin/env bash
set -euo pipefail

# Fetch one example from the apps repo and leave a standalone example directory.
#
# Usage:
#   get-example.sh multimodal-assistant
#   get-example.sh genai/multimodal-assistant
#   get-example.sh examples/genai/multimodal-assistant

REPO_URL="${NEAT_APPS_REPO_URL:-https://github.com/sima-neat/apps.git}"
BRANCH="${NEAT_APPS_BRANCH:-develop}"
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
  NEAT_APPS_BRANCH            Git branch, tag, or ref to fetch
                              default: develop
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

if ! command -v git >/dev/null 2>&1; then
  echo "git is required." >&2
  exit 1
fi

normalize_example_path() {
  local raw="$1"
  case "${raw}" in
    multimodal-assistant|multimodal-assist)
      printf '%s\n' "examples/genai/multimodal-assistant"
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

EXAMPLE_PATH="$(normalize_example_path "${EXAMPLE_ARG}")"
EXAMPLE_NAME="$(basename "${EXAMPLE_PATH}")"
DEST_DIR="${DEST_DIR:-${EXAMPLE_NAME}}"
TMP_DIR="$(mktemp -d /tmp/neat-apps-example.XXXXXX)"

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

echo "Fetching ${EXAMPLE_PATH} from ${REPO_URL} (${BRANCH}) ..."
git clone --filter=blob:none --sparse --branch "${BRANCH}" "${REPO_URL}" "${TMP_DIR}/apps"
git -C "${TMP_DIR}/apps" sparse-checkout set "${EXAMPLE_PATH}"

if [[ ! -d "${TMP_DIR}/apps/${EXAMPLE_PATH}" ]]; then
  echo "Example not found in ${REPO_URL}: ${EXAMPLE_PATH}" >&2
  exit 1
fi

cp -a "${TMP_DIR}/apps/${EXAMPLE_PATH}" "${DEST_DIR}"

echo ""
echo "Created:"
echo "  ${DEST_DIR}"
echo ""
echo "Next:"
echo "  cd ${DEST_DIR}"
echo "  ./install.sh"
echo "  ./run.sh"
