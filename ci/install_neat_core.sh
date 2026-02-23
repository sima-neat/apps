#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LINK_FILE="${1:-${ROOT_DIR}/ci/neat-core-link.json}"
CLI_BIN="${SIMA_CLI_BIN:-sima-cli}"

if ! command -v "${CLI_BIN}" >/dev/null 2>&1; then
  echo "Required command not found: ${CLI_BIN}" >&2
  exit 1
fi

METADATA_URL="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding=\"utf-8\"))[\"neat_core\"][\"metadata_url\"])' "${LINK_FILE}")"
echo "Installing SiMa NEAT core from ${METADATA_URL}"
"${CLI_BIN}" install -m "${METADATA_URL}"
