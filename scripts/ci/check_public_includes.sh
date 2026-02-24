#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGETS=(
  "${ROOT_DIR}/examples/cpp"
  "${ROOT_DIR}/support"
)

if ! command -v rg >/dev/null 2>&1; then
  echo "rg is required for include boundary checks." >&2
  exit 1
fi

# Apps examples must not depend on core source-tree internals/test headers.
PATTERN='^[[:space:]]*#include[[:space:]]+"([^"]*internal/[^"]*|tests/[^"]*|e2e_pipelines/[^"]*|model/[^"]*|nodes/[^"]*|pipeline/[^"]*)"'

set +e
HITS="$(rg -n --glob '*.h' --glob '*.hpp' --glob '*.cpp' --glob '*.cc' --glob '*.cxx' "${PATTERN}" "${TARGETS[@]}")"
RG_STATUS=$?
set -e

if [[ ${RG_STATUS} -eq 0 ]]; then
  echo "Forbidden include paths detected in apps sources:" >&2
  echo "${HITS}" >&2
  exit 1
fi

if [[ ${RG_STATUS} -gt 1 ]]; then
  echo "rg failed while checking includes." >&2
  exit ${RG_STATUS}
fi

echo "Public include boundary check passed."
