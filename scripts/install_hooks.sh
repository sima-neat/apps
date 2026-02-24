#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

GIT_HOOKS_DIR="$(git rev-parse --git-path hooks)"
mkdir -p "${GIT_HOOKS_DIR}"

HOOK_SRC="${ROOT_DIR}/.githooks/pre-commit"
if [[ ! -f "${HOOK_SRC}" ]]; then
  echo "ERROR: Missing hook source: ${HOOK_SRC}" >&2
  exit 1
fi

install -m 0755 "${HOOK_SRC}" "${GIT_HOOKS_DIR}/pre-commit"
echo "Installed pre-commit hook at ${GIT_HOOKS_DIR}/pre-commit"
