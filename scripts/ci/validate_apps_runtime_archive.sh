#!/usr/bin/env bash
set -euo pipefail

archive="${1:-}"
if [[ -z "${archive}" || ! -f "${archive}" ]]; then
  echo "Usage: validate_apps_runtime_archive.sh <runtime-archive>" >&2
  exit 1
fi

members="$(tar -tzf "${archive}")"

if ! grep -qx 'neat-apps-runtime/neat-core.json' <<<"${members}"; then
  echo "Runtime archive is missing neat-apps-runtime/neat-core.json." >&2
  exit 1
fi

forbidden='(^|/)(tests|test-scope\.yaml|[^/]+_(unit|e2e)_test|sandbox[^/]*|__pycache__)(/|$)|\.(pyc|pyo|log|db|lock)$'
if grep -E "${forbidden}" <<<"${members}"; then
  echo "Runtime archive contains test or generated files." >&2
  exit 1
fi

echo "Runtime archive contains production files only: ${archive}"
