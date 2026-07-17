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

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required to validate neat-apps-runtime/neat-core.json." >&2
  exit 1
fi

if ! tar -xOzf "${archive}" neat-apps-runtime/neat-core.json | python3 -c '
import json
import sys

try:
    data = json.load(sys.stdin)
except (json.JSONDecodeError, UnicodeDecodeError):
    raise SystemExit(1)

if not isinstance(data, dict):
    raise SystemExit(1)
neat_core = data.get("neat-core")
if not isinstance(neat_core, dict):
    raise SystemExit(1)
branch = neat_core.get("branch")
version = neat_core.get("version")
if not isinstance(branch, str) or not isinstance(version, str):
    raise SystemExit(1)
if not branch.strip() or not version.strip() or version.strip().lower() == "latest":
    raise SystemExit(1)
'; then
  echo "Runtime archive does not contain a valid exact Core dependency pin." >&2
  exit 1
fi

forbidden='(^|/)(tests|test-scope\.yaml|[^/]+_(unit|e2e)_test|sandbox[^/]*|__pycache__)(/|$)|\.(pyc|pyo|log|db|lock)$'
if grep -E "${forbidden}" <<<"${members}"; then
  echo "Runtime archive contains test or generated files." >&2
  exit 1
fi

echo "Runtime archive contains production files only: ${archive}"
