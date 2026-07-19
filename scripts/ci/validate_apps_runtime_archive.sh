#!/usr/bin/env bash
set -euo pipefail

archive="${1:-}"
if [[ -z "${archive}" || ! -f "${archive}" ]]; then
  echo "Usage: validate_apps_runtime_archive.sh <runtime-archive>" >&2
  exit 1
fi

members="$(tar -tzf "${archive}")"

for required_file in neat-core.json manifest.json; do
  if ! grep -qx "neat-apps-runtime/${required_file}" <<<"${members}"; then
    echo "Runtime archive is missing neat-apps-runtime/${required_file}." >&2
    exit 1
  fi
done

forbidden='(^|/)(models|portal|tests|test-scope\.yaml|CMakeLists\.txt|CTestTestfile\.cmake|cmake_install\.cmake|Makefile|[^/]+_(unit|e2e)_test|sandbox[^/]*|__pycache__)(/|$)|\.(a|pyc|pyo|log|db|lock)$'
if grep -E "${forbidden}" <<<"${members}"; then
  echo "Runtime archive contains non-runtime or generated files." >&2
  exit 1
fi

root_example_binaries="$(
  awk -F/ 'NF == 5 && $2 == "examples" && ($4 == $5 || $4 == $5 "_cpp")' <<<"${members}"
)"
if [[ -n "${root_example_binaries}" ]]; then
  printf '%s\n' "${root_example_binaries}" >&2
  echo "Runtime archive contains a root-level example binary." >&2
  exit 1
fi

invalid_assets="$(
  grep '^neat-apps-runtime/assets/' <<<"${members}" \
    | grep -Ev '^neat-apps-runtime/assets/$|^neat-apps-runtime/assets/datasets(/|$)' \
    || true
)"
if [[ -n "${invalid_assets}" ]]; then
  printf '%s\n' "${invalid_assets}" >&2
  echo "Runtime archive contains non-runtime assets." >&2
  exit 1
fi

echo "Runtime archive contains production files only: ${archive}"
