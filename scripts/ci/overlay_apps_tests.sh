#!/usr/bin/env bash
set -euo pipefail

archive="${1:-}"
runtime_dir="${2:-}"
if [[ -z "${archive}" || ! -f "${archive}" || -z "${runtime_dir}" || ! -d "${runtime_dir}" ]]; then
  echo "Usage: overlay_apps_tests.sh <test-archive> <runtime-dir>" >&2
  exit 1
fi

extract_dir="$(mktemp -d /tmp/neat-apps-tests.XXXXXX)"
trap 'rm -rf "${extract_dir}"' EXIT

tar -xzf "${archive}" -C "${extract_dir}"
test_root="${extract_dir}/neat-apps-tests"
if [[ ! -x "${test_root}/tests/test.sh" ]]; then
  echo "Test archive is missing executable tests/test.sh." >&2
  exit 1
fi

cp -a "${test_root}/." "${runtime_dir}/"
