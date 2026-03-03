#!/usr/bin/env bash
set -euo pipefail

TARGET="/data/simaai/coprocessing"

run_root() {
  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
  else
    sudo "$@"
  fi
}

if [[ "${TARGET}" != "/data/simaai/coprocessing" ]]; then
  echo "[empty-coprocessing] refusing to run: unexpected target ${TARGET}"
  exit 1
fi

if [[ ! -d "${TARGET}" ]]; then
  echo "[empty-coprocessing] directory not found: ${TARGET}"
  exit 0
fi

echo "[empty-coprocessing] removing all contents under ${TARGET}"
run_root find "${TARGET}" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
echo "[empty-coprocessing] done"
