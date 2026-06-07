#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

workspace_root="/workspace"
workspace_real="$(readlink -f "${workspace_root}" 2>/dev/null || true)"
remote_root=""

if [[ "${ROOT_DIR}" == "${workspace_root}"* ]]; then
  remote_root="${ROOT_DIR}"
elif [[ -n "${workspace_real}" && "${ROOT_DIR}" == "${workspace_real}"* ]]; then
  remote_root="${workspace_root}${ROOT_DIR#${workspace_real}}"
fi

if [[ -z "${remote_root}" || "${remote_root}" != /workspace/* ]]; then
  echo "[task] this VS Code task must be launched from the eLxr SDK workspace under /workspace" >&2
  echo "[task] current repo path: ${ROOT_DIR}" >&2
  exit 2
fi

remote_helper="${remote_root}/tests/scripts/testing/run_vscode_test_task.py"

build_dir="${BUILD_DIR:-build}"
skip_build=0
forward_args=()
has_build_dir_arg=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-dir)
      if [[ $# -lt 2 ]]; then
        echo "[task] --build-dir requires a directory path" >&2
        exit 2
      fi
      build_dir="$2"
      has_build_dir_arg=1
      forward_args+=("$1" "$2")
      shift 2
      ;;
    --skip-build)
      skip_build=1
      shift
      ;;
    *)
      forward_args+=("$1")
      shift
      ;;
  esac
done

if [[ "${has_build_dir_arg}" == "0" && "${build_dir}" != "build" ]]; then
  forward_args+=(--build-dir "${build_dir}")
fi

cmd="dk $(printf '%q' "${remote_helper}")"
for arg in "${forward_args[@]}"; do
  cmd+=" $(printf '%q' "${arg}")"
done

if bash -ic 'type dk >/dev/null 2>&1'; then
  :
else
  echo "[task] dk is not available in the current SDK shell" >&2
  exit 2
fi

remove_stale_build_dir() {
  local build_path
  if [[ "${build_dir}" == /* ]]; then
    build_path="${build_dir}"
  else
    build_path="${ROOT_DIR}/${build_dir}"
  fi

  local cache_path="${build_path}/CMakeCache.txt"
  if [[ ! -f "${cache_path}" ]]; then
    return 0
  fi

  local expected="CMAKE_HOME_DIRECTORY:INTERNAL=${ROOT_DIR}"
  if ! grep -Fxq "${expected}" "${cache_path}"; then
    echo "[task] removing stale build directory created from a different checkout path"
    rm -rf "${build_path:?}"
  fi
}

if [[ "${skip_build}" == "0" ]]; then
  echo "[task] building apps in the SDK with the installed NEAT core"
  remove_stale_build_dir
  if ! (cd "${ROOT_DIR}" && ./build.sh --build-dir "${build_dir}"); then
    rc=$?
    echo ""
    echo "[task] SDK build failed. Press Enter to close this task terminal."
    if [[ "${SIMANEAT_APPS_TEST_NO_PAUSE:-0}" != "1" ]]; then
      read -r
    fi
    exit "${rc}"
  fi
else
  echo "[task] skipping SDK build"
fi

if bash -ic "${cmd}"; then
  exit 0
else
  rc=$?
  echo ""
  echo "[task] board-side tests failed. Press Enter to close this task terminal."
  if [[ "${SIMANEAT_APPS_TEST_NO_PAUSE:-0}" != "1" ]]; then
    read -r
  fi
  exit "${rc}"
fi
