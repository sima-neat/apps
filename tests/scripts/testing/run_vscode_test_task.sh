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
remote_recovery_helper="${remote_root}/tests/scripts/testing/run_fix_devkit_runtime.py"
recovery_cmd="dk $(printf '%q' "${remote_recovery_helper}")"

cmd="dk $(printf '%q' "${remote_helper}")"
for arg in "$@"; do
  cmd+=" $(printf '%q' "${arg}")"
done

if bash -ic 'type dk >/dev/null 2>&1'; then
  :
else
  echo "[task] dk is not available in the current SDK shell" >&2
  exit 2
fi

remove_stale_build_dir() {
  local build_dir="${BUILD_DIR:-build}"
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

echo "[task] building apps in the SDK with the installed NEAT core"
remove_stale_build_dir
if ! (cd "${ROOT_DIR}" && ./build.sh); then
  rc=$?
  echo ""
  echo "[task] SDK build failed. Press Enter to close this task terminal."
  read -r
  exit "${rc}"
fi

if ! bash -ic "${recovery_cmd}"; then
  rc=$?
  echo ""
  echo "[task] board recovery failed. Press Enter to close this task terminal."
  read -r
  exit "${rc}"
fi

if bash -ic "${cmd}"; then
  exit 0
else
  rc=$?
  echo ""
  echo "[task] board-side tests failed. Press Enter to close this task terminal."
  read -r
  exit "${rc}"
fi
