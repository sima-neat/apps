#!/usr/bin/env bash
set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${EXAMPLE_DIR}/common/config.yaml}"
PYTHON_DIR="${EXAMPLE_DIR}/python"

if [[ -z "${PYNEAT_PYTHON:-}" ]]; then
  if [[ -x "${HOME}/pyneat/bin/python" ]]; then
    PYNEAT_PYTHON="${HOME}/pyneat/bin/python"
  else
    PYNEAT_PYTHON="python3"
  fi
fi

APP_PYTHON="${APP_PYTHON:-python3}"
SHUTDOWN_GRACE_SECONDS="${SHUTDOWN_GRACE_SECONDS:-10}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "config does not exist: ${CONFIG_PATH}" >&2
  exit 2
fi

pids=()

any_child_running() {
  local running_pids
  running_pids="$(jobs -r -p || true)"
  for pid in "${pids[@]}"; do
    if grep -qx "${pid}" <<<"${running_pids}"; then
      return 0
    fi
  done
  return 1
}

cleanup() {
  trap - EXIT INT TERM
  for pid in "${pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -INT "${pid}" 2>/dev/null || true
    fi
  done

  local deadline=$((SECONDS + SHUTDOWN_GRACE_SECONDS))
  while any_child_running && [[ "${SECONDS}" -lt "${deadline}" ]]; do
    sleep 1
  done

  for pid in "${pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
  for pid in "${pids[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "Starting Neat OpenAI-compatible model server..."
"${PYNEAT_PYTHON}" "${PYTHON_DIR}/serve_models.py" --config "${CONFIG_PATH}" &
pids+=("$!")

sleep "${MODEL_SERVER_START_DELAY:-2}"

echo "Starting Multimodal Assistant Flask UI..."
"${APP_PYTHON}" "${PYTHON_DIR}/serve_web.py" --config "${CONFIG_PATH}" &
pids+=("$!")

echo "Multimodal Assistant is starting. Press Ctrl+C to stop both processes."

while [[ "$(jobs -r -p | wc -l | tr -d ' ')" -eq "${#pids[@]}" ]]; do
  sleep 1
done

status=0
running_pids="$(jobs -r -p)"
for pid in "${pids[@]}"; do
  if ! grep -qx "${pid}" <<<"${running_pids}"; then
    set +e
    wait "${pid}"
    status=$?
    set -e
    break
  fi
done

exit "${status}"
