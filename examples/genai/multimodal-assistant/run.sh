#!/usr/bin/env bash
set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_DIR="${EXAMPLE_DIR}/src/python"
DEFAULT_APP_VENV="${EXAMPLE_DIR}/.venv"
DEFAULT_LOCAL_CONFIG="${EXAMPLE_DIR}/config.local.yaml"

if [[ -z "${CONFIG_PATH:-}" ]]; then
  if [[ -f "${DEFAULT_LOCAL_CONFIG}" ]]; then
    CONFIG_PATH="${DEFAULT_LOCAL_CONFIG}"
  else
    CONFIG_PATH="${EXAMPLE_DIR}/src/common/config.yaml"
  fi
fi

if [[ -z "${PYNEAT_PYTHON:-}" ]]; then
  if [[ -x "${HOME}/pyneat/bin/python" ]]; then
    PYNEAT_PYTHON="${HOME}/pyneat/bin/python"
  else
    PYNEAT_PYTHON="python3"
  fi
fi

if [[ -z "${APP_PYTHON:-}" ]]; then
  if [[ -x "${DEFAULT_APP_VENV}/bin/python" ]]; then
    APP_PYTHON="${DEFAULT_APP_VENV}/bin/python"
  else
    APP_PYTHON="python3"
  fi
fi
SHUTDOWN_GRACE_SECONDS="${SHUTDOWN_GRACE_SECONDS:-10}"
RAG_WORKER_PATTERN="${PYTHON_DIR}/rag/vectordb_worker.py"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "config does not exist: ${CONFIG_PATH}" >&2
  exit 2
fi

pids=()
process_groups=()

stop_stale_rag_worker() {
  if ! command -v pkill >/dev/null 2>&1; then
    return 0
  fi

  pkill -TERM -f "${RAG_WORKER_PATTERN}" 2>/dev/null || true
  sleep 1
  pkill -KILL -f "${RAG_WORKER_PATTERN}" 2>/dev/null || true
}

any_child_running() {
  for pid in "${pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      return 0
    fi
  done
  return 1
}

child_running() {
  local pid="$1"
  kill -0 "${pid}" 2>/dev/null
}

remember_process_group() {
  local pid="$1"
  local pgid
  pgid="$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d ' ' || true)"
  if [[ -n "${pgid}" ]]; then
    process_groups+=("${pgid}")
  fi
}

signal_process_groups() {
  local signal_name="$1"
  for pgid in "${process_groups[@]}"; do
    kill -"${signal_name}" "-${pgid}" 2>/dev/null || true
  done
}

cleanup() {
  trap - EXIT INT TERM
  signal_process_groups INT

  local deadline=$((SECONDS + SHUTDOWN_GRACE_SECONDS))
  while any_child_running && [[ "${SECONDS}" -lt "${deadline}" ]]; do
    sleep 1
  done

  signal_process_groups TERM

  deadline=$((SECONDS + 3))
  while any_child_running && [[ "${SECONDS}" -lt "${deadline}" ]]; do
    sleep 1
  done

  signal_process_groups KILL
  for pid in "${pids[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
  stop_stale_rag_worker
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

stop_stale_rag_worker

echo "Starting Neat OpenAI-compatible model server..."
setsid "${PYNEAT_PYTHON}" "${PYTHON_DIR}/server/main.py" --config "${CONFIG_PATH}" &
pids+=("$!")
server_pid="${pids[0]}"
remember_process_group "${server_pid}"

sleep "${MODEL_SERVER_START_DELAY:-2}"

if ! child_running "${server_pid}"; then
  set +e
  wait "${server_pid}"
  status=$?
  set -e
  exit "${status}"
fi

echo "Starting Multimodal Assistant Flask UI..."
setsid "${APP_PYTHON}" "${PYTHON_DIR}/ui/main.py" --config "${CONFIG_PATH}" &
pids+=("$!")
remember_process_group "${pids[1]}"

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
