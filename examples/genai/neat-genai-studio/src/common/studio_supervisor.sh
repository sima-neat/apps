# Sourced by run.sh to supervise and restart Studio's own processes.

any_process_group_running() {
  local pgid
  for pgid in "${process_groups[@]}"; do
    if kill -0 -- "-${pgid}" 2>/dev/null; then
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

stop_children() {
  # RAG starts a separate session; include it even if the API parent was killed.
  local pid
  while read -r pid; do
    remember_process_group "${pid}"
  done < <(pgrep -u "$(id -u)" -f "${RAG_WORKER_PATTERN}" || true)

  # Graceful: both Python entrypoints handle SIGTERM — the model server releases
  # its models on the MLA and the backend stops the RAG worker before exiting.
  signal_process_groups TERM

  local deadline=$((SECONDS + SHUTDOWN_GRACE_SECONDS))
  while any_process_group_running && [[ "${SECONDS}" -lt "${deadline}" ]]; do
    sleep 1
  done

  if any_process_group_running; then
    warn "Processes still running after ${SHUTDOWN_GRACE_SECONDS}s — forcing stop…"
    signal_process_groups KILL
    sleep 1
  fi

  for pid in "${pids[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
  pids=()
  process_groups=()
}

supervise_children() {
  # Supervisor: restart both owning processes after a failed accelerator load.
  # SIGKILL recovery also covers the OOM killer, which cannot send an HTTP error.
  local status=0 ready=0
  while true; do
    if [[ "${restart_requested}" -eq 1 ]]; then
      warn "Restarting GenAI Studio after a model-load failure…"
      restart_requested=0
      ready=0
      stop_children
      start_children
      continue
    fi
    if ! child_running "${server_pid}"; then
      set +e
      wait "${server_pid}"
      status=$?
      set -e
      if [[ "${status}" -eq 137 && "${ready}" -eq 1 ]]; then
        warn "Model server was killed (possibly out of memory); restarting GenAI Studio."
        # Let the surviving API answer an interrupted load before stopping it.
        sleep 2
        restart_requested=1
        continue
      fi
      if [[ "${status}" -ne 0 ]]; then
        errln "Model server exited with status ${status}; shutting down."
      fi
      break
    fi
    if ! child_running "${api_pid}"; then
      set +e
      wait "${api_pid}"
      status=$?
      set -e
      if [[ "${status}" -eq 137 && "${ready}" -eq 1 ]]; then
        warn "Studio backend was killed (possibly out of memory); restarting GenAI Studio."
        restart_requested=1
        continue
      fi
      break
    fi
    if [[ "${ready}" == 0 ]] && "${APP_PYTHON}" - "${_self_url}/health" >/dev/null 2>&1 <<'HEALTH'
import sys
import requests

try:
    response = requests.get(sys.argv[1], timeout=1, verify=False)
    sys.exit(0 if response.ok and response.json().get('ok') else 1)
except (requests.RequestException, ValueError):
    sys.exit(1)
HEALTH
    then
      ready=1
      ok "Neat GenAI Studio is ready."
      studio_ui_hint "${_url}"
    fi
    sleep 1
  done

  exit "${status}"
}
