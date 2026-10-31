# Sourced by run.sh. PID 1 owns the Studio so a killed launcher cannot orphan
# its workers. The service runs as the invoking user, not as root.
STUDIO_UNIT="neat-genai-studio-$(id -u).service"

run_supervised() {
  local name status waiter journal
  local names=() environment=()
  # Keep runtime/config overrides without putting their values on the command line.
  while IFS= read -r name; do
    case "${name}" in
      INVOCATION_ID|JOURNAL_STREAM|NOTIFY_SOCKET|NEAT_RUN_PID|STUDIO_SUPERVISED|SUDO_*|SSH_*|DBUS_*|XDG_RUNTIME_DIR) continue ;;
    esac
    names+=("${name}")
    environment+=("--setenv=${name}")
  done < <(compgen -e)
  local preserve; preserve="$(IFS=,; echo "${names[*]}")"

  sudo -n --preserve-env="${preserve}" systemd-run --quiet --wait --collect \
    --unit="${STUDIO_UNIT}" --uid="$(id -u)" --gid="$(id -g)" \
    --working-directory="${PWD}" --expand-environment=no \
    --property=Type=exec --property=Restart=on-abnormal --property=RestartSec=3 \
    --property=RestartForceExitStatus=75 --property=StartLimitIntervalSec=60 --property=StartLimitBurst=3 \
    --property=OOMPolicy=continue --property=OOMScoreAdjust=-500 --property=MemoryAccounting=yes \
    --property=KillMode=control-group --property=TimeoutStopSec=20 \
    --property='SuccessExitStatus=129 130 143' \
    --property=StandardOutput=journal --property=StandardError=journal \
    "${environment[@]}" --setenv="TMPDIR=${TMPDIR:-}" \
    --setenv="LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}" --setenv=STUDIO_SUPERVISED=1 \
    /bin/bash "${EXAMPLE_DIR}/run.sh" run &
  waiter=$!
  sudo -n journalctl --follow --output=cat --no-pager --since=now --unit="${STUDIO_UNIT}" &
  journal=$!
  trap 'sudo -n systemctl stop "${STUDIO_UNIT}"; kill "${journal}" 2>/dev/null || true' INT TERM HUP
  status=0
  wait "${waiter}" || status=$?
  kill "${journal}" 2>/dev/null || true
  wait "${journal}" 2>/dev/null || true
  trap - INT TERM HUP
  return "${status}"
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

stop_children() {
  # Graceful: both Python entrypoints handle SIGTERM — the model server releases
  # its models on the MLA and the backend stops the RAG worker before exiting.
  signal_process_groups TERM

  local deadline=$((SECONDS + SHUTDOWN_GRACE_SECONDS))
  while any_child_running && [[ "${SECONDS}" -lt "${deadline}" ]]; do
    sleep 1
  done

  if any_child_running; then
    warn "Processes still running after ${SHUTDOWN_GRACE_SECONDS}s — forcing stop…"
    signal_process_groups KILL
    sleep 1
  fi

  for pid in "${pids[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
  stop_stale_rag_worker
}

supervise_children() {
  # Supervisor: restart both owning processes after a failed accelerator load.
  # SIGKILL recovery also covers the OOM killer, which cannot send an HTTP error.
  local status=0 ready=0
  while true; do
    if [[ "${restart_requested}" -eq 1 ]]; then
      warn "Restarting GenAI Studio after a model-load failure…"
      # PID 1 cleans up the whole service, including workers whose parent died.
      exit 75
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
