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
SERVER_PATTERN="${PYTHON_DIR}/server/main.py"
UI_PATTERN="${PYTHON_DIR}/ui/main.py"
# Reset the MLA runtime dispatcher on launch so no models are left resident.
# MLA_RESET=0 disables it; MLA_RESET_CMD overrides how the reset is performed.
MLA_RESET="${MLA_RESET:-1}"
MLA_RESET_CMD="${MLA_RESET_CMD:-}"
# The MLA shared-memory dispatcher service that holds loaded models across
# client processes; restarting it releases every model on the MLA.
MLA_DISPATCHER_SERVICE="${MLA_DISPATCHER_SERVICE:-simaai-appcomplex.service}"
# Password fed to sudo (via -S) so the privileged MLA reset never blocks on a
# prompt. Defaults to the SiMa Modalix board's default login ("edgeai");
# override with MLA_SUDO_PASSWORD if your board differs.
MLA_SUDO_PASSWORD="${MLA_SUDO_PASSWORD:-edgeai}"
# PID file for the running instance (enables `./run.sh stop`).
PID_FILE="${RUN_PID_FILE:-${EXAMPLE_DIR}/.neat-genai-studio.pid}"
STOP_TIMEOUT="${STOP_TIMEOUT:-20}"

# ---------------------------------------------------------------------------
# Pretty output: the Neat sparkle banner + colourised status lines. Everything
# degrades to plain text when stdout is not a TTY, TERM is "dumb", or NO_COLOR
# is set, so piped/redirected logs stay clean.
# ---------------------------------------------------------------------------
if [[ -t 1 && -z "${NO_COLOR:-}" && "${TERM:-}" != "dumb" ]]; then
  C_RESET=$'\033[0m'; C_BOLD=$'\033[1m'; C_DIM=$'\033[2m'
  # Neat logo palette (truecolor): teal/green up-point, blue + lime sides,
  # orange down-point, ink centre.
  P_TEAL=$'\033[38;2;61;179;138m'
  P_GREEN=$'\033[38;2;74;168;54m'
  P_LIME=$'\033[38;2;154;190;30m'
  P_BLUE=$'\033[38;2;58;125;216m'
  P_ORANGE=$'\033[38;2;223;108;30m'
  P_INK=$'\033[38;2;60;66;74m'
  C_ACCENT="${P_LIME}"; C_MUTED=$'\033[38;2;140;150;160m'
  C_OK=$'\033[38;2;53;196;137m'; C_WARN=$'\033[38;2;224;173;74m'; C_ERR=$'\033[38;2;239;91;98m'
else
  C_RESET=''; C_BOLD=''; C_DIM=''
  P_TEAL=''; P_GREEN=''; P_LIME=''; P_BLUE=''; P_ORANGE=''; P_INK=''
  C_ACCENT=''; C_MUTED=''; C_OK=''; C_WARN=''; C_ERR=''
fi

# step() announces a major action and is preceded by a blank line so each step
# stands clearly apart from the previous step's output (including subprocess
# logs). info/ok/warn are the detail lines that sit directly under a step.
step() { printf '\n%s\n' "${C_ACCENT}${C_BOLD}▸${C_RESET} $*"; }
info() { printf '%s\n' "${C_MUTED}·${C_RESET} $*"; }
ok()   { printf '%s\n' "${C_OK}✔${C_RESET} $*"; }
warn() { printf '%s\n' "${C_WARN}⚠${C_RESET} $*" >&2; }
errln(){ printf '%s\n' "${C_ERR}✘${C_RESET} $*" >&2; }

# The Neat sparkle, formed from the logo palette, plus the wordmark + tagline.
banner() {
  printf '\n'
  printf '        %s▲%s\n'                   "${P_TEAL}"   "${C_RESET}"
  printf '       %s███%s\n'                  "${P_GREEN}"  "${C_RESET}"
  printf '      %s█████%s\n'                 "${P_GREEN}"  "${C_RESET}"
  printf '   %s◀████%s█%s████▶%s\n'          "${P_BLUE}" "${P_INK}" "${P_LIME}" "${C_RESET}"
  printf '      %s█████%s\n'                 "${P_ORANGE}" "${C_RESET}"
  printf '       %s███%s\n'                  "${P_ORANGE}" "${C_RESET}"
  printf '        %s▼%s\n'                   "${P_ORANGE}" "${C_RESET}"
  printf '\n'
  printf '   %sNEAT%s %sGenAI Studio%s\n'    "${C_BOLD}" "${C_RESET}" "${C_ACCENT}${C_BOLD}" "${C_RESET}"
  printf '   %sRuns LLMs/VLMs entirely on your SiMa.ai Modalix MLSoC%s\n' "${C_MUTED}" "${C_RESET}"
  printf '\n'
}

# A slim divider formed from the palette Neat sparkle (✦), cycling the logo
# colours — used as the spacer between explicit startup sections.
spacer() {
  local palette=("${P_TEAL}" "${P_GREEN}" "${P_LIME}" "${P_BLUE}" "${P_ORANGE}")
  local line="   " i
  for ((i = 0; i < 12; i++)); do
    line+="${palette[i % 5]}✦${C_RESET} "
  done
  printf '%s\n' "${line}"
}

# A titled section break: a blank line, a sparkle divider, then a bold heading.
# A following step() adds its own leading blank, so the title terminates in a
# single newline to keep exactly one blank line between heading and first step.
section() {
  printf '\n'
  spacer
  printf '   %s%s%s\n' "${C_BOLD}" "$1" "${C_RESET}"
}

# Best-effort browser URL from the app.web block of the config (scheme/host/port).
web_url() {
  local host port https scheme ip
  host="$(awk '/^  web:/{f=1;next} f&&/^  [a-z]/{f=0} f&&/host:/{print $2;exit}' "${CONFIG_PATH}" 2>/dev/null || true)"
  port="$(awk '/^  web:/{f=1;next} f&&/^  [a-z]/{f=0} f&&/port:/{print $2;exit}' "${CONFIG_PATH}" 2>/dev/null || true)"
  https="$(awk '/^  web:/{f=1;next} f&&/^  [a-z]/{f=0} f&&/https:/{print $2;exit}' "${CONFIG_PATH}" 2>/dev/null || true)"
  [[ -n "${port}" ]] || return 1
  scheme="http"; [[ "${https}" == "true" ]] && scheme="https"
  if [[ -z "${host}" || "${host}" == "0.0.0.0" || "${host}" == "::" ]]; then
    ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
    host="${ip:-localhost}"
  fi
  printf '%s://%s:%s' "${scheme}" "${host}" "${port}"
}

# Aligned "label   value" line for the System section.
_kv() { printf '   %s%-12s%s %s\n' "${C_MUTED}" "$1" "${C_RESET}" "$2"; }

# From `neat`'s report, the first value token of a labelled row (or "not
# installed"); labels may contain spaces (e.g. "Neat core", "DevKit Version").
_neat_field() {
  awk -v lbl="$1" '
    { s=$0; sub(/^[[:space:]]+/,"",s)
      if (substr(s,1,length(lbl))==lbl && (length(s)==length(lbl) || substr(s,length(lbl)+1,1)==" ")) {
        rest=substr(s,length(lbl)+1); sub(/^[[:space:]]+/,"",rest)
        if (rest ~ /^not installed/) { print "not installed"; exit }
        if (split(rest,a,/[[:space:]]+/) >= 1) print a[1]
        exit
      }
    }'
}
# The whole (trimmed) value of a labelled row — for multi-word values like Mode.
_neat_rest() {
  awk -v lbl="$1" '
    { s=$0; sub(/^[[:space:]]+/,"",s)
      if (substr(s,1,length(lbl))==lbl && (length(s)==length(lbl) || substr(s,length(lbl)+1,1)==" ")) {
        rest=substr(s,length(lbl)+1); sub(/^[[:space:]]+/,"",rest); sub(/[[:space:]]+$/,"",rest)
        print rest; exit
      }
    }'
}

# Print system logistics from the `neat` CLI: the DevKit/firmware version and
# the neat/pyneat/neat-llima component versions. All probes are best-effort and
# never block startup — `neat` may run an online update check, so it is
# time-bounded.
system_info() {
  local neat_out=""
  if command -v neat >/dev/null 2>&1; then
    if command -v timeout >/dev/null 2>&1; then
      neat_out="$(timeout "${NEAT_INFO_TIMEOUT:-8}" neat 2>/dev/null || true)"
    else
      neat_out="$(neat 2>/dev/null || true)"
    fi
  fi

  local mode devkit neat_ver pyneat_ver llima_ver runtime_ver py_ver
  if [[ -n "${neat_out}" ]]; then
    mode="$(_neat_rest  "Mode"           <<<"${neat_out}")"
    devkit="$(_neat_field "DevKit Version" <<<"${neat_out}")"
    neat_ver="$(_neat_field "Neat core"     <<<"${neat_out}")"
    pyneat_ver="$(_neat_field "PyNeat"        <<<"${neat_out}")"
    llima_ver="$(_neat_field "neat-llima"    <<<"${neat_out}")"
    runtime_ver="$(_neat_field "neat-runtime"  <<<"${neat_out}")"
  fi
  # Fallback for pyneat when the `neat` CLI is unavailable (e.g. off-board).
  if [[ -z "${pyneat_ver}" ]]; then
    pyneat_ver="$("${PYNEAT_PYTHON}" -c 'import importlib.metadata as m; print(m.version("pyneat"))' 2>/dev/null || true)"
  fi
  py_ver="$("${PYNEAT_PYTHON}" -c 'import platform; print(platform.python_version())' 2>/dev/null || true)"

  [[ -n "${mode}"   ]] && _kv "mode" "${mode}"
  # DevKit Version is the board firmware/build (captured by `neat`).
  [[ -n "${devkit}" ]] && _kv "firmware" "${devkit}"
  _kv "neat" "${neat_ver:-unknown}"
  _kv "pyneat" "${pyneat_ver:-unknown}"
  _kv "neat-llima" "${llima_ver:-unknown}"
  [[ -n "${runtime_ver}" ]] && _kv "neat-runtime" "${runtime_ver}"
  _kv "python" "${py_ver:-unknown}"
  _kv "host" "$(uname -sm 2>/dev/null || echo unknown)"
}

usage() {
  cat <<'USAGE'
Usage:
  ./run.sh            Start the model server and web UI (Ctrl+C to stop).
  ./run.sh stop       Cleanly stop a running instance.
  ./run.sh status     Report whether the studio is running.
USAGE
}

do_status() {
  if [[ -f "${PID_FILE}" ]]; then
    local pid; pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      ok "Neat GenAI Studio is running (pid ${pid})."
      local url; url="$(web_url || true)"
      [[ -n "${url}" ]] && info "Web UI: ${C_ACCENT}${url}${C_RESET}"
      return 0
    fi
  fi
  info "Neat GenAI Studio is not running."
  return 1
}

do_stop() {
  if [[ -f "${PID_FILE}" ]]; then
    local pid; pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      step "Stopping Neat GenAI Studio (pid ${pid})…"
      # SIGTERM triggers run.sh's cleanup: a graceful shutdown of both processes.
      kill -TERM "${pid}" 2>/dev/null || true
      local i
      for ((i = 0; i < STOP_TIMEOUT; i++)); do
        kill -0 "${pid}" 2>/dev/null || break
        sleep 1
      done
      if kill -0 "${pid}" 2>/dev/null; then
        warn "Not responding after ${STOP_TIMEOUT}s; sending KILL…"
        kill -KILL "${pid}" 2>/dev/null || true
      fi
      rm -f "${PID_FILE}"
      ok "Stopped."
      return 0
    fi
    rm -f "${PID_FILE}"
  fi
  # No recorded instance — best-effort cleanup of any stray studio processes.
  info "No running instance recorded; cleaning up any stray processes…"
  if command -v pkill >/dev/null 2>&1; then
    pkill -TERM -f "${SERVER_PATTERN}" 2>/dev/null || true
    pkill -TERM -f "${UI_PATTERN}" 2>/dev/null || true
    pkill -TERM -f "${RAG_WORKER_PATTERN}" 2>/dev/null || true
    sleep 2
    pkill -KILL -f "${SERVER_PATTERN}" 2>/dev/null || true
    pkill -KILL -f "${UI_PATTERN}" 2>/dev/null || true
    pkill -KILL -f "${RAG_WORKER_PATTERN}" 2>/dev/null || true
  fi
  ok "Done."
}

case "${1:-run}" in
  stop) do_stop; exit 0 ;;
  status) if do_status; then exit 0; else exit 1; fi ;;
  -h|--help|help) usage; exit 0 ;;
  run|start) ;;  # fall through to launch
  *) errln "unknown command: $1"; usage; exit 2 ;;
esac

banner

if [[ ! -f "${CONFIG_PATH}" ]]; then
  errln "config does not exist: ${CONFIG_PATH}"
  exit 2
fi
info "Config: ${C_DIM}${CONFIG_PATH}${C_RESET}"

# `neat` runs an online update check, so allow skipping this with SHOW_SYSTEM_INFO=0.
if [[ "${SHOW_SYSTEM_INFO:-1}" != "0" ]]; then
  section "System"
  system_info
fi

pids=()
process_groups=()
cleanup_started=0

stop_stale_rag_worker() {
  if ! command -v pkill >/dev/null 2>&1; then
    return 0
  fi

  pkill -TERM -f "${RAG_WORKER_PATTERN}" 2>/dev/null || true
  sleep 1
  pkill -KILL -f "${RAG_WORKER_PATTERN}" 2>/dev/null || true
}

# Return 0 (true) if 127.0.0.1:<port> still accepts connections.
port_in_use() {
  "${PYNEAT_PYTHON}" -c '
import socket, sys
s = socket.socket(); s.settimeout(0.3)
try:
    s.connect(("127.0.0.1", int(sys.argv[1]))); sys.exit(0)
except OSError:
    sys.exit(1)
finally:
    s.close()
' "$1" 2>/dev/null
}

# Run a privileged command best-effort. As root: run directly. Otherwise feed
# the board password to sudo over stdin (-S) so the MLA reset never blocks on a
# prompt — whether or not there is a terminal. Callers tolerate failure.
mla_sudo() {
  if [[ "$(id -u)" == "0" ]]; then
    "$@"
  elif command -v sudo >/dev/null 2>&1; then
    sudo -S -p '' "$@" <<<"${MLA_SUDO_PASSWORD}"
  else
    return 1
  fi
}

# Clear models held by the MLA shared-memory dispatcher. Models are loaded into
# the dispatcher daemon (mlashmcomplex), which persists across our client
# processes, so killing our server is not enough — a stale/crashed run leaves
# models resident and the next load fails with MLA_LOAD_FAILED. This restarts the
# dispatcher (which frees the MLA) best-effort; it needs privileges.
reset_mla_dispatcher() {
  [[ "${MLA_RESET}" == "1" ]] || return 0

  # 1) Explicit override wins.
  if [[ -n "${MLA_RESET_CMD}" ]]; then
    info "Resetting MLA via MLA_RESET_CMD: ${C_DIM}${MLA_RESET_CMD}${C_RESET}"
    bash -c "${MLA_RESET_CMD}" || warn "MLA_RESET_CMD failed (continuing)"
    return 0
  fi

  # 2) The SDK's official runtime-recovery script, if present on the board.
  local fixer
  for fixer in \
    "$(command -v fix_devkit_runtime.sh 2>/dev/null || true)" \
    /usr/bin/fix_devkit_runtime.sh /usr/local/bin/fix_devkit_runtime.sh; do
    if [[ -n "${fixer}" && -x "${fixer}" ]]; then
      info "Resetting MLA runtime via ${C_DIM}${fixer}${C_RESET}…"
      mla_sudo "${fixer}" || warn "MLA runtime reset failed (continuing)"
      return 0
    fi
  done

  # 3) Minimal fallback: restart the dispatcher service + re-init MLA memory.
  if command -v systemctl >/dev/null 2>&1; then
    info "Restarting MLA dispatcher (${MLA_DISPATCHER_SERVICE})…"
    mla_sudo systemctl restart "${MLA_DISPATCHER_SERVICE}" 2>/dev/null \
      || warn "could not restart ${MLA_DISPATCHER_SERVICE} (need privileges? set MLA_RESET_CMD or MLA_RESET=0)"
  fi
  if [[ -x /usr/bin/init_mla_memory.sh ]]; then
    mla_sudo /usr/bin/init_mla_memory.sh 2>/dev/null || true
  fi
}

# Put the MLA in a clean slate before starting: stop any stale model processes
# from a previous (e.g. crashed) run, reset the MLA dispatcher so no models are
# left resident, and wait for the OpenAI port to free up.
reset_mla_clean_slate() {
  step "Preparing the MLA (clearing any stale model processes)…"
  if command -v pkill >/dev/null 2>&1; then
    pkill -TERM -f "${SERVER_PATTERN}" 2>/dev/null || true
    pkill -TERM -f "${UI_PATTERN}" 2>/dev/null || true
    sleep 1
    pkill -KILL -f "${SERVER_PATTERN}" 2>/dev/null || true
    pkill -KILL -f "${UI_PATTERN}" 2>/dev/null || true
  fi
  stop_stale_rag_worker

  reset_mla_dispatcher

  # Wait for the OpenAI port to be released so the new server can bind it.
  local port deadline
  port="$(sed -n 's/^[[:space:]]*port:[[:space:]]*\([0-9][0-9]*\).*/\1/p' "${CONFIG_PATH}" | head -n 1)"
  port="${port:-9998}"
  deadline=$((SECONDS + 10))
  while port_in_use "${port}" && [[ "${SECONDS}" -lt "${deadline}" ]]; do
    sleep 1
  done
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
  if [[ "${cleanup_started}" -eq 1 ]]; then
    return
  fi
  cleanup_started=1

  trap - EXIT
  trap '' INT TERM

  step "Shutting down Neat GenAI Studio…"

  # Graceful: both Python entrypoints handle SIGTERM — the model server releases
  # its models on the MLA and the UI stops the RAG worker before exiting.
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
  rm -f "${PID_FILE}"
  ok "Neat GenAI Studio stopped."
}

# Refuse to start a second instance (checked before the EXIT trap so bailing out
# here never touches the other instance's pid file).
if [[ -f "${PID_FILE}" ]]; then
  existing="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${existing}" ]] && kill -0 "${existing}" 2>/dev/null; then
    errln "Neat GenAI Studio is already running (pid ${existing})."
    info "Run './run.sh stop' first, or './run.sh status' to check."
    exit 1
  fi
  rm -f "${PID_FILE}"
fi

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Record this instance so `./run.sh stop` can find it (removed by cleanup).
echo "$$" > "${PID_FILE}"

# Sentinel exit code the model server uses to ask for an MLA reset + relaunch.
MLA_RESET_EXIT_CODE="${MLA_RESET_EXIT_CODE:-75}"

launch_server() {
  step "Starting the Neat model server (OpenAI-compatible)…"
  setsid "${PYNEAT_PYTHON}" "${PYTHON_DIR}/server/main.py" --config "${CONFIG_PATH}" &
  server_pid="$!"
  pids[0]="${server_pid}"
  remember_process_group "${server_pid}"
}

section "Accelerator"
reset_mla_clean_slate

section "Model server"
launch_server

sleep "${MODEL_SERVER_START_DELAY:-2}"

if ! child_running "${server_pid}"; then
  set +e
  wait "${server_pid}"
  status=$?
  set -e
  if [[ "${status}" -eq "${MLA_RESET_EXIT_CODE}" ]]; then
    warn "Model server requested an MLA reset during startup — resetting and relaunching…"
    reset_mla_dispatcher
    launch_server
    sleep "${MODEL_SERVER_START_DELAY:-2}"
  else
    exit "${status}"
  fi
fi

section "Web UI"
step "Starting the Neat GenAI Studio web UI…"
setsid "${APP_PYTHON}" "${PYTHON_DIR}/ui/main.py" --config "${CONFIG_PATH}" &
pids[1]="$!"
ui_pid="${pids[1]}"
remember_process_group "${ui_pid}"

printf '\n'
ok "Neat GenAI Studio is starting up."
_url="$(web_url || true)"
if [[ -n "${_url}" ]]; then
  info "Open ${C_ACCENT}${C_BOLD}${_url}${C_RESET} in your browser once it finishes loading."
fi
info "Press ${C_BOLD}Ctrl+C${C_RESET} to stop, or run ${C_BOLD}./run.sh stop${C_RESET} from another shell."
printf '\n'

# Supervisor: keep both processes alive. If the model server exits with the
# MLA-reset sentinel (a switch or the UI's Reset button hit a wedged
# accelerator), reset the MLA dispatcher and relaunch just the server — the UI
# keeps running and its socket reconnects automatically.
#
# A relaunched server can die at startup if the freshly-reset dispatcher isn't
# ready yet, so retry a BOUNDED number of times (resetting the dispatcher again)
# before giving up — otherwise one transient post-reset failure would tear down
# the whole studio, including the still-healthy UI. The budget refreshes once the
# server has stayed up for a while, so an unrelated later reset gets a clean slate.
status=0
restart_attempts=0
healthy_ticks=0
MLA_MAX_RESTART_RETRIES="${MLA_MAX_RESTART_RETRIES:-4}"
while true; do
  if ! child_running "${server_pid}"; then
    set +e
    wait "${server_pid}"
    status=$?
    set -e
    healthy_ticks=0
    if [[ "${status}" -eq 0 ]]; then
      break   # clean model-server shutdown — do not relaunch
    fi
    if [[ "${restart_attempts}" -ge "${MLA_MAX_RESTART_RETRIES}" ]]; then
      errln "Model server did not stay up after ${restart_attempts} restart attempts (last status ${status}); shutting down."
      break
    fi
    restart_attempts=$((restart_attempts + 1))
    if [[ "${status}" -eq "${MLA_RESET_EXIT_CODE}" ]]; then
      info "Model server requested an MLA reset — resetting dispatcher and relaunching (attempt ${restart_attempts}/${MLA_MAX_RESTART_RETRIES})…"
    else
      warn "Model server exited (status ${status}) — resetting dispatcher and relaunching (attempt ${restart_attempts}/${MLA_MAX_RESTART_RETRIES})…"
    fi
    reset_mla_dispatcher
    launch_server
    sleep "${MODEL_SERVER_START_DELAY:-2}"
    continue
  fi
  if ! child_running "${ui_pid}"; then
    set +e
    wait "${ui_pid}"
    status=$?
    set -e
    break
  fi
  # Both healthy — once the server has stayed up a while, refresh the restart
  # budget so a later, unrelated reset isn't starved of retries.
  healthy_ticks=$((healthy_ticks + 1))
  if [[ "${healthy_ticks}" -ge 15 ]]; then
    restart_attempts=0
  fi
  sleep 1
done

exit "${status}"
