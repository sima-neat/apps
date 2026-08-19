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

# Interpreter of the isolated piper-tts venv — the UI's subprocess TTS worker
# uses it for the de/it/no/vi rhasspy voices (piper-tts and piper-plus can't
# share a venv). Exported so the UI child process inherits it.
if [[ -z "${PIPERTTS_PYTHON:-}" && -x "${EXAMPLE_DIR}/.venv-pipertts/bin/python" ]]; then
  PIPERTTS_PYTHON="${EXAMPLE_DIR}/.venv-pipertts/bin/python"
fi
export PIPERTTS_PYTHON="${PIPERTTS_PYTHON:-}"
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
# Terminal chat instead of the web UI (set by `./run.sh --cli`).
CLI_MODE="${CLI_MODE:-0}"
CLI_EXTRA_ARGS=()   # flags forwarded to cli/main.py (e.g. --chat, --download, --benchmark)
# In CLI mode the model server's logs go here (kept off the chat terminal).
SERVER_LOG="${SERVER_LOG:-${EXAMPLE_DIR}/.neat-genai-server.log}"

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
  banner
  cat <<'USAGE'
Usage:
  ./run.sh            Start the model server and web UI (Ctrl+C to stop).
                      Runs ./setup.sh automatically on the first launch.
  ./run.sh --cli      Start the model server and a terminal chat (no web UI).
  ./run.sh --chat [MODEL]      Terminal chat; load MODEL and chat (skips the menu).
  ./run.sh --download [REPO]   Terminal chat; download REPO (or prompt) first.
  ./run.sh --benchmark [MODEL] Terminal chat; benchmark MODEL (or prompt). --bench.
  ./run.sh stop       Cleanly stop a running instance.
  ./run.sh status     Report whether the studio is running.
  ./run.sh update     Update to the latest version (preserves models, config,
                      venvs, RAG db), then refresh dependencies.
  ./run.sh --clean    Remove app-generated data (venvs, config, RAG db, TTS
                      voices, caches, logs). Add -y to skip the prompt.

Environment:
  AUTO_SETUP=0        Do not auto-run ./setup.sh on first launch (error instead).
  NEAT_APPS_BRANCH    Branch to pull for `update` (default: main).
  UPDATE_DEPS=0       Skip the setup.sh dependency refresh during `update`.
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

# Remove app-generated data (venvs, generated config, RAG db, downloaded TTS
# voices, pid, caches, logs). Confirms first unless -y/--yes or CLEAN_YES=1.
# Downloaded chat/VLM/ASR models under catalog_dir are left intact.
do_clean() {
  local yes="${1:-}"
  # Never delete files out from under a running instance.
  if do_status >/dev/null 2>&1; then
    info "Stopping the running instance first…"
    do_stop >/dev/null 2>&1 || true
  fi

  local -a targets=() t
  for t in \
    "${DEFAULT_APP_VENV}" \
    "${EXAMPLE_DIR}/.venv-pipertts" \
    "${DEFAULT_LOCAL_CONFIG}" \
    "${PID_FILE}" \
    "${PYTHON_DIR}/ui/milvus.db" \
    "${PYTHON_DIR}/ui/milvus.meta.json" \
    "${PYTHON_DIR}/ui/assets/piper-plus" \
    "${PYTHON_DIR}/ui/assets/mms-tts-kor"; do
    [[ -e "$t" ]] && targets+=("$t")
  done
  # Downloaded rhasspy .onnx voices (committed .onnx.json configs are kept),
  # __pycache__ dirs, and any *.log files the app produced.
  while IFS= read -r -d '' t; do targets+=("$t"); done \
    < <(find "${PYTHON_DIR}/ui/assets" -maxdepth 1 -name '*.onnx' -print0 2>/dev/null)
  while IFS= read -r -d '' t; do targets+=("$t"); done \
    < <(find "${PYTHON_DIR}" -type d -name '__pycache__' -print0 2>/dev/null)
  while IFS= read -r -d '' t; do targets+=("$t"); done \
    < <(find "${EXAMPLE_DIR}" -maxdepth 2 -name '*.log' -print0 2>/dev/null)

  step "Cleaning generated data"
  if [[ ${#targets[@]} -eq 0 ]]; then
    ok "Nothing to clean — no generated data found."
    return 0
  fi

  info "These will be removed:"
  for t in "${targets[@]}"; do
    printf '     %s%s%s\n' "${C_DIM}" "${t#"${EXAMPLE_DIR}"/}" "${C_RESET}"
  done
  local size; size="$(du -sch "${targets[@]}" 2>/dev/null | tail -n1 | awk '{print $1}')"
  [[ -n "${size}" ]] && info "Total size: ${C_BOLD}${size}${C_RESET}"
  local catalog; catalog="$(sed -n 's/^[[:space:]]*catalog_dir:[[:space:]]*\(.*\)/\1/p' \
    "${CONFIG_PATH}" 2>/dev/null | head -n1)"
  [[ -n "${catalog}" ]] && info "Downloaded models under ${C_DIM}${catalog}${C_RESET} are kept."

  if [[ "${yes}" != "-y" && "${yes}" != "--yes" && "${CLEAN_YES:-0}" != "1" ]]; then
    printf '   Remove these? [y/N] '
    local reply=""; read -r reply || true
    if [[ ! "${reply}" =~ ^[Yy] ]]; then
      info "Aborted — nothing removed."
      return 0
    fi
  fi

  for t in "${targets[@]}"; do rm -rf "$t"; done
  ok "Clean complete. Re-run ./setup.sh to reinstall."
}

# Update the example's source to the latest published version. User data — venvs,
# config.local.yaml, RAG db, and downloaded models (which live outside this dir) —
# is preserved. In a git checkout this is `git pull`; standalone (fetched via
# get-example.sh) it re-fetches the release archive and overlays the source. Set
# NEAT_APPS_BRANCH to choose a branch (default main); UPDATE_DEPS=0 skips setup.sh.
do_update() {
  local branch="${NEAT_APPS_BRANCH:-main}"
  if do_status >/dev/null 2>&1; then
    warn "The studio is running — restart it (./run.sh stop, then start) after updating."
  fi
  section "Update"

  if command -v git >/dev/null 2>&1 \
       && git -C "${EXAMPLE_DIR}" ls-files --error-unmatch run.sh >/dev/null 2>&1; then
    # Part of an apps git checkout — pull is the cleanest update.
    step "Pulling the latest changes (git)…"
    if git -C "${EXAMPLE_DIR}" pull --ff-only; then
      ok "Source updated."
    else
      errln "git pull failed (local changes or a diverged branch). Resolve it manually."
      return 1
    fi
  else
    # Standalone (fetched via get-example.sh): re-download the archive and overlay.
    command -v tar >/dev/null 2>&1 || { errln "tar is required to update."; return 1; }
    local repo repo_url archive_url tmp root ex_path
    repo_url="${NEAT_APPS_REPO_URL:-https://github.com/sima-neat/apps.git}"
    repo="${repo_url%.git}"; repo="${repo#https://github.com/}"; repo="${repo#git@github.com:}"
    archive_url="${NEAT_APPS_ARCHIVE_URL:-https://github.com/${repo}/archive/refs/heads/${branch}.tar.gz}"
    tmp="$(mktemp -d)" || { errln "mktemp failed."; return 1; }
    step "Fetching the latest source (${branch})…"
    if [[ -f "${archive_url}" ]]; then
      cp "${archive_url}" "${tmp}/src.tar.gz" \
        || { errln "Cannot read archive: ${archive_url}"; rm -rf "${tmp}"; return 1; }
    elif command -v curl >/dev/null 2>&1; then
      curl -fsSL "${archive_url}" -o "${tmp}/src.tar.gz" \
        || { errln "Download failed: ${archive_url}"; rm -rf "${tmp}"; return 1; }
    elif command -v wget >/dev/null 2>&1; then
      wget -qO "${tmp}/src.tar.gz" "${archive_url}" \
        || { errln "Download failed: ${archive_url}"; rm -rf "${tmp}"; return 1; }
    else
      errln "curl or wget is required to update."; rm -rf "${tmp}"; return 1
    fi
    root="$(tar -tzf "${tmp}/src.tar.gz" 2>/dev/null | head -1 | cut -d/ -f1)"
    ex_path="${root}/examples/genai/neat-genai-studio"
    if ! tar -tzf "${tmp}/src.tar.gz" 2>/dev/null | grep -qxF "${ex_path}/"; then
      errln "neat-genai-studio not found in the archive (branch ${branch})."
      rm -rf "${tmp}"; return 1
    fi
    tar -xzf "${tmp}/src.tar.gz" -C "${tmp}" "${ex_path}" \
      || { errln "Extract failed."; rm -rf "${tmp}"; return 1; }
    step "Applying update (keeping your models, venvs, config and RAG db)…"
    # The archive holds only tracked source, so this overlays new code without
    # touching venvs / config.local.yaml / milvus.db (none are in the archive).
    cp -a "${tmp}/${ex_path}/." "${EXAMPLE_DIR}/" \
      || { errln "Copy failed — the update may be incomplete."; rm -rf "${tmp}"; return 1; }
    rm -rf "${tmp}"
    chmod +x "${EXAMPLE_DIR}/run.sh" "${EXAMPLE_DIR}/setup.sh" 2>/dev/null || true
    ok "Source updated from ${branch}."
  fi

  # Refresh dependencies (setup.sh is idempotent) unless opted out.
  if [[ "${UPDATE_DEPS:-1}" == "0" ]]; then
    ok "Update complete. Run ${C_BOLD}./setup.sh${C_RESET} if dependencies changed."
  else
    step "Refreshing dependencies (./setup.sh)…"
    if "${EXAMPLE_DIR}/setup.sh"; then
      ok "Update complete."
    else
      warn "Dependency refresh reported an error — re-run ./setup.sh manually."
    fi
  fi
}

case "${1:-run}" in
  stop) do_stop; exit 0 ;;
  status) if do_status; then exit 0; else exit 1; fi ;;
  --clean|clean) do_clean "${2:-}"; exit 0 ;;
  update|--update|upgrade) do_update; exit 0 ;;
  -h|--help|help) usage; exit 0 ;;
  --cli|cli) CLI_MODE=1 ;;   # fall through to launch, then run the terminal chat
  # CLI shortcuts: launch the terminal chat straight into a mode. An optional
  # second argument (a model name, or HF repo for download) is forwarded too.
  --chat|chat)
    CLI_MODE=1; CLI_EXTRA_ARGS+=(--chat)
    if [[ -n "${2:-}" ]]; then CLI_EXTRA_ARGS+=("$2"); fi ;;
  --download|download|dl)
    CLI_MODE=1; CLI_EXTRA_ARGS+=(--download)
    if [[ -n "${2:-}" ]]; then CLI_EXTRA_ARGS+=("$2"); fi ;;
  --benchmark|--bench|benchmark|bench)
    CLI_MODE=1; CLI_EXTRA_ARGS+=(--benchmark)
    if [[ -n "${2:-}" ]]; then CLI_EXTRA_ARGS+=("$2"); fi ;;
  run|start) ;;  # fall through to launch
  *) errln "unknown command: $1"; usage; exit 2 ;;
esac

banner

# First launch: if setup.sh hasn't created the app venv (and local config) yet,
# run it now so `./run.sh` works out of the box. AUTO_SETUP=0 opts out.
if [[ ! -x "${DEFAULT_APP_VENV}/bin/python" || ! -f "${DEFAULT_LOCAL_CONFIG}" ]]; then
  if [[ "${AUTO_SETUP:-1}" == "0" ]]; then
    errln "Setup has not been run yet (no ${DEFAULT_APP_VENV##*/} / config.local.yaml)."
    errln "Run ./setup.sh first, or unset AUTO_SETUP=0 to let run.sh do it."
    exit 1
  fi
  step "First launch — running ./setup.sh (this installs deps and downloads a model)…"
  if ! bash "${EXAMPLE_DIR}/setup.sh"; then
    errln "setup.sh failed — fix the errors above and re-run ./run.sh."
    exit 1
  fi
  ok "Setup complete."
  # Pick up what setup just created (these were resolved to fallbacks at startup).
  [[ -x "${DEFAULT_APP_VENV}/bin/python" ]] && APP_PYTHON="${DEFAULT_APP_VENV}/bin/python"
  if [[ "${CONFIG_PATH}" == "${EXAMPLE_DIR}/src/common/config.yaml" && -f "${DEFAULT_LOCAL_CONFIG}" ]]; then
    CONFIG_PATH="${DEFAULT_LOCAL_CONFIG}"
  fi
  if [[ -z "${PIPERTTS_PYTHON:-}" && -x "${EXAMPLE_DIR}/.venv-pipertts/bin/python" ]]; then
    export PIPERTTS_PYTHON="${EXAMPLE_DIR}/.venv-pipertts/bin/python"
  fi
fi

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

  # 2) Restart only the MLA dispatcher. Its systemd unit owns all hardware
  # initialization; the application must not invoke an initializer itself.
  if command -v systemctl >/dev/null 2>&1; then
    info "Restarting MLA dispatcher (${MLA_DISPATCHER_SERVICE})…"
    mla_sudo systemctl restart "${MLA_DISPATCHER_SERVICE}" 2>/dev/null \
      || warn "could not restart ${MLA_DISPATCHER_SERVICE} (need privileges? set MLA_RESET_CMD or MLA_RESET=0)"
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
  # In --cli mode the watchdog may have relaunched the model server under a new
  # process group (not in our remembered groups) — sweep any stray one.
  if [[ "${CLI_MODE}" == "1" ]] && command -v pkill >/dev/null 2>&1; then
    pkill -TERM -f "${SERVER_PATTERN}" 2>/dev/null || true
    sleep 1
    pkill -KILL -f "${SERVER_PATTERN}" 2>/dev/null || true
  fi
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
# Expose the supervisor PID + PID file to the UI so it can offer a GUI "Shut down"
# button (it SIGTERMs this process, which runs cleanup — same as `./run.sh stop`).
export NEAT_RUN_PID="$$"
export RUN_PID_FILE="${PID_FILE}"

# Sentinel exit code the model server uses to ask for an MLA reset + relaunch.
MLA_RESET_EXIT_CODE="${MLA_RESET_EXIT_CODE:-75}"

launch_server() {
  step "Starting the Neat model server (OpenAI-compatible)…"
  # In CLI mode the model server shares the terminal with the interactive chat,
  # so its logs would land on the "you ▸" prompt. Send them to a log file
  # instead; the CLI drives the server over HTTP and doesn't need its stdout.
  if [[ "${CLI_MODE}" == "1" ]]; then
    setsid "${PYNEAT_PYTHON}" "${PYTHON_DIR}/server/main.py" --config "${CONFIG_PATH}" \
      >"${SERVER_LOG}" 2>&1 &
  else
    setsid "${PYNEAT_PYTHON}" "${PYTHON_DIR}/server/main.py" --config "${CONFIG_PATH}" &
  fi
  server_pid="$!"
  pids[0]="${server_pid}"
  remember_process_group "${server_pid}"
}

# Background watchdog for --cli mode. CLI mode has no supervisor loop, so if the
# model server exits to service an MLA reset (the CLI's /reset) — or crashes —
# relaunch it here so the CLI can reconnect. Poll-based, since a backgrounded
# subshell can't `wait` a sibling PID; bounded so a broken server won't respawn
# forever. Strays it may spawn are swept by cleanup().
cli_supervise() {
  local tries=0
  while true; do
    sleep 2
    if kill -0 "${server_pid}" 2>/dev/null; then
      tries=0
      continue
    fi
    [[ "${tries}" -ge "${MLA_MAX_RESTART_RETRIES:-4}" ]] && return 0
    tries=$((tries + 1))
    reset_mla_dispatcher
    launch_server
    sleep "${MODEL_SERVER_START_DELAY:-2}"
  done
}

section "Accelerator"
reset_mla_clean_slate

section "Model Server"
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

# CLI mode: skip the web UI and run an interactive terminal chat in the
# foreground against the model server we just started. When it exits, the EXIT
# trap tears the model server down.
if [[ "${CLI_MODE}" == "1" ]]; then
  section "Terminal chat"
  step "Launching the CLI — type ${C_BOLD}/help${C_RESET} for commands, ${C_BOLD}/quit${C_RESET} to exit."
  info "Model server logs → ${C_DIM}${SERVER_LOG}${C_RESET}"
  printf '\n'
  cli_supervise &       # relaunch the server on an MLA-reset request so /reset works
  cli_sup_pid="$!"
  trap - INT            # let the Python CLI own Ctrl+C (abort a reply, not exit)
  "${APP_PYTHON}" "${PYTHON_DIR}/cli/main.py" --config "${CONFIG_PATH}" \
    ${CLI_EXTRA_ARGS[@]+"${CLI_EXTRA_ARGS[@]}"} || true
  kill "${cli_sup_pid}" 2>/dev/null || true
  exit 0                # -> EXIT trap stops the model server
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
# keeps running and its socket reconnects automatically. Other failures are
# deterministic application/configuration errors and stop without an MLA reset.
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
    if [[ "${status}" -ne "${MLA_RESET_EXIT_CODE}" ]]; then
      errln "Model server exited with status ${status}; shutting down without resetting the MLA."
      break
    fi
    if [[ "${restart_attempts}" -ge "${MLA_MAX_RESTART_RETRIES}" ]]; then
      errln "Model server did not stay up after ${restart_attempts} restart attempts (last status ${status}); shutting down."
      break
    fi
    restart_attempts=$((restart_attempts + 1))
    info "Model server requested an MLA reset — resetting dispatcher and relaunching (attempt ${restart_attempts}/${MLA_MAX_RESTART_RETRIES})…"
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
