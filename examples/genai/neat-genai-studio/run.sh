#!/usr/bin/env bash
set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_DIR="${EXAMPLE_DIR}/src/python"
DEFAULT_APP_VENV="${EXAMPLE_DIR}/.venv"
DEFAULT_LOCAL_CONFIG="${EXAMPLE_DIR}/config.local.yaml"
# shellcheck source=src/common/config_value.sh
source "${EXAMPLE_DIR}/src/common/config_value.sh"

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

# Interpreter of the isolated piper-tts venv — the backend's subprocess TTS worker
# uses it for the de/it/no/vi rhasspy voices (piper-tts and piper-plus can't
# share a venv). Exported so the backend process inherits it.
if [[ -z "${PIPERTTS_PYTHON:-}" && -x "${EXAMPLE_DIR}/.venv-pipertts/bin/python" ]]; then
  PIPERTTS_PYTHON="${EXAMPLE_DIR}/.venv-pipertts/bin/python"
fi
export PIPERTTS_PYTHON="${PIPERTTS_PYTHON:-}"
# Supertonic 3 (MLA TTS): the vendored runtime runs from ./.venv-supertonic (see
# setup.sh); its model files' location is persisted under app.tts.supertonic in
# the local config, with SUPERTONIC_MODELS_ROOT in the environment overriding.
# Resolved by resolve_supertonic_env once the config path is final. The backend
# spawns supertonic_worker.py with the resolved interpreter; when it is absent
# the engine is simply not offered.
SUPERTONIC_PYTHON="${SUPERTONIC_PYTHON:-}"
SUPERTONIC_VENV_EXPLICIT="${SUPERTONIC_VENV:-}"
SUPERTONIC_VENV="${SUPERTONIC_VENV:-${EXAMPLE_DIR}/.venv-supertonic}"
SHUTDOWN_GRACE_SECONDS="${SHUTDOWN_GRACE_SECONDS:-10}"
RAG_WORKER_PATTERN="${PYTHON_DIR}/rag/vectordb_worker.py"
SERVER_PATTERN="${PYTHON_DIR}/server/main.py"
UI_PATTERN="${PYTHON_DIR}/ui/main.py"
# PID file for the running instance (enables `./run.sh stop`).
PID_FILE="${RUN_PID_FILE:-${EXAMPLE_DIR}/.neat-genai-studio.pid}"
# The running instance records the URL it is actually serving, so `status`
# reports that, not a URL recomputed from its own (possibly different)
# CONFIG_PATH.
URL_FILE="${PID_FILE%.pid}.url"
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
web_config_value() { web_config_scalar "${CONFIG_PATH}" "$1"; }

# A host as it must appear in a URL: IPv6 literals in brackets.
url_host() {
  local h="$1"
  if [[ "${h}" == *:* && "${h}" != \[*\] ]]; then printf '[%s]' "${h}"; else printf '%s' "${h}"; fi
}
# True for the "listen on every interface" hosts (or none configured).
wildcard_host() { [[ -z "$1" || "$1" == "0.0.0.0" || "$1" == "::" || "$1" == "[::]" ]]; }

# _web_url_for <host-or-empty>: the backend URL with that host (scheme and port from
# the config). Returns 1 when the config has no port.
_web_url_for() {
  local port https scheme
  port="$(web_config_value port)"
  https="$(web_config_value https)"
  [[ -n "${port}" ]] || return 1
  # https defaults to true in shared.config when the key is absent.
  scheme="https"; [[ -n "${https}" ]] && ! config_true "${https}" && scheme="http"
  printf '%s://%s:%s' "${scheme}" "$(url_host "$1")" "${port}"
}

# The backend URL to give other machines (Insight): the configured host, or this
# board's first LAN address (IPv4 preferred) when it listens on every interface.
web_url() {
  local host ip
  host="$(web_config_value host)"
  if wildcard_host "${host}"; then
    ip="$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -v ':' | grep -m1 . || true)"
    [[ -n "${ip}" ]] || ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
    host="${ip:-localhost}"
  fi
  _web_url_for "${host}"
}

# The backend URL on this board: localhost when it listens on every interface
# (it keeps working if the IP changes), else the configured host.
local_web_url() {
  local host
  host="$(web_config_value host)"
  wildcard_host "${host}" && host="localhost"
  _web_url_for "${host}"
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
  _kv "supertonic" "$(supertonic_installed && echo "models ${SUPERTONIC_MODELS_ROOT_RESOLVED:-}" || echo "not installed (./setup.sh installs it)")"
  _kv "host" "$(uname -sm 2>/dev/null || echo unknown)"
}

usage() {
  banner
  cat <<'USAGE'
Usage:
  ./run.sh            Start the model server and the Studio backend: the API
                      Insight's GenAI tab calls (chat, speech, transcription,
                      translation, voices, models, RAG, /health). Ctrl+C stops.
                      Runs ./setup.sh automatically on the first launch.
  ./run.sh stop       Cleanly stop a running instance.
  ./run.sh status     Report whether the studio is running.
  ./run.sh update     Update to the latest version (preserves models, config,
                      venvs and RAG db). Dependency refresh is opt-in.
  ./run.sh --clean    Remove app-generated data (venvs, config, RAG db, TTS
                      voices, caches, logs). Add -y to skip the prompt.

Environment:
  AUTO_SETUP=0        Do not auto-run ./setup.sh on first launch (error instead).
  NEAT_APPS_BRANCH    Branch to pull for `update` (default: main).
  UPDATE_DEPS=1       Run full setup.sh dependency refresh during `update`.
  BACKEND_CORS_ORIGINS
                      Origins (comma-separated, or *) whose browser pages may
                      call the API (CORS). Default: app.web.cors_origins.
USAGE
}

do_status() {
  if [[ -f "${PID_FILE}" ]]; then
    local pid; pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      # Report the URL the RUNNING instance recorded: recomputing it here would
      # describe this invocation's config, which a CONFIG_PATH override makes a
      # different (possibly unreachable) endpoint. Fall back for an instance
      # started before the url file existed.
      local url; url="$(cat "${URL_FILE}" 2>/dev/null || true)"
      [[ -n "${url}" ]] || url="$(web_url || true)"
      ok "Neat GenAI Studio is running (pid ${pid})."
      [[ -n "${url}" ]] && info "API: ${C_ACCENT}${url}${C_RESET}  health: ${C_ACCENT}${url}/health${C_RESET}"
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
      rm -f "${PID_FILE}" "${URL_FILE}"
      ok "Stopped."
      return 0
    fi
    rm -f "${PID_FILE}" "${URL_FILE}"
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

# Supertonic paths: environment > app.tts.supertonic in the config > defaults.
# Only values that are set are exported, so the backend applies the same precedence.
_supertonic_config_value() { supertonic_config_value "${CONFIG_PATH}" "$1"; }
resolve_supertonic_env() {
  local models legacy
  # Pre-vendoring config/env: everything lived under app_root (<app_root>/models
  # and the runtime venv <app_root>/.venv), until setup.sh migrates the install.
  legacy="${SUPERTONIC_APP_ROOT:-$(_supertonic_config_value app_root)}"
  models="${SUPERTONIC_MODELS_ROOT:-$(_supertonic_config_value models_root)}"
  if [[ -z "${models}" && -n "${legacy}" ]]; then
    models="${legacy}/models"
  fi
  [[ -n "${models}" ]] && export SUPERTONIC_MODELS_ROOT="${models}"
  # The venv setup.sh built (persisted as app.tts.supertonic.venv when it was
  # not the default); SUPERTONIC_VENV / SUPERTONIC_PYTHON in the environment win.
  if [[ -z "${SUPERTONIC_VENV_EXPLICIT}" ]]; then
    local venv
    venv="$(_supertonic_config_value venv)"
    [[ -n "${venv}" ]] && SUPERTONIC_VENV="${venv}"
  fi
  if [[ -z "${SUPERTONIC_PYTHON}" && -x "${SUPERTONIC_VENV}/bin/python" ]]; then
    SUPERTONIC_PYTHON="${SUPERTONIC_VENV}/bin/python"
  fi
  # An install from before the runtime was vendored has no new venv yet: keep
  # using its <app_root>/.venv (it has numpy/onnxruntime/pyneat; the worker
  # imports the vendored package) so updating never drops the default engine.
  if [[ -z "${SUPERTONIC_PYTHON}" && -n "${legacy}" && -x "${legacy}/.venv/bin/python" ]]; then
    SUPERTONIC_PYTHON="${legacy}/.venv/bin/python"
  fi
  export SUPERTONIC_PYTHON SUPERTONIC_VENV
  # Effective path (default applied) for run.sh's own use: the banner and --clean.
  SUPERTONIC_MODELS_ROOT_RESOLVED="${models:-/media/nvme/supertonic-tts/models}"
}
# Same test as supertonic_tts.available(): runtime interpreter plus the two
# marker model files (setup.sh verifies every file's checksum).
supertonic_installed() {
  [[ -n "${SUPERTONIC_PYTHON}" \
     && -f "${SUPERTONIC_MODELS_ROOT_RESOLVED}/supertonic-3/onnx/tts.json" \
     && -f "${SUPERTONIC_MODELS_ROOT_RESOLVED}/supertonic-3-sima/supertonic_vector_field_sima_mpk.tar.gz" ]]
}
# What a dependency refresh would change for Supertonic: its pinned requirements
# and the reviewed model revisions/checksums in setup.sh.
supertonic_fingerprint() {
  {
    cat "${PYTHON_DIR}/requirements-supertonic.txt" 2>/dev/null
    grep -E '^SUPERTONIC_(UPSTREAM|SIMA)_HF_REVISION=|^[0-9a-f]{64}  supertonic-3' "${EXAMPLE_DIR}/setup.sh" 2>/dev/null
  } | sha256sum | cut -d' ' -f1
}
# Desktop icons and menu entries an earlier setup.sh installed for this
# checkout. They started the removed web UI through src/common/desktop-launch.sh,
# so they no longer work.
old_desktop_entries() {
  local t
  for t in "${XDG_DATA_HOME:-${HOME}/.local/share}/applications/neat-genai-studio.desktop" \
           "${XDG_DESKTOP_DIR:-${HOME}/Desktop}/neat-genai-studio.desktop"; do
    [[ -f "$t" ]] && grep -qF "${EXAMPLE_DIR}/" "$t" && printf '%s\n' "$t"
  done
  return 0
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

  resolve_supertonic_env
  local -a targets=() t
  # The Supertonic venv under the example directory is app-generated and always
  # removed. One configured elsewhere (SUPERTONIC_VENV / app.tts.supertonic.venv)
  # and the downloaded model files are kept unless CLEAN_SUPERTONIC=1.
  case "${SUPERTONIC_VENV}/" in
    "${EXAMPLE_DIR}"/*) [[ -e "${SUPERTONIC_VENV}" ]] && targets+=("${SUPERTONIC_VENV}") ;;
    *)
      if [[ "${CLEAN_SUPERTONIC:-0}" == "1" ]]; then
        [[ -e "${SUPERTONIC_VENV}" ]] && targets+=("${SUPERTONIC_VENV}")
      elif [[ -e "${SUPERTONIC_VENV}" ]]; then
        info "Keeping the Supertonic venv outside this directory: ${SUPERTONIC_VENV} (CLEAN_SUPERTONIC=1 removes it)."
      fi ;;
  esac
  if [[ "${CLEAN_SUPERTONIC:-0}" == "1" && -e "${SUPERTONIC_MODELS_ROOT_RESOLVED}" ]]; then
    targets+=("${SUPERTONIC_MODELS_ROOT_RESOLVED}")
  fi
  for t in \
    "${DEFAULT_APP_VENV}" \
    "${EXAMPLE_DIR}/.venv-pipertts" \
    "${DEFAULT_LOCAL_CONFIG}" \
    "${DEFAULT_LOCAL_CONFIG}.bak" \
    "${PID_FILE}" \
    "${PYTHON_DIR}/ui/milvus.db" \
    "${PYTHON_DIR}/ui/milvus.meta.json" \
    "${PYTHON_DIR}/ui/assets/piper-plus" \
    "${PYTHON_DIR}/ui/assets/mms-tts-kor"; do
    [[ -e "$t" ]] && targets+=("$t")
  done
  # A desktop icon / menu entry an earlier setup.sh installed for this checkout.
  while IFS= read -r t; do targets+=("$t"); done < <(old_desktop_entries)
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
  if [[ "${CLEAN_SUPERTONIC:-0}" != "1" && -e "${SUPERTONIC_MODELS_ROOT_RESOLVED}" ]]; then
    info "Supertonic models under ${C_DIM}${SUPERTONIC_MODELS_ROOT_RESOLVED}${C_RESET} are kept (CLEAN_SUPERTONIC=1 removes them)."
  fi

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
# get-example.sh) it re-fetches the release archive and mirrors the source. Set
# NEAT_APPS_BRANCH chooses a branch (default main); UPDATE_DEPS=1 refreshes only
# the Python dependencies.
migrate_piper_voices() {
  local assets="${PYTHON_DIR}/ui/assets" py voice encoder decoder needs_split=0
  py="${PIPERTTS_PYTHON:-${EXAMPLE_DIR}/.venv-pipertts/bin/python}"
  for voice in "${assets}"/*.onnx; do
    [[ -e "${voice}" ]] || continue
    [[ "${voice}" == *.enc.onnx || "${voice}" == *.dec.onnx ]] && continue
    encoder="${voice%.onnx}.enc.onnx"
    decoder="${voice%.onnx}.dec.onnx"
    if [[ ! -f "${encoder}" || ! -f "${decoder}" \
          || "${voice}" -nt "${encoder}" || "${voice}" -nt "${decoder}" ]]; then
      needs_split=1
      break
    fi
  done
  [[ "${needs_split}" == "1" ]] || return 0

  step "Preparing installed Piper voices for streaming…"
  if [[ ! -x "${py}" ]]; then
    errln "Piper environment not found; run ./setup.sh before updating."
    return 1
  fi
  if ! "${py}" -c "import onnx" >/dev/null 2>&1; then
    "${py}" -m pip install "onnx>=1.17,<2" || return 1
  fi
  "${py}" "${PYTHON_DIR}/ui/tts/split_voices.py" "${assets}"
}

do_update() {
  local branch="${NEAT_APPS_BRANCH:-main}"
  local supertonic_before
  supertonic_before="$(supertonic_fingerprint)"
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
    # Standalone (fetched via get-example.sh): re-download and mirror the source.
    command -v tar >/dev/null 2>&1 || { errln "tar is required to update."; return 1; }
    command -v rsync >/dev/null 2>&1 || { errln "rsync is required to update a standalone install."; return 1; }
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
    # Read the whole listing (awk, not head/grep -q): under pipefail, a reader
    # that exits early kills tar with SIGPIPE and the update aborts with 141.
    root="$(tar -tzf "${tmp}/src.tar.gz" 2>/dev/null | awk -F/ 'NR == 1 { print $1 }')"
    ex_path="${root}/examples/genai/neat-genai-studio"
    if ! tar -tzf "${tmp}/src.tar.gz" 2>/dev/null | grep -xF "${ex_path}/" >/dev/null; then
      errln "neat-genai-studio not found in the archive (branch ${branch})."
      rm -rf "${tmp}"; return 1
    fi
    tar -xzf "${tmp}/src.tar.gz" -C "${tmp}" "${ex_path}" \
      || { errln "Extract failed."; rm -rf "${tmp}"; return 1; }
    step "Applying update (keeping your models, venvs, config and RAG db)…"
    # The Supertonic venv and models root are configurable and may be pointed at
    # a non-default directory INSIDE the example, where --delete-delay would
    # erase them — a custom runtime and potentially gigabytes of models — while
    # this step claims to keep them. Resolve the configured paths and exclude any
    # that fall under EXAMPLE_DIR, on top of the default locations below.
    # Resolve the persisted paths FIRST: they are only populated by
    # resolve_supertonic_env, and reading them unset aborts under `set -u` —
    # which would break the documented update rather than protect anything. It
    # also picks up a custom venv recorded only in config.local.yaml.
    resolve_supertonic_env
    local -a keep=()
    local cfg_path abs rel
    for cfg_path in "${SUPERTONIC_VENV:-}" "${SUPERTONIC_MODELS_ROOT:-}"; do
      [[ -n "${cfg_path}" ]] || continue
      # Resolve without requiring the directory to exist yet; `local x=$(...)`
      # always returns 0, so the status is checked separately.
      abs=""
      if [[ -d "${cfg_path}" ]]; then
        abs="$(cd "${cfg_path}" 2>/dev/null && pwd -P)" || abs=""
      elif [[ -d "$(dirname "${cfg_path}")" ]]; then
        abs="$(cd "$(dirname "${cfg_path}")" 2>/dev/null && pwd -P)" || abs=""
        [[ -n "${abs}" ]] && abs="${abs}/$(basename "${cfg_path}")"
      else
        abs="${cfg_path}"
      fi
      [[ -n "${abs}" ]] || continue
      case "${abs}" in
        "${EXAMPLE_DIR}"/*)
          rel="${abs#"${EXAMPLE_DIR}"/}"
          keep+=( "--exclude=/${rel}/" )
          info "Keeping configured path: ${C_DIM}${rel}${C_RESET}"
          ;;
      esac
    done
    # Mirror tracked source so files deleted by a release disappear locally.
    # Explicitly exclude every install-owned path from transfer and deletion.
    rsync -a --delete-delay --delay-updates \
      ${keep[@]+"${keep[@]}"} \
      --exclude='/.venv/' \
      --exclude='/.venv-pipertts/' \
      --exclude='/.venv-supertonic/' \
      --exclude='/config.local.yaml' \
      --exclude='/config.local.yaml.bak' \
      --exclude='/.local-certs/' \
      --exclude='/.neat-genai-studio.pid' \
      --exclude='/.neat-genai-studio.mode' \
      --exclude='/.neat-genai-studio.url' \
      --exclude='*.log' \
      --exclude='/src/python/ui/uploads/' \
      --exclude='/src/python/ui/.milvus.db.lock' \
      --exclude='/src/python/ui/milvus.db' \
      --exclude='/src/python/ui/milvus.meta.json' \
      --exclude='/src/python/ui/assets/*.onnx' \
      --exclude='/src/python/ui/assets/*.onnx.json' \
      --exclude='/src/python/ui/assets/piper-plus/' \
      --exclude='/src/python/ui/assets/mms-tts-kor/' \
      "${tmp}/${ex_path}/" "${EXAMPLE_DIR}/" \
      || { errln "Source replacement failed."; rm -rf "${tmp}"; return 1; }
    rm -rf "${tmp}"
    chmod +x "${EXAMPLE_DIR}/run.sh" "${EXAMPLE_DIR}/setup.sh" 2>/dev/null || true
    ok "Source updated from ${branch}."
  fi

  # Dependency refresh is opt-in and deliberately avoids setup's data-writing
  # model, voice, config and RAG stages.
  if [[ "${UPDATE_DEPS:-0}" == "0" ]]; then
    info "Python dependency refresh skipped (set UPDATE_DEPS=1 to enable it)."
    if [[ "$(supertonic_fingerprint)" != "${supertonic_before}" ]]; then
      warn "This update changes the Supertonic requirements or model revisions; run UPDATE_DEPS=1 ./run.sh update (or ./setup.sh) to apply them."
    fi
  else
    step "Refreshing Python dependencies…"
    # setup.sh must refresh the Supertonic runtime this installation actually
    # uses: hand it the persisted venv and models root (config.local.yaml),
    # not its defaults, unless the environment already names them.
    resolve_supertonic_env
    if "${EXAMPLE_DIR}/setup.sh" --dependencies-only; then
      ok "Python dependencies refreshed."
    else
      errln "Dependency refresh failed."
      return 1
    fi
  fi
  migrate_piper_voices || { errln "Piper voice migration failed."; return 1; }
  ok "Update complete."
}

case "${1:-run}" in
  stop) do_stop; exit 0 ;;
  status) if do_status; then exit 0; else exit 1; fi ;;
  --clean|clean) do_clean "${2:-}"; exit 0 ;;
  update|--update|upgrade) do_update; exit 0 ;;
  -h|--help|help) usage; exit 0 ;;
  run|start) ;;  # fall through to launch
  *) errln "unknown command: $1"; usage; exit 2 ;;
esac

banner

# An icon from an earlier setup.sh would start the removed web UI: remove it.
while IFS= read -r t; do rm -f "$t" && info "Removed an old desktop icon: $t"; done < <(old_desktop_entries)

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

# Also covers the first launch after updating from a run.sh version that did
# not yet invoke the migration itself.
if ! migrate_piper_voices; then
  errln "Installed Piper voices could not be prepared; fix the error above and retry."
  exit 1
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  errln "config does not exist: ${CONFIG_PATH}"
  exit 2
fi
info "Config: ${C_DIM}${CONFIG_PATH}${C_RESET}"

resolve_supertonic_env

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

# Stop stale Studio processes from a previous (e.g. crashed) run and wait for
# the OpenAI port to become available. Do not touch the board runtime during
# normal application startup.
prepare_clean_start() {
  step "Clearing stale Neat GenAI Studio processes…"
  if command -v pkill >/dev/null 2>&1; then
    pkill -TERM -f "${SERVER_PATTERN}" 2>/dev/null || true
    pkill -TERM -f "${UI_PATTERN}" 2>/dev/null || true
    sleep 1
    pkill -KILL -f "${SERVER_PATTERN}" 2>/dev/null || true
    pkill -KILL -f "${UI_PATTERN}" 2>/dev/null || true
  fi
  stop_stale_rag_worker

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
  rm -f "${PID_FILE}" "${URL_FILE}"
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
  rm -f "${PID_FILE}" "${URL_FILE}"
fi

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
# Closing the terminal window stops the Studio cleanly instead of leaving the
# setsid'd processes behind.
trap 'exit 129' HUP

# Record this instance so `./run.sh stop` can find it (removed by cleanup).
echo "$$" > "${PID_FILE}"
_self_url="$(local_web_url || true)"
[[ -n "${_self_url}" ]] && printf '%s\n' "${_self_url}" > "${URL_FILE}"
# Expose the supervisor PID + PID file to the backend for POST /shutdown (it
# SIGTERMs this process, which runs cleanup — same as `./run.sh stop`).
export NEAT_RUN_PID="$$"
export RUN_PID_FILE="${PID_FILE}"

# The model server runs in the Neat environment (PYNEAT_PYTHON), which may not
# carry PyYAML (Platform 3.0's ~/pyneat does not). The app's own Python converts
# the YAML config to JSON for it, so the Neat environment is never modified.
SERVER_CONFIG_JSON="${EXAMPLE_DIR}/.neat-genai-server-config.json"
write_server_config() {
  "${APP_PYTHON}" - "${CONFIG_PATH}" "${SERVER_CONFIG_JSON}" <<'YAML_TO_JSON'
import json
import sys

import yaml

with open(sys.argv[1], encoding="utf-8") as source:
    config = yaml.safe_load(source) or {}
with open(sys.argv[2], "w", encoding="utf-8") as target:
    json.dump(config, target)
YAML_TO_JSON
}

launch_server() {
  step "Starting the Neat model server (OpenAI-compatible)…"
  if ! write_server_config; then
    errln "Could not read ${CONFIG_PATH} with ${APP_PYTHON}; run ./setup.sh first."
    return 1
  fi
  setsid "${PYNEAT_PYTHON}" "${PYTHON_DIR}/server/main.py" --config "${SERVER_CONFIG_JSON}" &
  server_pid="$!"
  pids[0]="${server_pid}"
  remember_process_group "${server_pid}"
}

prepare_clean_start

section "Model Server"
launch_server

sleep "${MODEL_SERVER_START_DELAY:-2}"

if ! child_running "${server_pid}"; then
  set +e
  wait "${server_pid}"
  status=$?
  set -e
  exit "${status}"
fi

section "Backend"
step "Starting the Neat GenAI Studio backend (API endpoints)…"
setsid "${APP_PYTHON}" "${PYTHON_DIR}/ui/main.py" --config "${CONFIG_PATH}" &
pids[1]="$!"
ui_pid="${pids[1]}"
remember_process_group "${ui_pid}"

printf '\n'
ok "Neat GenAI Studio is starting up."
_url="$(web_url || true)"
if [[ -n "${_url}" ]]; then
  info "API on ${C_ACCENT}${C_BOLD}${_url}${C_RESET}; readiness at ${C_ACCENT}${_url}/health${C_RESET}. Point Insight's GenAI tab at it."
  # Set at all (even empty) wins over the config, as in the backend process.
  if [[ -n "${BACKEND_CORS_ORIGINS+set}" ]]; then _cors="${BACKEND_CORS_ORIGINS}"; else _cors="$(web_config_value cors_origins)"; fi
  info "CORS for browser front ends: ${_cors:-off (app.web.cors_origins or BACKEND_CORS_ORIGINS allows origins)}."
fi
info "Press ${C_BOLD}Ctrl+C${C_RESET} to stop, or run ${C_BOLD}./run.sh stop${C_RESET} from another shell."
printf '\n'

# Supervisor: watch both processes; when either exits, the Studio stops.
status=0
while true; do
  if ! child_running "${server_pid}"; then
    set +e
    wait "${server_pid}"
    status=$?
    set -e
    if [[ "${status}" -ne 0 ]]; then
      errln "Model server exited with status ${status}; shutting down."
    fi
    break
  fi
  if ! child_running "${ui_pid}"; then
    set +e
    wait "${ui_pid}"
    status=$?
    set -e
    break
  fi
  sleep 1
done

exit "${status}"
