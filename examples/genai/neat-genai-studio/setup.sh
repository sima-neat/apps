#!/usr/bin/env bash
set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# MODELS_DIR is the model catalog: every compatible model directory under it is
# discoverable and loadable on the fly from the UI (no restart).
MODELS_DIR="${LLIMA_MODELS_PATH:-/media/nvme/llima/models}"
APP_VENV="${APP_VENV:-${EXAMPLE_DIR}/.venv}"
# Isolated venv for piper-tts (it and piper-plus both own the `piper` package).
PIPERTTS_VENV="${PIPERTTS_VENV:-${EXAMPLE_DIR}/.venv-pipertts}"
CONFIG_PATH="${CONFIG_PATH:-${EXAMPLE_DIR}/config.local.yaml}"
PYNEAT_PYTHON="${PYNEAT_PYTHON:-${HOME}/pyneat/bin/python}"
# No chat/VLM model is downloaded by default — the UI starts decoupled and you
# download/load one on demand. Set CHAT_MODEL_REPO to also fetch + preload one.
CHAT_MODEL_REPO="${CHAT_MODEL_REPO:-}"
# Extra compatible chat/VLM models to seed the catalog (space-separated HF repos).
# Example: CATALOG_MODEL_REPOS="simaai/Llama-3.2-3B-Instruct-... simaai/..."
CATALOG_MODEL_REPOS="${CATALOG_MODEL_REPOS:-}"
# Speech-to-text model downloaded and made active at startup.
#
# A LAYERED-encoder build: current LLiMa splits the Whisper encoder into one ELF
# per layer and rejects the older monolithic builds as "Unsupported legacy
# Whisper model". Move this to the simaai layered build once published; on an
# older runtime that needs a monolithic encoder, pass
# ASR_MODEL_REPO=simaai/whisper-small-a16w8 instead. Setup checks the model
# actually loads and will not configure it at startup if the runtime refuses it.
#
# Set ASR_MODEL_REPO="" to install none (note the `-`, not `:-`, so an
# explicitly empty value is honoured rather than falling back to the default).
ASR_MODEL_REPO="${ASR_MODEL_REPO-florianvoss/whisper-small-a16w8-layered-encoder}"
# Extra ASR models to seed the catalog (space-separated HF repos); switch
# between them at runtime in Settings -> Models. Example:
#   ASR_CATALOG_MODEL_REPOS="florianvoss/whisper-medium-a16w8-layered-encoder"
ASR_CATALOG_MODEL_REPOS="${ASR_CATALOG_MODEL_REPOS:-}"
RAG_EMBEDDING_REPO="thenlper/gte-small"
CHAT_MODEL_NAME="${CHAT_MODEL_NAME:-${CHAT_MODEL_REPO##*/}}"
# Only one chat/VLM model is resident at a time — loading a new one clears all
# other chat/VLM models (ASR is always kept). Kept configurable for advanced use.
MAX_RESIDENT_CHAT_MODELS="${MAX_RESIDENT_CHAT_MODELS:-1}"
ALLOW_HUB_DOWNLOAD="${ALLOW_HUB_DOWNLOAD:-true}"
# Hugging Face accounts searched for compatible models (space-separated):
# simaai (official precompiled) + TDoSiMa and florianvoss (community).
HUB_ORGS="${HUB_ORGS:-simaai TDoSiMa florianvoss}"
INSTALL_TTS_VOICES="${INSTALL_TTS_VOICES:-1}"
DEFAULT_TTS_LANGUAGES="en,de,es,fr,it,ja,pt,vi,zh"
TTS_LANGUAGES="${TTS_LANGUAGES:-}"
TTS_OPTIONAL_VOICES="${TTS_OPTIONAL_VOICES:-}"
# Supertonic 3 (MLA-accelerated multilingual TTS). The runtime is vendored under
# src/python/ui/supertonic_sima/ and runs in its own venv (pyneat + onnxruntime +
# numpy 1.26, which the UI venv cannot host); the model files come from Hugging
# Face at pinned revisions and are checksum-verified. INSTALL_SUPERTONIC=0 skips it.
INSTALL_SUPERTONIC="${INSTALL_SUPERTONIC:-1}"
# Paths: environment > what an earlier setup persisted in CONFIG_PATH
# (app.tts.supertonic.{venv,models_root}, or the pre-vendoring app_root) > defaults,
# so re-running setup without the variables keeps a custom install where it is.
# The persisted values are read by resolve_supertonic_paths once the UI venv
# exists, through the application's own config loader (any YAML form).
# shellcheck source=src/common/config_value.sh
source "${EXAMPLE_DIR}/src/common/config_value.sh"
SUPERTONIC_DEFAULT_VENV="${EXAMPLE_DIR}/.venv-supertonic"
SUPERTONIC_DEFAULT_MODELS_ROOT="/media/nvme/supertonic-tts/models"
SUPERTONIC_VENV="${SUPERTONIC_VENV:-}"
# An explicit PyNeat wheel for the Supertonic venv, and where to look for one
# matching the installed runtime when it is not given.
PYNEAT_WHEEL="${PYNEAT_WHEEL:-}"
PYNEAT_WHEEL_DIRS="${PYNEAT_WHEEL_DIRS:-/media/nvme/neat /data/neat ${HOME}/neat ${HOME}/Downloads}"
# Model files (SUPERTONIC_APP_ROOT is the pre-vendoring name: its models/ subdir).
SUPERTONIC_MODELS_ROOT="${SUPERTONIC_MODELS_ROOT:-${SUPERTONIC_APP_ROOT:+${SUPERTONIC_APP_ROOT}/models}}"
resolve_supertonic_paths() {
  # The UI venv (built just before) has PyYAML and runs config_query.py.
  [[ -x "${APP_VENV}/bin/python" ]] && export STUDIO_CONFIG_PYTHON="${APP_VENV}/bin/python"
  SUPERTONIC_VENV="${SUPERTONIC_VENV:-$(supertonic_config_value "${CONFIG_PATH}" venv)}"
  SUPERTONIC_VENV="${SUPERTONIC_VENV:-${SUPERTONIC_DEFAULT_VENV}}"
  SUPERTONIC_MODELS_ROOT="${SUPERTONIC_MODELS_ROOT:-$(supertonic_config_value "${CONFIG_PATH}" models_root)}"
  if [[ -z "${SUPERTONIC_MODELS_ROOT}" ]]; then
    local legacy_root
    legacy_root="$(supertonic_config_value "${CONFIG_PATH}" app_root)"
    SUPERTONIC_MODELS_ROOT="${legacy_root:+${legacy_root}/models}"
  fi
  SUPERTONIC_MODELS_ROOT="${SUPERTONIC_MODELS_ROOT:-${SUPERTONIC_DEFAULT_MODELS_ROOT}}"
}
# Reviewed Hugging Face revisions: the upstream CPU models and voice styles, and
# the compiled MLA packages. Bump deliberately, with review, together with the
# checksums in _supertonic_checksums (a bump without them fails verification).
SUPERTONIC_UPSTREAM_HF_REPO="Supertone/supertonic-3"
SUPERTONIC_UPSTREAM_HF_REVISION="${SUPERTONIC_UPSTREAM_HF_REVISION:-724fb5abbf5502583fb520898d45929e62f02c0b}"
SUPERTONIC_SIMA_HF_REPO="florianvoss/supertonic-3-sima"
SUPERTONIC_SIMA_HF_REVISION="${SUPERTONIC_SIMA_HF_REVISION:-9229108c974ef57810abce0889ac713a59e341a1}"
SKIP_MODEL_DOWNLOAD="${SKIP_MODEL_DOWNLOAD:-0}"
CPU_TORCH_VERSION="${CPU_TORCH_VERSION:-2.8.0+cpu}"
DEPENDENCIES_ONLY=0

# ---------------------------------------------------------------------------
# Pretty output: the Neat sparkle banner + colourised status lines. Degrades to
# plain text when stdout is not a TTY, TERM is "dumb", or NO_COLOR is set.
# ---------------------------------------------------------------------------
if [[ -t 1 && -z "${NO_COLOR:-}" && "${TERM:-}" != "dumb" ]]; then
  C_RESET=$'\033[0m'; C_BOLD=$'\033[1m'; C_DIM=$'\033[2m'
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

step() { printf '\n%s\n' "${C_ACCENT}${C_BOLD}▸${C_RESET} $*"; }
info() { printf '%s\n' "${C_MUTED}·${C_RESET} $*"; }
ok()   { printf '%s\n' "${C_OK}✔${C_RESET} $*"; }
warn() { printf '%s\n' "${C_WARN}⚠${C_RESET} $*" >&2; }
errln(){ printf '%s\n' "${C_ERR}✘${C_RESET} $*" >&2; }

catalog_dir_name() {
  local repo="$1" org name
  org="${repo%%/*}"
  name="${repo##*/}"
  if [[ "${org,,}" == "simaai" ]]; then
    printf '%s\n' "${name}"
  else
    printf '%s@%s\n' "${org}" "${name}"
  fi
}

write_repo_marker() {
  printf '%s\n' "$1" > "$2/.neat-hub-repo"
}

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
  printf '   %sNEAT%s %sGenAI Studio%s  %s· setup%s\n' \
    "${C_BOLD}" "${C_RESET}" "${C_ACCENT}${C_BOLD}" "${C_RESET}" "${C_MUTED}" "${C_RESET}"
  printf '   %sInstalls the environment, models, voices and local config%s\n' "${C_MUTED}" "${C_RESET}"
  printf '\n'
}

# A slim divider formed from the palette Neat sparkle (✦), cycling the logo
# colours — used as the spacer between explicit setup sections.
spacer() {
  local palette=("${P_TEAL}" "${P_GREEN}" "${P_LIME}" "${P_BLUE}" "${P_ORANGE}")
  local line="   " i
  for ((i = 0; i < 12; i++)); do
    line+="${palette[i % 5]}✦${C_RESET} "
  done
  printf '%s\n' "${line}"
}

# A titled section break: a blank line, a sparkle divider, then a bold heading.
section() {
  printf '\n'
  spacer
  printf '   %s%s%s\n' "${C_BOLD}" "$1" "${C_RESET}"
}

usage() {
  cat <<'USAGE'
Usage:
  ./setup.sh [--dependencies-only]

Options:
  --dependencies-only             Refresh the two Python environments only;
                                  do not download models/voices or rewrite data

Environment:
  PYNEAT_PYTHON                 Python interpreter with pyneat
                                default: ~/pyneat/bin/python
  APP_VENV                      UI virtual environment path
                                default: ./.venv
  CONFIG_PATH                   Generated local config path
                                default: ./config.local.yaml
  LLIMA_MODELS_PATH             Model download directory
                                default: /media/nvme/llima/models
  CHAT_MODEL_REPO               Optional HF repo to also download + preload a
                                chat/VLM model. default: empty (none)
  CATALOG_MODEL_REPOS           Extra compatible HF repos to seed the catalog
                                (space-separated). default: empty
  MAX_RESIDENT_CHAT_MODELS      Chat/VLM models kept resident in RAM at once
                                default: 1
  CREATE_ALIAS                  Create the neat-ai shell alias, 1 or 0.
                                If unset, interactive setup prompts; otherwise 0.
  WEB_HOST, WEB_PORT, WEB_HTTPS, WEB_HEADLESS, WEB_CORS_ORIGINS
                                app.web settings written to the config. Default:
                                the values in an existing config (kept on re-run),
                                else 0.0.0.0, 5000, true, false and none.
  CREATE_DESKTOP_ICON           Create the desktop icon / menu entry that starts
                                the Studio and opens it in the browser, 1 or 0.
                                If unset, interactive setup prompts (only where a
                                desktop exists); otherwise 0. An installed icon
                                is kept up to date on later runs.
  ALLOW_HUB_DOWNLOAD            Allow in-UI Hugging Face downloads, true or false
                                default: true
  INSTALL_TTS_VOICES            Download Piper TTS voices, 1 or 0
                                default: 1
  TTS_LANGUAGES                 Comma/space-separated language codes to install.
                                If unset in a terminal, setup prompts; otherwise:
                                en,de,es,fr,it,ja,pt,vi,zh
  TTS_OPTIONAL_VOICES           Optional catalogued voice ids to also install,
                                e.g. mera,en_US-ljspeech-medium
  INSTALL_SUPERTONIC            Install Supertonic 3, the MLA-accelerated TTS
                                engine (builds .venv-supertonic and downloads
                                its models from Hugging Face), 1 or 0. default: 1
  SUPERTONIC_MODELS_ROOT        Where the Supertonic model files are stored
                                default: /media/nvme/supertonic-tts/models
  SUPERTONIC_VENV               Runtime venv path. default: ./.venv-supertonic
  CPU_TORCH_VERSION             CPU-only PyTorch version for RAG installs
                                default: 2.8.0+cpu
  SKIP_MODEL_DOWNLOAD           Write config without downloading models, 1 or 0
                                default: 0
USAGE
}

case "${1:-}" in
  "") ;;
  -h|--help) usage; exit 0 ;;
  --dependencies-only) DEPENDENCIES_ONLY=1 ;;
  *) errln "Unknown argument: $1"; usage; exit 2 ;;
esac
if [[ $# -gt 1 ]]; then
  errln "Too many arguments."
  usage
  exit 2
fi

banner

section "Environment"
if ! command -v python3 >/dev/null 2>&1; then
  errln "python3 is required."
  exit 1
fi
info "python3: ${C_DIM}$(command -v python3)${C_RESET}"

if [[ ! -x "${PYNEAT_PYTHON}" ]]; then
  errln "Neat Python environment not found: ${PYNEAT_PYTHON}"
  info "Install Neat first, or set PYNEAT_PYTHON=/path/to/python-with-pyneat."
  exit 1
fi

if ! "${PYNEAT_PYTHON}" - <<'PY' >/dev/null 2>&1
import pyneat
PY
then
  errln "pyneat is not importable from ${PYNEAT_PYTHON}."
  info "Install Neat first, or set PYNEAT_PYTHON=/path/to/python-with-pyneat."
  exit 1
fi
ok "pyneat available (${C_DIM}${PYNEAT_PYTHON}${C_RESET})"

install_cpu_torch_if_needed() {
  case "$(uname -m)" in
    x86_64|amd64|aarch64|arm64)
      info "Installing CPU-only PyTorch for RAG embeddings…"
      "${APP_VENV}/bin/python" -m pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch==${CPU_TORCH_VERSION}"
      ;;
  esac
}

section "Python environment"
step "Creating UI virtual environment: ${C_DIM}${APP_VENV}${C_RESET}"
python3 -m venv "${APP_VENV}"
"${APP_VENV}/bin/python" -m pip install --upgrade pip
install_cpu_torch_if_needed
info "Installing UI + RAG requirements (incl. piper-plus)…"
"${APP_VENV}/bin/python" -m pip install \
  -r "${EXAMPLE_DIR}/src/python/requirements.txt" \
  -r "${EXAMPLE_DIR}/src/python/requirements-rag.txt"
ok "UI virtual environment ready."
resolve_supertonic_paths

# Supertonic 3: hybrid TTS whose vector field and vocoder run on the MLA through
# PyNeat. The runtime package is vendored in src/python/ui/supertonic_sima/; its
# dependencies (pyneat, onnxruntime, numpy 1.26) cannot share the UI venv, so an
# isolated venv is built here and the model files are downloaded from Hugging
# Face at pinned revisions and verified by checksum. The UI talks to it through
# supertonic_worker.py. Optional: a failure here only leaves the CPU engines in
# place.
# Every file the worker reads, relative to the models root, with the SHA-256 of
# the reviewed revisions above. Verified on every setup run (not only after a
# download), so a stale, truncated or replaced file is re-fetched.
_supertonic_checksums() {
  cat <<'SUMS'
42078d3aef1cd43ab43021f3c54f47d2d75ceb4e75f627f118890128b06a0d09  supertonic-3/onnx/tts.json
9bf7346e43883a81f8645c81224f786d43c5b57f3641f6e7671a7d6c493cb24f  supertonic-3/onnx/unicode_indexer.json
c3eb91414d5ff8a7a239b7fe9e34e7e2bf8a8140d8375ffb14718b1c639325db  supertonic-3/onnx/duration_predictor.onnx
c7befd5ea8c3119769e8a6c1486c4edc6a3bc8365c67621c881bbb774b9902ff  supertonic-3/onnx/text_encoder.onnx
bbdec6ee00231c2c742ad05483df5334cab3b52fda3ba38e6a07059c4563dbc2  supertonic-3/voice_styles/F1.json
7c722c6a72707b1a77f035d67f0d1351ba187738e06f7683e8c72b1df3477fc6  supertonic-3/voice_styles/F2.json
12f6ef2573baa2defa1128069cb59f203e3ab67c92af77b42df8a0e3a2f7c6ab  supertonic-3/voice_styles/F3.json
c2fa764c1225a76dfc3e2c73e8aa4f70d9ee48793860eb34c295fff01c2e032b  supertonic-3/voice_styles/F4.json
45966e73316415626cf41a7d1c6f3b4c70dbc1ba2bee5c1978ef0ce33244fc8d  supertonic-3/voice_styles/F5.json
e35604687f5d23694b8e91593a93eec0e4eca6c0b02bb8ed69139ab2ea6b0a5b  supertonic-3/voice_styles/M1.json
b76cbf62bac707c710cf0ae5aba5e31eea1a6339a9734bfae33ab98499534a50  supertonic-3/voice_styles/M2.json
ea1ac35ccb91b0d7ecad533a2fbd0eec10c91513d8951e3b25fbba99954e159b  supertonic-3/voice_styles/M3.json
ca8eefad4fcd989c9379032ff3e50738adc547eeb5e221b82593a6d7b3bac303  supertonic-3/voice_styles/M4.json
dd22b92740314321f8ae11c5e87f8dd60d060f15dd3a632b5adf77f471f77af2  supertonic-3/voice_styles/M5.json
2f6b8c918e0c402453e48bd2686dbea429e6ce1dd98151c940d88229980e8dd2  supertonic-3-sima/supertonic_vector_field_sima_mpk.tar.gz
90b2d6a089c8527826dd1d0cb5b557316ac703045f422d6e1332deeabd84e0cb  supertonic-3-sima/supertonic_vocoder_sima_bf16_mpk.tar.gz
4e9a4d85592f720c9a497cf94164290ccdddd0c5d104577fb810c936d2abf9f9  supertonic-3-sima/supertonic_runtime_data.npz
31285c6d2ed7c1bca84a78c6b3477edc0ac9f0442a8062adc2627574555a4e3e  supertonic-3-sima/artifact_manifest.json
1659e891f4da0cc80c6dd7a5fbb3ca87f3a12ad7f0f2bce0bfb0963433709628  supertonic-3-sima/vocoder_bf16_manifest.json
SUMS
}

# Files that are missing or do not match their pinned checksum (one per line,
# relative to the models root). Empty output means the install is intact.
_supertonic_bad_files() {
  local sum rel actual
  while read -r sum rel; do
    [[ -n "${rel}" ]] || continue
    if [[ ! -f "${SUPERTONIC_MODELS_ROOT}/${rel}" ]]; then
      printf '%s\n' "${rel}"; continue
    fi
    actual="$(sha256sum "${SUPERTONIC_MODELS_ROOT}/${rel}" 2>/dev/null | cut -d' ' -f1)"
    [[ "${actual}" == "${sum}" ]] || printf '%s\n' "${rel}"
  done < <(_supertonic_checksums)
}

_supertonic_runtime_ok() {
  local py="${SUPERTONIC_VENV}/bin/python"
  [[ -x "${py}" ]] || return 1
  PYTHONPATH="${EXAMPLE_DIR}/src/python/ui" "${py}" - <<'PY' >/dev/null 2>&1
import numpy, onnxruntime, pyneat
import supertonic_sima
PY
}

# A venv whose build failed part-way is removed so the Studio sees the engine as
# cleanly unavailable instead of spawning a worker that cannot import.
_supertonic_venv_failed() {
  warn "$1"
  rm -rf "${SUPERTONIC_VENV}"
  return 1
}

# The Supertonic venv needs the SAME PyNeat as the system runtime: its extension
# links a versioned libsima_neat.so, so a wheel from a different build fails to
# import and the engine silently disappears. sima-cli serves one channel, which
# is not necessarily the runtime that is installed, so prefer a local wheel
# matching the installed PyNeat and fall back to sima-cli.
_installed_pyneat_version() {
  "${PYNEAT_PYTHON}" - <<'PYV' 2>/dev/null
try:
    from importlib.metadata import version
    print(version("pyneat"))
except Exception:
    pass
PYV
}

_local_pyneat_wheel() {
  local want="$1" d
  [[ -n "${want}" ]] || return 1
  for d in ${PYNEAT_WHEEL_DIRS}; do
    [[ -d "${d}" ]] || continue
    local hit
    hit="$(find "${d}" -maxdepth 2 -type f -name "pyneat-${want}-*.whl" -print -quit 2>/dev/null)"
    [[ -n "${hit}" ]] && { printf '%s\n' "${hit}"; return 0; }
  done
  return 1
}

_supertonic_venv() {
  # The PyNeat wheel comes from sima-cli, which the DevKit exposes on PATH only
  # for login shells.
  if ! command -v sima-cli >/dev/null 2>&1 && [[ -x /data/sima-cli/.venv/bin/sima-cli ]]; then
    export PATH="${PATH}:/data/sima-cli/.venv/bin"
  fi
  local want_version wheel_override="${PYNEAT_WHEEL}"
  want_version="$(_installed_pyneat_version)"
  if [[ -z "${wheel_override}" && -n "${want_version}" ]]; then
    wheel_override="$(_local_pyneat_wheel "${want_version}" || true)"
    [[ -n "${wheel_override}" ]] && info "Using the PyNeat wheel matching the installed runtime (${want_version})."
  fi
  if [[ -z "${wheel_override}" ]] && ! command -v sima-cli >/dev/null 2>&1; then
    warn "sima-cli is required to fetch the PyNeat wheel for Supertonic; Supertonic TTS skipped."
    return 1
  fi
  step "Creating isolated Supertonic venv: ${C_DIM}${SUPERTONIC_VENV}${C_RESET}"
  python3 -m venv --clear "${SUPERTONIC_VENV}" \
    || _supertonic_venv_failed "Could not create ${SUPERTONIC_VENV}; Supertonic TTS skipped." || return 1
  "${SUPERTONIC_VENV}/bin/python" -m pip install --upgrade pip >/dev/null \
    || _supertonic_venv_failed "pip upgrade failed in the Supertonic venv; Supertonic TTS skipped." || return 1
  info "Installing the Supertonic runtime requirements (isolated)…"
  "${SUPERTONIC_VENV}/bin/python" -m pip install -r "${EXAMPLE_DIR}/src/python/requirements-supertonic.txt" \
    || _supertonic_venv_failed "Supertonic requirements failed to install; Supertonic TTS skipped." || return 1
  local wheel_dir status=0
  if [[ -n "${wheel_override}" ]]; then
    # --no-deps: the wheel must not move the pinned numpy/onnxruntime.
    "${SUPERTONIC_VENV}/bin/python" -m pip install --no-deps "${wheel_override}" \
      || _supertonic_venv_failed "The PyNeat wheel ${wheel_override} did not install; Supertonic TTS skipped." \
      || return 1
    _supertonic_runtime_ok \
      || _supertonic_venv_failed "The Supertonic runtime does not import with ${wheel_override##*/}; Supertonic TTS skipped." \
      || return 1
    ok "Supertonic venv ready."
    return 0
  fi
  wheel_dir="$(mktemp -d)"
  info "Fetching the PyNeat wheel with sima-cli…"
  local -a wheels=()
  if sima-cli neat install core -t pyneat --install-dir "${wheel_dir}"; then
    mapfile -t wheels < <(find "${wheel_dir}" -maxdepth 1 -type f -name 'pyneat-*.whl' -print | sort)
    if [[ "${#wheels[@]}" -ne 1 ]]; then
      status=1; warn "Expected one PyNeat wheel from sima-cli, found ${#wheels[@]}."
    # --no-deps: the wheel must not move the pinned numpy/onnxruntime.
    elif ! "${SUPERTONIC_VENV}/bin/python" -m pip install --no-deps "${wheels[0]}"; then
      status=1; warn "The PyNeat wheel did not install into the Supertonic venv."
    fi
  else
    status=1; warn "sima-cli could not fetch the PyNeat wheel."
  fi
  rm -rf "${wheel_dir}"
  [[ "${status}" -eq 0 ]] || _supertonic_venv_failed "Supertonic TTS skipped." || return 1
  if ! _supertonic_runtime_ok; then
    local got
    got="$("${SUPERTONIC_VENV}/bin/python" -c 'from importlib.metadata import version; print(version("pyneat"))' 2>/dev/null || true)"
    warn "sima-cli's PyNeat (${got:-unknown}) does not match the installed runtime (${want_version:-unknown})."
    warn "Point PYNEAT_WHEEL at the wheel for the installed runtime, or search PYNEAT_WHEEL_DIRS (${PYNEAT_WHEEL_DIRS})."
    _supertonic_venv_failed "The Supertonic runtime does not import in the new venv; Supertonic TTS skipped." || return 1
  fi
  ok "Supertonic venv ready."
}

_supertonic_models() {
  local hf="${SUPERTONIC_VENV}/bin/hf"
  if [[ ! -x "${hf}" ]]; then
    warn "Hugging Face CLI missing from ${SUPERTONIC_VENV}; Supertonic TTS skipped."
    return 1
  fi
  mkdir -p "${SUPERTONIC_MODELS_ROOT}" || return 1
  export HF_HOME="${SUPERTONIC_MODELS_ROOT}/.hf-cache"   # keep the cache off the root filesystem
  step "Downloading the compiled MLA packages (${SUPERTONIC_SIMA_HF_REPO}@${SUPERTONIC_SIMA_HF_REVISION:0:12})…"
  "${hf}" download "${SUPERTONIC_SIMA_HF_REPO}" \
    supertonic_vector_field_sima_mpk.tar.gz \
    supertonic_vocoder_sima_bf16_mpk.tar.gz \
    supertonic_runtime_data.npz \
    artifact_manifest.json \
    vocoder_bf16_manifest.json \
    --revision "${SUPERTONIC_SIMA_HF_REVISION}" \
    --local-dir "${SUPERTONIC_MODELS_ROOT}/supertonic-3-sima" || return 1
  step "Downloading the upstream CPU models and voice styles (${SUPERTONIC_UPSTREAM_HF_REPO}@${SUPERTONIC_UPSTREAM_HF_REVISION:0:12})…"
  "${hf}" download "${SUPERTONIC_UPSTREAM_HF_REPO}" \
    onnx/duration_predictor.onnx onnx/text_encoder.onnx onnx/tts.json onnx/unicode_indexer.json \
    voice_styles/F1.json voice_styles/F2.json voice_styles/F3.json voice_styles/F4.json voice_styles/F5.json \
    voice_styles/M1.json voice_styles/M2.json voice_styles/M3.json voice_styles/M4.json voice_styles/M5.json \
    --revision "${SUPERTONIC_UPSTREAM_HF_REVISION}" \
    --local-dir "${SUPERTONIC_MODELS_ROOT}/supertonic-3" || return 1
  step "Verifying all Supertonic model files against their pinned checksums…"
  local -a bad=()
  mapfile -t bad < <(_supertonic_bad_files)
  if [[ "${#bad[@]}" -gt 0 ]]; then
    warn "Checksum mismatch after download: ${bad[*]}"
    return 1
  fi
  ok "All $(_supertonic_checksums | wc -l) Supertonic files verified."
}

# Re-apply the pinned requirements to an existing, importable venv (a
# dependency refresh must pick up changed pins; the PyNeat wheel is left alone
# unless the runtime stops importing, in which case the venv is rebuilt).
_supertonic_refresh_requirements() {
  info "Refreshing the Supertonic runtime requirements in ${C_DIM}${SUPERTONIC_VENV}${C_RESET}…"
  if ! "${SUPERTONIC_VENV}/bin/python" -m pip install -r "${EXAMPLE_DIR}/src/python/requirements-supertonic.txt"; then
    warn "Supertonic requirements refresh failed; rebuilding the venv."
    return 1
  fi
  _supertonic_runtime_ok && return 0
  warn "Supertonic runtime no longer imports after the refresh; rebuilding the venv."
  return 1
}

# install_supertonic [refresh]: builds/repairs the venv and verifies every model
# file against its pinned checksum, re-fetching anything missing or different.
# `refresh` (dependency-only mode) also re-applies the pins to a usable venv.
# Always returns 0: Supertonic is optional and the CPU engines remain.
install_supertonic() {
  local refresh="${1:-}"
  section "Supertonic 3 (MLA text-to-speech)"
  if _supertonic_runtime_ok; then
    if [[ "${refresh}" == "refresh" ]] && ! _supertonic_refresh_requirements; then
      _supertonic_venv || return 0
    fi
  else
    _supertonic_venv || return 0
  fi
  info "Verifying the Supertonic model files in ${C_DIM}${SUPERTONIC_MODELS_ROOT}${C_RESET}…"
  local -a bad=()
  mapfile -t bad < <(_supertonic_bad_files)
  if [[ "${#bad[@]}" -gt 0 ]]; then
    info "Supertonic model files to fetch (missing or not the pinned revision): ${bad[*]}"
    local rel
    for rel in "${bad[@]}"; do rm -f "${SUPERTONIC_MODELS_ROOT}/${rel}"; done   # never keep a bad copy
    if ! _supertonic_models; then
      warn "Supertonic model download or verification failed; the CPU TTS engines remain available."
      return 0
    fi
  fi
  if ! _supertonic_runtime_ok; then
    warn "The Supertonic runtime does not import; the CPU TTS engines remain available."
    return 0
  fi
  ok "Supertonic 3 ${refresh:+refreshed and }ready (venv ${C_DIM}${SUPERTONIC_VENV}${C_RESET}, models ${C_DIM}${SUPERTONIC_MODELS_ROOT}${C_RESET}, all files verified)."
  # Pre-vendoring installs (default location only) used an external checkout
  # and a venv beside the models; neither is read any more.
  if [[ "${SUPERTONIC_MODELS_ROOT}" == "${SUPERTONIC_DEFAULT_MODELS_ROOT}" ]]; then
    local legacy
    for legacy in /media/nvme/supertonic-tts/.venv /media/nvme/repos/supertonic-sima; do
      [[ -e "${legacy}" ]] && info "${legacy} is no longer used by the Studio; remove it with rm -rf when convenient."
    done
  fi
  return 0
}

# piper-tts ships the same top-level `piper` package as piper-plus, so it gets
# its own venv; the UI reaches it through a subprocess worker.
step "Creating isolated piper-tts venv: ${C_DIM}${PIPERTTS_VENV}${C_RESET}"
python3 -m venv "${PIPERTTS_VENV}"
"${PIPERTTS_VENV}/bin/python" -m pip install --upgrade pip >/dev/null
info "Installing piper-tts (isolated)…"
"${PIPERTTS_VENV}/bin/python" -m pip install \
  -r "${EXAMPLE_DIR}/src/python/requirements-pipertts.txt"
ok "piper-tts venv ready."

if [[ "${DEPENDENCIES_ONLY}" == "1" ]]; then
  # The Supertonic runtime is a dependency too (its own venv under the example
  # directory); `UPDATE_DEPS=1 ./run.sh update` must refresh it or an upgraded
  # checkout loses the default TTS engine. Its model files are only fetched when
  # missing, which a dependency refresh should not normally trigger.
  if [[ "${INSTALL_SUPERTONIC}" == "1" ]]; then
    install_supertonic refresh
  else
    info "Supertonic 3 install skipped (INSTALL_SUPERTONIC=0)."
  fi
  ok "Dependency refresh complete; models, voices, config and RAG data were unchanged."
  exit 0
fi

mkdir -p "${MODELS_DIR}"
if [[ -n "${CHAT_MODEL_REPO}" ]]; then
  CHAT_MODEL_DIR="${MODELS_DIR}/$(catalog_dir_name "${CHAT_MODEL_REPO}")"
else
  CHAT_MODEL_DIR=""
fi
if [[ -n "${ASR_MODEL_REPO}" ]]; then
  ASR_MODEL_NAME="$(catalog_dir_name "${ASR_MODEL_REPO}")"
  ASR_MODEL_DIR="${MODELS_DIR}/${ASR_MODEL_NAME}"
else
  ASR_MODEL_NAME=""
  ASR_MODEL_DIR=""
fi
RAG_EMBEDDING_DIR="${MODELS_DIR}/gte-small"

section "Models"
if [[ "${SKIP_MODEL_DOWNLOAD}" != "1" ]]; then
  if [[ -n "${CHAT_MODEL_REPO}" ]]; then
    step "Downloading chat/VLM model: ${CHAT_MODEL_REPO}"
    "${APP_VENV}/bin/hf" download "${CHAT_MODEL_REPO}" --local-dir "${CHAT_MODEL_DIR}"
    write_repo_marker "${CHAT_MODEL_REPO}" "${CHAT_MODEL_DIR}"
  else
    info "No default chat/VLM model — download one from the UI (or set CHAT_MODEL_REPO)."
  fi

  if [[ -n "${ASR_MODEL_REPO}" ]]; then
    step "Downloading ASR model: ${ASR_MODEL_REPO}"
    "${APP_VENV}/bin/hf" download "${ASR_MODEL_REPO}" --local-dir "${ASR_MODEL_DIR}"
    write_repo_marker "${ASR_MODEL_REPO}" "${ASR_MODEL_DIR}"
  else
    info "No default ASR model — download one from the UI (or set ASR_MODEL_REPO)."
  fi

  for repo in ${ASR_CATALOG_MODEL_REPOS}; do
    name="$(catalog_dir_name "${repo}")"
    step "Downloading ASR catalog model: ${repo}"
    "${APP_VENV}/bin/hf" download "${repo}" --local-dir "${MODELS_DIR}/${name}"
    write_repo_marker "${repo}" "${MODELS_DIR}/${name}"
  done

  step "Downloading RAG embedding model: ${RAG_EMBEDDING_REPO}"
  "${APP_VENV}/bin/hf" download "${RAG_EMBEDDING_REPO}" --local-dir "${RAG_EMBEDDING_DIR}"

  for repo in ${CATALOG_MODEL_REPOS}; do
    name="$(catalog_dir_name "${repo}")"
    step "Downloading catalog model: ${repo}"
    "${APP_VENV}/bin/hf" download "${repo}" --local-dir "${MODELS_DIR}/${name}"
    write_repo_marker "${repo}" "${MODELS_DIR}/${name}"
  done
  ok "Model downloads complete."
else
  info "Skipping model downloads because SKIP_MODEL_DOWNLOAD=1."
fi

if [[ -n "${CHAT_MODEL_REPO}" ]]; then
  CHAT_YAML=$(cat <<CHAT
    chat:               # Loaded at startup (a subset of the catalog).
      - name: ${CHAT_MODEL_NAME}
        path: ${CHAT_MODEL_DIR}
CHAT
)
else
  CHAT_YAML="    chat: []            # No model preloaded; load on demand from the UI."
fi

# Ask the runtime whether it can actually load the model, before naming it as the
# startup ASR. Encoder layout requirements have changed between runtime builds in
# both directions, so this asks rather than infers: an explicit refusal means the
# config should not point here, while an inconclusive result (busy accelerator,
# no runtime) leaves the configuration alone.
asr_model_loads() {
  local dir="$1" out rc
  [[ -d "${dir}" ]] || return 2
  [[ -x "${PYNEAT_PYTHON}" ]] || return 2
  out="$("${PYNEAT_PYTHON}" "${EXAMPLE_DIR}/scripts/probe_asr.py" "${dir}" 2>&1)"
  rc=$?
  case "${rc}" in
    0) return 0 ;;
    3) printf '%s\n' "${out}" ; return 1 ;;   # the runtime refused it
    *) printf '%s\n' "${out}" ; return 2 ;;   # could not tell
  esac
}

ASR_STARTUP_OK=1
if [[ -n "${ASR_MODEL_REPO}" && "${SKIP_MODEL_DOWNLOAD}" != "1" ]]; then
  step "Checking that ${ASR_MODEL_NAME} loads on this runtime…"
  # `out=$(f)` under `set -e` aborts the script when f returns nonzero — which
  # is precisely the refused/inconclusive outcomes this case exists to handle.
  # An if/else keeps errexit from firing and still captures the status.
  if probe_out="$(asr_model_loads "${ASR_MODEL_DIR}")"; then
    probe_rc=0
  else
    probe_rc=$?
  fi
  case "${probe_rc}" in
    0) ok "Speech-to-text model loads." ;;
    1) ASR_STARTUP_OK=0
       warn "This runtime refuses ${ASR_MODEL_NAME}:"
       printf '%s\n' "${probe_out}" | sed 's/^/    /' >&2
       warn "Leaving it in the catalog but NOT configuring it at startup."
       warn "Install one this runtime accepts, e.g. ASR_MODEL_REPO=simaai/whisper-small-a16w8 ./setup.sh,"
       warn "or pick a model in Settings -> Models once the Studio is running." ;;
    *) warn "Could not verify ${ASR_MODEL_NAME} (accelerator busy or no runtime); configuring it anyway." ;;
  esac
fi

if [[ -n "${ASR_MODEL_REPO}" && "${ASR_STARTUP_OK}" == "1" ]]; then
  ASR_YAML=$(cat <<ASR
    asr:                # Active at startup; switch at runtime in Settings -> Models.
      name: ${ASR_MODEL_NAME}
      path: ${ASR_MODEL_DIR}
ASR
)
elif [[ -n "${ASR_MODEL_REPO}" ]]; then
  ASR_YAML="    # asr: omitted — ${ASR_MODEL_NAME} is in the catalog but this runtime refuses it."
else
  ASR_YAML="    # asr: omitted — no speech-to-text model is preloaded."
fi

# Render "simaai TDoSiMa" -> "simaai, TDoSiMa" for the YAML flow list.
HUB_ORGS_YAML="$(echo "${HUB_ORGS}" | tr -s ' ' | sed 's/^ //; s/ $//; s/ /, /g')"
section "Configuration"
step "Writing local config: ${C_DIM}${CONFIG_PATH}${C_RESET}"
mkdir -p "$(dirname "${CONFIG_PATH}")"
# Web settings a customer may have changed (including backend-only mode and its
# CORS allowlist) survive a re-run: environment > the existing config (read by
# the app's own loader) > defaults. The previous file is kept as .bak so any
# other hand edits can be carried over.
_web_setting() {   # _web_setting <key> <default>
  local v=""
  [[ -f "${CONFIG_PATH}" ]] && v="$(web_config_scalar "${CONFIG_PATH}" "$1")"
  printf '%s' "${v:-$2}"
}
WEB_HOST="${WEB_HOST:-$(_web_setting host 0.0.0.0)}"
WEB_PORT="${WEB_PORT:-$(_web_setting port 5000)}"
WEB_HTTPS="${WEB_HTTPS:-$(_web_setting https true)}"
WEB_HEADLESS="${WEB_HEADLESS:-$(_web_setting headless false)}"
WEB_CORS_ORIGINS="${WEB_CORS_ORIGINS-$(_web_setting cors_origins '')}"
config_true "${WEB_HTTPS}" && WEB_HTTPS=true || WEB_HTTPS=false
config_true "${WEB_HEADLESS}" && WEB_HEADLESS=true || WEB_HEADLESS=false
if [[ -f "${CONFIG_PATH}" ]]; then
  cp -p "${CONFIG_PATH}" "${CONFIG_PATH}.bak"
  info "Previous config kept as ${C_DIM}${CONFIG_PATH}.bak${C_RESET} (web settings carried over: host ${WEB_HOST}, port ${WEB_PORT}, https ${WEB_HTTPS}, headless ${WEB_HEADLESS}${WEB_CORS_ORIGINS:+, cors_origins ${WEB_CORS_ORIGINS}})."
fi
cat > "${CONFIG_PATH}" <<YAML
server:
  openai:
    host: 0.0.0.0
    port: 9998

  control:
    host: 127.0.0.1
    port: 9997

  models:
    # Every compatible model directory under catalog_dir can be loaded on the fly.
    catalog_dir: ${MODELS_DIR}
    max_resident_chat_models: ${MAX_RESIDENT_CHAT_MODELS}
${CHAT_YAML}
${ASR_YAML}

  hub:
    allow_download: ${ALLOW_HUB_DOWNLOAD}
    orgs: [${HUB_ORGS_YAML}]

app:
  openai:
    client_host: 127.0.0.1
    port: 9998

  control:
    client_host: 127.0.0.1
    port: 9997

  request:
    max_tokens: 512
    system_prompt: >-
      Answer clearly and concisely. Use Markdown formatting when it helps.
      Answer the question in the language it was asked in.

  web:
    host: "${WEB_HOST}"
    port: ${WEB_PORT}
    https: ${WEB_HTTPS}
    headless: ${WEB_HEADLESS}   # true: API endpoints only, no web UI (same as ./run.sh --backend-only)
    cors_origins: "${WEB_CORS_ORIGINS}"  # backend-only: origins (comma list or *) whose browser pages may call the API

  ui:
    font_family: Inter
    font_size: 15

  tts:
    supertonic:
      # Supertonic 3 (MLA TTS) model files and the runtime venv setup.sh built
      # (run.sh and the UI read both; SUPERTONIC_MODELS_ROOT / SUPERTONIC_VENV
      # in the environment override).
      models_root: "${SUPERTONIC_MODELS_ROOT}"
      venv: "$([[ "${SUPERTONIC_VENV}" == "${SUPERTONIC_DEFAULT_VENV}" ]] || printf '%s' "${SUPERTONIC_VENV}")"

  rag:
    enabled: true
    embedding_model_dir: ${RAG_EMBEDDING_DIR}

YAML
ok "Config written."

if [[ "${SKIP_MODEL_DOWNLOAD}" != "1" ]]; then
  section "RAG database"
  step "Creating default RAG database…"
  (
    cd "${EXAMPLE_DIR}/src/python"
    "${APP_VENV}/bin/python" rag/create_db.py \
      --input "${EXAMPLE_DIR}/src/common/rag/neat.md" \
      --output ui/milvus.db \
      --embedding-model "${RAG_EMBEDDING_DIR}"
  )
  ok "RAG database ready."
fi

if [[ "${INSTALL_TTS_VOICES}" == "1" ]]; then
  if [[ -z "${TTS_LANGUAGES}" ]]; then
    if [[ -t 0 ]]; then
      printf '\nAvailable server TTS languages: en de es fr it ja no pt vi zh\n'
      printf 'Korean remains browser/text-only (no catalogued server voice).\n'
      read -r -p "Languages to install [${DEFAULT_TTS_LANGUAGES}]: " TTS_LANGUAGES
      TTS_LANGUAGES="${TTS_LANGUAGES:-${DEFAULT_TTS_LANGUAGES}}"
    else
      TTS_LANGUAGES="${DEFAULT_TTS_LANGUAGES}"
    fi
  fi
  section "Text-to-speech"
  step "Installing catalogued voices for: ${TTS_LANGUAGES}"
  (
    cd "${EXAMPLE_DIR}/src/python"
    TTS_LANGUAGES="${TTS_LANGUAGES}" \
      TTS_OPTIONAL_VOICES="${TTS_OPTIONAL_VOICES}" \
      PYTHON="${PIPERTTS_VENV}/bin/python" \
      bash voice_install.sh
  )
  # piper-plus English G2P (g2p-en) pulls NLTK tagger data at runtime; fetch it
  # now so English TTS works offline on the board.
  info "Pre-fetching NLTK data for piper-plus English G2P…"
  "${APP_VENV}/bin/python" - <<'PY' 2>/dev/null || warn "NLTK data prefetch skipped (English piper-plus may need it on first online run)"
try:
    import nltk
    for pkg in ("averaged_perceptron_tagger_eng", "averaged_perceptron_tagger", "cmudict"):
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass
except Exception:
    pass
PY
  ok "Voices installed."
fi

if [[ "${INSTALL_SUPERTONIC}" == "1" ]]; then
  install_supertonic
else
  info "Supertonic 3 install skipped (INSTALL_SUPERTONIC=0)."
fi

# Offer a convenient `neat-ai` shell alias for ./run.sh. Noninteractive setup
# does not modify the shell profile unless CREATE_ALIAS=1 is explicitly set.
maybe_create_alias() {
  local run_path="${EXAMPLE_DIR}/run.sh"
  # Target is eLxr, where interactive board sessions (e.g. over SSH) are login
  # shells that read ~/.bash_profile — so put the alias there for bash.
  local rc
  case "$(basename "${SHELL:-bash}")" in
    zsh) rc="${HOME}/.zshrc" ;;
    *)   rc="${HOME}/.bash_profile" ;;
  esac

  chmod +x "${run_path}" 2>/dev/null || true
  if [[ -f "${rc}" ]] && grep -q "alias neat-ai=" "${rc}"; then
    ok "'neat-ai' alias already set in ${rc}."
    return
  fi

  local create_alias="${CREATE_ALIAS:-}"
  if [[ -z "${create_alias}" ]]; then
    if [[ -t 0 ]]; then
      read -r -p "Create a 'neat-ai' shell alias in ${rc}? [y/N] " create_alias
      case "${create_alias}" in
        y|Y|yes|YES) create_alias=1 ;;
        *) create_alias=0 ;;
      esac
    else
      create_alias=0
    fi
  fi
  if [[ "${create_alias}" != "1" ]]; then
    info "Shell alias not created (set CREATE_ALIAS=1 to enable it)."
    return
  fi

  {
    printf '\n# Neat GenAI Studio — added by setup.sh\n'
    printf "alias neat-ai='%s'\n" "${run_path}"
  } >> "${rc}"
  ok "Added a 'neat-ai' alias to ${rc}."
  info "Start the studio with ${C_BOLD}neat-ai${C_RESET} (alias for ./run.sh) — run ${C_BOLD}source ${rc}${C_RESET} or open a new shell first."
  info "Alias: ${C_DIM}neat-ai='${run_path}'${C_RESET}"
}

# Offer a desktop launcher (double-click icon + application menu entry) for a
# user at the board with a display, keyboard and mouse. It runs
# src/common/desktop-launch.sh in a terminal: the Studio starts (or is found
# running) and the browser opens on the UI. Noninteractive setup only creates
# it with CREATE_DESKTOP_ICON=1.
desktop_entry() {
  local launch="${EXAMPLE_DIR}/src/common/desktop-launch.sh"
  local exec_line
  if command -v x-terminal-emulator >/dev/null 2>&1; then
    exec_line="Exec=x-terminal-emulator -T \"Neat GenAI Studio\" -e \"${launch}\""
    printf '%s\n' "[Desktop Entry]" "Version=1.0" "Type=Application" "Name=Neat GenAI Studio" \
      "Comment=Run LLMs, VLMs, speech-to-text and text-to-speech on the Modalix MLA" \
      "${exec_line}" "Path=${EXAMPLE_DIR}" "Icon=${EXAMPLE_DIR}/src/python/ui/static/icons/neat-logo.png" \
      "Terminal=false" "Categories=Development;" "Keywords=LLM;GenAI;SiMa;Modalix;Neat;" \
      "StartupNotify=false"
  else
    printf '%s\n' "[Desktop Entry]" "Version=1.0" "Type=Application" "Name=Neat GenAI Studio" \
      "Comment=Run LLMs, VLMs, speech-to-text and text-to-speech on the Modalix MLA" \
      "Exec=\"${launch}\"" "Path=${EXAMPLE_DIR}" "Icon=${EXAMPLE_DIR}/src/python/ui/static/icons/neat-logo.png" \
      "Terminal=true" "Categories=Development;" "Keywords=LLM;GenAI;SiMa;Modalix;Neat;" \
      "StartupNotify=false"
  fi
}

# XFCE (and other file managers) ask before running a launcher that is not
# marked trusted; mark it through the desktop session's bus when one exists
# (a user logged in at the board). Without a session the first double-click
# asks once ("Mark Executable").
_trust_launcher() {
  local file="$1" bus="${DBUS_SESSION_BUS_ADDRESS:-}"
  command -v gio >/dev/null 2>&1 || return 1
  if [[ -z "${bus}" && -S "/run/user/$(id -u)/bus" ]]; then
    bus="unix:path=/run/user/$(id -u)/bus"
  fi
  [[ -n "${bus}" ]] || return 1
  DBUS_SESSION_BUS_ADDRESS="${bus}" gio set -t string "${file}" \
    metadata::xfce-exe-checksum "$(sha256sum "${file}" | cut -d' ' -f1)" >/dev/null 2>&1 || return 1
  DBUS_SESSION_BUS_ADDRESS="${bus}" gio set -t string "${file}" metadata::trusted true >/dev/null 2>&1 || true
  return 0
}

maybe_create_desktop_icon() {
  local apps_dir="${XDG_DATA_HOME:-${HOME}/.local/share}/applications"
  local desktop_dir="${XDG_DESKTOP_DIR:-${HOME}/Desktop}"
  if [[ ! -d "${desktop_dir}" && ! -d /usr/share/xsessions ]]; then
    return 0                        # headless board: nothing to offer
  fi
  local want="${CREATE_DESKTOP_ICON:-}"
  local installed="${apps_dir}/neat-genai-studio.desktop"
  if [[ -z "${want}" && -f "${installed}" ]]; then
    want=1                          # keep an existing launcher up to date
  fi
  if [[ -z "${want}" ]]; then
    if [[ -t 0 ]]; then
      read -r -p "Create a desktop icon that starts the Studio and opens it in the browser? [y/N] " want
      case "${want}" in
        y|Y|yes|YES) want=1 ;;
        *) want=0 ;;
      esac
    else
      want=0
    fi
  fi
  if [[ "${want}" != "1" ]]; then
    info "Desktop icon not created (set CREATE_DESKTOP_ICON=1 to enable it)."
    return 0
  fi

  chmod +x "${EXAMPLE_DIR}/src/common/desktop-launch.sh" "${EXAMPLE_DIR}/run.sh" 2>/dev/null || true
  local entry target changed=0 trusted=1
  entry="$(desktop_entry)"
  mkdir -p "${apps_dir}"
  local -a targets=("${installed}")
  [[ -d "${desktop_dir}" ]] && targets+=("${desktop_dir}/neat-genai-studio.desktop")
  for target in "${targets[@]}"; do
    if [[ ! -f "${target}" ]] || [[ "$(cat "${target}")" != "${entry}" ]]; then
      printf '%s\n' "${entry}" > "${target}"
      changed=1
    fi
    chmod 755 "${target}"
    _trust_launcher "${target}" || trusted=0
  done
  command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "${apps_dir}" >/dev/null 2>&1 || true
  if [[ "${changed}" == "1" ]]; then
    ok "Desktop icon installed: ${C_DIM}${targets[*]}${C_RESET}"
  else
    ok "Desktop icon already installed."
  fi
  info "Double-click ${C_BOLD}Neat GenAI Studio${C_RESET} on the desktop (or in the applications menu): a terminal starts the Studio and the browser opens on it. Closing that terminal stops the Studio."
  if [[ "${trusted}" != "1" ]]; then
    info "On the first double-click the desktop may ask to trust the launcher; choose ${C_BOLD}Mark Executable${C_RESET} (or Launch Anyway)."
  fi
}

section "Done"
ok "Install complete."
maybe_create_alias
maybe_create_desktop_icon
info "Config: ${C_DIM}${CONFIG_PATH}${C_RESET}"
info "Start the studio with ${C_BOLD}./run.sh${C_RESET}"
printf '\n'
