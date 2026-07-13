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
ASR_MODEL_REPO="simaai/whisper-small-a16w8"
RAG_EMBEDDING_REPO="thenlper/gte-small"
CHAT_MODEL_NAME="${CHAT_MODEL_NAME:-${CHAT_MODEL_REPO##*/}}"
# Only one chat/VLM model is resident at a time — loading a new one clears all
# other chat/VLM models (ASR is always kept). Kept configurable for advanced use.
MAX_RESIDENT_CHAT_MODELS="${MAX_RESIDENT_CHAT_MODELS:-1}"
ALLOW_HUB_DOWNLOAD="${ALLOW_HUB_DOWNLOAD:-true}"
# Hugging Face accounts searched for compatible models (space-separated):
# simaai (official precompiled) + TDoSiMa (community).
HUB_ORGS="${HUB_ORGS:-simaai TDoSiMa}"
INSTALL_TTS_VOICES="${INSTALL_TTS_VOICES:-1}"
SKIP_MODEL_DOWNLOAD="${SKIP_MODEL_DOWNLOAD:-0}"
CPU_TORCH_VERSION="${CPU_TORCH_VERSION:-2.8.0+cpu}"

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
  ./setup.sh

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
  ALLOW_HUB_DOWNLOAD            Allow in-UI Hugging Face downloads, true or false
                                default: true
  INSTALL_TTS_VOICES            Download Piper TTS voices, 1 or 0
                                default: 1
  CPU_TORCH_VERSION             CPU-only PyTorch version for RAG installs
                                default: 2.8.0+cpu
  SKIP_MODEL_DOWNLOAD           Write config without downloading models, 1 or 0
                                default: 0
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -gt 0 ]]; then
  errln "Unknown argument: $1"
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

# piper-tts ships the same top-level `piper` package as piper-plus, so it gets
# its own venv; the UI reaches it through a subprocess worker.
step "Creating isolated piper-tts venv: ${C_DIM}${PIPERTTS_VENV}${C_RESET}"
python3 -m venv "${PIPERTTS_VENV}"
"${PIPERTTS_VENV}/bin/python" -m pip install --upgrade pip >/dev/null
info "Installing piper-tts (isolated)…"
"${PIPERTTS_VENV}/bin/python" -m pip install \
  -r "${EXAMPLE_DIR}/src/python/requirements-pipertts.txt"
ok "piper-tts venv ready."

mkdir -p "${MODELS_DIR}"
CHAT_MODEL_DIR="${MODELS_DIR}/${CHAT_MODEL_NAME}"
ASR_MODEL_DIR="${MODELS_DIR}/whisper-small-a16w8"
RAG_EMBEDDING_DIR="${MODELS_DIR}/gte-small"

section "Models"
if [[ "${SKIP_MODEL_DOWNLOAD}" != "1" ]]; then
  if [[ -n "${CHAT_MODEL_REPO}" ]]; then
    step "Downloading chat/VLM model: ${CHAT_MODEL_REPO}"
    "${APP_VENV}/bin/hf" download "${CHAT_MODEL_REPO}" --local-dir "${CHAT_MODEL_DIR}"
  else
    info "No default chat/VLM model — download one from the UI (or set CHAT_MODEL_REPO)."
  fi

  step "Downloading ASR model: ${ASR_MODEL_REPO}"
  "${APP_VENV}/bin/hf" download "${ASR_MODEL_REPO}" --local-dir "${ASR_MODEL_DIR}"

  step "Downloading RAG embedding model: ${RAG_EMBEDDING_REPO}"
  "${APP_VENV}/bin/hf" download "${RAG_EMBEDDING_REPO}" --local-dir "${RAG_EMBEDDING_DIR}"

  for repo in ${CATALOG_MODEL_REPOS}; do
    name="$(basename "${repo}")"
    step "Downloading catalog model: ${repo}"
    "${APP_VENV}/bin/hf" download "${repo}" --local-dir "${MODELS_DIR}/${name}"
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

# Render "simaai TDoSiMa" -> "simaai, TDoSiMa" for the YAML flow list.
HUB_ORGS_YAML="$(echo "${HUB_ORGS}" | tr -s ' ' | sed 's/^ //; s/ $//; s/ /, /g')"

section "Configuration"
step "Writing local config: ${C_DIM}${CONFIG_PATH}${C_RESET}"
mkdir -p "$(dirname "${CONFIG_PATH}")"
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
    asr:
      name: whisper-small-a16w8
      path: ${ASR_MODEL_DIR}

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
    host: 0.0.0.0
    port: 5000
    https: true

  ui:
    font_family: Inter
    font_size: 15

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
  section "Text-to-speech"
  step "Installing piper-tts voices + the piper-plus model…"
  (
    cd "${EXAMPLE_DIR}/src/python"
    ENABLE_KOREAN_TTS="${ENABLE_KOREAN_TTS:-0}" bash voice_install.sh
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

# Always set up (and show) a convenient `neat-ai` shell alias for ./run.sh.
# Set CREATE_ALIAS=0 to opt out.
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
  if [[ "${CREATE_ALIAS:-1}" != "0" ]]; then
    if [[ -f "${rc}" ]] && grep -q "alias neat-ai=" "${rc}"; then
      ok "'neat-ai' alias already set in ${rc}."
    else
      {
        printf '\n# Neat GenAI Studio — added by setup.sh\n'
        printf "alias neat-ai='%s'\n" "${run_path}"
      } >> "${rc}"
      ok "Added a 'neat-ai' alias to ${rc}."
    fi
  fi
  # Always show how to start it.
  info "Start the studio with ${C_BOLD}neat-ai${C_RESET} (alias for ./run.sh) — run ${C_BOLD}source ${rc}${C_RESET} or open a new shell first."
  info "Alias: ${C_DIM}neat-ai='${run_path}'${C_RESET}"
}

section "Done"
ok "Install complete."
maybe_create_alias
info "Config: ${C_DIM}${CONFIG_PATH}${C_RESET}"
info "Start the studio with ${C_BOLD}./run.sh${C_RESET}"
printf '\n'
