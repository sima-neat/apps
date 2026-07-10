#!/usr/bin/env bash
set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# MODELS_DIR is the model catalog: every compatible model directory under it is
# discoverable and loadable on the fly from the UI (no restart).
MODELS_DIR="${LLIMA_MODELS_PATH:-/media/nvme/llima/models}"
APP_VENV="${APP_VENV:-${EXAMPLE_DIR}/.venv}"
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
  echo "Unknown argument: $1" >&2
  usage
  exit 2
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required." >&2
  exit 1
fi

if [[ ! -x "${PYNEAT_PYTHON}" ]]; then
  echo "Neat Python environment not found: ${PYNEAT_PYTHON}" >&2
  echo "Install Neat first, or set PYNEAT_PYTHON=/path/to/python-with-pyneat." >&2
  exit 1
fi

if ! "${PYNEAT_PYTHON}" - <<'PY' >/dev/null 2>&1
import pyneat
PY
then
  echo "pyneat is not importable from ${PYNEAT_PYTHON}." >&2
  echo "Install Neat first, or set PYNEAT_PYTHON=/path/to/python-with-pyneat." >&2
  exit 1
fi

install_cpu_torch_if_needed() {
  case "$(uname -m)" in
    x86_64|amd64|aarch64|arm64)
      echo "Installing CPU-only PyTorch for RAG embeddings..."
      "${APP_VENV}/bin/python" -m pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch==${CPU_TORCH_VERSION}"
      ;;
  esac
}

echo "Creating UI virtual environment:"
echo "  ${APP_VENV}"
python3 -m venv "${APP_VENV}"
"${APP_VENV}/bin/python" -m pip install --upgrade pip
install_cpu_torch_if_needed
"${APP_VENV}/bin/python" -m pip install \
  -r "${EXAMPLE_DIR}/src/python/requirements.txt" \
  -r "${EXAMPLE_DIR}/src/python/requirements-rag.txt"

mkdir -p "${MODELS_DIR}"
CHAT_MODEL_DIR="${MODELS_DIR}/${CHAT_MODEL_NAME}"
ASR_MODEL_DIR="${MODELS_DIR}/whisper-small-a16w8"
RAG_EMBEDDING_DIR="${MODELS_DIR}/gte-small"

if [[ "${SKIP_MODEL_DOWNLOAD}" != "1" ]]; then
  if [[ -n "${CHAT_MODEL_REPO}" ]]; then
    echo ""
    echo "Downloading chat/VLM model:"
    echo "  ${CHAT_MODEL_REPO}"
    "${APP_VENV}/bin/hf" download "${CHAT_MODEL_REPO}" --local-dir "${CHAT_MODEL_DIR}"
  else
    echo ""
    echo "No default chat/VLM model — download one from the UI (or set CHAT_MODEL_REPO)."
  fi

  echo ""
  echo "Downloading ASR model:"
  echo "  ${ASR_MODEL_REPO}"
  "${APP_VENV}/bin/hf" download "${ASR_MODEL_REPO}" --local-dir "${ASR_MODEL_DIR}"

  echo ""
  echo "Downloading RAG embedding model:"
  echo "  ${RAG_EMBEDDING_REPO}"
  "${APP_VENV}/bin/hf" download "${RAG_EMBEDDING_REPO}" --local-dir "${RAG_EMBEDDING_DIR}"

  for repo in ${CATALOG_MODEL_REPOS}; do
    name="$(basename "${repo}")"
    echo ""
    echo "Downloading catalog model:"
    echo "  ${repo}"
    "${APP_VENV}/bin/hf" download "${repo}" --local-dir "${MODELS_DIR}/${name}"
  done
else
  echo ""
  echo "Skipping model downloads because SKIP_MODEL_DOWNLOAD=1."
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
    max_tokens: 128
    system_prompt: >-
      Answer clearly and concisely. Use Markdown formatting when it helps.

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

if [[ "${SKIP_MODEL_DOWNLOAD}" != "1" ]]; then
  echo ""
  echo "Creating default RAG database..."
  (
    cd "${EXAMPLE_DIR}/src/python"
    "${APP_VENV}/bin/python" rag/create_db.py \
      --input "${EXAMPLE_DIR}/src/common/rag/neat.md" \
      --output ui/milvus.db \
      --embedding-model "${RAG_EMBEDDING_DIR}"
  )
fi

if [[ "${INSTALL_TTS_VOICES}" == "1" ]]; then
  echo ""
  echo "Installing Piper TTS voices..."
  (
    cd "${EXAMPLE_DIR}/src/python"
    bash voice_install.sh
  )
fi

echo ""
echo "Install complete."
echo "Generated config:"
echo "  ${CONFIG_PATH}"
echo ""
echo "Run:"
echo "  ./run.sh"
