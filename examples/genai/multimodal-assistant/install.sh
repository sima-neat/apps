#!/usr/bin/env bash
set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE_DIR="${MULTIMODAL_ASSISTANT_CACHE_DIR:-${HOME}/.cache/sima-neat/multimodal-assistant}"
APP_VENV="${APP_VENV:-${CACHE_DIR}/venv}"
MODELS_DIR="${LLIMA_MODELS_PATH:-/media/nvme/llima/models}"
PYNEAT_PYTHON="${PYNEAT_PYTHON:-${HOME}/pyneat/bin/python}"
CHAT_MODEL_REPO="${CHAT_MODEL_REPO:-simaai/Qwen3-VL-2B-Instruct-GPTQ-a16w4}"
ASR_MODEL_REPO="simaai/whisper-small-a16w8"
RAG_EMBEDDING_REPO="thenlper/gte-small"
CHAT_MODEL_NAME="${CHAT_MODEL_NAME:-$(basename "${CHAT_MODEL_REPO}")}"
INSTALL_TTS_VOICES="${INSTALL_TTS_VOICES:-1}"
SKIP_MODEL_DOWNLOAD="${SKIP_MODEL_DOWNLOAD:-0}"

usage() {
  cat <<'USAGE'
Usage:
  ./install.sh

Environment:
  PYNEAT_PYTHON                 Python interpreter with pyneat
                                default: ~/pyneat/bin/python
  APP_VENV                      UI virtual environment path
                                default: ~/.cache/sima-neat/multimodal-assistant/venv
  LLIMA_MODELS_PATH             Model download directory
                                default: /media/nvme/llima/models
  CHAT_MODEL_REPO               Hugging Face repo for the default chat/VLM model
                                default: simaai/Qwen3-VL-2B-Instruct-GPTQ-a16w4
  INSTALL_TTS_VOICES            Download Piper TTS voices, 1 or 0
                                default: 1
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

echo "Creating UI virtual environment:"
echo "  ${APP_VENV}"
python3 -m venv "${APP_VENV}"
"${APP_VENV}/bin/python" -m pip install --upgrade pip
"${APP_VENV}/bin/python" -m pip install -r "${EXAMPLE_DIR}/src/python/requirements.txt"
"${APP_VENV}/bin/python" -m pip install -r "${EXAMPLE_DIR}/src/python/requirements-rag.txt"
"${APP_VENV}/bin/python" -m pip install -U "huggingface_hub[cli]"

mkdir -p "${MODELS_DIR}"
CHAT_MODEL_DIR="${MODELS_DIR}/${CHAT_MODEL_NAME}"
ASR_MODEL_DIR="${MODELS_DIR}/whisper-small-a16w8"
RAG_EMBEDDING_DIR="${MODELS_DIR}/gte-small"

if [[ "${SKIP_MODEL_DOWNLOAD}" != "1" ]]; then
  echo ""
  echo "Downloading chat/VLM model:"
  echo "  ${CHAT_MODEL_REPO}"
  "${APP_VENV}/bin/hf" download "${CHAT_MODEL_REPO}" --local-dir "${CHAT_MODEL_DIR}"

  echo ""
  echo "Downloading ASR model:"
  echo "  ${ASR_MODEL_REPO}"
  "${APP_VENV}/bin/hf" download "${ASR_MODEL_REPO}" --local-dir "${ASR_MODEL_DIR}"

  echo ""
  echo "Downloading RAG embedding model:"
  echo "  ${RAG_EMBEDDING_REPO}"
  "${APP_VENV}/bin/hf" download "${RAG_EMBEDDING_REPO}" --local-dir "${RAG_EMBEDDING_DIR}"
else
  echo ""
  echo "Skipping model downloads because SKIP_MODEL_DOWNLOAD=1."
fi

cat > "${EXAMPLE_DIR}/src/common/config.yaml" <<YAML
server:
  openai:
    host: 0.0.0.0
    port: 9998

  models:
    chat:
      - name: ${CHAT_MODEL_NAME}
        path: ${CHAT_MODEL_DIR}
    asr:
      name: whisper-small-a16w8
      path: ${ASR_MODEL_DIR}

app:
  openai:
    client_host: 127.0.0.1
    port: 9998

  request:
    max_tokens: 128
    system_prompt: >-
      Answer clearly and concisely.

  web:
    host: 0.0.0.0
    port: 5000
    https: true

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
echo ""
echo "Run:"
echo "  APP_PYTHON=${APP_VENV}/bin/python PYNEAT_PYTHON=${PYNEAT_PYTHON} ./run.sh"
