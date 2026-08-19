#!/usr/bin/env bash
set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_MODELS_DIR="/media/nvme/llima/models"
MODELS_DIR="${LLIMA_MODELS_PATH:-${DEFAULT_MODELS_DIR}}"
APP_VENV="${APP_VENV:-${EXAMPLE_DIR}/.venv}"
CONFIG_PATH="${CONFIG_PATH:-${EXAMPLE_DIR}/config.local.yaml}"
PYNEAT_PYTHON="${PYNEAT_PYTHON:-${HOME}/pyneat/bin/python}"
CHAT_MODEL_REPO="${CHAT_MODEL_REPO:-simaai/Qwen3-VL-4B-Instruct-GPTQ-a16w4}"
ASR_MODEL_REPO="simaai/whisper-small-a16w8"
RAG_EMBEDDING_REPO="thenlper/gte-small"
CHAT_MODEL_NAME="${CHAT_MODEL_NAME:-$(basename "${CHAT_MODEL_REPO}")}"
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
  CHAT_MODEL_REPO               Hugging Face repo for the default chat/VLM model
                                default: simaai/Qwen3-VL-4B-Instruct-GPTQ-a16w4
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

if ! mkdir -p "${MODELS_DIR}" 2>/dev/null ||
  [[ ! -d "${MODELS_DIR}" || ! -w "${MODELS_DIR}" ]]
then
  echo "Model directory is not writable: ${MODELS_DIR}" >&2
  if [[ -z "${LLIMA_MODELS_PATH:-}" ]]; then
    echo "Mount NVMe and make ${DEFAULT_MODELS_DIR} writable, or set:" >&2
  else
    echo "Set LLIMA_MODELS_PATH to a writable directory and retry:" >&2
  fi
  echo "  LLIMA_MODELS_PATH=/writable/path ./setup.sh" >&2
  exit 1
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

mkdir -p "$(dirname "${CONFIG_PATH}")"
cat > "${CONFIG_PATH}" <<YAML
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
echo "Generated config:"
echo "  ${CONFIG_PATH}"
echo ""
echo "Run:"
echo "  ./run.sh"
