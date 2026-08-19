#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CATALOG_TOOL="${SCRIPT_DIR}/ui/voice_catalog.py"
ASSETS_DIR="${SCRIPT_DIR}/ui/assets"

# Comma- or space-separated ISO 639-1 codes. Korean intentionally has no
# catalogued server-side voice and will remain browser/text-only.
TTS_LANGUAGES="${TTS_LANGUAGES:-en,de,es,fr,it,ja,pt,vi,zh}"
# Optional catalogued voice ids, for example: "mera,en_US-ljspeech-medium".
TTS_OPTIONAL_VOICES="${TTS_OPTIONAL_VOICES:-}"
PYTHON="${PYTHON:-python3}"

python3 "${CATALOG_TOOL}" validate
python3 "${CATALOG_TOOL}" install \
  --languages "${TTS_LANGUAGES}" \
  --optional "${TTS_OPTIONAL_VOICES}" \
  --assets "${ASSETS_DIR}"

if ! "${PYTHON}" -c "import onnx" >/dev/null 2>&1; then
  echo "Cannot split voices: ${PYTHON} does not provide the onnx package." >&2
  echo "Run setup.sh or set PYTHON to .venv-pipertts/bin/python." >&2
  exit 1
fi
printf "\n🔪 Splitting Piper voices for streaming synthesis\n"
"${PYTHON}" "${SCRIPT_DIR}/split_voices.py" "${ASSETS_DIR}"
