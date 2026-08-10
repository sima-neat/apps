#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Directory where voices are stored
ASSETS_DIR="ui/assets"

# Interpreter used to split voices for streaming synthesis. setup.sh passes the
# app virtual environment; a manual run falls back to whatever python3 is on PATH.
PYTHON="${PYTHON:-python3}"

# Edit this list with the voices you want to install.
# Use the format: "<lang_COUNTRY>-<voice>-<quality>"
# Examples:
#   en_US-amy-low
#   de_DE-thorsten-medium
#   es_ES-davefx-medium
#   fr_FR-siwis-medium
#   it_IT-paola-medium
#   zh_CN-huayan-medium
#   no_NO-talesyntese-medium
VOICES=(
  "en_US-hfc_female-medium"
  "en_US-hfc_male-medium"
  "en_US-kristin-medium"
  "en_GB-northern_english_male-medium"
  "vi_VN-vivos-x_low"
  "vi_VN-vais1000-medium"
  "de_DE-thorsten-medium"
  "es_ES-davefx-medium"
  "fr_FR-siwis-medium"
  "it_IT-paola-medium"
  "zh_CN-huayan-medium"
  "no_NO-talesyntese-medium"
)

if [[ ${#VOICES[@]} -eq 0 ]]; then
  echo "No voices selected. Edit VOICES[] in voice_install.sh to add voices to download."
  exit 0
fi

mkdir -p "$ASSETS_DIR"

for entry in "${VOICES[@]}"; do
  IFS='-' read -r lang_country voice quality <<< "$entry"
  if [[ -z "${lang_country:-}" || -z "${voice:-}" || -z "${quality:-}" ]]; then
    echo "❌ Invalid entry: '$entry'. Expected format '<lang_COUNTRY>-<voice>-<quality>'"
    exit 1
  fi

  language="${lang_country:0:2}"
  onnx_file="${lang_country}-${voice}-${quality}.onnx"
  base_url="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/${language}/${lang_country}/${voice}/${quality}"

  target_onnx="${ASSETS_DIR}/${onnx_file}"
  target_json="${ASSETS_DIR}/${onnx_file}.json"

  url_onnx="${base_url}/${onnx_file}?download=true"
  url_json="${base_url}/${onnx_file}.json?download=true"

  printf "\n📦 Voice: %s\n" "${entry}"

  if [[ -f "$target_json" ]]; then
    echo "✅ JSON exists: ${target_json}"
  else
    echo "⬇️  Downloading JSON: ${url_json}"
    curl -L --fail -o "$target_json" "$url_json"
  fi

  if [[ -f "$target_onnx" ]]; then
    echo "✅ ONNX exists: ${target_onnx}"
  else
    echo "⬇️  Downloading ONNX: ${url_onnx}"
    curl -L --fail -o "$target_onnx" "$url_onnx"
  fi

done

# PiperTTS decodes the latent in slices so playback starts before the sentence is
# finished, which means it loads an encoder/decoder pair rather than the whole
# voice. Split every voice here so the app never sees one it cannot load.
if ! "${PYTHON}" -c "import onnx" >/dev/null 2>&1; then
  echo ""
  echo "❌ Cannot split voices: '${PYTHON}' has no 'onnx' package."
  echo "   Use the app environment, or point PYTHON at one:"
  echo "     PYTHON=/path/to/.venv/bin/python bash voice_install.sh"
  exit 1
fi

printf "\n🔪 Splitting voices for streaming synthesis\n"
"${PYTHON}" "${SCRIPT_DIR}/split_voices.py" "${ASSETS_DIR}"

printf "\n✅ Done. Voices are available in '%s/'.\n" "${ASSETS_DIR}"
