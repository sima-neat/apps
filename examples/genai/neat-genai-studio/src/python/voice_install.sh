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

python3 "${CATALOG_TOOL}" validate
python3 "${CATALOG_TOOL}" install \
  --languages "${TTS_LANGUAGES}" \
  --optional "${TTS_OPTIONAL_VOICES}" \
  --assets "${ASSETS_DIR}"
