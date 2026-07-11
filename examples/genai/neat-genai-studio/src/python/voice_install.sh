#!/bin/bash
set -euo pipefail

# Directory where voices are stored
ASSETS_DIR="ui/assets"

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
# rhasspy piper-tts voices. Kept only for languages piper-plus has no trained
# model for (German, Italian, Norwegian, Vietnamese) plus an English fallback.
# piper-plus (below) is preferred for en/ja/zh/es/fr/pt.
VOICES=(
  "en_US-hfc_female-medium"       # English fallback (piper-plus preferred for en)
  "en_US-hfc_male-medium"
  "de_DE-thorsten-medium"         # German     — no piper-plus model
  "it_IT-paola-medium"            # Italian    — no piper-plus model
  "no_NO-talesyntese-medium"      # Norwegian  — no piper-plus model
  "vi_VN-vais1000-medium"         # Vietnamese — no piper-plus model
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

# --- piper-plus multilingual voices (each covers ja/en/zh/es/fr/pt) ----------
# MIT-licensed, onnxruntime. Every piper-plus checkpoint is the same 6-language
# model; they differ by VOICE, so we fetch a few and let the UI pick the active
# one. Saved as ui/assets/piper-plus/<key>/{model.onnx,config.json}. Edit this
# list ("<key>|<hf-repo>|<onnx-filename>") to change which voices are installed.
PIPER_PLUS_MODELS=(
  "css10|ayousanz/piper-plus-css10-ja-6lang|css10-ja-6lang-fp16.onnx"
  "tsukuyomi|ayousanz/piper-plus-tsukuyomi-chan|tsukuyomi-chan-6lang-fp16.onnx"
  "mera|kizuna-intelligence/piper-plus-mera-multilingual|mera-multilingual.onnx"
)
for entry in "${PIPER_PLUS_MODELS[@]}"; do
  IFS='|' read -r pp_key pp_repo pp_onnx_name <<< "$entry"
  pp_dir="${ASSETS_DIR}/piper-plus/${pp_key}"
  mkdir -p "$pp_dir"
  pp_base="https://huggingface.co/${pp_repo}/resolve/main"
  printf "\n📦 piper-plus voice '%s': %s\n" "$pp_key" "$pp_repo"
  if [[ -f "${pp_dir}/model.onnx" && -f "${pp_dir}/config.json" ]]; then
    echo "✅ exists: ${pp_dir}"
  else
    echo "⬇️  Downloading ${pp_onnx_name} + config.json…"
    curl -L --fail -o "${pp_dir}/model.onnx" "${pp_base}/${pp_onnx_name}?download=true"
    curl -L --fail -o "${pp_dir}/config.json" "${pp_base}/config.json?download=true"
  fi
done

# --- Optional Korean TTS (Meta MMS, CC-BY-NC 4.0 — non-commercial) -----------
# Off by default. Set ENABLE_KOREAN_TTS=1 (and accept the non-commercial
# license) to fetch it; the app also needs ENABLE_KOREAN_TTS=1 at runtime.
if [[ "${ENABLE_KOREAN_TTS:-0}" == "1" ]]; then
  printf "\n📦 Korean MMS model: facebook/mms-tts-kor (CC-BY-NC 4.0)\n"
  mms_dir="${ASSETS_DIR}/mms-tts-kor"
  mkdir -p "$mms_dir"
  mms_base="https://huggingface.co/facebook/mms-tts-kor/resolve/main"
  for f in config.json model.safetensors tokenizer_config.json vocab.json special_tokens_map.json; do
    if [[ -f "${mms_dir}/${f}" ]]; then
      echo "✅ ${f} exists"
    else
      echo "⬇️  Downloading ${f}…"
      curl -L --fail -o "${mms_dir}/${f}" "${mms_base}/${f}?download=true"
    fi
  done
fi

printf "\n✅ Done. Voices are available in '%s/'.\n" "${ASSETS_DIR}"
