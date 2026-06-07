#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

APP_DIR="${1:-}"
if [[ -z "$APP_DIR" ]]; then
    if [[ -f "$SCRIPT_DIR/app.py" && -f "$SCRIPT_DIR/requirements.txt" ]]; then
        APP_DIR="$SCRIPT_DIR"
    elif [[ -d "$SCRIPT_DIR/simaai-genai-demo" ]]; then
        APP_DIR="$SCRIPT_DIR/simaai-genai-demo"
    else
        echo "ℹ️  No demo web app folder found. Skipping demo web app setup."
        exit 0
    fi
fi

if [[ ! -d "$APP_DIR" ]]; then
    echo "ℹ️  Demo web app folder not found: $APP_DIR"
    exit 0
fi

VENV_DIR="${2:-$SCRIPT_DIR/.venv}"
if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    echo "❌ Shared virtual environment not found: $VENV_DIR"
    echo "   Run base install.sh first."
    exit 1
fi

source "$VENV_DIR/bin/activate"

copy_dir() {
    local src="$1"
    local dst="$2"
    if command -v rsync >/dev/null 2>&1; then
        rsync -a "$src/" "$dst/"
    else
        cp -a "$src/." "$dst/"
    fi
}

check_gte_model_location() {
    if [[ -d "$SCRIPT_DIR/models/gte-small-local" ]]; then
        echo "✅ GTE model found at $SCRIPT_DIR/models/gte-small-local"
    else
        echo "⚠️  GTE model not found at $SCRIPT_DIR/models/gte-small-local"
    fi
}

download_piper_onnx() {
    local assets_dir="$APP_DIR/assets"
    [[ -d "$assets_dir" ]] || return 0

    echo "🔎 Scanning for *.onnx.json files in $assets_dir ..."
    for json_file in "$assets_dir"/*.onnx.json; do
        [[ -f "$json_file" ]] || continue
        local filename lang_country voice quality language onnx_file download_url
        filename="$(basename "$json_file" .onnx.json)"
        IFS='-' read -r lang_country voice quality <<< "$filename"
        language="${lang_country:0:2}"
        onnx_file="${lang_country}-${voice}-${quality}.onnx"
        download_url="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/${language}/${lang_country}/${voice}/${quality}/${onnx_file}?download=true"

        if [[ -f "${assets_dir}/${onnx_file}" ]]; then
            echo "✅ ${onnx_file} already exists, skipping."
        else
            echo "⬇️  Downloading ${onnx_file}"
            curl -L -o "${assets_dir}/${onnx_file}" "$download_url"
        fi
    done
}

install_demo_requirements() {
    local req="$APP_DIR/requirements.txt"
    [[ -f "$req" ]] || return 0

    local nproc_val=1
    if command -v nproc >/dev/null 2>&1; then
        nproc_val="$(nproc)"
    elif command -v getconf >/dev/null 2>&1; then
        nproc_val="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"
    fi

    export CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF -DCMAKE_BUILD_PARALLEL_LEVEL=${nproc_val}"
    export CMAKE_BUILD_PARALLEL_LEVEL="${nproc_val}"

    echo "📦 Installing demo web app dependencies from $req ..."
    python3 -m pip install -r "$req"
}

check_gte_model_location
download_piper_onnx
install_demo_requirements

echo "✅ Demo web app installation complete for $APP_DIR."
