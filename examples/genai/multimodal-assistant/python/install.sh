#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST_FILE="${SCRIPT_DIR}/manifest.txt"
cd "$SCRIPT_DIR"

manifest_contains_resource() {
    local resource="$1"

    [[ -f "$MANIFEST_FILE" ]] || return 1
    grep -Fx -- "$resource" "$MANIFEST_FILE" >/dev/null
}

manifest_contains_hf_model_folder() {
    local foldername="$1"
    local resource
    local repo_name

    [[ -f "$MANIFEST_FILE" ]] || return 1
    while IFS= read -r resource || [[ -n "$resource" ]]; do
        [[ "$resource" == hf:* ]] || continue
        resource="${resource%@*}"
        repo_name="${resource##*/}"
        if [[ "$repo_name" == "$foldername" ]]; then
            return 0
        fi
    done < "$MANIFEST_FILE"

    return 1
}

normalize_model_layout() {
    local models_dir="./models"
    local moved_any=false

    if [[ ! -f "$MANIFEST_FILE" ]]; then
        echo "⚠️  manifest.txt not found under $SCRIPT_DIR. Skipping model folder scan."
        return 0
    fi

    mkdir -p "$models_dir"
    echo "🗂️  Normalizing model layout into $models_dir ..."

    for search_root in "." ".."; do
        [[ -d "$search_root" ]] || continue
        for dir in "$search_root"/*/; do
            [[ -d "$dir" ]] || continue

            local foldername
            foldername="$(basename "$dir")"
            [[ "$foldername" == "models" || "$foldername" == .* ]] && continue
            manifest_contains_hf_model_folder "$foldername" || continue

            if { compgen -G "${dir}elf_files/*.elf" > /dev/null && [[ -f "${dir}devkit/vlm_config.json" ]]; } \
               || [[ "$foldername" == *"whisper-small-a16w8"* ]]; then
                local target="${models_dir}/${foldername}"
                [[ "$dir" == "$target/" ]] && continue
                if [[ -d "$target" ]]; then
                    echo "ℹ️  ${target} already exists. Keeping existing target and source ${dir%/}."
                else
                    echo "➡️  Moving ${dir%/} -> ${target}"
                    mv "${dir%/}" "$target"
                    moved_any=true
                fi
            fi
        done
    done

    if [[ "$moved_any" == false ]]; then
        echo "ℹ️  No model folders were moved during normalization."
    else
        echo "✅ Model normalization complete under ${models_dir}."
    fi
}

normalize_gte_model_layout() {
    local models_dir="./models"
    local gte_target="${models_dir}/gte-small-local"
    mkdir -p "$models_dir"

    if ! manifest_contains_resource "models/gte-small-local.tar.gz"; then
        echo "ℹ️  GTE model archive is not listed in manifest.txt. Skipping GTE normalization."
        return
    fi

    if [[ -d "$gte_target" && -f "$gte_target/config.json" ]]; then
        echo "✅ GTE model already normalized at ${gte_target}."
        return
    fi

    if [[ -f "${models_dir}/gte-small-local.tar.gz" ]]; then
        echo "📦 Extracting ${models_dir}/gte-small-local.tar.gz into ${models_dir} ..."
        tar -xzf "${models_dir}/gte-small-local.tar.gz" -C "${models_dir}"
        if [[ -d "${models_dir}/gte-small-local/gte-small-local" ]]; then
            mv "${models_dir}/gte-small-local/gte-small-local" "${models_dir}/gte-small-local.tmp"
            rm -rf "${models_dir}/gte-small-local"
            mv "${models_dir}/gte-small-local.tmp" "${gte_target}"
        fi
    fi

    if [[ -d "$gte_target" && -f "$gte_target/config.json" ]]; then
        echo "✅ GTE model normalized at ${gte_target}."
    else
        echo "⚠️  GTE model was not found for normalization."
    fi
}

install_base_venv() {
    echo "🐍 Setting up Python virtual environment (.venv) with --system-site-packages ..."
    python3 -m venv .venv --system-site-packages
    source .venv/bin/activate
    python3 -m pip install --upgrade pip setuptools wheel
}

install_local_llima_wheels() {
    local system_python="python3"
    local -a pip_cmd
    local -a install_args
    local installed_any=false
    if [[ -x "/usr/bin/python3" ]]; then
        system_python="/usr/bin/python3"
    fi

    pip_cmd=("$system_python" -m pip)
    if [[ "${EUID}" -ne 0 ]]; then
        if command -v sudo >/dev/null 2>&1; then
            pip_cmd=(sudo "$system_python" -m pip)
        else
            echo "❌ System wheel install requires root privileges (or sudo)."
            return 1
        fi
    fi

    install_args=(install)
    if "${pip_cmd[@]}" install --help 2>/dev/null | grep -q -- "--break-system-packages"; then
        install_args+=(--break-system-packages)
    fi

    if [[ ! -f "$MANIFEST_FILE" ]]; then
        echo "⚠️  manifest.txt not found under $SCRIPT_DIR. Refusing to install local sima_lmm wheels by filesystem scan."
        return 0
    fi

    while IFS= read -r resource || [[ -n "$resource" ]]; do
        [[ "$resource" == sima_lmm-*-cp311-cp311-linux_aarch64.whl ]] || continue

        local whl="./${resource//%2B/+}"
        if [[ ! -f "$whl" && -f "./$resource" ]]; then
            whl="./$resource"
        fi
        if [[ ! -f "$whl" ]]; then
            continue
        fi

        echo "📦 Installing $whl"
        "${pip_cmd[@]}" "${install_args[@]}" "$whl"
        installed_any=true
        break
    done < "$MANIFEST_FILE"

    if [[ "$installed_any" == false ]]; then
        echo "⚠️  No manifest-listed sima_lmm cp311 aarch64 wheel was found in this package."
    fi
}

ensure_executable_tools() {
    local paths=(
        "$SCRIPT_DIR/run.sh"
        "$SCRIPT_DIR/backend_runtime_test.sh"
    )

    if [[ -n "${DEMO_APP_DIR:-}" ]]; then
        paths+=(
            "$DEMO_APP_DIR/run.sh"
            "$DEMO_APP_DIR/backend_runtime_test.sh"
            "$DEMO_APP_DIR/apitest/backend_runtime_test.sh"
        )
    fi

    for p in "${paths[@]}"; do
        if [[ -f "$p" ]]; then
            chmod +x "$p"
            echo "✅ Set executable: $p"
        fi
    done
}

# Optional platform check can be skipped when packaging/testing on host.
if [[ "${SKIP_MODALIX_CHECK:-0}" != "1" ]]; then
    model_path="/proc/device-tree/model"
    if [[ -f "$model_path" ]]; then
        model="$(tr -d '\0' < "$model_path" | head -n 1)"
        if [[ "$model" != *"Modalix"* ]]; then
            echo "❌ This installer must be run on a Modalix device."
            echo "💡 model reported: '$model'"
            exit 1
        fi
    else
        echo "❌ Modalix device model file not found: $model_path"
        echo "💡 Set SKIP_MODALIX_CHECK=1 only for host-side packaging checks."
        exit 1
    fi
fi

if command -v apt >/dev/null 2>&1; then
    echo "✅ apt detected. Installing system tools required for LLiMa..."
    sudo apt update -y && sudo apt install -y git gawk ffmpeg rsync
fi

echo "🚀 Installing base LLiMa runtime..."
normalize_model_layout
normalize_gte_model_layout
install_base_venv
install_local_llima_wheels

DEMO_APP_DIR=""
if [[ -d "$SCRIPT_DIR/simaai-genai-demo" ]]; then
    DEMO_APP_DIR="$SCRIPT_DIR/simaai-genai-demo"
elif [[ -f "$SCRIPT_DIR/app.py" && -f "$SCRIPT_DIR/requirements.txt" ]]; then
    DEMO_APP_DIR="$SCRIPT_DIR"
fi

if [[ -n "$DEMO_APP_DIR" && -f "$DEMO_APP_DIR/install-demo-web-app.sh" ]]; then
    echo "🧩 Demo app detected at $DEMO_APP_DIR"
    "$DEMO_APP_DIR/install-demo-web-app.sh" "$DEMO_APP_DIR" "$SCRIPT_DIR/.venv"
else
    echo "ℹ️  Demo app not found (expected ./simaai-genai-demo or current dir app files)."
    echo "ℹ️  Base LLiMa installation completed."
fi

ensure_executable_tools
