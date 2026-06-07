#!/bin/bash
set -euo pipefail

PROD_MODE=false
if [[ "${1:-}" == "--prod" ]]; then
    PROD_MODE=true
    shift
elif [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: $0 [--prod]"
    echo "  default: local mode (metadata uses '+' wheel names; works with python -m http.server)"
    echo "  --prod : production mode (metadata uses '%2B' escaped wheel names)"
    exit 0
elif [[ $# -gt 0 ]]; then
    echo "❌ Unknown argument: $1"
    echo "Usage: $0 [--prod]"
    exit 1
fi

# Version input
MAIN_VERSION="${MAIN_VERSION:-2.1.0}"

# LLiMa build artifact names
DEVELOP_AARCH64_WHL="${DEVELOP_AARCH64_WHL:-sima_lmm-${MAIN_VERSION}.dev0+pr.199.2-cp311-cp311-linux_aarch64.whl}"
MASTER_AARCH64_WHL="${MASTER_AARCH64_WHL:-sima_lmm-${MAIN_VERSION}.dev0+pr.198.1-cp311-cp311-linux_aarch64.whl}"

# URL-encoded wheel names for package metadata (%2B instead of +)
MASTER_AARCH64_WHL_ENC="${MASTER_AARCH64_WHL/+/%2B}"
DEVELOP_AARCH64_WHL_ENC="${DEVELOP_AARCH64_WHL/+/%2B}"

ARTIFACT_BASE_URL="https://artifacts.eng.sima.ai:443/artifactory/sima-pypi/swml-auto-lmm"
MASTER_AARCH64_URL="${ARTIFACT_BASE_URL}/${MASTER_AARCH64_WHL_ENC}"
DEVELOP_AARCH64_URL="${ARTIFACT_BASE_URL}/${DEVELOP_AARCH64_WHL_ENC}"
GTE_URL="https://docs.sima.ai/pkg_downloads/SDK2.0.0/samples/llima/models/gte-small-local.tar.gz"

STAGING_DIR="${HOME}/Downloads/llima-staging"
MODELS_DIR="${STAGING_DIR}/models"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CHECKOUT_BRANCH="$(git -C "${SCRIPT_DIR}" branch --show-current)"
if [[ -z "${CHECKOUT_BRANCH}" ]]; then
    CHECKOUT_BRANCH="detached"
fi
CHECKOUT_BRANCH_SAFE="${CHECKOUT_BRANCH//[!A-Za-z0-9._-]/-}"
CHECKOUT_SHORT_HASH="$(git -C "${SCRIPT_DIR}" rev-parse --short HEAD)"
METADATA_VERSION="${MAIN_VERSION}-${CHECKOUT_BRANCH_SAFE}-${CHECKOUT_SHORT_HASH}"

if ! command -v sima-cli >/dev/null 2>&1; then
    echo "❌ sima-cli not found in PATH."
    exit 1
fi

mkdir -p "${MODELS_DIR}"
cd "${STAGING_DIR}"

echo "📦 Building demo archive via build-dist.sh..."
(
    cd "${SCRIPT_DIR}"
    ./build-dist.sh
)

if [[ -f "/tmp/simaai-genai-demo.tar.gz" ]]; then
    cp -f "/tmp/simaai-genai-demo.tar.gz" "${STAGING_DIR}/simaai-genai-demo.tar.gz"
else
    echo "❌ build-dist.sh did not produce /tmp/simaai-genai-demo.tar.gz"
    exit 1
fi

# Stage root scripts used by metadata installation/runtime.
for root_script in install.sh run.sh; do
    if [[ -f "${SCRIPT_DIR}/${root_script}" ]]; then
        cp -f "${SCRIPT_DIR}/${root_script}" "${STAGING_DIR}/${root_script}"
        chmod +x "${STAGING_DIR}/${root_script}"
    else
        echo "❌ Missing required root script: ${SCRIPT_DIR}/${root_script}"
        exit 1
    fi
done

# Stage backend runtime validation tool as an individually downloadable resource.
BACKEND_TEST_SRC="${SCRIPT_DIR}/apitest/backend_runtime_test.sh"
BACKEND_TEST_NAME="backend_runtime_test.sh"
if [[ -f "${BACKEND_TEST_SRC}" ]]; then
    cp -f "${BACKEND_TEST_SRC}" "${STAGING_DIR}/${BACKEND_TEST_NAME}"
    chmod +x "${STAGING_DIR}/${BACKEND_TEST_NAME}"
else
    echo "❌ Missing required backend test script: ${BACKEND_TEST_SRC}"
    exit 1
fi

download_with_index_mode() {
    local url="$1"
    echo "⬇️  sima-cli -i download ${url}"
    sima-cli -i download "${url}"
}

normalize_wheel_filename() {
    local decoded_name="$1"  # filename with +
    local encoded_name="$2"  # filename with %2B
    local decoded_base
    local encoded_base
    local decoded_dir

    # Always keep staged artifact filenames decoded (+) so object keys remain canonical.
    # --prod only changes metadata resource strings (URL-escaped), not on-disk filenames.
    decoded_base="$(basename "${decoded_name}")"
    encoded_base="$(basename "${encoded_name}")"
    decoded_dir="$(dirname "${decoded_name}")"
    if [[ "${decoded_dir}" != "." ]]; then
        mkdir -p "${decoded_dir}"
    fi

    if [[ -f "${encoded_name}" && ! -f "${decoded_name}" ]]; then
        mv -f "${encoded_name}" "${decoded_name}"
    elif [[ -f "${encoded_base}" && ! -f "${decoded_name}" ]]; then
        mv -f "${encoded_base}" "${decoded_name}"
    elif [[ -f "${decoded_base}" && "${decoded_base}" != "${decoded_name}" && ! -f "${decoded_name}" ]]; then
        mv -f "${decoded_base}" "${decoded_name}"
    fi
}

sha256_file() {
    local file_path="$1"

    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "${file_path}" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "${file_path}" | awk '{print $1}'
    else
        echo "❌ Neither sha256sum nor shasum is available." >&2
        exit 1
    fi
}

download_with_index_mode "${MASTER_AARCH64_URL}"
download_with_index_mode "${DEVELOP_AARCH64_URL}"

# Normalize wheel filenames to decoded '+' names.
normalize_wheel_filename "${MASTER_AARCH64_WHL}" "${MASTER_AARCH64_WHL_ENC}"
normalize_wheel_filename "${DEVELOP_AARCH64_WHL}" "${DEVELOP_AARCH64_WHL_ENC}"

if [[ "${PROD_MODE}" == true ]]; then
    MASTER_AARCH64_RESOURCE="${MASTER_AARCH64_WHL_ENC}"
    DEVELOP_AARCH64_RESOURCE="${DEVELOP_AARCH64_WHL_ENC}"
else
    MASTER_AARCH64_RESOURCE="${MASTER_AARCH64_WHL}"
    DEVELOP_AARCH64_RESOURCE="${DEVELOP_AARCH64_WHL}"
fi

echo "⬇️  sima-cli download ${GTE_URL}"
sima-cli download "${GTE_URL}"

# Ensure gte archive is under models/.
if [[ -f "gte-small-local.tar.gz" ]]; then
    mv -f "gte-small-local.tar.gz" "${MODELS_DIR}/gte-small-local.tar.gz"
elif [[ -f "${MODELS_DIR}/gte-small-local.tar.gz" ]]; then
    :
else
    echo "❌ gte-small-local.tar.gz was not downloaded."
    exit 1
fi

cat > "${STAGING_DIR}/manifest.txt" <<EOF
manifest.txt
install.sh
run.sh
backend_runtime_test.sh
hf:simaai/gemma3-siglip448-a16w4@Release-2.1.0
hf:simaai/whisper-small-a16w8
models/gte-small-local.tar.gz
${MASTER_AARCH64_RESOURCE}
${DEVELOP_AARCH64_RESOURCE}
simaai-genai-demo.tar.gz
hf:simaai/llava-1.5-7b-hf-a16w4@Release-2.1.0
hf:simaai/Mistral-7B-Instruct-v0.3-a16w4@Release-2.1.0
hf:simaai/Llama-3.1-8B-Instruct-a16w4@Release-2.1.0
hf:simaai/Llama-2-7b-chat-hf-a16w4@Release-2.1.0
hf:simaai/Llama-3.2-3B-Instruct-a16w4@Release-2.1.0
hf:simaai/phi-3.5-mini-instruct-a16w4@Release-2.1.0
EOF

INSTALL_CHECKSUM="$(sha256_file "${STAGING_DIR}/install.sh")"
RUN_CHECKSUM="$(sha256_file "${STAGING_DIR}/run.sh")"
BACKEND_TEST_CHECKSUM="$(sha256_file "${STAGING_DIR}/${BACKEND_TEST_NAME}")"
MANIFEST_CHECKSUM="$(sha256_file "${STAGING_DIR}/manifest.txt")"
GTE_CHECKSUM="$(sha256_file "${MODELS_DIR}/gte-small-local.tar.gz")"
MASTER_AARCH64_CHECKSUM="$(sha256_file "${STAGING_DIR}/${MASTER_AARCH64_WHL}")"
DEVELOP_AARCH64_CHECKSUM="$(sha256_file "${STAGING_DIR}/${DEVELOP_AARCH64_WHL}")"
DEMO_APP_CHECKSUM="$(sha256_file "${STAGING_DIR}/simaai-genai-demo.tar.gz")"

METADATA_PLATFORMS="$(cat <<EOF
        {
            "type": "board",
            "compatible_with": [
                "modalix"
            ],
            "version": "2.1.0"
        },
        {
            "type": "board",
            "compatible_with": [
                "modalix"
            ],
            "version": "2.1.1"
        },
        {
            "type": "host",
            "os": ["mac", "linux"]
        }
EOF
)"
METADATA_INSTALL_SCRIPT='rm -Rf ~/.cache && mkdir -p simaai-genai-demo/models && rm -Rf simaai-genai-demo/gte-small-local simaai-genai-demo/models/gte-small-local && if [ -d gte-small-local/gte-small-local ]; then mv gte-small-local/gte-small-local simaai-genai-demo/models/; elif [ -d gts-small-local/gte-small-local ]; then mv gts-small-local/gte-small-local simaai-genai-demo/models/; fi && bash ./install.sh && rm -Rf ~/.cache'

cat > "${STAGING_DIR}/metadata.json" <<EOF
{
    "name": "llima",
    "version": "${METADATA_VERSION}",
    "release": "stable",
    "description": "SiMa.ai Lean Language & Image Modalix Application",
    "platforms": [
${METADATA_PLATFORMS}
    ],
    "resources": [
        "install.sh",
        "run.sh",
        "backend_runtime_test.sh",
        "manifest.txt",
        "hf:simaai/gemma3-siglip448-a16w4@Release-2.0.0",
        "hf:simaai/whisper-small-a16w8",
        "models/gte-small-local.tar.gz",
        "${MASTER_AARCH64_RESOURCE}"
    ],
    "resources-checksum": {
        "install.sh": "${INSTALL_CHECKSUM}",
        "run.sh": "${RUN_CHECKSUM}",
        "backend_runtime_test.sh": "${BACKEND_TEST_CHECKSUM}",
        "manifest.txt": "${MANIFEST_CHECKSUM}",
        "models/gte-small-local.tar.gz": "${GTE_CHECKSUM}",
        "${MASTER_AARCH64_RESOURCE}": "${MASTER_AARCH64_CHECKSUM}"
    },
    "selectable-resources": [
        {
            "name": "Demo Web App",
            "url": "",
            "resource": "simaai-genai-demo.tar.gz",
            "checksum": "${DEMO_APP_CHECKSUM}"
        }
    ],
    "size": {
        "download": "30GB",
        "install": "40GB"
    },
    "installation": {
        "script": "${METADATA_INSTALL_SCRIPT}",
        "post-message": "[bold]To run this package:[/bold]\\n\\n1. [green]./run.sh[/green] (in web interaction mode)\\n2. [green]./run.sh -cli[/green] (in CLI mode) \\n\\n"
    }
}
EOF

cat > "${STAGING_DIR}/metadata-develop.json" <<EOF
{
    "name": "llima",
    "version": "${METADATA_VERSION}",
    "release": "develop",
    "description": "SiMa.ai Lean Language & Image Modalix Application",
    "platforms": [
${METADATA_PLATFORMS}
    ],
    "resources": [
        "install.sh",
        "run.sh",
        "backend_runtime_test.sh",
        "manifest.txt",
        "hf:simaai/gemma3-siglip448-a16w4@Release-2.0.0",
        "hf:simaai/whisper-small-a16w8",
        "models/gte-small-local.tar.gz",
        "${DEVELOP_AARCH64_RESOURCE}"
    ],
    "resources-checksum": {
        "install.sh": "${INSTALL_CHECKSUM}",
        "run.sh": "${RUN_CHECKSUM}",
        "backend_runtime_test.sh": "${BACKEND_TEST_CHECKSUM}",
        "manifest.txt": "${MANIFEST_CHECKSUM}",
        "models/gte-small-local.tar.gz": "${GTE_CHECKSUM}",
        "${DEVELOP_AARCH64_RESOURCE}": "${DEVELOP_AARCH64_CHECKSUM}"
    },
    "selectable-resources": [
        {
            "name": "Demo Web App",
            "url": "",
            "resource": "simaai-genai-demo.tar.gz",
            "checksum": "${DEMO_APP_CHECKSUM}"
        }
    ],
    "size": {
        "download": "30GB",
        "install": "40GB"
    },
    "installation": {
        "script": "${METADATA_INSTALL_SCRIPT}",
        "post-message": "[bold]To run this package:[/bold]\\n\\n1. [green]cd simaai-genai-demo[/green]\\n2. [green]./run.sh[/green] (in web interaction mode)\\n3. [green]./run.sh -cli[/green] (in CLI mode) \\n\\n"
    }
}
EOF

cat > "${STAGING_DIR}/metadata-minimal.json" <<EOF
{
    "name": "llima",
    "version": "${METADATA_VERSION}",
    "release": "stable",
    "description": "SiMa.ai Lean Language & Image Modalix Application",
    "platforms": [
${METADATA_PLATFORMS}
    ],
    "resources": [
        "install.sh",
        "run.sh",
        "backend_runtime_test.sh",
        "manifest.txt",
        "hf:simaai/gemma3-siglip448-a16w4@Release-2.0.0",
        "hf:simaai/whisper-small-a16w8",
        "models/gte-small-local.tar.gz",
        "${MASTER_AARCH64_RESOURCE}"
    ],
    "resources-checksum": {
        "install.sh": "${INSTALL_CHECKSUM}",
        "run.sh": "${RUN_CHECKSUM}",
        "backend_runtime_test.sh": "${BACKEND_TEST_CHECKSUM}",
        "manifest.txt": "${MANIFEST_CHECKSUM}",
        "models/gte-small-local.tar.gz": "${GTE_CHECKSUM}",
        "${MASTER_AARCH64_RESOURCE}": "${MASTER_AARCH64_CHECKSUM}"
    },
    "selectable-resources": [],
    "size": {
        "download": "30GB",
        "install": "40GB"
    },
    "installation": {
        "script": "${METADATA_INSTALL_SCRIPT}",
        "post-message": "[bold]To run this package:[/bold]\\n\\n1. [green]./run.sh[/green] (in web interaction mode)\\n2. [green]./run.sh -cli[/green] (in CLI mode) \\n\\n"
    }
}
EOF

cat > "${STAGING_DIR}/metadata-develop-minimal.json" <<EOF
{
    "name": "llima",
    "version": "${METADATA_VERSION}",
    "release": "develop",
    "description": "SiMa.ai Lean Language & Image Modalix Application",
    "platforms": [
${METADATA_PLATFORMS}
    ],
    "resources": [
        "install.sh",
        "run.sh",
        "backend_runtime_test.sh",
        "manifest.txt",
        "hf:simaai/gemma3-siglip448-a16w4@Release-2.0.0",
        "hf:simaai/whisper-small-a16w8",
        "models/gte-small-local.tar.gz",
        "${DEVELOP_AARCH64_RESOURCE}"
    ],
    "resources-checksum": {
        "install.sh": "${INSTALL_CHECKSUM}",
        "run.sh": "${RUN_CHECKSUM}",
        "backend_runtime_test.sh": "${BACKEND_TEST_CHECKSUM}",
        "manifest.txt": "${MANIFEST_CHECKSUM}",
        "models/gte-small-local.tar.gz": "${GTE_CHECKSUM}",
        "${DEVELOP_AARCH64_RESOURCE}": "${DEVELOP_AARCH64_CHECKSUM}"
    },
    "selectable-resources": [],
    "size": {
        "download": "30GB",
        "install": "40GB"
    },
    "installation": {
        "script": "${METADATA_INSTALL_SCRIPT}",
        "post-message": "[bold]To run this package:[/bold]\\n\\n1. [green]./run.sh[/green] (in web interaction mode)\\n2. [green]./run.sh -cli[/green] (in CLI mode) \\n\\n"
    }
}
EOF

cat > "${STAGING_DIR}/metadata-full.json" <<EOF
{
    "name": "llima",
    "version": "${METADATA_VERSION}",
    "release": "stable",
    "description": "SiMa.ai Lean Language & Image Modalix Application",
    "platforms": [
${METADATA_PLATFORMS}
    ],
    "resources": [
        "install.sh",
        "run.sh",
        "backend_runtime_test.sh",
        "manifest.txt",
        "hf:simaai/gemma3-siglip448-a16w4@Release-2.0.0",
        "hf:simaai/whisper-small-a16w8",
        "models/gte-small-local.tar.gz",
        "${MASTER_AARCH64_RESOURCE}",
        "simaai-genai-demo.tar.gz"
    ],
    "resources-checksum": {
        "install.sh": "${INSTALL_CHECKSUM}",
        "run.sh": "${RUN_CHECKSUM}",
        "backend_runtime_test.sh": "${BACKEND_TEST_CHECKSUM}",
        "manifest.txt": "${MANIFEST_CHECKSUM}",
        "models/gte-small-local.tar.gz": "${GTE_CHECKSUM}",
        "${MASTER_AARCH64_RESOURCE}": "${MASTER_AARCH64_CHECKSUM}",
        "simaai-genai-demo.tar.gz": "${DEMO_APP_CHECKSUM}"
    },
    "selectable-resources": [],
    "size": {
        "download": "30GB",
        "install": "40GB"
    },
    "installation": {
        "script": "${METADATA_INSTALL_SCRIPT}",
        "post-message": "[bold]To run this package:[/bold]\\n\\n1. [green]./run.sh[/green] (in web interaction mode)\\n2. [green]./run.sh -cli[/green] (in CLI mode) \\n\\n"
    }
}
EOF

cat > "${STAGING_DIR}/metadata-develop-full.json" <<EOF
{
    "name": "llima",
    "version": "${METADATA_VERSION}",
    "release": "develop",
    "description": "SiMa.ai Lean Language & Image Modalix Application",
    "platforms": [
${METADATA_PLATFORMS}
    ],
    "resources": [
        "install.sh",
        "run.sh",
        "backend_runtime_test.sh",
        "manifest.txt",
        "hf:simaai/gemma3-siglip448-a16w4@Release-2.0.0",
        "hf:simaai/whisper-small-a16w8",
        "models/gte-small-local.tar.gz",
        "${DEVELOP_AARCH64_RESOURCE}",
        "simaai-genai-demo.tar.gz"
    ],
    "resources-checksum": {
        "install.sh": "${INSTALL_CHECKSUM}",
        "run.sh": "${RUN_CHECKSUM}",
        "backend_runtime_test.sh": "${BACKEND_TEST_CHECKSUM}",
        "manifest.txt": "${MANIFEST_CHECKSUM}",
        "models/gte-small-local.tar.gz": "${GTE_CHECKSUM}",
        "${DEVELOP_AARCH64_RESOURCE}": "${DEVELOP_AARCH64_CHECKSUM}",
        "simaai-genai-demo.tar.gz": "${DEMO_APP_CHECKSUM}"
    },
    "selectable-resources": [],
    "size": {
        "download": "30GB",
        "install": "40GB"
    },
    "installation": {
        "script": "${METADATA_INSTALL_SCRIPT}",
        "post-message": "[bold]To run this package:[/bold]\\n\\n1. [green]./run.sh[/green] (in web interaction mode)\\n2. [green]./run.sh -cli[/green] (in CLI mode) \\n\\n"
    }
}
EOF

cat > "${STAGING_DIR}/metadata-selection.json" <<EOF
{
    "name": "llima",
    "version": "${METADATA_VERSION}",
    "release": "stable",
    "description": "SiMa.ai Lean Language & Image Modalix Application",
    "platforms": [
${METADATA_PLATFORMS}
    ],
    "resources": [
        "install.sh",
        "run.sh",
        "backend_runtime_test.sh",
        "manifest.txt",
        "hf:simaai/whisper-small-a16w8",
        "models/gte-small-local.tar.gz",
        "${MASTER_AARCH64_RESOURCE}"
    ],
    "resources-checksum": {
        "install.sh": "${INSTALL_CHECKSUM}",
        "run.sh": "${RUN_CHECKSUM}",
        "backend_runtime_test.sh": "${BACKEND_TEST_CHECKSUM}",
        "manifest.txt": "${MANIFEST_CHECKSUM}",
        "models/gte-small-local.tar.gz": "${GTE_CHECKSUM}",
        "${MASTER_AARCH64_RESOURCE}": "${MASTER_AARCH64_CHECKSUM}"
    },
    "selectable-resources": [
        {
            "name": "Demo Web App",
            "url": "",
            "resource": "simaai-genai-demo.tar.gz",
            "checksum": "${DEMO_APP_CHECKSUM}"
        },
        {
            "name": "gemma3-siglip448-a16w4 (Image/Text To Text)",
            "url": "https://huggingface.co/simaai/gemma3-siglip448-a16w4",
            "resource": "hf:simaai/gemma3-siglip448-a16w4@Release-2.0.0"
        },
        {
            "name": "llava-1.5-7b-hf-a16w4 LLM model (Image/Text To Text)",
            "url": "https://huggingface.co/simaai/llava-1.5-7b-hf-a16w4",
            "resource": "hf:simaai/llava-1.5-7b-hf-a16w4@Release-2.0.0"
        },
        {
            "name": "Mistral-7B-Instruct-v0.3-a16w4 LLM model (Text to Text)",
            "url": "https://huggingface.co/simaai/Mistral-7B-Instruct-v0.3-a16w4",
            "resource": "hf:simaai/Mistral-7B-Instruct-v0.3-a16w4@Release-2.0.0"
        },
        {
            "name": "Llama-3.1-8B-Instruct-a16w4 (Text To Text)",
            "url": "https://huggingface.co/simaai/Llama-3.1-8B-Instruct-a16w4",
            "resource": "hf:simaai/Llama-3.1-8B-Instruct-a16w4@Release-2.0.0"
        },
        {
            "name": "Llama-2-7b-chat-hf-a16w4 (Text To Text)",
            "url": "https://huggingface.co/simaai/llama-2-7b-chat-hf-a16w4",
            "resource": "hf:simaai/Llama-2-7b-chat-hf-a16w4@Release-2.0.0"
        },
        {
            "name": "Llama-3.2-3B-Instruct-a16w4 (Text To Text)",
            "url": "https://huggingface.co/simaai/llama-3.2-3B-Instruct-a16w4",
            "resource": "hf:simaai/Llama-3.2-3B-Instruct-a16w4@Release-2.0.0"
        },
        {
            "name": "phi-3.5-mini-instruct-a16w4 LLM model (Text to Text)",
            "url": "https://huggingface.co/simaai/phi-3.5-mini-instruct-a16w4",
            "resource": "hf:simaai/phi-3.5-mini-instruct-a16w4@Release-2.0.0"
        }
    ],
    "size": {
        "download": "30GB",
        "install": "40GB"
    },
    "installation": {
        "script": "${METADATA_INSTALL_SCRIPT}",
        "post-message": "[bold]To run this package:[/bold]\\n\\n1. [green]./run.sh[/green] (in web interaction mode)\\n2. [green]./run.sh -cli[/green] (in CLI mode) \\n\\n"
    }
}
EOF

echo "✅ Staging completed in ${STAGING_DIR}"
echo "   Metadata version: ${METADATA_VERSION}"
if [[ "${PROD_MODE}" == true ]]; then
    echo "   Mode: production (--prod, escaped %2B resource names)"
else
    echo "   Mode: local (decoded + resource names)"
fi
echo "   - ${STAGING_DIR}/metadata.json"
echo "   - ${STAGING_DIR}/metadata-develop.json"
echo "   - ${STAGING_DIR}/metadata-minimal.json"
echo "   - ${STAGING_DIR}/metadata-develop-minimal.json"
echo "   - ${STAGING_DIR}/metadata-full.json"
echo "   - ${STAGING_DIR}/metadata-develop-full.json"
echo "   - ${STAGING_DIR}/metadata-selection.json"
echo "   - ${STAGING_DIR}/manifest.txt"
echo "   - ${STAGING_DIR}/models/gte-small-local.tar.gz"
echo "   - ${STAGING_DIR}/simaai-genai-demo.tar.gz"
echo "   - ${STAGING_DIR}/backend_runtime_test.sh"
