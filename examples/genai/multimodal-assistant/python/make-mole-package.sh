#!/bin/bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: make-mole-package.sh [--channel master|develop] [--prod]

Stages a MOLE package under ~/Downloads/mole-stage by:
1) copying install-mole.sh
2) downloading sima_lmm wheel from Artifactory via sima-cli
3) writing metadata.json

Environment overrides:
  MAIN_VERSION        (default: 2.1.0)
  MASTER_BUILD_ID     (default: 10)
  DEVELOPER_BUILD_ID  (default: 129)
  STAGING_DIR         (default: ~/Downloads/mole-stage)
EOF
}

CHANNEL="master"
PROD_MODE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --channel)
            [[ $# -ge 2 ]] || { echo "❌ --channel requires a value"; usage; exit 1; }
            CHANNEL="$2"
            shift 2
            ;;
        --prod)
            PROD_MODE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "❌ Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

if [[ "$CHANNEL" != "master" && "$CHANNEL" != "develop" ]]; then
    echo "❌ Invalid --channel value: $CHANNEL (expected master or develop)"
    exit 1
fi

MAIN_VERSION="${MAIN_VERSION:-2.1.0}"
MASTER_BUILD_ID="${MASTER_BUILD_ID:-12}"
DEVELOPER_BUILD_ID="${DEVELOPER_BUILD_ID:-135}"
STAGING_DIR="${STAGING_DIR:-$HOME/Downloads/mole-stage}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CHECKOUT_BRANCH="$(git -C "${SCRIPT_DIR}" branch --show-current)"
if [[ -z "${CHECKOUT_BRANCH}" ]]; then
    CHECKOUT_BRANCH="detached"
fi
CHECKOUT_BRANCH_SAFE="${CHECKOUT_BRANCH//[!A-Za-z0-9._-]/-}"
CHECKOUT_SHORT_HASH="$(git -C "${SCRIPT_DIR}" rev-parse --short HEAD)"
METADATA_VERSION="${MAIN_VERSION}-${CHECKOUT_BRANCH_SAFE}-${CHECKOUT_SHORT_HASH}"

if [[ "$CHANNEL" == "master" ]]; then
    BUILD_ID="$MASTER_BUILD_ID"
    RELEASE="stable"
else
    BUILD_ID="$DEVELOPER_BUILD_ID"
    RELEASE="develop"
fi

WHEEL_NAME="sima_lmm-${MAIN_VERSION}.dev0+${CHANNEL}.${BUILD_ID}-py3-none-any.whl"
WHEEL_NAME_ENC="${WHEEL_NAME/+/%2B}"
ARTIFACT_BASE_URL="https://artifacts.eng.sima.ai:443/artifactory/sima-pypi/swml-auto-lmm"
WHEEL_URL="${ARTIFACT_BASE_URL}/${WHEEL_NAME_ENC}"

if ! command -v sima-cli >/dev/null 2>&1; then
    echo "❌ sima-cli not found in PATH."
    exit 1
fi

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

mkdir -p "${STAGING_DIR}"
cd "${STAGING_DIR}"

# Remove legacy staged scripts from previous package formats.
rm -f "${STAGING_DIR}/install.sh" "${STAGING_DIR}/run.sh" "${STAGING_DIR}/backend_runtime_test.sh"
# Remove legacy MOLE wheel type from previous revisions.
rm -f "${STAGING_DIR}"/sima_lmm-*-cp311-cp311-linux_aarch64.whl

if [[ -f "${SCRIPT_DIR}/install-mole.sh" ]]; then
    cp -f "${SCRIPT_DIR}/install-mole.sh" "${STAGING_DIR}/install-mole.sh"
    chmod +x "${STAGING_DIR}/install-mole.sh"
else
    echo "❌ Missing required script: ${SCRIPT_DIR}/install-mole.sh"
    exit 1
fi

echo "⬇️  sima-cli -i download ${WHEEL_URL}"
sima-cli -i download "${WHEEL_URL}"

# Keep canonical on-disk artifact name decoded (+).
if [[ -f "${WHEEL_NAME_ENC}" ]]; then
    mv -f "${WHEEL_NAME_ENC}" "${WHEEL_NAME}"
fi

if [[ "${PROD_MODE}" == true ]]; then
    cp -f "${WHEEL_NAME}" "${WHEEL_NAME_ENC}"
    WHEEL_RESOURCE="${WHEEL_NAME_ENC}"
else
    WHEEL_RESOURCE="${WHEEL_NAME}"
fi

INSTALL_MOLE_CHECKSUM="$(sha256_file "${STAGING_DIR}/install-mole.sh")"
WHEEL_CHECKSUM="$(sha256_file "${STAGING_DIR}/${WHEEL_NAME}")"

cat > "${STAGING_DIR}/metadata.json" <<EOF
{
    "name": "mole",
    "version": "${METADATA_VERSION}",
    "release": "${RELEASE}",
    "description": "SiMa.ai MOLE package",
    "platforms": [
        {
            "type": "host",
            "os": ["mac", "linux"]
        }
    ],
    "resources": [
        "install-mole.sh",
        "${WHEEL_RESOURCE}"
    ],
    "resources-checksum": {
        "install-mole.sh": "${INSTALL_MOLE_CHECKSUM}",
        "${WHEEL_RESOURCE}": "${WHEEL_CHECKSUM}"
    },
    "installation": {
        "script": "bash ./install-mole.sh",
        "post-message": "[bold]MOLE package installed.[/bold]\\n\\n"
    }
}
EOF

echo "✅ MOLE package staged in ${STAGING_DIR}"
echo "   Metadata version: ${METADATA_VERSION}"
echo "   - ${STAGING_DIR}/metadata.json"
echo "   - ${STAGING_DIR}/${WHEEL_NAME}"
