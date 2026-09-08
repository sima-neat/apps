#!/usr/bin/env bash
# Create a test gallery.bin from a directory of reference face images.
#
# Directory layout expected:
#   <images_dir>/
#     Alice/    ← person name (used as identity label)
#       img1.jpg
#       img2.jpg
#     Bob/
#       img1.jpg
#
# Usage:
#   ./create_test_gallery.sh <images_dir> [output_gallery.bin]
#
# The script copies the whole images directory to the Modalix device, runs
# face-recognizer --enroll (which walks sub-directories, using each sub-directory
# name as the identity label), and copies the resulting gallery back.
set -euo pipefail

IMAGES_DIR="${1:-}"
OUTPUT="${2:-tests/test_data/test_gallery.bin}"

if [[ -z "${IMAGES_DIR}" ]]; then
    echo "Usage: $0 <images_dir> [output_gallery.bin]" >&2
    exit 1
fi

if [[ ! -d "${IMAGES_DIR}" ]]; then
    echo "ERROR: images directory not found: ${IMAGES_DIR}" >&2
    exit 1
fi

DEVICE="${SIMA_DEVICE:?ERROR: set SIMA_DEVICE=sima@<device-ip> before running this script}"
APPS_BIN="/workspace/sima-neat/apps/build/examples/face-recognition/face-recognizer_cpp"
GALLERY_BIN="${APPS_BIN}/face-recognizer"
REMOTE_TMP="/tmp/face_recog_test_data"

# Resolve SDK container for SSH key
SDK_CONTAINER="$(docker ps --format '{{.Names}}' | grep -E 'sima-neat-sdk|sdk' | head -1)"
if [[ -z "${SDK_CONTAINER}" ]]; then
    echo "ERROR: SDK container not running" >&2
    exit 1
fi

SSH_KEY="$(docker inspect "${SDK_CONTAINER}" \
    --format '{{range .Mounts}}{{if eq .Destination "/root/.ssh"}}{{.Source}}{{end}}{{end}}')/id_rsa"

sima_ssh() {
    ssh -i "${SSH_KEY}" -o StrictHostKeyChecking=no "${DEVICE}" "$@"
}
sima_scp_to() {
    scp -i "${SSH_KEY}" -o StrictHostKeyChecking=no -r "$1" "${DEVICE}:$2"
}
sima_scp_from() {
    scp -i "${SSH_KEY}" -o StrictHostKeyChecking=no "${DEVICE}:$1" "$2"
}

# Verify at least one person sub-directory with images exists before uploading
has_images=0
while IFS= read -r person_dir; do
    [[ -d "${person_dir}" ]] || continue
    image_count="$(find "${person_dir}" -maxdepth 1 -type f \
        \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)"
    if [[ "${image_count}" -gt 0 ]]; then
        has_images=1
        break
    fi
done < <(find "${IMAGES_DIR}" -mindepth 1 -maxdepth 1 -type d | sort)

if [[ "${has_images}" -eq 0 ]]; then
    echo "ERROR: no person directories with images found in ${IMAGES_DIR}" >&2
    exit 1
fi

echo "Preparing remote workspace..."
REMOTE_IMAGES="${REMOTE_TMP}/images"
sima_ssh "rm -rf '${REMOTE_TMP}' && mkdir -p '${REMOTE_IMAGES}'"

echo "Uploading images to device..."
sima_scp_to "${IMAGES_DIR}/." "${REMOTE_IMAGES}/"

REMOTE_GALLERY="${REMOTE_TMP}/test_gallery.bin"
echo "Running face-recognizer --enroll on device..."
sima_ssh "QT_QPA_PLATFORM=offscreen '${GALLERY_BIN}' --enroll \
    --images '${REMOTE_IMAGES}' --gallery '${REMOTE_GALLERY}'"

mkdir -p "$(dirname "${OUTPUT}")"
echo "Copying gallery.bin to ${OUTPUT}..."
sima_scp_from "${REMOTE_GALLERY}" "${OUTPUT}"

echo ""
echo "Done. Test gallery written to: ${OUTPUT}"
echo ""
echo "To run the e2e test locally:"
echo "  export SIMANEAT_APPS_TEST_MODELS_DIR=/path/to/models"
echo "  export SIMANEAT_APPS_TEST_GALLERY_BIN=${OUTPUT}"
echo "  export SIMANEAT_TEST_RTSP_H264_URL=rtsp://<host>:<port>/<stream>"
echo "  ctest --test-dir build -L e2e -R face-recognizer --output-on-failure"
