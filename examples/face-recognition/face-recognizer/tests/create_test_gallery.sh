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
# The script enrolls each sub-directory as one identity by running face-enroll
# on the Modalix device via SSH and copying the result back.
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

DEVICE="${SIMA_DEVICE:-sima@192.168.135.41}"
APPS_BIN="/workspace/sima-neat/apps/build/examples/face-recognition/face-recognizer_cpp"
ENROLL_BIN="${APPS_BIN}/face-enroll"
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

echo "Preparing remote workspace..."
sima_ssh "rm -rf '${REMOTE_TMP}' && mkdir -p '${REMOTE_TMP}/images'"

GALLERY_ARGS=()
while IFS= read -r person_dir; do
    person="$(basename "${person_dir}")"
    [[ -d "${person_dir}" ]] || continue
    image_count="$(find "${person_dir}" -maxdepth 1 -type f \
        \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)"
    if [[ "${image_count}" -eq 0 ]]; then
        echo "  SKIP ${person}: no images"
        continue
    fi
    echo "  Uploading ${image_count} image(s) for ${person}..."
    sima_ssh "mkdir -p '${REMOTE_TMP}/images/${person}'"
    sima_scp_to "${person_dir}/." "${REMOTE_TMP}/images/${person}/"
    GALLERY_ARGS+=("--name" "${person}" "--images" "${REMOTE_TMP}/images/${person}")
done < <(find "${IMAGES_DIR}" -mindepth 1 -maxdepth 1 -type d | sort)

if [[ "${#GALLERY_ARGS[@]}" -eq 0 ]]; then
    echo "ERROR: no person directories with images found in ${IMAGES_DIR}" >&2
    exit 1
fi

REMOTE_GALLERY="${REMOTE_TMP}/test_gallery.bin"
echo "Running face-enroll on device..."
sima_ssh "QT_QPA_PLATFORM=offscreen '${ENROLL_BIN}' ${GALLERY_ARGS[*]} \
    --output '${REMOTE_GALLERY}'"

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
