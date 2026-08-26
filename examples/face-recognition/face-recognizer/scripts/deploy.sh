#!/usr/bin/env bash
# deploy.sh — Cross-compile and deploy face-recognizer to the DevKit over NFS/SSH.
#
# Prerequisites:
#   - Neat SDK container running with NFS mounted at /workspace
#   - DevKit reachable at DEVKIT_IP (default: set via --ip or DEVKIT_IP env var)
#   - Models placed at: examples/face-recognition/face-recognizer/assets/models/
#       scrfd_2.5g_bnkps.mla_mpk.tar.gz, w600k_r50.surgery_mpk.tar.gz
#
# Usage:
#   ./scripts/deploy.sh [--ip <devkit-ip>] [--build-only] [--gallery <gallery.bin>]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOD_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"          # face-recognizer/
APPS_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)" # apps/
DEVKIT_IP="${DEVKIT_IP:-}"
DEVKIT_USER="${DEVKIT_USER:-sima}"
NFS_ROOT="/workspace"
BUILD_ONLY=0
GALLERY_FILE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ip)        DEVKIT_IP="$2"; shift 2 ;;
    --build-only) BUILD_ONLY=1; shift ;;
    --gallery)   GALLERY_FILE="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [[ -z "${DEVKIT_IP}" ]]; then
  echo "ERROR: DevKit IP not set. Pass --ip <ip> or export DEVKIT_IP=<ip>."
  exit 1
fi

echo "=== face-recognizer deploy ==="
echo "  APPS_ROOT  : ${APPS_ROOT}"
echo "  DEVKIT     : ${DEVKIT_USER}@${DEVKIT_IP}"

# ── Build ─────────────────────────────────────────────────────────────────────
BUILD_DIR="${APPS_ROOT}/build"
echo ""
echo "[1/3] Configuring CMake..."
# The SDK toolchain keeps multiarch cmake configs under lib/aarch64-linux-gnu/cmake/
# which cmake does not search automatically in cross-compile mode. Pass explicit
# _DIR vars and PKG_CONFIG_SYSROOT_DIR so GStreamer/OpenCV headers resolve
# from the sysroot rather than the host.
_ARCH_CMAKE="${SYSROOT:-/opt/toolchain/aarch64/modalix}/usr/lib/aarch64-linux-gnu/cmake"
PKG_CONFIG_SYSROOT_DIR="${SYSROOT:-/opt/toolchain/aarch64/modalix}" \
cmake -S "${APPS_ROOT}" \
      -B "${BUILD_DIR}" \
      -DCMAKE_BUILD_TYPE=Release \
      -DSIMANEAT_APPS_BUILD_CPP=ON \
      -DCMAKE_PREFIX_PATH="${SYSROOT:-/opt/toolchain/aarch64/modalix}/usr" \
      -DCMAKE_SYSROOT="${SYSROOT:-/opt/toolchain/aarch64/modalix}" \
      -DPKG_CONFIG_EXECUTABLE=/usr/bin/pkg-config \
      -DSimaLMM_DIR="${_ARCH_CMAKE}/SimaLMM" \
      -Dfmt_DIR="${_ARCH_CMAKE}/fmt" \
      -Dspdlog_DIR="${_ARCH_CMAKE}/spdlog" \
      -DNeatInternals_DIR="${_ARCH_CMAKE}/NeatInternals" \
      2>&1 | tail -5

echo "[2/3] Building face-recognizer, face-enroll, face-model-test..."
cmake --build "${BUILD_DIR}" --target face-recognizer face-enroll face-model-test \
      --parallel 2 2>&1 | tail -10

echo "[2/3] Build complete."

if [[ $BUILD_ONLY -eq 1 ]]; then
  echo "Build-only mode; skipping deploy."; exit 0
fi

# ── Deploy to DevKit via NFS ──────────────────────────────────────────────────
DEPLOY_DIR="${NFS_ROOT}/face-recognizer"
echo ""
echo "[3/3] Deploying to ${DEVKIT_USER}@${DEVKIT_IP}:${DEPLOY_DIR} ..."

# Create directories
ssh -o StrictHostKeyChecking=no "${DEVKIT_USER}@${DEVKIT_IP}" \
    "mkdir -p ${DEPLOY_DIR}/models ${DEPLOY_DIR}/gallery_images"

# Copy binaries
for BIN in face-recognizer face-enroll face-model-test; do
  BIN_PATH="${BUILD_DIR}/examples/face-recognition/face-recognizer_cpp/${BIN}"
  if [[ -f "${BIN_PATH}" ]]; then
    scp -o StrictHostKeyChecking=no "${BIN_PATH}" "${DEVKIT_USER}@${DEVKIT_IP}:${DEPLOY_DIR}/"
    echo "  Deployed: ${BIN}"
  else
    echo "  WARNING: binary not found: ${BIN_PATH}"
  fi
done

# Copy models — config.yaml uses models/ relative to DEPLOY_DIR so they land here.
MODELS_DIR="${MOD_ROOT}/models"
for MODEL in scrfd_2.5g_bnkps.mla_mpk.tar.gz w600k_r50.surgery_mpk.tar.gz; do
  if [[ -f "${MODELS_DIR}/${MODEL}" ]]; then
    scp -o StrictHostKeyChecking=no "${MODELS_DIR}/${MODEL}" \
        "${DEVKIT_USER}@${DEVKIT_IP}:${DEPLOY_DIR}/models/"
    echo "  Deployed model: ${MODEL}"
  else
    echo "  WARNING: model not found: ${MODELS_DIR}/${MODEL}"
  fi
done

# Copy config — paths in config.yaml are already relative to DEPLOY_DIR
# (models/ and gallery.bin), so only the display sink needs to be cleared.
CONF="${SCRIPT_DIR}/../src/common/config.yaml"
scp -o StrictHostKeyChecking=no "${CONF}" "${DEVKIT_USER}@${DEVKIT_IP}:${DEPLOY_DIR}/config.yaml"
ssh -o StrictHostKeyChecking=no "${DEVKIT_USER}@${DEVKIT_IP}" \
    "sed -i 's|sink:[[:space:]]*display|sink: \"\"|g' ${DEPLOY_DIR}/config.yaml"
echo "  Deployed: config.yaml"

# Copy gallery if provided
if [[ -n "${GALLERY_FILE}" && -f "${GALLERY_FILE}" ]]; then
  scp -o StrictHostKeyChecking=no "${GALLERY_FILE}" "${DEVKIT_USER}@${DEVKIT_IP}:${DEPLOY_DIR}/gallery.bin"
  echo "  Deployed: gallery.bin"
fi

echo ""
echo "=== Deploy complete ==="
echo ""
echo "Run on DevKit:"
echo "  ssh ${DEVKIT_USER}@${DEVKIT_IP}"
echo "  cd ${DEPLOY_DIR}"
echo "  export SIMA_PROCESSCVU_RUN_TARGET=EV74"
echo "  ./face-recognizer --config config.yaml --input rtsp://... --test"
