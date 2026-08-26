#!/usr/bin/env bash
# smoke_test.sh — End-to-end sanity check on the DevKit.
# Runs model-test, enroll, and a short recognizer --test pass on a clip.
#
# Usage (run on DevKit):
#   ./scripts/smoke_test.sh --test-image <path.jpg> --test-video <clip.mp4>
#
# Expected: exits 0 if all checks pass, non-zero on first failure.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG="${DEPLOY_DIR}/config.yaml"
TEST_IMAGE=""
TEST_VIDEO=""
GALLERY_IMAGES=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --test-image)    TEST_IMAGE="$2";     shift 2 ;;
    --test-video)    TEST_VIDEO="$2";     shift 2 ;;
    --gallery-images) GALLERY_IMAGES="$2"; shift 2 ;;
    *) echo "Unknown: $1"; exit 1 ;;
  esac
done

export SIMA_PROCESSCVU_RUN_TARGET="${SIMA_PROCESSCVU_RUN_TARGET:-EV74}"

PASS=0; FAIL=0
check() {
  local label="$1"; shift
  echo -n "  [CHECK] ${label} ... "
  if eval "$@" >/dev/null 2>&1; then
    echo "PASS"; (( PASS++ )) || true
  else
    echo "FAIL"; (( FAIL++ )) || true
  fi
}

echo "=== face-recognizer smoke test ==="
echo "  Deploy dir : ${DEPLOY_DIR}"
echo "  Config     : ${CONFIG}"
echo ""

# ── 1. Binary existence ───────────────────────────────────────────────────────
check "face-recognizer binary exists"  "[[ -f ${DEPLOY_DIR}/face-recognizer ]]"
check "face-enroll binary exists"      "[[ -f ${DEPLOY_DIR}/face-enroll ]]"
check "face-model-test binary exists"  "[[ -f ${DEPLOY_DIR}/face-model-test ]]"
check "SCRFD model exists"    "[[ -f ${DEPLOY_DIR}/models/scrfd_2.5g_bnkps.mla_mpk.tar.gz ]]"
check "ArcFace model exists"  "[[ -f ${DEPLOY_DIR}/models/w600k_r50.surgery_mpk.tar.gz ]]"

# ── 2. Unit tests ─────────────────────────────────────────────────────────────
if [[ -f "${DEPLOY_DIR}/face_recog_unit_test" ]]; then
  echo ""
  echo "[UNIT TESTS]"
  "${DEPLOY_DIR}/face_recog_unit_test"
  check "unit tests pass" "${DEPLOY_DIR}/face_recog_unit_test"
fi

# ── 3. Model-test with a real image ──────────────────────────────────────────
if [[ -n "${TEST_IMAGE}" ]]; then
  echo ""
  echo "[MODEL TEST]"
  check "SCRFD on test image" \
    "${DEPLOY_DIR}/face-model-test --config ${CONFIG} --model scrfd --image ${TEST_IMAGE}"
  check "ArcFace on test image" \
    "${DEPLOY_DIR}/face-model-test --config ${CONFIG} --model arcface --image ${TEST_IMAGE}"
fi

# ── 4. Enrollment ─────────────────────────────────────────────────────────────
if [[ -n "${GALLERY_IMAGES}" ]]; then
  echo ""
  echo "[ENROLLMENT]"
  TMP_GALLERY="/tmp/smoke_gallery.bin"
  check "enroll gallery" \
    "${DEPLOY_DIR}/face-enroll --config ${CONFIG} --images ${GALLERY_IMAGES} --gallery ${TMP_GALLERY}"
  check "gallery file written" "[[ -f ${TMP_GALLERY} ]]"
fi

# ── 5. Short recognizer --test run on video ───────────────────────────────────
if [[ -n "${TEST_VIDEO}" ]]; then
  echo ""
  echo "[RECOGNIZER TEST RUN]"
  LOG="/tmp/smoke_recognizer.log"
  "${DEPLOY_DIR}/face-recognizer" \
    --config "${CONFIG}" \
    --input  "${TEST_VIDEO}" \
    --gallery "${TMP_GALLERY:-${DEPLOY_DIR}/gallery.bin}" \
    --test \
    --no-display > "${LOG}" 2>&1 || true

  check "recognizer completed" "grep -q 'Done:' ${LOG}"
  check "FPS printed"          "grep -qE 'FPS|fps' ${LOG}"
  check "no crash (no SIGSEGV)" "! grep -q 'Segmentation fault' ${LOG}"
  echo "  Recognizer output (tail):"
  tail -20 "${LOG}" | sed 's/^/    /'
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=== Smoke test: ${PASS} passed, ${FAIL} failed ==="
[[ $FAIL -eq 0 ]] && exit 0 || exit 1
