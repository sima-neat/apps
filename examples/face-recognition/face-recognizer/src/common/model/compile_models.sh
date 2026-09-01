#!/usr/bin/env bash
# compile_models.sh — Compile SCRFD + ArcFace ONNX models for SiMa Modalix.
#
# Run this script INSIDE the Neat Development Environment (SDK container).
# It activates the Model Compiler virtual environment and compiles both models
# with BF16 activations and MLA tessellation.
#
# Usage:
#   bash examples/face-recognition/scripts/compile_models.sh \
#     --models-dir <path-to-prepared-onnx-dir> \
#     --build-dir  <output-dir> \
#     [--calib-dir <calibration-images-dir>] \
#     [--scrfd-only | --arcface-only]
#
# Prerequisites:
#   - Neat Development Environment with Model Compiler installed.
#   - Prepared ONNX files (output of prepare_models.py / scrfd_to_mla.py / arcface_to_mla.py):
#       <models-dir>/scrfd_2.5g_bnkps.mla.onnx
#       <models-dir>/w600k_r50.surgery.onnx
#   - (Optional) calibration images in --calib-dir for real-data quantization.
#
# Output:
#   <build-dir>/scrfd_2.5g_bnkps.mla/<basename>_mpk.tar.gz
#   <build-dir>/w600k_r50.surgery/<basename>_mpk.tar.gz
#
# After compilation, copy the packages to the model assets directory:
#   cp <build-dir>/scrfd_2.5g_bnkps.mla/*_mpk.tar.gz \
#      examples/face-recognition/face-recognizer/assets/models/scrfd_2.5g_bnkps.mla_mpk.tar.gz
#   cp <build-dir>/w600k_r50.surgery/*_mpk.tar.gz \
#      examples/face-recognition/face-recognizer/assets/models/w600k_r50.surgery_mpk.tar.gz

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# src/common/model → src/common → src → face-recognizer → face-recognition → examples → apps
APPS_ROOT="$(cd "${SCRIPT_DIR}/../../../../../../.." && pwd)"
MOD_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"  # face-recognizer/

# ── Defaults ──────────────────────────────────────────────────────────────────
MODELS_DIR=""
BUILD_DIR=""
CALIB_DIR=""
COMPILE_SCRFD=1
COMPILE_ARCFACE=1
MC_VENV="${MODEL_COMPILER_VENV:-/sdk-extensions/model-compiler}"
QC_SCRIPT="${QUANTIZE_COMPILE_SCRIPT:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models-dir)   MODELS_DIR="$2";   shift 2 ;;
    --build-dir)    BUILD_DIR="$2";    shift 2 ;;
    --calib-dir)    CALIB_DIR="$2";    shift 2 ;;
    --scrfd-only)   COMPILE_ARCFACE=0; shift ;;
    --arcface-only) COMPILE_SCRFD=0;   shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [[ -z "${MODELS_DIR}" ]]; then
  echo "ERROR: --models-dir is required."
  echo "  Run prepare_models.py first to produce the prepared ONNX files."
  exit 1
fi
if [[ -z "${BUILD_DIR}" ]]; then
  BUILD_DIR="${MODELS_DIR}/build_bf16_mlatess"
  echo "  [info] --build-dir not set; using ${BUILD_DIR}"
fi

# ── Resolve quantize_compile.py ───────────────────────────────────────────────
if [[ -z "${QC_SCRIPT}" ]]; then
  # Standard SDK location
  CANDIDATE="${APPS_ROOT}/scripts/quantize_compile.py"
  if [[ ! -f "${CANDIDATE}" ]]; then
    # Fall back: search the model-compiler venv
    CANDIDATE="$(find "${MC_VENV}" -name "quantize_compile.py" 2>/dev/null | head -1)"
  fi
  QC_SCRIPT="${CANDIDATE}"
fi
if [[ -z "${QC_SCRIPT}" || ! -f "${QC_SCRIPT}" ]]; then
  echo "ERROR: quantize_compile.py not found."
  echo "  Set QUANTIZE_COMPILE_SCRIPT=/path/to/quantize_compile.py"
  exit 1
fi

# ── Activate model compiler ───────────────────────────────────────────────────
if [[ -f "${MC_VENV}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${MC_VENV}/bin/activate"
  echo "  [info] Model Compiler venv: ${MC_VENV}"
else
  echo "WARNING: Model Compiler venv not found at ${MC_VENV}."
  echo "  Set MODEL_COMPILER_VENV=<path> or activate manually before running."
fi

PYTHON="${MC_VENV}/bin/python3"
[[ -f "${PYTHON}" ]] || PYTHON="python3"

mkdir -p "${BUILD_DIR}"
echo "=== Face Recognition — Model Compilation ==="
echo "  Models dir : ${MODELS_DIR}"
echo "  Build dir  : ${BUILD_DIR}"
echo "  Calib dir  : ${CALIB_DIR:-<none — using random data>}"
echo "  Script     : ${QC_SCRIPT}"
echo ""

# ── Build helper ──────────────────────────────────────────────────────────────
compile_model() {
  local model_path="$1"
  local input_name="$2"
  local input_shape="$3"
  local output_names="$4"
  local label="$5"

  echo "── Compiling ${label} ──────────────────────────────"
  echo "   Model  : ${model_path}"
  echo "   Input  : ${input_name}  ${input_shape}"
  echo "   Outputs: ${output_names}"
  echo ""

  local calib_args=()
  if [[ -n "${CALIB_DIR}" && -d "${CALIB_DIR}" ]]; then
    calib_args=(--real_data --dataset_images "${CALIB_DIR}" --num_calib_samples 16)
  fi

  "${PYTHON}" "${QC_SCRIPT}" \
    --model_path    "${model_path}" \
    --model_format  onnx \
    --model_layout  NCHW \
    --input_names   "${input_name}" \
    --input_shapes  "${input_shape}" \
    --output_names  "${output_names}" \
    --device        modalix \
    --build_dir     "${BUILD_DIR}" \
    --bf16-weights \
    --bf16-activations \
    --mla-tesselation \
    "${calib_args[@]+"${calib_args[@]}"}"

  echo ""
  echo "   Done: ${label}"
  echo ""
}

# ── SCRFD 2.5G ────────────────────────────────────────────────────────────────
if [[ $COMPILE_SCRFD -eq 1 ]]; then
  SCRFD_ONNX="${MODELS_DIR}/scrfd_2.5g_bnkps.mla.onnx"
  if [[ ! -f "${SCRFD_ONNX}" ]]; then
    echo "ERROR: SCRFD ONNX not found: ${SCRFD_ONNX}"
    echo "  Run scrfd_to_mla.py first:  python3 scrfd_to_mla.py scrfd_2.5g_bnkps.onnx"
    exit 1
  fi
  compile_model \
    "${SCRFD_ONNX}" \
    "input.1" \
    "1,3,640,640" \
    "stride_8_cls stride_16_cls stride_32_cls stride_8_bbox stride_16_bbox stride_32_bbox stride_8_kps stride_16_kps stride_32_kps" \
    "SCRFD 2.5G"
fi

# ── ArcFace W600K R50 ─────────────────────────────────────────────────────────
if [[ $COMPILE_ARCFACE -eq 1 ]]; then
  ARCFACE_ONNX="${MODELS_DIR}/w600k_r50.surgery.onnx"
  if [[ ! -f "${ARCFACE_ONNX}" ]]; then
    echo "ERROR: ArcFace ONNX not found: ${ARCFACE_ONNX}"
    echo "  Run arcface_to_mla.py first:  python3 arcface_to_mla.py w600k_r50.onnx"
    exit 1
  fi
  compile_model \
    "${ARCFACE_ONNX}" \
    "input.1" \
    "1,3,112,112" \
    "683" \
    "ArcFace W600K R50"
fi

# ── Copy to models/ under face-recognizer ────────────────────────────────────
MODELS_OUT="${MOD_ROOT}/models"
mkdir -p "${MODELS_OUT}"

echo "=== Copying compiled packages to face-recognizer/models/ ==="

if [[ $COMPILE_SCRFD -eq 1 ]]; then
  SCRFD_PKG="$(find "${BUILD_DIR}/scrfd_2.5g_bnkps.mla"* -name "*_mpk.tar.gz" 2>/dev/null | head -1)"
  if [[ -n "${SCRFD_PKG}" ]]; then
    cp "${SCRFD_PKG}" "${MODELS_OUT}/scrfd_2.5g_bnkps.mla_mpk.tar.gz"
    echo "  Copied SCRFD → ${MODELS_OUT}/scrfd_2.5g_bnkps.mla_mpk.tar.gz"
  else
    echo "  WARNING: SCRFD compiled package not found in ${BUILD_DIR}"
  fi
fi

if [[ $COMPILE_ARCFACE -eq 1 ]]; then
  ARCFACE_PKG="$(find "${BUILD_DIR}/w600k_r50.surgery"* -name "*_mpk.tar.gz" 2>/dev/null | head -1)"
  if [[ -n "${ARCFACE_PKG}" ]]; then
    cp "${ARCFACE_PKG}" "${MODELS_OUT}/w600k_r50.surgery_mpk.tar.gz"
    echo "  Copied ArcFace → ${MODELS_OUT}/w600k_r50.surgery_mpk.tar.gz"
  else
    echo "  WARNING: ArcFace compiled package not found in ${BUILD_DIR}"
  fi
fi

echo ""
echo "=== Compilation complete ==="
echo "  Models ready at: ${ASSETS_DIR}/"
