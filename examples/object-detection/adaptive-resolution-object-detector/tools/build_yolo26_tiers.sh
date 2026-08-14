#!/usr/bin/env bash
# Build YOLO26 detection models at multiple input resolutions -> int8 -> modalix
# MPK archives, one archive per tier, for adaptive-resolution-object-detector.
#
# Produces (default sizes 320 640 960), under the apps repo assets/models/:
#   yolo26n-<size>-det-int8-mla_tess-b1.tar.gz
#
# Requires the SiMa Model Compiler plus the model-surgery and quantize-compile
# helpers. Run inside the Neat Development Environment. Override any tool path,
# the size list, or the weights via the env vars below.
#
#   SIZES="320 640"  WEIGHTS=yolo26n.pt  ./build_yolo26_tiers.sh
#
# NOTE ON MODEL VARIANT: the core box-decode surgery tool
# (yolo26_boxdecode_surgery.py) bakes in the attention channel dims of the
# YOLO26 *nano* head, so this script defaults to yolo26n. Larger variants
# (s/m/l/x) have wider attention and are NOT supported by that surgery tool;
# use SiMa's pre-compiled packs for those. Only the P3/P4/P5 grid *sizes* are
# input-dependent, and this script derives them per size (see the surgery patch
# below), so any multiple-of-32 input size works for yolo26n.
set -euo pipefail

read -r -a SIZES <<< "${SIZES:-320 640 960}"
WEIGHTS="${WEIGHTS:-yolo26n.pt}"
VARIANT="${VARIANT:-yolo26n}"

APPS_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
WORK="${WORK:-$APPS_ROOT/.build/yolo26_tiers}"
MC="${MC:-/sdk-extensions/model-compiler/bin}"
CALIB="${CALIB:-$WORK/calib_images}"
OUT_DIR="${OUT_DIR:-$APPS_ROOT/assets/models}"

# Core helpers (override if your install differs).
SURGERY_SRC="${SURGERY_SRC:-/neat-resources/core-src/tools/yolo26_boxdecode_surgery.py}"
GUARD="${GUARD:-$HOME/.claude/skills/sima-model-surgery/scripts/model_surgery_guard.py}"
QUANTIZE="${QUANTIZE:-$HOME/.claude/skills/sima-model-quantize-compile/scripts/quantize_compile.py}"

export PATH="$MC:$PATH"
mkdir -p "$WORK/models" "$CALIB" "$OUT_DIR"
cd "$WORK"

# 1. Derive a size-aware surgery from the core tool. The core tool hardcodes the
#    640 P5 grid (20x20 / 400 tokens) in the attention rewrite and the 640 output
#    grids (80/40/20) in its validator; both are derived here from the input size.
SURGERY="$WORK/yolo26_boxdecode_surgery_anysize.py"
"$MC/python" - "$SURGERY_SRC" "$SURGERY" <<'PY'
import sys
src = open(sys.argv[1]).read()
old1 = ('        add_int64_initializer(model, qkv_shape, [1, 2, 128, 400])\n'
        '        add_int64_initializer(model, v4_shape, [1, 128, 20, 20])')
new1 = ('        _p5 = (model.graph.input[0].type.tensor_type.shape.dim[2].dim_value or 640) // 32\n'
        '        add_int64_initializer(model, qkv_shape, [1, 2, 128, _p5 * _p5])\n'
        '        add_int64_initializer(model, v4_shape, [1, 128, _p5, _p5])')
old2 = ('    expected = {\n'
        '        "bbox_0": [1, 4, 80, 80],\n'
        '        "bbox_1": [1, 4, 40, 40],\n'
        '        "bbox_2": [1, 4, 20, 20],\n'
        '        "class_logit_0": [1, 80, 80, 80],\n'
        '        "class_logit_1": [1, 80, 40, 40],\n'
        '        "class_logit_2": [1, 80, 20, 20],\n'
        '    }')
new2 = ('    _s = model.graph.input[0].type.tensor_type.shape.dim[2].dim_value or 640\n'
        '    _g8, _g16, _g32 = _s // 8, _s // 16, _s // 32\n'
        '    expected = {\n'
        '        "bbox_0": [1, 4, _g8, _g8],\n'
        '        "bbox_1": [1, 4, _g16, _g16],\n'
        '        "bbox_2": [1, 4, _g32, _g32],\n'
        '        "class_logit_0": [1, 80, _g8, _g8],\n'
        '        "class_logit_1": [1, 80, _g16, _g16],\n'
        '        "class_logit_2": [1, 80, _g32, _g32],\n'
        '    }')
assert src.count(old1) == 1 and src.count(old2) == 1, "core surgery tool layout changed; update this patch"
open(sys.argv[2], "w").write(src.replace(old1, new1).replace(old2, new2))
print("generated size-aware surgery:", sys.argv[2])
PY

# 2. Ultralytics export venv (isolated: newer torch than the compiler env).
if [ ! -d "$WORK/ul_venv" ]; then
  python3 -m venv "$WORK/ul_venv"
  "$WORK/ul_venv/bin/pip" install -q --upgrade pip
  "$WORK/ul_venv/bin/pip" install -q ultralytics onnx onnxsim
fi

# 3. Calibration images. Any representative set works; accuracy scales with it.
if [ -z "$(ls -A "$CALIB" 2>/dev/null)" ]; then
  for d in "$APPS_ROOT/assets/coco-images" "$APPS_ROOT/assets/test_images" "$APPS_ROOT/assets/images"; do
    [ -d "$d" ] && find "$d" -maxdepth 1 -type f \( -name '*.jpg' -o -name '*.png' \) \
      -exec cp -n {} "$CALIB/" \; || true
  done
fi
echo "calibration images: $(ls "$CALIB" 2>/dev/null | wc -l)"

for S in "${SIZES[@]}"; do
  echo "===== ${VARIANT} @ ${S}x${S} ====="
  base="${VARIANT}_${S}"

  "$WORK/ul_venv/bin/python" - "$WORK" "$WEIGHTS" "$S" "$base" <<'PY'
import sys, shutil, pathlib
from ultralytics import YOLO
work, weights, size, base = pathlib.Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4]
model = YOLO(weights)
path = model.export(format="onnx", imgsz=size, opset=17, simplify=False, dynamic=False, nms=False)
shutil.copy(path, work / "models" / f"{base}.onnx")
print("exported:", base)
PY

  "$MC/python" "$SURGERY" \
    --input  "$WORK/models/${base}.onnx" \
    --output "$WORK/models/${base}_raw_supported_einsum.onnx" \
    --attention-mode supported-einsum --check-mla-only

  # Audit is informational: the guard exits non-zero on any "unknown" op
  # (Constant/Identity are always unknown but benign). Abort only on a real
  # unsupported op.
  AUDIT_OUT="$("$MC/python" "$GUARD" audit-model \
    --model "$WORK/models/${base}_raw_supported_einsum.onnx" --dtype int8 2>&1 || true)"
  echo "$AUDIT_OUT" | grep -E "supported=" || true
  if echo "$AUDIT_OUT" | grep -qE "unsupported=[1-9]"; then
    echo "[audit] real unsupported ops for ${S}; aborting"; exit 1
  fi

  "$MC/python" "$QUANTIZE" \
    --model_path "$WORK/models/${base}_raw_supported_einsum.onnx" \
    --model_format onnx --model_layout NCHW \
    --input_names images --input_shapes "1,3,${S},${S}" \
    --output_names bbox_0 bbox_1 bbox_2 class_logit_0 class_logit_1 class_logit_2 \
    --device modalix --build_dir "$WORK/build_${S}" \
    --real_data --dataset_images "$CALIB" --num_calib_samples 128 \
    --calib_method mse --requant_mode sima --any_shape_on_mla --mla-tesselation

  cp -f "$WORK/build_${S}/${base}_raw_supported_einsum/${base}_raw_supported_einsum_mpk.tar.gz" \
        "$OUT_DIR/${VARIANT}-${S}-det-int8-mla_tess-b1.tar.gz"
  echo "OK -> $OUT_DIR/${VARIANT}-${S}-det-int8-mla_tess-b1.tar.gz"
done

echo
echo "Done. Point model.tiers in src/common/config.yaml at the archives above."
