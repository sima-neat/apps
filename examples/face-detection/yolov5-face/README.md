# yolov5s-face Detection + Keypoints Pipeline

## Metadata
| Field | Value |
| --- | --- |
| Category | face-detection |
| Difficulty | Intermediate |
| Tags | face-detection, keypoints, yolov5-face, folder-inference |
| Languages | C++, Python |
| Status | experimental |
| Binary Name | yolov5-face |
| Model | yolov5s_face_raw_split |

## Concept
Image-folder face detection pipeline using `yolov5s_face_raw_split`. Each image is fed raw to the compiled MPK; the EV74 CVU preproc plugin (`simaaiprocesspreproc_1` in the MPK's `pipeline_sequence.json`) handles letterbox, resize, BGR→RGB, and INT8 quantize on-device. The six raw split heads (3 box heads at 18ch + 3 landmark heads at 30ch) are decoded on the host into face boxes plus 5 facial keypoints (left eye, right eye, nose, left mouth corner, right mouth corner). Detections are inverse-letterboxed back to source pixels and written as annotated PNGs.

Synchronous `model.run()` (`Session` + `Run` with `RunMode::Sync` in C++) — the data flow stays obvious in both language entrypoints, mirroring the rest of the apps repo's reference examples.

Host-side decode is required, not a choice. The MPK ships a 6-input `0_boxdecoder.json` config, but its `pipeline_sequence.json` ends at `detess_dequant`, and even when boxdecode is wired in, NEAT's BBOX wire format (24-byte `RawBox`: int32 `x, y, w, h`, float32 `score`, int32 `class_id`) has no slots for the 5 landmark coordinates — `SimaBoxDecode` would drop them. Decoding the raw heads on the host is the only way to keep landmarks for this MPK. The C++ decoder mirrors the math from [`python/compilation.py`](python/compilation.py) line-for-line; the Python decoder is filter-first (compute `sigmoid(obj) * sigmoid(cls)` over all anchors, threshold via `np.nonzero`, decode only survivors) which is functionally equivalent but skips ~99% of the per-anchor sigmoid + grid + landmark arithmetic on real inputs.

## Supported Models
Validated with: `yolov5s_face_raw_split`

The compiled package `yolov5s_face_raw_split_mpk.tar.gz` is shipped at `assets/models/yolov5s_face_raw_split_mpk.tar.gz`. To rebuild it from the ONNX source, see [examples/face-detection/yolov5-face/python/compilation.py](python/compilation.py) and run with `--postprocess yolov5face_split`.

## Prerequisites
- Installed NEAT SDK.
- Model artifacts are user-managed and should be downloaded into `assets/models/`.
- Labels file: `examples/face-detection/yolov5-face/common/face_label.txt` (single class: `face`).
- Test images: bundled at `assets/images/thermal_test/`.

## Important Behavior
- Both C++ and Python use named flags (`--model`, `--labels`, `--input-dir`, `--output-dir`).
- The compiled model's canvas is **800×800**; the on-device CVU preproc plugin scales and center-pads the input frame to that canvas before quantization. The host computes the same `(scale, pad_l, pad_t)` parameters only to inverse-map model-canvas coordinates back to original-image pixels.
- Maximum input frame size is **1280×720** (from the MPK's `0_preproc.json`). Frames larger than that are skipped with a warning.
- The on-device preproc kernel caches its parameters per input shape. Warmup runs the first real image so the cache is primed for the loop; an unprimed cache pays a one-time ~2 s reconfig on its first frame.
- Output images are written as `.png` files with green face boxes, score text, and 5 colored landmark dots per face (red, green, blue, magenta, orange — left eye, right eye, nose, left mouth corner, right mouth corner).
- Use `--profile` to print per-stage `preprocess / inference / decode / overlay+save / total` timing plus aggregate FPS.
- Use `--num-runs N` to repeat the image set N times for benchmarking.
- Use `--no-overlay` to skip drawing/saving for pure-throughput mode.
- The decode path is anchor-based and assumes the same anchors as `yolov5s_face.pt`. If you retrain with different anchors, update `_ANCHORS`/`kAnchors` in both `python/main.py` and `cpp/main.cpp`.

## Command-Line Options
### C++
- Invocation:
  `./build/examples/face-detection/yolov5-face/yolov5-face --model <model.tar.gz> --labels <labels.txt> --input-dir <dir> --output-dir <dir> [--min-score 0.25] [--nms-iou 0.45] [--profile] [--no-overlay] [--num-runs 1]`
- Required arguments:
  `--model <model.tar.gz>`, `--labels <labels.txt>`, `--input-dir <dir>`, `--output-dir <dir>`
- Optional arguments:
  `--min-score <float>` (default: `0.25`), `--nms-iou <float>` (default: `0.45`), `--profile`, `--no-overlay`, `--num-runs <int>` (default: `1`)

### Python
- Invocation:
  `python examples/face-detection/yolov5-face/python/main.py --model <model.tar.gz> --labels <labels.txt> --input-dir <dir> --output-dir <dir> [--min-score 0.25] [--nms-iou 0.45] [--profile] [--no-overlay] [--num-runs 1]`
- Required arguments:
  `--model <model.tar.gz>`, `--labels <labels.txt>`, `--input-dir <dir>`, `--output-dir <dir>`
- Optional arguments:
  `--min-score <float>` (default: `0.25`), `--nms-iou <float>` (default: `0.45`), `--profile`, `--no-overlay`, `--num-runs <int>` (default: `1`)

## Build
### Build From The Apps Repo
```bash
cd <apps-repo-root>
./build.sh
```

Binary output:
```bash
./build/examples/face-detection/yolov5-face/yolov5-face
```

### Build This Example Directly With CMake
```bash
cd <apps-repo-root>/examples/face-detection/yolov5-face
cmake -S cpp -B build
cmake --build build -j
```

Binary output:
```bash
./build/yolov5-face
```

## Run
### C++
```bash
./build/examples/face-detection/yolov5-face/yolov5-face \
  --model assets/models/yolov5s_face_raw_split_mpk.tar.gz \
  --labels examples/face-detection/yolov5-face/common/face_label.txt \
  --input-dir assets/images/thermal_test --output-dir tmp_output_folder
```

### Python
```bash
source ~/pyneat/bin/activate
pip install -r examples/face-detection/yolov5-face/python/requirements.txt
python examples/face-detection/yolov5-face/python/main.py \
  --model assets/models/yolov5s_face_raw_split_mpk.tar.gz \
  --labels examples/face-detection/yolov5-face/common/face_label.txt \
  --input-dir assets/images/thermal_test --output-dir tmp_output_folder
```

## Testing
Run from the apps repository root:

```bash
cd <apps-repo-root>
```

### C++
Unit test:
```bash
./build/examples/face-detection/yolov5-face/yolov5-face_unit_test \
  ./build/examples/face-detection/yolov5-face/yolov5-face
```

E2E test:
```bash
SIMANEAT_APPS_TEST_MODELS_DIR="$PWD/assets/models" \
SIMANEAT_APPS_TEST_INPUT_DIR="$PWD/assets/images/thermal_test" \
SIMANEAT_APPS_TEST_TIMEOUT_MS=60000 \
./build/examples/face-detection/yolov5-face/yolov5-face_e2e_test \
  ./build/examples/face-detection/yolov5-face/yolov5-face
```

### Python
Unit test:
```bash
source ~/pyneat/bin/activate
pip install -r examples/face-detection/yolov5-face/python/requirements.txt
pytest examples/face-detection/yolov5-face/python/tests/test_unit.py -v
```

E2E test:
```bash
source ~/pyneat/bin/activate
pip install -r examples/face-detection/yolov5-face/python/requirements.txt
SIMANEAT_APPS_TEST_MODELS_DIR="$PWD/assets/models" \
SIMANEAT_APPS_TEST_INPUT_DIR="$PWD/assets/images/thermal_test" \
SIMANEAT_APPS_TEST_TIMEOUT_MS=60000 \
SIMANEAT_APPS_TEST_REQUIRE_E2E=1 \
pytest examples/face-detection/yolov5-face/python/tests/test_e2e.py -v
```

## Debugging Notes
- If detections are missing, check `--min-score` (post-sigmoid `obj * cls`, typical range 0.2–0.6) and verify the model package exists at `assets/models/yolov5s_face_raw_split_mpk.tar.gz`.
- If keypoints look mis-anchored, the most likely cause is incorrect head pairing in `postprocess_yolov5face_split` (Python) / `decode_yolov5face_split` (C++). Print each output tensor's shape on the first frame and confirm the spatial sizes group as `{100×100, 50×50, 25×25}` with channel counts `{18, 30}` per group.
- If boxes look offset, the host's `letterbox_params(orig_w, orig_h, ...)` math must agree with the on-device preproc transform (CENTER pad, BILINEAR scale-to-fit). Both should produce identical `(scale, pad_l, pad_t)` for a given input size.
- If the first frame's `inference=` line shows hundreds of ms while subsequent frames are fast, the on-device preproc shape cache wasn't primed at warmup — make sure the warmup tensor matches the real input dims.
- Use `--profile` to identify whether the bottleneck is preprocess, decode, or overlay+save. On-device inference (MLA + preproc + detess) shows up as the `inference=` line.

## Source Files
- C++ source: `cpp/main.cpp`
- C++ tests: `cpp/tests/unit_test.cpp`, `cpp/tests/e2e_test.cpp`
- Python source: `python/main.py`
- Python tests: `python/tests/test_unit.py`, `python/tests/test_e2e.py`
- Compilation script (offline): `python/compilation.py`
- Shared assets: `common/face_label.txt`
