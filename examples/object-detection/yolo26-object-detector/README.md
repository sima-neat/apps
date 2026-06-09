# YOLO26 Object Detector

## Metadata
| Field | Value |
| --- | --- |
| Category | object-detection |
| Difficulty | Beginner |
| Tags | object-detection, yolo26m, folder-inference |
| Languages | C++, Python |
| Status | experimental |
| Binary Name | yolo26-object-detector |
| Model | yolo26m-det-bf16-mla_tess-b1 [https://docs.sima.ai/pkg_downloads/SDK2.0.0/models/modalix/yolo26-detection/yolo26m-det-bf16-mla_tess-b1.tar.gz] |

## Concept
Minimal image-folder object detection pipeline using yolo26m. Each image is inferred, annotated with bounding boxes and labels, and written to an output folder. The pipeline demonstrates the NEAT API for model loading, inference, and result decoding.

## Preview
Snippet from a pipeline run:

![YOLO26 object detector preview](../../../assets/portal/object-detection/yolo26-object-detector/image.png)

## Supported Models
Supported model:
- `yolo26m-det-bf16-mla_tess-b1.tar.gz`

Download the supported model:

```bash
mkdir -p assets/models
cd assets/models

sima-cli download https://docs.sima.ai/pkg_downloads/SDK2.0.0/models/modalix/yolo26-detection/yolo26m-det-bf16-mla_tess-b1.tar.gz

cd ../..
```

## Prerequisites
- Installed Neat SDK.
- Model artifacts are user-managed and should be downloaded into `assets/models/`.
- Labels file: `examples/object-detection/yolo26-object-detector/src/common/coco_label.txt`

## Important Behavior
- C++ and Python read runtime values from `src/common/config.yaml`.
- Labels file is configured under `model.labels`.
- Output images are written as `.png` files.
- Use `runtime.profile` to print per-image and aggregate timing plus FPS.
- Use `runtime.num_runs` to repeat the image set for benchmarking.
- Use `output.overlay: false` to skip drawing bounding boxes and writing output images.

## Command-Line Options
### C++
- Invocation:
  `./build/examples/object-detection/yolo26-object-detector/yolo26-object-detector [--config <path>]`
- Optional arguments:
  `--config <path>`: YAML config path. Defaults to `src/common/config.yaml`.

### Python
- Invocation:
  `python examples/object-detection/yolo26-object-detector/src/python/main.py [--config <path>]`
- Optional arguments:
  `--config <path>`: YAML config path. Defaults to `src/common/config.yaml`.

## Build
### Build From The Apps Repo
```bash
cd <apps-repo-root>
./build.sh
```

Binary output:
```bash
./build/examples/object-detection/yolo26-object-detector/yolo26-object-detector
```

### Build This Example Directly With CMake
```bash
cd <apps-repo-root>/examples/object-detection/yolo26-object-detector
cmake -S cpp -B build
cmake --build build -j
```

Binary output:
```bash
./build/yolo26-object-detector
```

## Run
### C++
```bash
./build/examples/object-detection/yolo26-object-detector/yolo26-object-detector
```

### Python
```bash
source ~/pyneat/bin/activate
pip install -r examples/object-detection/yolo26-object-detector/src/python/requirements.txt
python examples/object-detection/yolo26-object-detector/src/python/main.py
```

## Testing
Run from the apps repository root:

```bash
cd <apps-repo-root>
```

### C++
Unit test:
```bash
./build/examples/object-detection/yolo26-object-detector/yolo26-object-detector_unit_test \
  ./build/examples/object-detection/yolo26-object-detector/yolo26-object-detector
```

E2E test:
```bash
SIMANEAT_APPS_TEST_MODELS_DIR="$PWD/assets/models" \
SIMANEAT_APPS_TEST_INPUT_DIR="$PWD/assets/test_images" \
SIMANEAT_APPS_TEST_TIMEOUT_MS=60000 \
./build/examples/object-detection/yolo26-object-detector/yolo26-object-detector_e2e_test \
  ./build/examples/object-detection/yolo26-object-detector/yolo26-object-detector
```

### Python
Unit test:
```bash
source ~/pyneat/bin/activate
pip install -r examples/object-detection/yolo26-object-detector/src/python/requirements.txt
pytest examples/object-detection/yolo26-object-detector/tests/python/test_unit.py -v
```

E2E test:
```bash
source ~/pyneat/bin/activate
pip install -r examples/object-detection/yolo26-object-detector/src/python/requirements.txt
SIMANEAT_APPS_TEST_MODELS_DIR="$PWD/assets/models" \
SIMANEAT_APPS_TEST_INPUT_DIR="$PWD/assets/test_images" \
SIMANEAT_APPS_TEST_TIMEOUT_MS=60000 \
SIMANEAT_APPS_TEST_REQUIRE_E2E=1 \
pytest examples/object-detection/yolo26-object-detector/tests/python/test_e2e.py -v
```

## Debugging Notes
- If detections are missing, validate label file ordering and score thresholds.
- If model load fails, verify `assets/models/yolo26m-det-bf16-mla_tess-b1.tar.gz` exists.
- Ensure input folder contains supported image extensions (`.jpg`, `.jpeg`, `.png`, `.bmp`).
- Use `--profile` to identify bottlenecks in the pipeline.
- Use `decode.score_threshold` and `decode.nms_iou` to tune detection sensitivity.

## Source Files
- C++ source: `src/cpp/main.cpp`
- C++ tests: `tests/cpp/test_unit.cpp`, `tests/cpp/test_e2e.cpp`
- Python source: `src/python/main.py`
- Python tests: `tests/python/test_unit.py`, `tests/python/test_e2e.py`
- Shared assets: `src/common/coco_label.txt`
