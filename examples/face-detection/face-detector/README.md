# Face Detector

## Metadata
| Field | Value |
| --- | --- |
| Category | face-detection |
| Difficulty | Beginner |
| Tags | retinaface, face-detection |
| Languages | C++, Python |
| Status | experimental |
| Binary Name | face-detector |
| Model | retinaface_mobilenet25 |

## Concept
This example demonstrates folder-based face detection with **RetinaFace**, a one-stage dense detector designed for robust face localization across pose, scale, and occlusion conditions.

RetinaFace predicts:
- Face bounding boxes
- Face confidence scores
- Five facial landmarks (left eye, right eye, nose tip, left mouth corner, right mouth corner)

Compared with generic object detectors, RetinaFace is specialized for facial geometry and alignment-sensitive tasks. Landmark outputs make it useful not only for drawing detection overlays, but also for downstream steps such as face alignment, tracking initialization, quality filtering, and recognition pre-processing.

In this app, the compiled `retinaface_mobilenet25` package is run on images from an input folder, raw outputs are decoded into candidate detections, low-confidence candidates are filtered, overlapping boxes are merged with Non-Maximum Suppression (NMS), and the final boxes/landmarks are rendered to output images.

## Preview
Snippet from a pipeline run:

![Face detector preview](../../../assets/portal/face-detection/face-detector/image.png)

## Supported Models
Use the SDK platform version wherever `<platform-version>` appears.

Validated with: `retinaface_mobilenet25`

Download the validated model:

```bash
mkdir -p assets/models
cd assets/models
sima-cli download https://docs.sima.ai/pkg_downloads/SDK<platform-version>/models/modalix/retinaface_mobilenet25_mod_0_mpk.tar.gz
cd ../..
```

The command stores the model under `assets/models/` as a repo-local convention. `model.path` can point to any readable model package path.

## Prerequisites
- Installed Neat Development Environment + Neat Library.
- Model artifacts are user-managed and should be downloaded into `assets/models/`. Download the default model, or set `model.path` to another readable model package.

## Get The Apps Repo
Use the [Neat Development Environment](https://developer.sima.ai/software/getting-started/dev-environment/) with the [Neat Library](https://developer.sima.ai/software/getting-started/neat-library/) installed for setup and compilation.

Clone and build the apps repo inside the Neat Development Environment:

```bash
git clone https://github.com/sima-neat/apps.git
cd apps
./build.sh --clean
```

After building, run the example commands below on the Modalix/DevKit board.

## Configure
Edit `examples/face-detection/face-detector/src/common/config.yaml`.

```yaml
model:
  path: <model-path>                                  # Path to the model package.

io:
  input_dir: <input-dir>                    # Folder containing input images.
  output_dir: <output-dir>                  # Folder for annotated images.

decode:
  confidence_threshold: 0.40                                  # Minimum face confidence.
  landmarks: true                                             # Draw five facial landmarks.
```

## Run
### C++
```bash
./build/examples/face-detection/face-detector_cpp/face-detector \
  --config examples/face-detection/face-detector/src/common/config.yaml
```

### Python
```bash
source ~/pyneat/bin/activate
pip install -r examples/face-detection/face-detector/src/python/requirements.txt
python3 examples/face-detection/face-detector/src/python/main.py \
  --config examples/face-detection/face-detector/src/common/config.yaml
```

## Testing
On the Modalix/DevKit board, run from the apps repository root:

```bash
cd <apps-repo-root>
```

### C++
Unit test:
```bash
ctest --test-dir build/examples/face-detection/face-detector_cpp \
  -R 'face-detector.unit' --output-on-failure -V
```

E2E test:
```bash
SIMANEAT_APPS_TEST_MODELS_DIR="$PWD/assets/models" \
SIMANEAT_APPS_TEST_INPUT_DIR="$PWD/assets/test_images" \
SIMANEAT_APPS_TEST_OUTPUT_DIR=/tmp \
SIMANEAT_APPS_TEST_TIMEOUT_MS=60000 \
ctest --test-dir build/examples/face-detection/face-detector_cpp \
  -R 'face-detector.e2e' --output-on-failure -V
```

### Python
Unit test:
```bash
source ~/pyneat/bin/activate
pip install -r examples/face-detection/face-detector/src/python/requirements.txt
pip install pytest
export PYTHONPATH="$PWD"
pytest -c tests/pytest.ini --rootdir="$PWD" -m unit \
  examples/face-detection/face-detector/tests/python/test_unit.py -v
```

E2E test:
```bash
source ~/pyneat/bin/activate
pip install -r examples/face-detection/face-detector/src/python/requirements.txt
pip install pytest
export PYTHONPATH="$PWD"
SIMANEAT_APPS_TEST_MODELS_DIR="$PWD/assets/models" \
SIMANEAT_APPS_TEST_INPUT_DIR="$PWD/assets/test_images" \
SIMANEAT_APPS_TEST_OUTPUT_DIR=/tmp/retinaface-python-e2e \
SIMANEAT_APPS_TEST_TIMEOUT_MS=60000 \
SIMANEAT_APPS_TEST_REQUIRE_E2E=1 \
pytest -c tests/pytest.ini --rootdir="$PWD" -m e2e \
  examples/face-detection/face-detector/tests/python/test_e2e.py -v
```

Notes:
- Use an absolute path for `SIMANEAT_APPS_TEST_MODELS_DIR` in Python e2e. The test runs `main.py` with `cwd` set to the example directory.
- `SIMANEAT_APPS_TEST_INPUT_DIR` is supported by both C++ and Python e2e tests.
- If `SIMANEAT_APPS_TEST_INPUT_DIR` is not set, both e2e tests fall back to `assets/test_images`.

## Debugging Notes
- If the model fails to load, verify `model.path` points to a readable model package.
- If no detections appear, lower `decode.confidence_threshold` and confirm the input actually contains faces.
- If too many duplicate boxes appear, reduce `decode.nms_iou` and/or lower `decode.top_k`.
- If output writing fails, ensure `io.output_dir` exists or can be created.

## Source Files
- C++ source: `src/cpp/main.cpp`
- C++ tests: `tests/cpp/test_unit.cpp`, `tests/cpp/test_e2e.cpp`
- Python source: `src/python/main.py`
- Python tests: `tests/python/test_unit.py`, `tests/python/test_e2e.py`
