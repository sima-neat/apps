# YOLO26 Object Detector

## Metadata
| Field | Value |
| --- | --- |
| Category | object-detection |
| Difficulty | Beginner |
| Tags | object-detection, yolo26, folder-inference |
| Languages | C++, Python |
| Status | experimental |
| Binary Name | yolo26-object-detector |
| Model | yolo26m-det-bf16-mla_tess-b1 |

## Concept
Minimal image-folder object detection pipeline using a YOLO26 detection model. Each image is inferred, annotated with bounding boxes and labels, and written to an output folder. The pipeline demonstrates the Neat Library API for model loading, inference, and result decoding.

## Preview
Snippet from a pipeline run:

![YOLO26 object detector preview](../../../assets/portal/object-detection/yolo26-object-detector/image.png)

## Supported Models
Use the SDK platform version wherever `<platform-version>` appears.

Default model: `yolo26m-det-bf16-mla_tess-b1.tar.gz`.

Download the default model:

```bash
mkdir -p assets/models
cd assets/models

sima-cli download https://docs.sima.ai/pkg_downloads/SDK<platform-version>/models/modalix/yolo26-detection/yolo26m-det-bf16-mla_tess-b1.tar.gz

cd ../..
```

The command stores the model under `assets/models/` as a repo-local convention. `model.path` can point to any readable model package path.

## Prerequisites
- Installed Neat Development Environment.
- Model artifacts are user-managed. Download the default model, or set `model.path` to another readable model package.
- Labels file: `examples/object-detection/yolo26-object-detector/src/common/coco_label.txt`

## Get The Apps Repo
Use the [Neat Development Environment](https://developer.sima.ai/software/getting-started/dev-environment/) for setup and compilation. Install the Neat Library first by following the [Neat Library guide](https://developer.sima.ai/software/getting-started/neat-library/).

Clone and build the apps repo in the Neat Development Environment:

```bash
git clone https://github.com/sima-neat/apps.git
cd apps
./build.sh --clean
```

After building, run the example commands below on the Modalix/DevKit board.

## Configure
Edit `examples/object-detection/yolo26-object-detector/src/common/config.yaml`.

```yaml
model:
  path: <model-path>                         # Path to the model package.
  labels: examples/object-detection/yolo26-object-detector/src/common/coco_label.txt

io:
  input_dir: assets/test_images                           # Folder containing input images.
  output_dir: sandbox/yolo26-object-detector              # Folder for annotated images.

decode:
  score_threshold: 0.40                                   # Minimum object confidence.
  nms_iou: 0.60                                           # Overlap threshold for NMS.
```

## Run
### C++
```bash
./build/examples/object-detection/yolo26-object-detector/yolo26-object-detector \
  --config examples/object-detection/yolo26-object-detector/src/common/config.yaml
```

### Python
```bash
source ~/pyneat/bin/activate
pip install -r examples/object-detection/yolo26-object-detector/src/python/requirements.txt
python3 examples/object-detection/yolo26-object-detector/src/python/main.py \
  --config examples/object-detection/yolo26-object-detector/src/common/config.yaml
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
- If model load fails, verify `model.path` points to a readable model package.
- Ensure input folder contains supported image extensions (`.jpg`, `.jpeg`, `.png`, `.bmp`).
- Use `--profile` to identify bottlenecks in the pipeline.
- Use `decode.score_threshold` and `decode.nms_iou` to tune detection sensitivity.

## Appendix: Additional Models
Other supported batch-1 YOLO26 detection models:
- `yolo26n-det-bf16-mla_tess-b1.tar.gz`
- `yolo26s-det-bf16-mla_tess-b1.tar.gz`
- `yolo26l-det-bf16-mla_tess-b1.tar.gz`
- `yolo26x-det-bf16-mla_tess-b1.tar.gz`
- `yolo26m-det-bf16-b1.tar.gz`
- `yolo26m-det-int8-b1.tar.gz`

Replace the default filename in the download command and `model.path`.

## Source Files
- C++ source: `src/cpp/main.cpp`
- C++ tests: `tests/cpp/test_unit.cpp`, `tests/cpp/test_e2e.cpp`
- Python source: `src/python/main.py`
- Python tests: `tests/python/test_unit.py`, `tests/python/test_e2e.py`
- Shared assets: `src/common/coco_label.txt`
