# DETR Object Detector

## Metadata
| Field | Value |
| --- | --- |
| Category | object-detection |
| Difficulty | Intermediate |
| Tags | detr, object-detection, folder-input, coco |
| Languages | C++, Python |
| Status | experimental |
| Binary Name | detr-object-detector |
| Model | detr_resnet50_modified_class_embed_bbox_embed |

## Concept
This example demonstrates folder-based object detection with a compiled **DETR** model. Each input image is resized with aspect-ratio preservation into the model frame, normalized with ImageNet mean and standard deviation, run through the Neat Library pipeline, and then decoded into object boxes and class scores.

The model emits two raw tensors: classification logits and normalized bounding boxes for a fixed set of object queries. The example applies `softmax` over the class logits, `sigmoid` over the box outputs, filters by confidence, optionally keeps only the COCO `person` class, maps detections back onto the original image, and writes annotated output images.

## Preview
Snippet from a pipeline run:

![DETR object detector preview](../../../portal/assets/examples/object-detection/detr-object-detector/image.png)

## Supported Models
Use the SDK platform version wherever `<platform-version>` appears.

Validated with: `detr_resnet50_modified_class_embed_bbox_embed`

Download the validated model:

```bash
mkdir -p models
cd models
sima-cli download https://docs.sima.ai/pkg_downloads/SDK<platform-version>/models/modalix/detr_resnet50_modified_class_embed_bbox_embed_mpk.tar.gz
cd ..
```

The command stores the model under `models/` as a repo-local convention. `model.path` can point to any readable model package path.

## Prerequisites
- Installed Neat Development Environment + Neat Library.
- Model artifacts are user-managed and should be downloaded into `models/`. Download the default model, or set `model.path` to another readable model package.

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
Edit `examples/object-detection/detr-object-detector/src/common/config.yaml`.

```yaml
model:
  path: <model-path>                                  # Path to the model package.

io:
  input_dir: assets/datasets/coco                                                # Folder containing input images.
  output_dir: sandbox/detr-object-detector                                     # Folder for annotated images.

decode:
  confidence_threshold: 0.70                                                   # Minimum object confidence.
  person_only: false                                                           # Keep only COCO person detections when true.
```

## Run
### C++
```bash
./build/examples/object-detection/detr-object-detector/detr-object-detector \
  --config examples/object-detection/detr-object-detector/src/common/config.yaml
```

### Python
```bash
source ~/pyneat/bin/activate
pip install -r examples/object-detection/detr-object-detector/src/python/requirements.txt
python3 examples/object-detection/detr-object-detector/src/python/main.py \
  --config examples/object-detection/detr-object-detector/src/common/config.yaml
```

## Testing
On the Modalix/DevKit board, run from the apps repository root:

```bash
cd <apps-repo-root>
```

### C++
Unit test:
```bash
ctest --test-dir build/examples/object-detection/detr-object-detector \
  -R 'detr-object-detector.unit' --output-on-failure -V
```

E2E test:
```bash
SIMANEAT_APPS_TEST_MODELS_DIR="$PWD/models" \
SIMANEAT_APPS_TEST_INPUT_DIR="$PWD/assets/datasets-test/coco" \
SIMANEAT_APPS_TEST_OUTPUT_DIR=/tmp \
SIMANEAT_APPS_TEST_TIMEOUT_MS=180000 \
ctest --test-dir build/examples/object-detection/detr-object-detector \
  -R 'detr-object-detector.e2e' --output-on-failure -V
```

### Python
Unit test:
```bash
source ~/pyneat/bin/activate
pip install -r examples/object-detection/detr-object-detector/src/python/requirements.txt
pip install pytest
export PYTHONPATH="$PWD"
pytest -c tests/pytest.ini --rootdir="$PWD" -m unit \
  examples/object-detection/detr-object-detector/tests/python/test_unit.py -v
```

E2E test:
```bash
source ~/pyneat/bin/activate
pip install -r examples/object-detection/detr-object-detector/src/python/requirements.txt
pip install pytest
export PYTHONPATH="$PWD"
SIMANEAT_APPS_TEST_MODELS_DIR="$PWD/models" \
SIMANEAT_APPS_TEST_INPUT_DIR="$PWD/assets/datasets-test/coco" \
SIMANEAT_APPS_TEST_OUTPUT_DIR=/tmp/detr-python-e2e \
SIMANEAT_APPS_TEST_TIMEOUT_MS=180000 \
SIMANEAT_APPS_TEST_REQUIRE_E2E=1 \
pytest -c tests/pytest.ini --rootdir="$PWD" -m e2e \
  examples/object-detection/detr-object-detector/tests/python/test_e2e.py -v
```

## Debugging Notes
- If the model fails to load, verify `model.path` points to a readable model package.
- First-run model initialization can exceed 60 seconds on some setups. Increase `SIMANEAT_APPS_TEST_TIMEOUT_MS` (for example `180000`) for e2e runs.
- If no detections appear, lower `decode.confidence_threshold` and confirm the input images contain supported COCO objects.
- If detections are visibly offset, verify the model frame assumptions (`1333x800`) still match the compiled package.
- If output writing fails, ensure `io.output_dir` exists or can be created.

## Source Files
- C++ source: `src/cpp/main.cpp`
- C++ tests: `tests/cpp/test_unit.cpp`, `tests/cpp/test_e2e.cpp`
- Python source: `src/python/main.py`
- Python tests: `tests/python/test_unit.py`, `tests/python/test_e2e.py`
