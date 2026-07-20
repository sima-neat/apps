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

This example runs folder-based object detection with a compiled DETR model. Each image is resized with preserved aspect ratio, normalized, inferred, and decoded into object boxes and class scores.

The model emits classification logits and normalized boxes for a fixed set of queries. The example applies softmax to the logits, sigmoid to the boxes, filters by confidence, optionally keeps only the COCO `person` class, maps detections to the original image, and writes annotated output.

## Preview

![DETR object detector preview](../../../portal/assets/examples/object-detection/detr-object-detector/image.png)

## Prerequisites

- `sima-cli` ([documentation](https://developer.sima.ai/software/tools/sima-cli/)) on a supported Modalix or DevKit target.

## Install Apps

1. Choose a version from the [Neat Apps releases](https://github.com/sima-neat/apps/releases).
2. Install the selected version and enter the installed bundle. We recommend using the latest release:

```bash
sima-cli neat install apps@<release-version>
cd prebuilt-apps
```

Run the remaining commands from `prebuilt-apps/`.

## Prepare the Model

| Model | Role | Source |
| --- | --- | --- |
| `detr_resnet50_modified_class_embed_bbox_embed_mpk.tar.gz` | Default | Direct artifact |

Set `PLATFORM_VERSION` to the `platform-version` value recorded in `manifest.json`.

```bash
export PLATFORM_VERSION="<platform-version>"
mkdir -p models
cd models
sima-cli download "https://docs.sima.ai/pkg_downloads/SDK${PLATFORM_VERSION}/models/modalix/detr_resnet50_modified_class_embed_bbox_embed_mpk.tar.gz"
cd ..
```

Set `model.path` in the config to the downloaded package.

## Configure

Edit `examples/object-detection/detr-object-detector/src/common/config.yaml`.

```yaml
model:
  path: <model-path>

io:
  input_dir: assets/datasets/coco
  output_dir: sandbox/detr-object-detector

decode:
  confidence_threshold: 0.70
  person_only: false
```

## Run

### C++

```bash
./examples/object-detection/detr-object-detector/src/cpp/pre-built/detr-object-detector \
  --config examples/object-detection/detr-object-detector/src/common/config.yaml
```

### Python

```bash
source ~/pyneat/bin/activate
pip install -r examples/object-detection/detr-object-detector/src/python/requirements.txt
python3 examples/object-detection/detr-object-detector/src/python/main.py \
  --config examples/object-detection/detr-object-detector/src/common/config.yaml
```

## Troubleshooting

- Verify `model.path` if model loading fails.
- Lower `decode.confidence_threshold` if expected objects are missing.
- Confirm the compiled package still uses the documented `1333x800` frame.
- Confirm `io.output_dir` can be created.

## Source Files

- C++ reference source: `src/cpp/main.cpp`
- Python source: `src/python/main.py`
- Shared config: `src/common/config.yaml`

The packaged C++ source is an implementation reference. Run the executable under `src/cpp/pre-built/`; the installed bundle does not include CMake files.

## Development From Source

To modify, compile, or test this example, use the [Apps contributor workflow](https://github.com/sima-neat/apps/blob/main/CONTRIBUTING.md).
