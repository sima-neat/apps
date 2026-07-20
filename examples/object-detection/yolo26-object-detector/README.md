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

Minimal image-folder detection with a YOLO26 model. Each image is inferred, annotated with boxes and labels, and written to an output folder.

## Preview

![YOLO26 object detector preview](../../../portal/assets/examples/object-detection/yolo26-object-detector/image.png)

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

| Model file | Role |
| --- | --- |
| `yolo26m-det-bf16-mla_tess-b1.tar.gz` | Default |
| `yolo26n-det-bf16-mla_tess-b1.tar.gz` | Supported |
| `yolo26s-det-bf16-mla_tess-b1.tar.gz` | Supported |
| `yolo26l-det-bf16-mla_tess-b1.tar.gz` | Supported |
| `yolo26x-det-bf16-mla_tess-b1.tar.gz` | Supported |
| `yolo26m-det-bf16-b1.tar.gz` | Supported |
| `yolo26m-det-int8-b1.tar.gz` | Supported |

Set `PLATFORM_VERSION` to the `platform-version` value recorded in `manifest.json`. Replace `<model-file>` with a file from the table.

```bash
export PLATFORM_VERSION="<platform-version>"
mkdir -p models
cd models
sima-cli download "https://docs.sima.ai/pkg_downloads/SDK${PLATFORM_VERSION}/models/modalix/yolo26-detection/<model-file>"
cd ..
```

Set `model.path` in the config to the downloaded package.

## Configure

Edit `examples/object-detection/yolo26-object-detector/src/common/config.yaml`.

```yaml
model:
  path: <model-path>
  labels: examples/object-detection/yolo26-object-detector/src/common/coco_label.txt

io:
  input_dir: assets/datasets/coco
  output_dir: sandbox/yolo26-object-detector

decode:
  score_threshold: 0.40
  nms_iou: 0.60
```

## Run

### C++

```bash
./examples/object-detection/yolo26-object-detector/src/cpp/pre-built/yolo26-object-detector \
  --config examples/object-detection/yolo26-object-detector/src/common/config.yaml
```

### Python

```bash
source ~/pyneat/bin/activate
pip install -r examples/object-detection/yolo26-object-detector/src/python/requirements.txt
python3 examples/object-detection/yolo26-object-detector/src/python/main.py \
  --config examples/object-detection/yolo26-object-detector/src/common/config.yaml
```

## Troubleshooting

- Verify `model.path` and the labels file if detections are missing.
- Confirm the input folder contains `.jpg`, `.jpeg`, `.png`, or `.bmp` files.
- Adjust `decode.score_threshold` and `decode.nms_iou` when tuning detections.
- Use `--profile` to inspect pipeline timing.

## Source Files

- C++ reference source: `src/cpp/main.cpp`
- Python source: `src/python/main.py`
- Shared config and labels: `src/common/`

The packaged C++ source is an implementation reference. Run the executable under `src/cpp/pre-built/`; the installed bundle does not include CMake files.

## Development From Source

To modify, compile, or test this example, use the [Apps contributor workflow](https://github.com/sima-neat/apps/blob/main/CONTRIBUTING.md).
