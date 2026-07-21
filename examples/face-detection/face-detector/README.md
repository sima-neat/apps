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

This example runs RetinaFace over an image folder. RetinaFace predicts face boxes, confidence scores, and five landmarks for eyes, nose, and mouth.

The example decodes the compiled model outputs, filters low-confidence candidates, applies non-maximum suppression, and writes annotated images.

## Preview

![Face detector preview](../../../portal/assets/examples/face-detection/face-detector/image.png)

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
| `retinaface_mobilenet25_mod_0_mpk.tar.gz` | Default | Direct artifact |

Check the installed platform version, then set `PLATFORM_VERSION` to the displayed `DISTRO_VERSION` value.

```bash
cat /etc/buildinfo
export PLATFORM_VERSION="<platform-version>"
mkdir -p models
cd models
sima-cli download "https://docs.sima.ai/pkg_downloads/SDK${PLATFORM_VERSION}/models/modalix/retinaface_mobilenet25_mod_0_mpk.tar.gz"
cd ..
```

Set `model.path` in the config to the downloaded package.

## Configure

Edit `examples/face-detection/face-detector/src/common/config.yaml`.

```yaml
model:
  path: <model-path>

io:
  input_dir: assets/datasets/coco
  output_dir: sandbox/face-detector

decode:
  confidence_threshold: 0.40
  landmarks: true
```

## Run

### C++

```bash
./examples/face-detection/face-detector/src/cpp/pre-built/face-detector \
  --config examples/face-detection/face-detector/src/common/config.yaml
```

### Python

```bash
source ~/pyneat/bin/activate
pip install -r examples/face-detection/face-detector/src/python/requirements.txt
python3 examples/face-detection/face-detector/src/python/main.py \
  --config examples/face-detection/face-detector/src/common/config.yaml
```

## Troubleshooting

- Verify `model.path` if model loading fails.
- Lower `decode.confidence_threshold` if expected faces are missing.
- Adjust `decode.nms_iou` or `decode.top_k` if duplicate boxes remain.
- Confirm `io.output_dir` can be created.

## Source Files

- C++ reference source: `src/cpp/main.cpp`
- Python source: `src/python/main.py`
- Shared config: `src/common/config.yaml`

The packaged C++ source is an implementation reference. Run the executable under `src/cpp/pre-built/`; the installed bundle does not include CMake files.

## Development From Source

To modify, compile, or test this example, use the [Apps contributor workflow](https://github.com/sima-neat/apps/blob/main/CONTRIBUTING.md).
