# YOLOv8 Instance Segmenter

## Metadata

| Field | Value |
| --- | --- |
| Category | segmentation |
| Difficulty | Intermediate |
| Tags | segmentation, yolov8, instance-segmentation, folder-inference |
| Languages | C++, Python |
| Status | experimental |
| Binary Name | yolov8-instance-segmenter |
| Model | yolo_v8n_seg |

## Concept

Offline YOLOv8 instance segmentation over image folders using YOLOv8 segmentation outputs and DetessDequant postprocessing.

## Preview

![Instance segmenter preview](../../../portal/assets/examples/segmentation/yolov8-instance-segmenter/image.jpg)

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

| Model package | Role | Model Zoo name |
| --- | --- | --- |
| `yolo_v8n_seg_mpk.tar.gz` | Default | `yolo_v8n_seg` |
| `yolo_v8s_seg_mpk.tar.gz` | Supported | `yolo_v8s_seg` |
| `yolo_v8m_seg_mpk.tar.gz` | Supported | `yolo_v8m_seg` |
| `yolo_v8l_seg_mpk.tar.gz` | Supported | `yolo_v8l_seg` |

Check the installed platform version, then set `PLATFORM_VERSION` to the displayed `DISTRO_VERSION` value. Replace `<model-name>` with a model from the table.

```bash
cat /etc/buildinfo
export PLATFORM_VERSION="<platform-version>"
mkdir -p models
cd models
sima-cli modelzoo -v "${PLATFORM_VERSION}" get <model-name>
cd ..
```

Set `model.path` in the config to the downloaded package.

## Configure

Edit `examples/segmentation/yolov8-instance-segmenter/src/common/config.yaml`.

```yaml
model:
  path: <model-path>

io:
  input_dir: assets/datasets/coco
  output_dir: sandbox/yolov8-instance-segmenter

decode:
  score_threshold: 0.25
```

## Run

### C++

```bash
./examples/segmentation/yolov8-instance-segmenter/src/cpp/pre-built/yolov8-instance-segmenter \
  --config examples/segmentation/yolov8-instance-segmenter/src/common/config.yaml
```

### Python

```bash
source ~/pyneat/bin/activate
pip install -r examples/segmentation/yolov8-instance-segmenter/src/python/requirements.txt
python3 examples/segmentation/yolov8-instance-segmenter/src/python/main.py \
  --config examples/segmentation/yolov8-instance-segmenter/src/common/config.yaml
```

## Troubleshooting

- Verify `model.path` if startup fails.
- Adjust `decode.score_threshold` if output is empty.
- Confirm `io.output_dir` is writable.

## Source Files

- C++ reference source: `src/cpp/main.cpp`
- Python source: `src/python/main.py`
- Shared config: `src/common/config.yaml`

The packaged C++ source is an implementation reference. Run the executable under `src/cpp/pre-built/`; the installed bundle does not include CMake files.

## Development From Source

To modify, compile, or test this example, use the [Apps contributor workflow](https://github.com/sima-neat/apps/blob/main/CONTRIBUTING.md).
