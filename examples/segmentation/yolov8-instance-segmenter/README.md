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

Segments objects in a folder of images with YOLOv8 and saves annotated images with a colored mask for each detected instance.

## Preview

![Instance segmenter preview](../../../portal/assets/examples/segmentation/yolov8-instance-segmenter/image.jpg)

## Prerequisites

- `sima-cli` ([documentation](https://developer.sima.ai/software/tools/sima-cli/)) on a supported Modalix or DevKit target.

## Install Apps

Install the latest Neat Apps runtime and enter the installed bundle:

```bash
sima-cli neat install apps
cd prebuilt-apps
APP_DIR=examples/segmentation/yolov8-instance-segmenter
```

Run the remaining commands from `prebuilt-apps/`.

## Prepare the Model

| Model package | Role | Model Zoo name |
| --- | --- | --- |
| `yolo_v8n_seg_mpk.tar.gz` | Default | `yolo_v8n_seg` |
| `yolo_v8s_seg_mpk.tar.gz` | Supported | `yolo_v8s_seg` |
| `yolo_v8m_seg_mpk.tar.gz` | Supported | `yolo_v8m_seg` |
| `yolo_v8l_seg_mpk.tar.gz` | Supported | `yolo_v8l_seg` |

Model packages come from the Model Zoo release below, which can differ from the installed platform version. Replace `<model-name>` with a model from the table.

```bash
export MODELZOO_VERSION="2.1.2"
mkdir -p models
cd models
sima-cli modelzoo -v "${MODELZOO_VERSION}" get <model-name>
cd ..
```

Set `model.path` in the config to the downloaded package.

## Configure

Open `${APP_DIR}/src/common/config.yaml` and set `model.path`, `io.input_dir`, and `io.output_dir`. Change the score threshold only if you want to show more or fewer segments.

## Run

### C++

```bash
./${APP_DIR}/src/cpp/pre-built/yolov8-instance-segmenter \
  --config ${APP_DIR}/src/common/config.yaml
```

### Python

```bash
source ~/pyneat/bin/activate
pip install -r ${APP_DIR}/src/python/requirements.txt
python3 ${APP_DIR}/src/python/main.py \
  --config ${APP_DIR}/src/common/config.yaml
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
