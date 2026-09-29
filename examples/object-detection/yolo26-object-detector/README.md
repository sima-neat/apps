# YOLO26 Object Detector

## Metadata

| Field | Value |
| --- | --- |
| Category | object-detection |
| Difficulty | Beginner |
| Tags | object-detection, yolo26, folder-inference |
| Languages | C++, Python |
| Status | stable |
| Binary Name | yolo26-object-detector |
| Model | yolo26m-det-bf16-mla_tess-b1 |

## Concept

Detects objects in a folder of images with YOLO26 and saves copies annotated with boxes, labels, and confidence scores.

## Preview

![YOLO26 object detector preview](../../../portal/assets/examples/object-detection/yolo26-object-detector/image.png)

## Prerequisites

- `sima-cli` ([documentation](https://developer.sima.ai/software/tools/sima-cli/)) on a supported Modalix or DevKit target.

## Install Apps

Install the latest Neat Apps runtime and enter the installed bundle:

```bash
sima-cli neat install apps
cd prebuilt-apps
APP_DIR=examples/object-detection/yolo26-object-detector
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

This model comes from the direct SDK artifact release below, which can differ from the installed platform version. Replace `<model-file>` with a file from the table.

```bash
export MODELZOO_VERSION="2.1.3"
mkdir -p models
cd models
sima-cli download "https://docs.sima.ai/pkg_downloads/SDK${MODELZOO_VERSION}/models/modalix/yolo26-detection/<model-file>"
cd ..
```

Set `model.path` in the config to the downloaded package.

## Configure

Open `${APP_DIR}/src/common/config.yaml` and set `model.path`, `io.input_dir`, and `io.output_dir`. Change the score threshold only if you want to show more or fewer detections.

## Run

### C++

```bash
./${APP_DIR}/src/cpp/pre-built/yolo26-object-detector \
  --config ${APP_DIR}/src/common/config.yaml
```

### Python

```bash
source ~/pyneat/bin/activate
pip install -r ${APP_DIR}/src/python/requirements.txt
python3 ${APP_DIR}/src/python/main.py \
  --config ${APP_DIR}/src/common/config.yaml
```

## Expected Result

The application prints the image count, one line per image, and a final total:

```text
Found 21 images
[1/21] 000000081061.jpg -> 000000081061.png (3 detections)
Done: 21 images processed
```

`io.output_dir` then holds one annotated PNG per input image; with the packaged
`assets/datasets/coco` folder that is 21 files. The C++ binary prints the same
lines with the input filename in quotes.

Running a second time adds a `Cleared N stale output images` line before the
rest, in both implementations, where `N` is however many image files the previous
run left in `io.output_dir`. Setting `io.output_dir` to the same folder as
`io.input_dir` skips that cleanup and reports
`Skipping output cleanup because output_dir matches input_dir` on stderr; keep
them separate.

## Troubleshooting

- Verify `model.path` and the labels file if detections are missing.
- Confirm the input folder contains `.jpg`, `.jpeg`, `.png`, or `.bmp` files.
- Adjust `decode.score_threshold` and `decode.nms_iou` when tuning detections.
- Set `runtime.profile: true` in the config to print pipeline timing summaries.

## Source Files

- C++ reference source: `src/cpp/main.cpp`
- Python source: `src/python/main.py`
- Shared config and labels: `src/common/`

The packaged C++ source is an implementation reference. Run the executable under `src/cpp/pre-built/`; the installed bundle does not include CMake files.

## Development From Source

To modify, compile, or test this example, use the [Apps contributor workflow](https://github.com/sima-neat/apps/blob/main/CONTRIBUTING.md).
