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
| Model | yolo26m (w16a16) |

## Concept

Detects objects in a folder of images with YOLO26 and saves copies annotated with boxes, labels, and confidence scores.

## Preview

![YOLO26 object detector preview](../../../portal/assets/examples/object-detection/yolo26-object-detector/image.png)

## Prerequisites

- `sima-cli` ([documentation](https://developer.sima.ai/software/tools/sima-cli/)) on the Modalix DevKit, or in a Neat SDK container paired with it.

## Install Apps

Install on the Modalix DevKit, or under `/workspace` in a Neat SDK container paired with it. Only the example itself must run on the DevKit. `sima-cli neat install apps` also offers to install the compatible Neat Library and Insight, and skips them when they are up to date. To keep the Neat Library and Insight you have without being asked, run the install as `NEAT_APPS_SKIP_DEPENDENCIES=1 sima-cli neat install apps`. See [Install Apps](https://github.com/sima-neat/apps/blob/main/README.md#install-apps).

Install the latest Neat Apps runtime and enter the installed bundle:

```bash
sima-cli neat install apps
cd prebuilt-apps
APP_DIR=examples/object-detection/yolo26-object-detector
```

Run the remaining commands from `prebuilt-apps/`.

## Prepare the Model

The [model browser](https://developer.sima.ai/models/) has the YOLO26 detection models in five sizes, from n (nano) to x (extra large). The default is `yolo26m` with the `w16a16` variant, the BF16 build of the model. Download it and unpack it:

```bash
sima-cli models download --id yolo26m --variant w16a16
unzip -o models/yolo26m/w16a16/yolo26m_w16a16_default.zip -d models/yolo26m/w16a16
```

The download is a ZIP file with the model package and its license. `unzip` puts the model package, `models/yolo26m/w16a16/yolo26m_w16a16_mpk.tar.gz`, next to the ZIP file. The example reads only the `.tar.gz` file, so set `model.path` in the config to it, not to the ZIP file.

| ID | Variants | Role |
| --- | --- | --- |
| `yolo26m` | `w8a8`, `w8a16`, `w16a16` | Default (`w16a16`) |
| `yolo26n` | `w8a8`, `w8a16`, `w16a16` | Supported |
| `yolo26s` | `w8a8`, `w8a16`, `w16a16` | Supported |
| `yolo26l` | `w8a8`, `w8a16`, `w16a16` | Supported |
| `yolo26x` | `w8a8`, `w8a16`, `w16a16` | Supported |

To use another model, replace `yolo26m` and `w16a16` in both commands and in `model.path`. The variant is the number format the model was compiled with: `w8a8` stores weights and activations as 8-bit integers, `w8a16` keeps 8-bit integer weights and stores activations in BF16, a 16-bit float format, and `w16a16` uses BF16 for both. See the model browser for the measured accuracy and performance of each size and variant.

## Configure

Open `${APP_DIR}/src/common/config.yaml` and set `model.path`. For the default model that is `models/yolo26m/w16a16/yolo26m_w16a16_mpk.tar.gz`. `io.input_dir` and `io.output_dir` are already set. Change the score threshold only if you want to show more or fewer detections.

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
lines with the input filename in quotes. The detection counts depend on the
model and variant.

Running a second time adds a `Cleared N stale output images` line before the
rest, in both implementations, where `N` is however many image files the previous
run left in `io.output_dir`. Setting `io.output_dir` to the same folder as
`io.input_dir` skips that cleanup and reports
`Skipping output cleanup because output_dir matches input_dir` on stderr; keep
them separate.

## Troubleshooting

- Verify `model.path` and the labels file if detections are missing.
- If the example stops with `model archive must use .tar.gz`, `model.path` points to the downloaded ZIP file. Set it to the `_mpk.tar.gz` file unpacked from it.
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
