# SSD-MobileNet Object Detector

## Metadata

| Field | Value |
| --- | --- |
| Category | object-detection |
| Difficulty | Intermediate |
| Tags | ssd, mobilenet, object-detection, folder-input, coco |
| Languages | C++, Python |
| Status | stable |
| Binary Name | ssd-mobilenet-object-detector |
| Model | SSD-MobileNet V1/V2 and SSDlite-MobileNetV3 |

## Concept

Detects objects in a folder of images with SSD MobileNet, saves annotated PNGs, and can write the same detections to a JSON report.

The application supports 300×300 SSD-MobileNet V1/V2 and 320×320 SSDlite-MobileNetV3 models. Neat owns
the recipe-specific priors, class scoring, and non-maximum suppression through
`BoxDecodeType::Ssd`; the example only selects the matching preprocessing profile.

## Preview

![SSD-MobileNetV2 object detector preview](../../../portal/assets/examples/object-detection/ssd-mobilenet-object-detector/image.png)

## Prerequisites

- `sima-cli` ([documentation](https://developer.sima.ai/software/tools/sima-cli/)) on a supported Modalix or DevKit target.

## Install Apps

Install the latest Neat Apps runtime and enter the installed bundle:

```bash
sima-cli neat install apps
cd prebuilt-apps
APP_DIR=examples/object-detection/ssd-mobilenet-object-detector
```

Run the remaining commands from `prebuilt-apps/`.

## Prepare the Model

The tested model is SSD-MobileNetV2 INT8 with tessellation inside the MLA. Download its SDK
2.1.3 artifact:

```bash
export MODELZOO_VERSION="2.1.3"
mkdir -p models
cd models
sima-cli download \
  "https://docs.sima.ai/pkg_downloads/SDK${MODELZOO_VERSION}/models/modalix/ssd_mobilenet_v2_modalix_int8_tess_mla_mpk.tar.gz"
cd ..
```

The configuration below uses
`models/ssd_mobilenet_v2_modalix_int8_tess_mla_mpk.tar.gz`.

## Configure

Open `${APP_DIR}/src/common/config.yaml`. Set `model.path`, `io.input_dir`, and `io.output_dir`. Set `io.detections_json` to a file path if you also want a JSON report.

`io.input_dir` is a folder, not a single image. The application processes every
`.jpg`, `.jpeg`, `.png`, and `.bmp` file directly inside it in filename order. Subdirectories are
not searched. Each input produces an annotated PNG under `io.output_dir`.

## Run

### C++

```bash
./${APP_DIR}/src/cpp/pre-built/ssd-mobilenet-object-detector \
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

- Verify `model.path` and the labels file if detections are missing.
- Confirm the input folder contains `.jpg`, `.jpeg`, `.png`, or `.bmp` files.
- Keep `io.input_dir` and `io.output_dir` different.
- If boxes look vertically squished or shifted toward the frame center, the resize mode is wrong.
  This model requires a **stretch** resize; the example already sets it.
- If detections are missing but the pipeline runs, lower `decode.score_threshold` because INT8
  quantization can drop weak detections.
- Set `io.detections_json` to write per-image detections (class, score, box) for offline checks.

## Source Files

- C++ reference source: `src/cpp/main.cpp`
- Python source: `src/python/main.py`
- Shared config and labels: `src/common/`

The packaged C++ source is an implementation reference. Run the executable under `src/cpp/pre-built/`; the installed bundle does not include CMake files.

## Development From Source

To modify, compile, or test this example, use the [Apps contributor workflow](https://github.com/sima-neat/apps/blob/main/CONTRIBUTING.md).
