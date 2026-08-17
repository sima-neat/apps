# SSD-MobileNet Object Detector

## Metadata

| Field | Value |
| --- | --- |
| Category | object-detection |
| Difficulty | Intermediate |
| Tags | ssd, mobilenet, object-detection, folder-input, coco |
| Languages | C++, Python |
| Status | experimental |
| Binary Name | ssd-mobilenet-object-detector |
| Model | SSD-MobileNet V1/V2 and SSDlite-MobileNetV3 |

## Concept

This example runs SSD object detection on every image in a folder. Neat resizes and normalizes each
image, runs the model, decodes the SSD outputs, and writes an annotated PNG to the output folder.
An optional JSON report contains the same detections as class, score, and bounding-box values.

The example supports 300×300 SSD-MobileNet V1/V2 and 320×320 SSDlite-MobileNetV3 models. Neat owns
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
```

Run the remaining commands from `prebuilt-apps/`.

## Prepare the Model

| Family | Frame | `model.preprocessing_profile` | Published variants |
| --- | --- | --- | --- |
| SSD-MobileNetV1 | 300 | `tensorflow_ssd` | INT8/BF16 × tessellation in/out of MLA |
| SSD-MobileNetV2 | 300 | `tensorflow_ssd` | INT8/BF16 × tessellation in/out of MLA |
| SSDlite-MobileNetV3 BF16 | 320 | `tensorflow_ssd` | BF16 × tessellation in/out of MLA |
| SSDlite-MobileNetV3 QAT INT8 | 320 | `torchvision_ssdlite` | INT8 × tessellation in/out of MLA |

Install all four compiled SSD-MobileNetV2 variants from the current develop catalog:

```bash
mkdir -p models
sima-cli neat install --stg \
  models/ssd_mobilenet_v2@develop:latest \
  --install-dir ./models
```

The configuration below uses
`models/ssd_mobilenet_v2_modalix_int8_tess_mla_mpk.tar.gz`.

## Configure

Edit `examples/object-detection/ssd-mobilenet-object-detector/src/common/config.yaml`.

```yaml
model:
  path: models/ssd_mobilenet_v2_modalix_int8_tess_mla_mpk.tar.gz
  preprocessing_profile: tensorflow_ssd
  labels: examples/object-detection/ssd-mobilenet-object-detector/src/common/coco_labels.txt

io:
  input_dir: assets/datasets/coco
  output_dir: sandbox/ssd-mobilenet-object-detector
  detections_json: ""        # Optional machine-readable detections report.

decode:
  score_threshold: 0.55
  nms_iou: 0.60
  max_detections: 100

runtime:
  timeout_ms: 20000
```

`model.path` is required. `io.input_dir` is a folder, not a single image: the app processes every
`.jpg`, `.jpeg`, `.png`, and `.bmp` file directly inside it in filename order. Subdirectories are
not searched. Each input produces an annotated PNG under `io.output_dir`.

## Run

### C++

```bash
./examples/object-detection/ssd-mobilenet-object-detector/src/cpp/pre-built/ssd-mobilenet-object-detector \
  --config examples/object-detection/ssd-mobilenet-object-detector/src/common/config.yaml
```

### Python

```bash
source ~/pyneat/bin/activate
pip install -r examples/object-detection/ssd-mobilenet-object-detector/src/python/requirements.txt
python3 examples/object-detection/ssd-mobilenet-object-detector/src/python/main.py \
  --config examples/object-detection/ssd-mobilenet-object-detector/src/common/config.yaml
```

## Troubleshooting

- Verify `model.path` and the labels file if detections are missing.
- Confirm the input folder contains `.jpg`, `.jpeg`, `.png`, or `.bmp` files.
- Keep `io.input_dir` and `io.output_dir` different.
- If boxes look vertically squished or shifted toward the frame center, the resize mode is wrong.
  This model requires a **stretch** resize; the example already sets it.
- If detections are missing but the pipeline runs, lower `decode.score_threshold` — int8
  quantization of the classification heads can drop weak detections.
- Set `io.detections_json` to write per-image detections (class, score, box) for offline checks.

## Source Files

- C++ reference source: `src/cpp/main.cpp`
- Python source: `src/python/main.py`
- Shared config and labels: `src/common/`

The packaged C++ source is an implementation reference. Run the executable under `src/cpp/pre-built/`; the installed bundle does not include CMake files.

## Development From Source

To modify, compile, or test this example, use the [Apps contributor workflow](https://github.com/sima-neat/apps/blob/main/CONTRIBUTING.md).
