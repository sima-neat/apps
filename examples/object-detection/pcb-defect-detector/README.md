# PCB Defect Detector

## Metadata

| Field | Value |
| --- | --- |
| Category | object-detection |
| Difficulty | Beginner |
| Tags | object-detection, yolo26, pcb, defect-detection, industrial-inspection |
| Languages | C++, Python |
| Status | stable |
| Binary Name | pcb-defect-detector |
| Model | yolo26n_plc |

## Concept

Batch visual inspection of printed circuit boards with a custom-trained YOLO26n
detector. Every image in a folder is run on the MLA and saved back annotated with
defect boxes and labels.

Images of any resolution are accepted and may be mixed in one folder. Core
letterboxes each one to the model input on device, so boxes are drawn on the
full-resolution source image.

Six manufacturing defect classes are detected:

```text
missing_hole, mouse_bite, open_circuit, short, spur, spurious_copper
```

## Preview

![PCB defect detection result](../../../portal/assets/examples/object-detection/pcb-defect-detector/image.jpg)

## Prerequisites

- `sima-cli` ([documentation](https://developer.sima.ai/software/tools/sima-cli/)) on a supported Modalix or DevKit target.
- Network access to `docs.sima.ai` to download the model package.

## Install Apps

Install the latest Neat Apps runtime and enter the installed bundle:

```bash
sima-cli neat install apps
cd prebuilt-apps
APP_DIR=examples/object-detection/pcb-defect-detector
```

Run the remaining commands from `prebuilt-apps/`.

## Prepare the Model

| Model | Role | Source |
| --- | --- | --- |
| `yolo26n_plc_mpk.tar.gz` | Default | Direct artifact |

Model packages come from the Model Zoo release below, which can differ from the installed platform version.

```bash
export MODELZOO_VERSION="2.1.3"
mkdir -p models
cd models
sima-cli download "https://docs.sima.ai/pkg_downloads/SDK${MODELZOO_VERSION}/models/modalix/yolo26n_plc_mpk.tar.gz"
cd ..
```

Set `model.path` in the config to the downloaded package.

The detector is trained on a custom PCB defect dataset rather than published in
Model Zoo. The pack is a BF16 build with MLA tessellation, compiled from the
trained YOLO26n ONNX export with the raw detection heads exposed and C2PSA
attention rewritten to BF16-friendly Einsum nodes. Class order is fixed by the
checkpoint and must match `src/common/pcb_label.txt`.

Dataset and source-model reference:
[PCB defects detection](https://platform.ultralytics.com/muhammadrizwanmunawar/datasets/pcb-defects-detection).

## Configure

Open `${APP_DIR}/src/common/config.yaml` and set `model.path`, `io.input_dir`, and `io.output_dir`. Five sample boards ship under `assets/datasets/pcb/`, so the example runs as soon as the model pack is in place.

`model.input_max_width` and `model.input_max_height` set the largest image the run accepts. A larger board is reported as a failed image and the run exits 4; raise them for larger scans.

Change `decode.score_threshold` only if you want to show more or fewer defects. Set `runtime.profile: true` for per-image timing, `runtime.num_runs` above 1 to repeat the folder for steadier timings, and `output.overlay: false` to measure inference without drawing or writing images.

## Run

### C++

```bash
./${APP_DIR}/src/cpp/pre-built/pcb-defect-detector \
  --config ${APP_DIR}/src/common/config.yaml
```

### Python

```bash
source ~/pyneat/bin/activate
pip install -r ${APP_DIR}/src/python/requirements.txt
python3 ${APP_DIR}/src/python/main.py \
  --config ${APP_DIR}/src/common/config.yaml
```

Both implementations also accept `--score` and `--nms` to override the configured thresholds without editing the file, and `--validate-config-only` to check the configuration and label file without loading the model.

## Expected Result

Each input image produces one annotated copy in `io.output_dir`, named `<stem>_pcb` with the source extension kept:

```text
pcb_01_missing_hole.jpg -> pcb_01_missing_hole_pcb.jpg
```

Each run prints a per-image line with the defect count and class breakdown, followed by per-class totals for the folder.

## Troubleshooting

- Run either implementation with `--validate-config-only` to check configuration values and the label file without loading the model.
- Confirm `model.path` points at a readable model package under `models/`.
- Confirm `io.input_dir` exists and contains `.jpg`, `.jpeg`, `.png`, or `.bmp` files, and that `io.output_dir` is writable.
- If no defects are reported, lower `decode.score_threshold`; if boxes overlap heavily, lower `decode.nms_iou`.
- Class names in `src/common/pcb_label.txt` are index-aligned with the trained checkpoint. Reordering them mislabels detections.
- Exit code 4 means one or more images could not be read, or their annotated
  output could not be written; the run names them on stderr.

## Source Files

- C++ reference source: `src/cpp/main.cpp`
- Python source: `src/python/main.py`
- Shared config: `src/common/config.yaml`
- Class labels: `src/common/pcb_label.txt`
- Sample images: `assets/datasets/pcb/`

The packaged C++ source is an implementation reference. Run the executable under `src/cpp/pre-built/`; the installed bundle does not include CMake files.

## Development From Source

To modify, compile, or test this example, use the [Apps contributor workflow](https://github.com/sima-neat/apps/blob/main/CONTRIBUTING.md).
