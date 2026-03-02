# Simple Object Detection Overlay Pipeline

## Metadata
| Field | Value |
| --- | --- |
| Category | object-detection |
| Difficulty | Beginner |
| Tags | object-detection, yolov8, folder-inference |
| Languages | C++, Python |
| Status | experimental |
| Binary Name | simple-object-detection-overlay-pipeline |
| Model | yolo_v8n |

## Concept
Minimal image-folder object detection pipeline. Each image is inferred, annotated with bounding boxes and labels, and written to an output folder.

## Supported Models
Also works with: `yolo_v8s`, `yolo_v8m`, `yolo_v8l`

Download any variant into `assets/models/`:
- `mkdir -p assets/models && cd assets/models && sima-cli modelzoo get yolo_v8n && cd ../..`

## Prerequisites
- Installed NEAT SDK.
- Model artifacts are user-managed and should be downloaded into `assets/models/`.
- Download command: `mkdir -p assets/models && cd assets/models && sima-cli modelzoo get yolo_v8n && cd ../..`
- Labels file: `examples/object-detection/simple-object-detection-overlay-pipeline/coco_label.txt`

## Important Behavior
- C++ and Python both use positional arguments.
- Labels file is required.
- Output images are written as `.png` files.

## Command-Line Options
### C++
- Invocation:
  `./build/examples/object-detection/simple-object-detection-overlay-pipeline/simple-object-detection-overlay-pipeline <model.tar.gz> <labels.txt> <input_dir> <output_dir>`
- Required arguments:
  `<model.tar.gz> <labels.txt> <input_dir> <output_dir>`
- Optional arguments:
  None.

### Python
- Invocation:
  `python examples/object-detection/simple-object-detection-overlay-pipeline/main.py <model.tar.gz> <labels.txt> <input_dir> <output_dir> [--min-score 0.55]`
- Required arguments:
  `<model.tar.gz> <labels.txt> <input_dir> <output_dir>`
- Optional arguments:
  `--min-score` (default: `0.55`)

## Build
### Build From The Apps Repo
```bash
cd <apps-repo-root>
./build.sh
```

Binary output:
```bash
./build/examples/object-detection/simple-object-detection-overlay-pipeline/simple-object-detection-overlay-pipeline
```

### Build This Example Directly With CMake
```bash
cd <apps-repo-root>/examples/object-detection/simple-object-detection-overlay-pipeline
cmake -S . -B build
cmake --build build -j
```

Binary output:
```bash
./build/simple-object-detection-overlay-pipeline
```

## Run
### C++
```bash
./build/examples/object-detection/simple-object-detection-overlay-pipeline/simple-object-detection-overlay-pipeline \
  assets/models/yolo_v8n_mpk.tar.gz \
  examples/object-detection/simple-object-detection-overlay-pipeline/coco_label.txt \
  <input_dir> <output_dir>
```

### Python
```bash
source ~/pyneat/.venv/bin/activate
pip install -r examples/object-detection/simple-object-detection-overlay-pipeline/requirements.txt
python examples/object-detection/simple-object-detection-overlay-pipeline/main.py \
  assets/models/yolo_v8n_mpk.tar.gz \
  examples/object-detection/simple-object-detection-overlay-pipeline/coco_label.txt \
  <input_dir> <output_dir>
```

## Debugging Notes
- If detections are missing, validate label file ordering and score thresholds.
- If model load fails, verify `assets/models/yolo_v8n_mpk.tar.gz` exists.
- Ensure input folder contains supported image extensions.

## Reference
- C++ source: `main.cpp`
- Python source: `main.py`
