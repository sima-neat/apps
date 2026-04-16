# Simple yolo26m Object Detection Overlay Pipeline

## Metadata
| Field | Value |
| --- | --- |
| Category | object-detection |
| Difficulty | Beginner |
| Tags | object-detection, yolo26m, folder-inference |
| Languages | C++, Python |
| Status | experimental |
| Binary Name | simple-newest-yolo-object-detection-overlay-pipeline |
| Model | yolo26m_mod |

## Concept
Minimal image-folder object detection pipeline using yolo26m, the newest iteration of the YOLO model family. Each image is inferred, annotated with bounding boxes and labels, and written to an output folder. The pipeline demonstrates the NEAT API for model loading, session building, synchronous inference, and result decoding.

## Preview
Snippet from a pipeline run:

![Simple newest yolo object detection overlay preview](../../../assets/portal/object-detection/simple-newest-yolo-object-detection-overlay-pipeline/image.png)

## Prerequisites
- Installed NEAT SDK.
- Model artifacts are user-managed and should be downloaded into `assets/models/`.
- yolo26m is not yet published in the SiMa modelzoo. Until it is available, obtain the compiled model pack (`yolo26m_mod_mpk.tar.gz`) from the shared models repository or contact the model owner.
- Labels file: `examples/object-detection/simple-newest-yolo-object-detection-overlay-pipeline/common/coco_label.txt`

## Important Behavior
- Both C++ and Python use named flags (`--model`, `--labels`, `--input-dir`, `--output-dir`).
- Labels file is required.
- Output images are written as `.png` files.
- Use `--profile` to print per-image and aggregate timing.
- Use `--no-overlay` to skip drawing bounding boxes (useful for benchmarking).

## Command-Line Options
### C++
- Invocation:
  `./build/examples/object-detection/simple-newest-yolo-object-detection-overlay-pipeline/simple-newest-yolo-object-detection-overlay-pipeline --model <model.tar.gz> --labels <labels.txt> --input-dir <dir> --output-dir <dir>`
- Required arguments:
  `--model <model.tar.gz>`, `--labels <labels.txt>`, `--input-dir <dir>`, `--output-dir <dir>`
- Optional arguments:
  `--min-score <float>` (default: `0.25`), `--nms-iou <float>` (default: `0.45`), `--profile`, `--no-overlay`

### Python
- Invocation:
  `python examples/object-detection/simple-newest-yolo-object-detection-overlay-pipeline/python/main.py --model <model.tar.gz> --labels <labels.txt> --input-dir <dir> --output-dir <dir>`
- Required arguments:
  `--model <model.tar.gz>`, `--labels <labels.txt>`, `--input-dir <dir>`, `--output-dir <dir>`
- Optional arguments:
  `--min-score <float>` (default: `0.25`), `--nms-iou <float>` (default: `0.45`), `--profile`, `--no-overlay`

## Build
### Build From The Apps Repo
```bash
cd <apps-repo-root>
./build.sh
```

Binary output:
```bash
./build/examples/object-detection/simple-newest-yolo-object-detection-overlay-pipeline/simple-newest-yolo-object-detection-overlay-pipeline
```

### Build This Example Directly With CMake
```bash
cd <apps-repo-root>/examples/object-detection/simple-newest-yolo-object-detection-overlay-pipeline
cmake -S cpp -B build
cmake --build build -j
```

Binary output:
```bash
./build/simple-newest-yolo-object-detection-overlay-pipeline
```

## Run
### C++
```bash
./build/examples/object-detection/simple-newest-yolo-object-detection-overlay-pipeline/simple-newest-yolo-object-detection-overlay-pipeline \
  --model assets/models/yolo26m_mod_mpk.tar.gz \
  --labels examples/object-detection/simple-newest-yolo-object-detection-overlay-pipeline/common/coco_label.txt \
  --input-dir <input_dir> --output-dir <output_dir>
```

### Python
```bash
source ~/pyneat/bin/activate
pip install -r examples/object-detection/simple-newest-yolo-object-detection-overlay-pipeline/python/requirements.txt
python examples/object-detection/simple-newest-yolo-object-detection-overlay-pipeline/python/main.py \
  --model assets/models/yolo26m_mod_mpk.tar.gz \
  --labels examples/object-detection/simple-newest-yolo-object-detection-overlay-pipeline/common/coco_label.txt \
  --input-dir <input_dir> --output-dir <output_dir>
```

## Debugging Notes
- If detections are missing, validate label file ordering and score thresholds.
- If model load fails, verify `assets/models/yolo26m_mod_mpk.tar.gz` exists.
- Ensure input folder contains supported image extensions (`.jpg`, `.jpeg`, `.png`, `.bmp`).
- Use `--profile` to identify bottlenecks in the pipeline.
- Use `--min-score` and `--nms-iou` to tune detection sensitivity.

## Source Files
- C++ source: `cpp/main.cpp`
- C++ tests: `cpp/tests/unit_test.cpp`, `cpp/tests/e2e_test.cpp`
- Python source: `python/main.py`
- Python tests: `python/tests/test_unit.py`, `python/tests/test_e2e.py`
- Shared assets: `common/coco_label.txt`
