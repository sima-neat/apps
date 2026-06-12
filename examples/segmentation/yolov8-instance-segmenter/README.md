# YOLOv8 Instance Segmenter

## Metadata
| Field | Value |
| --- | --- |
| Category | segmentation |
| Difficulty | Intermediate |
| Tags | segmentation |
| Status | experimental |
| Binary Name | yolov8-instance-segmenter |
| Model | yolo_v8n_seg |

## Concept
Offline YOLOv8 instance segmentation over image folders using YOLOv8 segmentation outputs and DetessDequant post-processing.

## Preview
Snippet from a pipeline run:

![Instance segmenter preview](../../../assets/portal/segmentation/yolov8-instance-segmenter/image.jpg)

## Supported Models
Use the platform version wherever `<platform-version>` appears.

Also works with: `yolo_v8s_seg`, `yolo_v8m_seg`, `yolo_v8l_seg`

Download any variant into `assets/models/`:
- `mkdir -p assets/models && cd assets/models && sima-cli modelzoo -v <platform-version> get yolo_v8n_seg && cd ../..`

## Prerequisites
- Installed Neat Development Environment.
- Model artifacts are user-managed and should be downloaded into `assets/models/`.
- Download command: `mkdir -p assets/models && cd assets/models && sima-cli modelzoo -v <platform-version> get yolo_v8n_seg && cd ../..`

## Important Behavior
- Model path is positional and required.
- Input directory is scanned for common image extensions.
- Output images include per-instance mask overlays plus bounding boxes/class labels.
- Runtime and decode settings live in `src/common/config.yaml`.
- Inference runs on a resized model input, but saved overlays preserve the original image resolution.
- Uses YOLOv8-seg tensors for box regression/class scores, mask coefficients, and prototype masks.
- Masks, mask contours, and bounding boxes share the same vivid class-color palette.

## Command-Line Options
### C++
- Invocation:
  `./build/examples/segmentation/yolov8-instance-segmenter/yolov8-instance-segmenter [--config <path>]`
- Required arguments:
  None. Defaults to `src/common/config.yaml`.
- Optional arguments:
  `--config <path>`

### Python
- Invocation:
  `python3 examples/segmentation/yolov8-instance-segmenter/src/python/main.py [--config <path>]`
- Required arguments:
  None. Defaults to `src/common/config.yaml`.
- Optional arguments:
  `--config <path>`

## Build
### Build From The Apps Repo
```bash
cd <apps-repo-root>
./build.sh
```

Binary output:
```bash
./build/examples/segmentation/yolov8-instance-segmenter/yolov8-instance-segmenter
```

### Build This Example Directly With CMake
```bash
cd <apps-repo-root>/examples/segmentation/yolov8-instance-segmenter
cmake -S cpp -B build
cmake --build build -j
```

Binary output:
```bash
./build/yolov8-instance-segmenter
```

## Run
### C++
```bash
./build/examples/segmentation/yolov8-instance-segmenter/yolov8-instance-segmenter \
  --config examples/segmentation/yolov8-instance-segmenter/src/common/config.yaml
```

### Python
```bash
source ~/pyneat/bin/activate
pip install -r examples/segmentation/yolov8-instance-segmenter/src/python/requirements.txt
python3 examples/segmentation/yolov8-instance-segmenter/src/python/main.py \
  --config examples/segmentation/yolov8-instance-segmenter/src/common/config.yaml
```

## Debugging Notes
- If startup fails, verify model file path and filename.
- If output is empty, check `decode.score_threshold` and `runtime.infer_size`.
- Ensure output directory is writable.

## Source Files
- C++ source: `src/cpp/main.cpp`
- Python source: `src/python/main.py`
- Shared config: `src/common/config.yaml`
