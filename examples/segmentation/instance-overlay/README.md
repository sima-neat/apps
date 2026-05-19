# Instance Overlay

## Metadata
| Field | Value |
| --- | --- |
| Category | segmentation |
| Difficulty | Intermediate |
| Tags | segmentation |
| Status | experimental |
| Binary Name | instance-overlay |
| Model | yolo_v8n_seg |

## Concept
Offline instance segmentation over image folders using YOLOv8 segmentation outputs and DetessDequant post-processing.

## Preview
Snippet from a pipeline run:

![Offline instance segmentation overlay preview](../../../assets/portal/segmentation/instance-overlay/image.jpg)

## Supported Models
Also works with: `yolo_v8s_seg`, `yolo_v8m_seg`, `yolo_v8l_seg`

Download any variant into `assets/models/`:
- `mkdir -p assets/models && cd assets/models && sima-cli modelzoo -v 2.0.0 get yolo_v8n_seg && cd ../..`

## Prerequisites
- Installed Neat SDK.
- Model artifacts are user-managed and should be downloaded into `assets/models/`.
- Download command: `mkdir -p assets/models && cd assets/models && sima-cli modelzoo -v 2.0.0 get yolo_v8n_seg && cd ../..`

## Important Behavior
- Model path is positional and required.
- Input directory is scanned for common image extensions.
- Output images include per-instance mask overlays plus bounding boxes/class labels.
- Inference runs on a resized model input, but saved overlays preserve the original image resolution.
- Uses YOLOv8-seg tensors for box regression/class scores, mask coefficients, and prototype masks.
- Masks, mask contours, and bounding boxes share the same vivid class-color palette.

## Command-Line Options
### C++
- Invocation:
  `./build/examples/segmentation/instance-overlay/instance-overlay <model.tar.gz> <input_dir> <output_dir>`
- Required arguments:
  `<model.tar.gz> <input_dir> <output_dir>`
- Optional arguments:
  None.

### Python
- Invocation:
  `python3 examples/segmentation/instance-overlay/python/main.py <model.tar.gz> <input_dir> <output_dir>`
- Required arguments:
  `<model.tar.gz> <input_dir> <output_dir>`
- Optional arguments:
  None.

## Build
### Build From The Apps Repo
```bash
cd <apps-repo-root>
./build.sh
```

Binary output:
```bash
./build/examples/segmentation/instance-overlay/instance-overlay
```

### Build This Example Directly With CMake
```bash
cd <apps-repo-root>/examples/segmentation/instance-overlay
cmake -S cpp -B build
cmake --build build -j
```

Binary output:
```bash
./build/instance-overlay
```

## Run
### C++
```bash
./build/examples/segmentation/instance-overlay/instance-overlay \
  assets/models/yolo_v8n_seg_mpk.tar.gz <input_dir> <output_dir>
```

### Python
```bash
source ~/pyneat/bin/activate
pip install -r examples/segmentation/instance-overlay/python/requirements.txt
python3 examples/segmentation/instance-overlay/python/main.py \
  assets/models/yolo_v8n_seg_mpk.tar.gz <input_dir> <output_dir>
```

## Debugging Notes
- If startup fails, verify model file path and filename.
- If output is empty, check score thresholds in code and input image resolution.
- Ensure output directory is writable.

## Source Files
- C++ source: `cpp/main.cpp`
- Python source: `python/main.py`
