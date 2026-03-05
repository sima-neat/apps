# Offline Instance Segmentation Overlay

## Metadata
| Field | Value |
| --- | --- |
| Category | instance-segmentation |
| Difficulty | Intermediate |
| Tags | instance-segmentation |
| Status | experimental |
| Binary Name | offline-instance-segmentation-overlay |
| Model | yolo_v8n_seg |

## Concept
Offline instance segmentation over image folders using YOLOv8 segmentation outputs and DetessDequant post-processing.

## Supported Models
Also works with: `yolo_v8s_seg`, `yolo_v8m_seg`, `yolo_v8l_seg`

Download any variant into `assets/models/`:
- `mkdir -p assets/models && cd assets/models && sima-cli modelzoo get yolo_v8n_seg && cd ../..`

## Prerequisites
- Installed NEAT SDK.
- Model artifacts are user-managed and should be downloaded into `assets/models/`.
- Download command: `mkdir -p assets/models && cd assets/models && sima-cli modelzoo get yolo_v8n_seg && cd ../..`

## Important Behavior
- Model path is positional and required.
- Input directory is scanned for common image extensions.
- Output images include segmentation overlays and per-object visualization.

## Command-Line Options
### C++
- Invocation:
  `./build/examples/instance-segmentation/offline-instance-segmentation-overlay/offline-instance-segmentation-overlay <model.tar.gz> <input_dir> <output_dir>`
- Required arguments:
  `<model.tar.gz> <input_dir> <output_dir>`
- Optional arguments:
  None.

### Python
- Invocation:
  `python3 examples/instance-segmentation/offline-instance-segmentation-overlay/python/main.py <model.tar.gz> <input_dir> <output_dir>`
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
./build/examples/instance-segmentation/offline-instance-segmentation-overlay/offline-instance-segmentation-overlay
```

### Build This Example Directly With CMake
```bash
cd <apps-repo-root>/examples/instance-segmentation/offline-instance-segmentation-overlay
cmake -S cpp -B build
cmake --build build -j
```

Binary output:
```bash
./build/offline-instance-segmentation-overlay
```

## Run
### C++
```bash
./build/examples/instance-segmentation/offline-instance-segmentation-overlay/offline-instance-segmentation-overlay \
  assets/models/yolo_v8n_seg_mpk.tar.gz <input_dir> <output_dir>
```

### Python
```bash
source ~/pyneat/bin/activate
pip install -r examples/instance-segmentation/offline-instance-segmentation-overlay/python/requirements.txt
python3 examples/instance-segmentation/offline-instance-segmentation-overlay/python/main.py \
  assets/models/yolo_v8n_seg_mpk.tar.gz <input_dir> <output_dir>
```

## Debugging Notes
- If startup fails, verify model file path and filename.
- If output is empty, check score thresholds in code and input image resolution.
- Ensure output directory is writable.

## Source Files
- C++ source: `cpp/main.cpp`
- Python source: `python/main.py`
