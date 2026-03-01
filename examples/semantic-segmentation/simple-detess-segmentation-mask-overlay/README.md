# Simple DetESS Segmentation Mask Overlay

## Metadata
| Field | Value |
| --- | --- |
| Category | semantic-segmentation |
| Difficulty | Intermediate |
| Tags | semantic-segmentation |
| Status | experimental |
| Binary Name | simple-detess-segmentation-mask-overlay |
| Model | yolov5n |

## Concept
Semantic segmentation mask overlay for image folders using YOLOv5 DetessDequant outputs.

## Supported Models
Also works with: `yolov5s`, `yolov5m`, `yolov5l`

Download any variant into `assets/models/`:
- `mkdir -p assets/models && cd assets/models && sima-cli modelzoo get yolov5n && cd ../..`

## Prerequisites
- Installed NEAT SDK.
- Model artifacts are user-managed and should be downloaded into `assets/models/`.
- Download command: `mkdir -p assets/models && cd assets/models && sima-cli modelzoo get yolov5n && cd ../..`

## Important Behavior
- Model path is positional and required.
- Input directory is scanned for image files.
- Output files are mask overlays.

## Command-Line Options
### C++
- Invocation:
  `./build/examples/semantic-segmentation/simple-detess-segmentation-mask-overlay/simple-detess-segmentation-mask-overlay <model.tar.gz> <input_dir> <output_dir>`
- Required arguments:
  `<model.tar.gz> <input_dir> <output_dir>`
- Optional arguments:
  None.

### Python
- Invocation:
  `python examples/semantic-segmentation/simple-detess-segmentation-mask-overlay/main.py <model.tar.gz> <input_dir> <output_dir>`
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
./build/examples/semantic-segmentation/simple-detess-segmentation-mask-overlay/simple-detess-segmentation-mask-overlay
```

### Build This Example Directly With CMake
```bash
cd <apps-repo-root>/examples/semantic-segmentation/simple-detess-segmentation-mask-overlay
cmake -S . -B build
cmake --build build -j
```

Binary output:
```bash
./build/simple-detess-segmentation-mask-overlay
```

## Run
### C++
```bash
./build/examples/semantic-segmentation/simple-detess-segmentation-mask-overlay/simple-detess-segmentation-mask-overlay \
  assets/models/yolov5n_mpk.tar.gz <input_dir> <output_dir>
```

### Python
```bash
source ~/pyneat/.venv/bin/activate
pip install -r examples/semantic-segmentation/simple-detess-segmentation-mask-overlay/requirements.txt
python examples/semantic-segmentation/simple-detess-segmentation-mask-overlay/main.py \
  assets/models/yolov5_seg_overlay_mpk.tar.gz <input_dir> <output_dir>
```

## Debugging Notes
- If masks look misaligned, check resize path and input dimensions.
- If startup fails, validate model path and tarball integrity.
- Ensure output directory is writable.

## Reference
- C++ source: `main.cpp`
- Python source: `main.py`
