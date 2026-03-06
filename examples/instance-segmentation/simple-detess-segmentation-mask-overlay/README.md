# Simple DetessDequant Instance Segmentation Mask Overlay

## Metadata
| Field | Value |
| --- | --- |
| Category | instance-segmentation |
| Difficulty | Intermediate |
| Tags | instance-segmentation |
| Status | experimental |
| Binary Name | simple-detess-segmentation-mask-overlay |
| Model | yolov5n |

## Concept
Instance segmentation mask overlay for image folders using YOLOv5 DetessDequant outputs.

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
- Output files are combined instance-segmentation overlays named `*_overlay.jpg`.
- Each overlay image contains class-colored mask fills, mask contours, bounding boxes, and labels.
- Masks are reconstructed only for detected objects. This example does not produce a dense semantic label for every pixel.
- Masks, contours, and boxes share the same class-color palette as the YOLOv8 offline instance-segmentation example.

## Command-Line Options
### C++
- Invocation:
  `./build/examples/instance-segmentation/simple-detess-segmentation-mask-overlay/simple-detess-segmentation-mask-overlay <model.tar.gz> <input_dir> <output_dir>`
- Required arguments:
  `<model.tar.gz> <input_dir> <output_dir>`
- Optional arguments:
  None.

### Python
- Invocation:
  `python examples/instance-segmentation/simple-detess-segmentation-mask-overlay/python/main.py <model.tar.gz> <input_dir> <output_dir>`
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
./build/examples/instance-segmentation/simple-detess-segmentation-mask-overlay/simple-detess-segmentation-mask-overlay
```

### Build This Example Directly With CMake
```bash
cd <apps-repo-root>/examples/instance-segmentation/simple-detess-segmentation-mask-overlay
cmake -S cpp -B build
cmake --build build -j
```

Binary output:
```bash
./build/simple-detess-segmentation-mask-overlay
```

## Run
### C++
```bash
./build/examples/instance-segmentation/simple-detess-segmentation-mask-overlay/simple-detess-segmentation-mask-overlay \
  assets/models/yolov5n_mpk.tar.gz <input_dir> <output_dir>
```

### Python
```bash
source ~/pyneat/bin/activate
pip install -r examples/instance-segmentation/simple-detess-segmentation-mask-overlay/python/requirements.txt
python examples/instance-segmentation/simple-detess-segmentation-mask-overlay/python/main.py \
  assets/models/yolov5n_mpk.tar.gz <input_dir> <output_dir>
```

## Debugging Notes
- If masks look misaligned, check resize path and input dimensions.
- If startup fails, validate model path and tarball integrity.
- Ensure output directory is writable.

## Source Files
- C++ source: `cpp/main.cpp`
- Python source: `python/main.py`
