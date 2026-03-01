# YOLOv5 Segmentation Overlay

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
Minimal YOLOv5 segmentation overlay from DetessDequant outputs. Processes images and writes segmentation overlays.

## Supported Models
Also works with: `yolov5s`, `yolov5m`, `yolov5l`

Download any variant into `assets/models/`: `sima-cli modelzoo get yolov5s`

## Prerequisites
- Installed NEAT SDK
- Model artifacts are user-managed and should be downloaded into `assets/models/`.
- Download command: `mkdir -p assets/models && cd assets/models && sima-cli modelzoo get yolov5n && cd ../..`

## Run
### C++
```bash
./build/examples/semantic-segmentation/simple-detess-segmentation-mask-overlay/simple-detess-segmentation-mask-overlay assets/models/yolov5n_mpk.tar.gz <input_dir> <output_dir>
```

### Python
```bash
source ~/pyneat/.venv/bin/activate
pip install -r examples/semantic-segmentation/simple-detess-segmentation-mask-overlay/requirements.txt
python examples/semantic-segmentation/simple-detess-segmentation-mask-overlay/main.py assets/models/yolov5_seg_overlay_mpk.tar.gz <input_dir> <output_dir>
```

## Source Files
- C++: `main.cpp`
- Python: `main.py`
