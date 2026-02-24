# YOLOv5 Segmentation Overlay

## Metadata
| Field | Value |
| --- | --- |
| Category | semantic-segmentation |
| Difficulty | Intermediate |
| Tags | semantic-segmentation |
| Status | experimental |
| Binary Name | yolov5_seg_overlay |
| Model | yolov5n |

## Concept
Minimal YOLOv5 segmentation overlay from DetessDequant outputs. Processes images and writes segmentation overlays.

## Supported Models
Also works with: `yolov5s`, `yolov5m`, `yolov5l`

Download any variant: `sima-cli modelzoo get yolov5s`

## Prerequisites
- Installed NEAT SDK
- Model downloaded: `./scripts/download_models.sh` (or `sima-cli modelzoo get yolov5n`)

## Run
### C++
```bash
./build/examples/semantic-segmentation/yolov5_seg_overlay/yolov5_seg_overlay models/yolov5n_mpk.tar.gz <input_dir> <output_dir>
```

## Source Files
- C++: `main.cpp`
