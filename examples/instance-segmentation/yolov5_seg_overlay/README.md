# YOLOv5 Instance Segmentation Overlay

## Metadata
| Field | Value |
| --- | --- |
| Category | instance-segmentation |
| Difficulty | Intermediate |
| Tags | instance-segmentation |
| Status | experimental |
| Binary Name | yolov5_seg_overlay |

## Concept
Minimal YOLOv5 instance segmentation overlay from DetessDequant outputs. Processes images and writes segmentation overlays.

## Prerequisites
- Compiled YOLOv5-seg MPK (`.tar.gz`)
- Installed NEAT SDK

## Run
### C++
```bash
./build/examples/instance-segmentation/yolov5_seg_overlay/yolov5_seg_overlay <model.tar.gz> <input_dir> <output_dir>
```

## Source Files
- C++: `main.cpp`
