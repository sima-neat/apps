# YOLOv8n Instance Segmentation (DetessDequant)

## Metadata
| Field | Value |
| --- | --- |
| Category | instance-segmentation |
| Difficulty | Intermediate |
| Tags | instance-segmentation |
| Status | experimental |
| Binary Name | yolov8n_seg_detess_simple |

## Concept
Minimal YOLOv8-seg pipeline using DetessDequant postprocessing (no boxdecode). Processes images and writes segmentation results.

## Prerequisites
- Compiled YOLOv8n-seg MPK (`.tar.gz`)
- Installed NEAT SDK

## Run
### C++
```bash
./build/examples/instance-segmentation/yolov8n_seg_detess_simple/yolov8n_seg_detess_simple <model.tar.gz> <input_dir> <output_dir>
```

## Source Files
- C++: `main.cpp`
