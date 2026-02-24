# YOLOv8 COCO Throughput Benchmark

## Metadata
| Field | Value |
| --- | --- |
| Category | throughput |
| Difficulty | Intermediate |
| Tags | throughput, object-detection |
| Status | experimental |
| Binary Name | yolov8_coco_tput |

## Concept
Throughput benchmark for YOLOv8 on COCO images using the Session API. Measures inference performance under sustained load.

## Prerequisites
- Compiled YOLOv8 MPK (`.tar.gz`)
- COCO dataset images
- Installed NEAT SDK

## Run
### C++
```bash
./build/examples/throughput/yolov8_coco_tput/yolov8_coco_tput <model.tar.gz> <image_dir>
```

## Source Files
- C++: `main.cpp`
