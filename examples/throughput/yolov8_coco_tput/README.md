# YOLOv8 COCO Throughput Benchmark

## Metadata
| Field | Value |
| --- | --- |
| Category | throughput |
| Difficulty | Intermediate |
| Tags | throughput, object-detection |
| Status | experimental |
| Binary Name | yolov8_coco_tput |
| Model | yolo_v8s |

## Concept
Throughput benchmark for YOLOv8 on COCO images using the Session API. Measures inference performance under sustained load.

## Prerequisites
- Installed NEAT SDK
- COCO dataset images
- Model downloaded: `./scripts/download_models.sh` (or `sima-cli modelzoo get yolo_v8s`)

## Run
### C++
```bash
./build/examples/throughput/yolov8_coco_tput/yolov8_coco_tput --model models/yolo_v8s_mpk.tar.gz --coco-dir <coco_val2017_dir>
```

## Source Files
- C++: `main.cpp`
