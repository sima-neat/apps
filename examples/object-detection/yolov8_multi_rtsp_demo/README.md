# YOLOv8 Multi-RTSP Demo

## Metadata
| Field | Value |
| --- | --- |
| Category | object-detection |
| Difficulty | Advanced |
| Tags | object-detection, rtsp, multistream |
| Status | experimental |
| Binary Name | yolov8_multi_rtsp_demo |
| Model | yolo_v8s |

## Concept
Multi-camera RTSP object detection using YOLOv8 with the Session API. Demonstrates concurrent stream handling.

## Prerequisites
- Installed NEAT SDK
- One or more RTSP camera sources
- Model downloaded: `sima-cli modelzoo get yolo_v8s`

## Run
### C++
```bash
./build/examples/object-detection/yolov8_multi_rtsp_demo/yolov8_multi_rtsp_demo --rtsp-list utils/rtsp/rtsp_list.sample.txt
```

## Source Files
- C++: `main.cpp`
