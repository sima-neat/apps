# YOLOv8 Multi-RTSP Demo

## Metadata
| Field | Value |
| --- | --- |
| Category | object-detection |
| Difficulty | Advanced |
| Tags | object-detection, rtsp, multistream |
| Status | experimental |
| Binary Name | yolov8_multi_rtsp_demo |

## Concept
Multi-camera RTSP object detection using YOLOv8 with the Session API. Demonstrates concurrent stream handling.

## Prerequisites
- Compiled YOLOv8 MPK (`.tar.gz`)
- One or more RTSP camera sources
- Installed NEAT SDK

## Run
### C++
```bash
./build/examples/object-detection/yolov8_multi_rtsp_demo/yolov8_multi_rtsp_demo <model.tar.gz> <rtsp_url> [rtsp_url...]
```

## Source Files
- C++: `main.cpp`
