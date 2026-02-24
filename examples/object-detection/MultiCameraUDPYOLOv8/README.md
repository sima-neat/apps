# Multi-Camera UDP YOLOv8

## Metadata
| Field | Value |
| --- | --- |
| Category | object-detection |
| Difficulty | Advanced |
| Tags | object-detection, rtsp |
| Status | experimental |
| Binary Name | MultiCameraUDPYOLOv8 |
| Model | yolo_v8s |

## Concept
Multi-camera YOLOv8 detection with UDP output using the Session API and GStreamer pipelines.

## Prerequisites
- Installed NEAT SDK
- RTSP camera sources
- Model downloaded: `./scripts/download_models.sh` (or `sima-cli modelzoo get yolo_v8s`)

## Run
### C++
```bash
./build/examples/object-detection/MultiCameraUDPYOLOv8/MultiCameraUDPYOLOv8 --rtsp <rtsp_url>
```

## Source Files
- C++: `main.cpp`
