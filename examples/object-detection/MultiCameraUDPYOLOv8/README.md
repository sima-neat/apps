# Multi-Camera UDP YOLOv8

## Metadata
| Field | Value |
| --- | --- |
| Category | object-detection |
| Difficulty | Advanced |
| Tags | object-detection, rtsp |
| Status | experimental |
| Binary Name | MultiCameraUDPYOLOv8 |

## Concept
Multi-camera YOLOv8 detection with UDP output using the Session API and GStreamer pipelines.

## Prerequisites
- Compiled YOLOv8 MPK (`.tar.gz`)
- RTSP camera sources
- Installed NEAT SDK

## Run
### C++
```bash
./build/examples/object-detection/MultiCameraUDPYOLOv8/MultiCameraUDPYOLOv8 <model.tar.gz> <rtsp_url> [rtsp_url...]
```

## Source Files
- C++: `main.cpp`
