# 4-Pipe YOLO OptiView

## Metadata
| Field | Value |
| --- | --- |
| Category | object-detection |
| Difficulty | Advanced |
| Tags | object-detection, rtsp, optiview, multistream |
| Status | experimental |
| Binary Name | 4PipesYOLOOptiview |
| Model | yolo_v8s |

## Concept
Four-camera YOLO detection pipeline with concurrent RTSP capture, decode, and UDP forwarding to OptiView. Uses bounded queues and frame dropping under load.

## Prerequisites
- Installed NEAT SDK
- Up to 4 RTSP camera sources
- OptiView endpoint
- Model downloaded: `sima-cli modelzoo get yolo_v8s`

## Run
### C++
```bash
./build/examples/object-detection/4PipesYOLOOptiview/4PipesYOLOOptiview --rtsp <rtsp_url> [--rtsp <rtsp_url>...]
```

## Source Files
- C++: `main.cpp`
