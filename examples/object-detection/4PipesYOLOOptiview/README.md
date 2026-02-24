# 4-Pipe YOLO OptiView

## Metadata
| Field | Value |
| --- | --- |
| Category | object-detection |
| Difficulty | Advanced |
| Tags | object-detection, rtsp, optiview, multistream |
| Status | experimental |
| Binary Name | 4PipesYOLOOptiview |

## Concept
Four-camera YOLO detection pipeline with concurrent RTSP capture, decode, and UDP forwarding to OptiView. Uses bounded queues and frame dropping under load.

## Prerequisites
- Compiled YOLO MPK (`.tar.gz`)
- Up to 4 RTSP camera sources
- OptiView endpoint
- Installed NEAT SDK

## Run
### C++
```bash
./build/examples/object-detection/4PipesYOLOOptiview/4PipesYOLOOptiview <model.tar.gz> <rtsp_url> [rtsp_url...]
```

## Source Files
- C++: `main.cpp`
