# Single-Camera YOLO OptiView

## Metadata
| Field | Value |
| --- | --- |
| Category | object-detection |
| Difficulty | Intermediate |
| Tags | object-detection, rtsp, optiview |
| Status | experimental |
| Binary Name | OneCameraYOLOOptiview |

## Concept
Single-camera YOLO detection with results forwarded to OptiView for visualization over RTSP.

## Prerequisites
- Compiled YOLO MPK (`.tar.gz`)
- RTSP camera source
- OptiView endpoint
- Installed NEAT SDK

## Run
### C++
```bash
./build/examples/object-detection/OneCameraYOLOOptiview/OneCameraYOLOOptiview <model.tar.gz> <rtsp_url>
```

## Source Files
- C++: `main.cpp`
