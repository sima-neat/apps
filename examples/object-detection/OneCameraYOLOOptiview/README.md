# Single-Camera YOLO OptiView

## Metadata
| Field | Value |
| --- | --- |
| Category | object-detection |
| Difficulty | Intermediate |
| Tags | object-detection, rtsp, optiview |
| Status | experimental |
| Binary Name | OneCameraYOLOOptiview |
| Model | yolo_v8s |

## Concept
Single-camera YOLO detection with results forwarded to OptiView for visualization over RTSP.

## Prerequisites
- Installed NEAT SDK
- RTSP camera source
- OptiView endpoint
- Model downloaded: `sima-cli modelzoo get yolo_v8s`

## Run
### C++
```bash
./build/examples/object-detection/OneCameraYOLOOptiview/OneCameraYOLOOptiview --rtsp <rtsp_url>
```

## Source Files
- C++: `main.cpp`
