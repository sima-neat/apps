# Graph Pipes YOLO OptiView

## Metadata
| Field | Value |
| --- | --- |
| Category | object-detection |
| Difficulty | Advanced |
| Tags | object-detection, rtsp, optiview, multistream, graph |
| Status | experimental |
| Binary Name | GraphPipesYOLOOptiview |
| Model | yolo_v8s |

## Concept
Multi-channel YOLO detection using the Graph API with strict frame synchronization. Forwards encoded streams to OptiView via UDP while running detection on decoded frames.

## Prerequisites
- Installed NEAT SDK
- RTSP camera sources
- OptiView endpoint
- Model downloaded: `./scripts/download_models.sh` (or `sima-cli modelzoo get yolo_v8s`)

## Run
### C++
```bash
./build/examples/object-detection/GraphPipesYOLOOptiview/GraphPipesYOLOOptiview --rtsp <rtsp_url> [--rtsp <rtsp_url>...]
```

## Source Files
- C++: `main.cpp`
