# Graph Pipes YOLO OptiView

## Metadata
| Field | Value |
| --- | --- |
| Category | object-detection |
| Difficulty | Advanced |
| Tags | object-detection, rtsp, optiview, multistream, graph |
| Status | experimental |
| Binary Name | GraphPipesYOLOOptiview |

## Concept
Multi-channel YOLO detection using the Graph API with strict frame synchronization. Forwards encoded streams to OptiView via UDP while running detection on decoded frames.

## Prerequisites
- Compiled YOLO MPK (`.tar.gz`)
- RTSP camera sources
- OptiView endpoint
- Installed NEAT SDK

## Run
### C++
```bash
./build/examples/object-detection/GraphPipesYOLOOptiview/GraphPipesYOLOOptiview <model.tar.gz> <rtsp_url> [rtsp_url...]
```

## Source Files
- C++: `main.cpp`
