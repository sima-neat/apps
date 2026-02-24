# MiDaS v2.1 RTSP Depth Estimation

## Metadata
| Field | Value |
| --- | --- |
| Category | depth-estimation |
| Difficulty | Intermediate |
| Tags | depth-estimation, rtsp |
| Status | experimental |
| Binary Name | midas_v21_rtsp |

## Concept
Depth estimation from an RTSP camera stream using a MiDaS v2.1 model with the Session API.

## Prerequisites
- Compiled MiDaS v2.1 MPK (`.tar.gz`)
- RTSP camera source
- Installed NEAT SDK

## Run
### C++
```bash
./build/examples/depth-estimation/midas_v21_rtsp/midas_v21_rtsp <model.tar.gz> <rtsp_url>
```

## Source Files
- C++: `main.cpp`
