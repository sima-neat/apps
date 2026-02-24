# MiDaS v2.1 RTSP Depth Estimation

## Metadata
| Field | Value |
| --- | --- |
| Category | depth-estimation |
| Difficulty | Intermediate |
| Tags | depth-estimation, rtsp |
| Status | experimental |
| Binary Name | midas_v21_rtsp |
| Model | midas_v21_small_256 |

## Concept
Depth estimation from an RTSP camera stream using a MiDaS v2.1 model with the Session API.

## Prerequisites
- Installed NEAT SDK
- RTSP camera source
- Model downloaded: `./scripts/download_models.sh` (or `sima-cli modelzoo get midas_v21_small_256`)

## Run
### C++
```bash
./build/examples/depth-estimation/midas_v21_rtsp/midas_v21_rtsp --model models/midas_v21_small_256_mpk.tar.gz --rtsp <rtsp_url>
```

## Source Files
- C++: `main.cpp`
