# YOLOv8 Multi-RTSP Demo

## Metadata
| Field | Value |
| --- | --- |
| Category | object-detection |
| Difficulty | Advanced |
| Tags | object-detection, rtsp, multistream |
| Status | experimental |
| Binary Name | multistream-rtsp-detection-pipeline |
| Model | yolo_v8m |

## Concept
Multi-camera RTSP object detection using YOLOv8 with the Session API. Demonstrates concurrent stream handling with real-time bounding box visualization via OpenCV.

Both C++ and Python versions process frames from each stream, run YOLOv8 inference, and can draw/save annotated JPEG outputs in per-stream subfolders under `--output`.

`--frames` controls how many frames are processed per stream.  
`--save-every` controls output cadence (default `10`), so saved images per stream are approximately `frames / save-every`.

## Supported Models
Also works with: `yolo_v8n`, `yolo_v8s`, `yolo_v8l`

Download any variant: `sima-cli modelzoo get yolo_v8n`

## Prerequisites
- Installed NEAT SDK
- One or more RTSP camera sources (or use the file-based RTSP server below)
- Model downloaded: `sima-cli modelzoo get yolo_v8m`

## Setting Up RTSP Streams (using a video file)

If you don't have live RTSP cameras, use the included multi-file RTSP server to simulate multiple camera streams from a single video file:

```bash
python utils/rtsp/rtsp_multi_file_server.py /path/to/video.mp4 --streams 4 --width 1280 --height 720 --fps 10
```

This will serve the video on `rtsp://127.0.0.1:8554/stream0` through `rtsp://127.0.0.1:8554/stream3`.

## Run
### C++
```bash
./build/examples/object-detection/multistream-rtsp-detection-pipeline/multistream-rtsp-detection-pipeline \
  --model /path/to/yolo_v8m_mpk.tar.gz \
  --output /path/to/output \
  --labels-file examples/object-detection/multistream-rtsp-detection-pipeline/coco_label.txt \
  --frames 100 \
  --tcp \
  --fps 10 \
  --sample-every 1 \
  --save-every 10 \
  --run-queue-depth 4 \
  --overflow-policy keep-latest \
  --output-memory owned \
  --min-score 0.60 \
  --nms-iou 0.50 \
  --max-det 100 \
  --model-timeout-ms 3000 \
  --frame-queue 128 \
  --result-queue 128 \
  --pull-timeout-ms 150 \
  --max-idle-ms 15000 \
  --reconnect-miss 3 \
  --rtsp rtsp://127.0.0.1:8554/stream0 \
  --rtsp rtsp://127.0.0.1:8554/stream1 \
  --rtsp rtsp://127.0.0.1:8554/stream2 \
  --rtsp rtsp://127.0.0.1:8554/stream3
```

### Python
```bash
source ~/pyneat/.venv/bin/activate
pip install -r examples/object-detection/multistream-rtsp-detection-pipeline/requirements.txt
python examples/object-detection/multistream-rtsp-detection-pipeline/main.py \
  --model /path/to/yolo_v8m_mpk.tar.gz \
  --output /path/to/output \
  --labels-file examples/object-detection/multistream-rtsp-detection-pipeline/coco_label.txt \
  --frames 100 \
  --tcp \
  --fps 10 \
  --sample-every 1 \
  --save-every 10 \
  --run-queue-depth 4 \
  --overflow-policy keep-latest \
  --output-memory owned \
  --infer-size 640 \
  --min-score 0.60 \
  --nms-iou 0.50 \
  --max-det 100 \
  --model-timeout-ms 3000 \
  --frame-queue 128 \
  --result-queue 128 \
  --pull-timeout-ms 150 \
  --max-idle-ms 15000 \
  --reconnect-miss 3 \
  --rtsp rtsp://127.0.0.1:8554/stream0 \
  --rtsp rtsp://127.0.0.1:8554/stream1 \
  --rtsp rtsp://127.0.0.1:8554/stream2 \
  --rtsp rtsp://127.0.0.1:8554/stream3
```

Optional flags:
```bash
--debug
--profile --profile-every 50
--save-every 10
```

## Source Files
- C++: `main.cpp`
- Python: `main.py`
