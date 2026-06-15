# Model Benchmark Results

This file is a manually maintained reference table for Apps-supported compiled model packages.
The numbers come from `pyneat.Model.benchmark()` through `src/python/main.py`.

These results measure the compiled model package only. They do not include camera ingest, RTSP decode, Insight output, metadata conversion, overlays, or application postprocessing.

## Run Context

| Field | Value |
| --- | --- |
| Target | Modalix (`aarch64`) |
| SDK | 2.1.2 |
| Date | 2026-06-12 |
| Frames | 1000 |
| Command | `python3 examples/benchmarking/model-benchmark/src/python/main.py --model <model-package>` |
| Refresh Script | `python3 examples/benchmarking/model-benchmark/scripts/refresh_results.py --run` |
| JSON Reports | `sandbox/model-benchmark/runs/<model-id>.json` |
| Power Columns | Omitted |

## General Models

| Model ID | Package | Used By | Latency / FPS |
| --- | --- | --- | ---: |
| `resnet_50` | `resnet_50_mpk.tar.gz` | image-classifier | 2.264 / 918.20 |
| `depth_anything_v2_vits` | `depth_anything_v2_vits_mpk.tar.gz` | depth-estimator | 20.695 / 54.56 |
| `retinaface_mobilenet25` | `retinaface_mobilenet25_mod_0_mpk.tar.gz` | face-detector | 4.383 / 608.04 |
| `detr_resnet50_modified_class_embed_bbox_embed` | `detr_resnet50_modified_class_embed_bbox_embed_mpk.tar.gz` | detr-object-detector | 23.672 / 68.15 |
| `yolo_v8n_seg` | `yolo_v8n_seg_mpk.tar.gz` | yolov8-instance-segmenter | 6.600 / 387.95 |

## YOLO26 Detection

| Model ID | Package | Used By | Latency / FPS |
| --- | --- | --- | ---: |
| `yolo26n-det-bf16-mla_tess-b1` | `yolo26n-det-bf16-mla_tess-b1.tar.gz` | single-stream-object-detector, multi-stream-object-detector, multi-stream-people-tracker | 6.427 / 348.86 |
| `yolo26s-det-bf16-mla_tess-b1` | `yolo26s-det-bf16-mla_tess-b1.tar.gz` | single-stream-object-detector, multi-stream-object-detector, multi-stream-people-tracker | 8.984 / 180.09 |
| `yolo26m-det-bf16-mla_tess-b1` | `yolo26m-det-bf16-mla_tess-b1.tar.gz` | single-stream-object-detector, multi-stream-object-detector, yolo26-object-detector, detection-to-vlm-assistant, multi-stream-people-tracker | 16.762 / 62.57 |
| `yolo26l-det-bf16-mla_tess-b1` | `yolo26l-det-bf16-mla_tess-b1.tar.gz` | single-stream-object-detector, multi-stream-object-detector, multi-stream-people-tracker | 20.107 / 50.71 |
| `yolo26x-det-bf16-mla_tess-b1` | `yolo26x-det-bf16-mla_tess-b1.tar.gz` | single-stream-object-detector, multi-stream-object-detector, multi-stream-people-tracker | 49.978 / 27.83 |
| `yolo26m-det-bf16-b1` | `yolo26m-det-bf16-b1.tar.gz` | single-stream-object-detector, multi-stream-object-detector, detection-to-vlm-assistant, multi-stream-people-tracker | 21.110 / 77.23 |
| `yolo26m-det-int8-b1` | `yolo26m-det-int8-b1.tar.gz` | single-stream-object-detector, multi-stream-object-detector, detection-to-vlm-assistant, multi-stream-people-tracker | 6.764 / 271.33 |

## YOLO26 Segmentation

| Model ID | Package | Used By | Latency / FPS |
| --- | --- | --- | ---: |
| `yolo26n-seg-bf16-mla_tess` | `yolo26n-seg-bf16-mla_tess.tar.gz` | single-stream-instance-segmenter | 9.290 / 238.74 |
| `yolo26s-seg-bf16-mla_tess` | `yolo26s-seg-bf16-mla_tess.tar.gz` | single-stream-instance-segmenter | 15.474 / 131.85 |
| `yolo26m-seg-bf16-mla_tess` | `yolo26m-seg-bf16-mla_tess.tar.gz` | single-stream-instance-segmenter | 32.663 / 50.02 |
| `yolo26l-seg-bf16-mla_tess` | `yolo26l-seg-bf16-mla_tess.tar.gz` | single-stream-instance-segmenter | 28.501 / 42.89 |
| `yolo26x-seg-bf16-mla_tess` | `yolo26x-seg-bf16-mla_tess.tar.gz` | single-stream-instance-segmenter | 55.701 / 19.84 |
| `yolo26m-seg-bf16-b1` | `yolo26m-seg-bf16-b1.tar.gz` | single-stream-instance-segmenter | 26.983 / 50.43 |
| `yolo26m-seg-bf16-mla_tess-b1` | `yolo26m-seg-bf16-mla_tess-b1.tar.gz` | single-stream-instance-segmenter | 25.012 / 50.53 |
| `yolo26m-seg-int8-b1` | `yolo26m-seg-int8-b1.tar.gz` | single-stream-instance-segmenter | 9.991 / 195.12 |
