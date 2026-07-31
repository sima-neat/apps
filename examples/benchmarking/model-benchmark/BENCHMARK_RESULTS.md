# Model Benchmark Results

This file is a manually maintained reference table for Apps-supported compiled model packages.
The numbers come from `pyneat.Model.benchmark()` through `src/python/main.py`.

These results measure the compiled model package only. They do not include camera ingest, RTSP decode, Insight output, metadata conversion, overlays, or application postprocessing.

## Run Context

| Field | Value |
| --- | --- |
| Target | Modalix (`aarch64`) |
| SDK | 2.1.3_master_B4040 (neat core 0.4.0+develop.0d77e009b73b) |
| Date | 2026-07-31 |
| Frames | 1000 |
| Command | `python3 examples/benchmarking/model-benchmark/src/python/main.py --model <model-package> [--decode-type <yolo26-det\|yolo26-seg>]` |
| Refresh Script | `python3 examples/benchmarking/model-benchmark/scripts/refresh_results.py --run` |
| JSON Reports | `sandbox/model-benchmark/runs/<model-id>.json` |
| Power Columns | Omitted |

## General Models

| Model ID | Package | Used By | Postprocess | Outputs | Latency / FPS |
| --- | --- | --- | --- | ---: | ---: |
| `resnet_50` | `resnet_50_mpk.tar.gz` | image-classifier | `detessdequant` | 1 | 2.360 / 1102.25 |
| `depth_anything_v2_vits` | `depth_anything_v2_vits_mpk.tar.gz` | depth-estimator | `detessdequant` | 1 | 20.841 / 55.01 |
| `retinaface_mobilenet25` | `retinaface_mobilenet25_mod_0_mpk.tar.gz` | face-detector | `detessdequant` | 9 | 4.540 / 637.29 |
| `detr_resnet50_modified_class_embed_bbox_embed` | `detr_resnet50_modified_class_embed_bbox_embed_mpk.tar.gz` | detr-object-detector | `detessdequant` | 2 | 23.426 / 68.86 |
| `yolo_v8n_seg` | `yolo_v8n_seg_mpk.tar.gz` | yolov8-instance-segmenter | `detessdequant` | 10 | 6.749 / 385.41 |

## YOLO26 Detection

| Model ID | Package | Used By | Postprocess | Outputs | Latency / FPS |
| --- | --- | --- | --- | ---: | ---: |
| `yolo26n-det-bf16-mla_tess-b1` | `yolo26n-det-bf16-mla_tess-b1.tar.gz` | single-stream-object-detector, multi-stream-object-detector, multi-stream-people-tracker | `cast` | 6 | 6.757 / 377.67 |
| `yolo26s-det-bf16-mla_tess-b1` | `yolo26s-det-bf16-mla_tess-b1.tar.gz` | single-stream-object-detector, multi-stream-object-detector, multi-stream-people-tracker | `cast` | 6 | 9.321 / 188.20 |
| `yolo26m-det-bf16-mla_tess-b1` | `yolo26m-det-bf16-mla_tess-b1.tar.gz` | single-stream-object-detector, multi-stream-object-detector, yolo26-object-detector, detection-to-vlm-assistant, multi-stream-people-tracker | `cast` | 6 | 17.024 / 77.33 |
| `yolo26l-det-bf16-mla_tess-b1` | `yolo26l-det-bf16-mla_tess-b1.tar.gz` | single-stream-object-detector, multi-stream-object-detector, multi-stream-people-tracker | `cast` | 6 | 20.331 / 61.34 |
| `yolo26x-det-bf16-mla_tess-b1` | `yolo26x-det-bf16-mla_tess-b1.tar.gz` | single-stream-object-detector, multi-stream-object-detector, multi-stream-people-tracker | `cast` | 6 | 39.772 / 27.98 |
| `yolo26m-det-bf16-b1` | `yolo26m-det-bf16-b1.tar.gz` | single-stream-object-detector, multi-stream-object-detector, detection-to-vlm-assistant, multi-stream-people-tracker | `unknown` | 6 | 18.143 / 78.15 |
| `yolo26m-det-int8-b1` | `yolo26m-det-int8-b1.tar.gz` | single-stream-object-detector, multi-stream-object-detector, detection-to-vlm-assistant, multi-stream-people-tracker | `detessdequant` | 6 | 7.076 / 300.09 |
| `yolo26m-det-int8-b1-boxdecode` | `yolo26m-det-int8-b1.tar.gz` | single-stream-object-detector, multi-stream-object-detector, detection-to-vlm-assistant, multi-stream-people-tracker | `boxdecode` | 1 | 6.898 / 304.88 |

## YOLO26 Segmentation

| Model ID | Package | Used By | Postprocess | Outputs | Latency / FPS |
| --- | --- | --- | --- | ---: | ---: |
| `yolo26n-seg-bf16-mla_tess` | `yolo26n-seg-bf16-mla_tess.tar.gz` | single-stream-instance-segmenter | `cast` | 10 | 9.062 / 296.73 |
| `yolo26s-seg-bf16-mla_tess` | `yolo26s-seg-bf16-mla_tess.tar.gz` | single-stream-instance-segmenter | `cast` | 10 | 13.146 / 135.03 |
| `yolo26m-seg-bf16-mla_tess` | `yolo26m-seg-bf16-mla_tess.tar.gz` | single-stream-instance-segmenter | `cast` | 10 | 25.548 / 50.48 |
| `yolo26l-seg-bf16-mla_tess` | `yolo26l-seg-bf16-mla_tess.tar.gz` | single-stream-instance-segmenter | `cast` | 10 | 28.980 / 43.14 |
| `yolo26x-seg-bf16-mla_tess` | `yolo26x-seg-bf16-mla_tess.tar.gz` | single-stream-instance-segmenter | `cast` | 10 | 56.054 / 19.91 |
| `yolo26m-seg-bf16-b1` | `yolo26m-seg-bf16-b1.tar.gz` | single-stream-instance-segmenter | `unknown` | 10 | 27.146 / 50.90 |
| `yolo26m-seg-bf16-mla_tess-b1` | `yolo26m-seg-bf16-mla_tess-b1.tar.gz` | single-stream-instance-segmenter | `cast` | 10 | 25.248 / 51.09 |
| `yolo26m-seg-int8-b1` | `yolo26m-seg-int8-b1.tar.gz` | single-stream-instance-segmenter | `detessdequant` | 10 | 10.403 / 201.88 |
| `yolo26m-seg-int8-b1-boxdecode` | `yolo26m-seg-int8-b1.tar.gz` | single-stream-instance-segmenter | `boxdecode` | 1 | 11.736 / 205.43 |
