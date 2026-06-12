# Model Benchmark Results

This file records model-only synthetic benchmark results from `pyneat.Model.benchmark()`.
It does not measure camera ingest, RTSP decode, Insight output, metadata conversion, overlays, or application postprocessing.

## Run Context

| Field | Value |
| --- | --- |
| Command | `python3 examples/benchmarking/model-benchmark/src/python/main.py --model <model-package>` |
| Config | `examples/benchmarking/model-benchmark/src/common/config.yaml` |
| Frames | 1000 |
| Report JSON | `sandbox/model-benchmark/report.json` |
| Power Columns | Omitted |
| Status | Pending: local DevKit `192.168.2.7:22` was unreachable, so benchmark runs did not start. |

## General Models

| Model ID | Package | Used By | Status | Latency ms | FPS | Date |
| --- | --- | --- | --- | ---: | ---: | --- |
| `resnet_50` | `resnet_50_mpk.tar.gz` | image-classifier | Pending | TBD | TBD | TBD |
| `depth_anything_v2_vits` | `depth_anything_v2_vits_mpk.tar.gz` | depth-estimator | Pending | TBD | TBD | TBD |
| `retinaface_mobilenet25` | `retinaface_mobilenet25_mod_0_mpk.tar.gz` | face-detector | Pending | TBD | TBD | TBD |
| `detr_resnet50_modified_class_embed_bbox_embed` | `detr_resnet50_modified_class_embed_bbox_embed_mpk.tar.gz` | detr-object-detector | Pending | TBD | TBD | TBD |
| `yolo_v8n_seg` | `yolo_v8n_seg_mpk.tar.gz` | yolov8-instance-segmenter | Pending | TBD | TBD | TBD |

## YOLO26 Detection

| Model ID | Package | Used By | Status | Latency ms | FPS | Date |
| --- | --- | --- | --- | ---: | ---: | --- |
| `yolo26n-det-bf16-mla_tess-b1` | `yolo26n-det-bf16-mla_tess-b1.tar.gz` | single-stream-object-detector, multi-stream-object-detector, multi-stream-people-tracker | Pending | TBD | TBD | TBD |
| `yolo26s-det-bf16-mla_tess-b1` | `yolo26s-det-bf16-mla_tess-b1.tar.gz` | single-stream-object-detector, multi-stream-object-detector, multi-stream-people-tracker | Pending | TBD | TBD | TBD |
| `yolo26m-det-bf16-mla_tess-b1` | `yolo26m-det-bf16-mla_tess-b1.tar.gz` | single-stream-object-detector, multi-stream-object-detector, yolo26-object-detector, detection-to-vlm-assistant, multi-stream-people-tracker | Pending | TBD | TBD | TBD |
| `yolo26l-det-bf16-mla_tess-b1` | `yolo26l-det-bf16-mla_tess-b1.tar.gz` | single-stream-object-detector, multi-stream-object-detector, multi-stream-people-tracker | Pending | TBD | TBD | TBD |
| `yolo26x-det-bf16-mla_tess-b1` | `yolo26x-det-bf16-mla_tess-b1.tar.gz` | single-stream-object-detector, multi-stream-object-detector, multi-stream-people-tracker | Pending | TBD | TBD | TBD |
| `yolo26m-det-bf16-b1` | `yolo26m-det-bf16-b1.tar.gz` | single-stream-object-detector, multi-stream-object-detector, detection-to-vlm-assistant, multi-stream-people-tracker | Pending | TBD | TBD | TBD |
| `yolo26m-det-int8-b1` | `yolo26m-det-int8-b1.tar.gz` | single-stream-object-detector, multi-stream-object-detector, detection-to-vlm-assistant, multi-stream-people-tracker | Pending | TBD | TBD | TBD |

## YOLO26 Segmentation

| Model ID | Package | Used By | Status | Latency ms | FPS | Date |
| --- | --- | --- | --- | ---: | ---: | --- |
| `yolo26n-seg-bf16-mla_tess` | `yolo26n-seg-bf16-mla_tess.tar.gz` | single-stream-instance-segmenter | Pending | TBD | TBD | TBD |
| `yolo26s-seg-bf16-mla_tess` | `yolo26s-seg-bf16-mla_tess.tar.gz` | single-stream-instance-segmenter | Pending | TBD | TBD | TBD |
| `yolo26m-seg-bf16-mla_tess` | `yolo26m-seg-bf16-mla_tess.tar.gz` | single-stream-instance-segmenter | Pending | TBD | TBD | TBD |
| `yolo26l-seg-bf16-mla_tess` | `yolo26l-seg-bf16-mla_tess.tar.gz` | single-stream-instance-segmenter | Pending | TBD | TBD | TBD |
| `yolo26x-seg-bf16-mla_tess` | `yolo26x-seg-bf16-mla_tess.tar.gz` | single-stream-instance-segmenter | Pending | TBD | TBD | TBD |
| `yolo26m-seg-bf16-b1` | `yolo26m-seg-bf16-b1.tar.gz` | single-stream-instance-segmenter | Pending | TBD | TBD | TBD |
| `yolo26m-seg-bf16-mla_tess-b1` | `yolo26m-seg-bf16-mla_tess-b1.tar.gz` | single-stream-instance-segmenter | Pending | TBD | TBD | TBD |
| `yolo26m-seg-int8-b1` | `yolo26m-seg-int8-b1.tar.gz` | single-stream-instance-segmenter | Pending | TBD | TBD | TBD |
