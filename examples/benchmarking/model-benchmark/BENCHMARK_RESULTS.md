# Model Benchmark Results

This file is a manually maintained reference table for Apps-supported compiled model packages: every package an example README downloads from the Model Zoo.
The numbers come from `pyneat.Model.benchmark()` through `src/python/main.py`.

These results measure the compiled model package only, through the route the package declares, with synthetic input. They do not include camera ingest, RTSP decode, Insight output, metadata conversion, overlays, application postprocessing, or decode and preprocess options an application sets.

Latency is per sample from the latency pass. FPS comes from the separate throughput pass, which keeps several samples in flight.

## Run Context

| Field | Value |
| --- | --- |
| Target | Modalix (`aarch64`) |
| SDK | 3.0.0_master_B2501 (neat core 0.6.0+develop.ece040d9a962), Model Zoo 3.0.0 |
| Date | 2026-10-30 |
| Minimum Frames per Pass | 100 |
| Minimum Duration per Pass | 30000 ms |
| Command | `python3 examples/benchmarking/model-benchmark/src/python/main.py --model <model-package> --frames <frames> --min-duration-ms <ms> [--decode-type <yolo26-det\|yolo26-seg>]` |
| Refresh Script | `python3 examples/benchmarking/model-benchmark/scripts/refresh_results.py --run --models-dir <dir>` |
| JSON Reports | `sandbox/model-benchmark/runs/<model-id>.json` |
| Power Columns | Omitted: board telemetry measures the whole board, not the model |
| BoxDecode Options | `top_k=100`; thresholds come from the model package |

## Classification

| Model ID | Package | Used By | Postprocess | Outputs | Latency ms (mean / p50 / p99) | FPS |
| --- | --- | --- | --- | ---: | ---: | ---: |
| `resnet_50` | `resnet_50_mpk.tar.gz` | image-classification-explorer, mipi-multi-model | `detessdequant` | 1 | 2.29 / 2.29 / 2.40 | 889.4 |
| `resnet_18` | `resnet_18_mpk.tar.gz` | image-classification-explorer | `detessdequant` | 1 | 1.70 / 1.70 / 1.78 | 1838.8 |
| `efficientnet_b0` | `efficientnet_b0_mpk.tar.gz` | image-classification-explorer | `detessdequant` | 1 | 1.84 / 1.84 / 1.91 | 1528.0 |
| `densenet_121` | `densenet_121_mpk.tar.gz` | image-classification-explorer | `detessdequant` | 1 | 2.55 / 2.51 / 2.59 | 726.4 |

## Detection

| Model ID | Package | Used By | Postprocess | Outputs | Latency ms (mean / p50 / p99) | FPS |
| --- | --- | --- | --- | ---: | ---: | ---: |
| `detr_resnet50_modified_class_embed_bbox_embed` | `detr_resnet50_modified_class_embed_bbox_embed_mpk.tar.gz` | detr-object-detector | `detessdequant` | 2 | 29.52 / 29.51 / 29.89 | 42.8 |
| `ssd_mobilenet_v2_modalix_int8_tess_mla` | `ssd_mobilenet_v2_modalix_int8_tess_mla_mpk.tar.gz` | ssd-mobilenet-object-detector, mipi-multi-model | `dequantize` | 12 | 3.23 / 3.20 / 3.83 | 1035.6 |
| `yolo_11s` | `yolo_11s_mpk.tar.gz` | multi-stream-multi-model | `detessdequant` | 6 | 6.34 / 6.25 / 6.86 | 440.7 |
| `yolo26n_plc` | `yolo26n_plc_mpk.tar.gz` | pcb-defect-detector | `cast` | 6 | 6.15 / 6.15 / 6.76 | 267.0 |
| `yolo26n_p2_tiny_drone_int8_qat_b1` | `yolo26n_p2_tiny_drone_int8_qat_b1_mpk.tar.gz` | yolo26-tiny-drone-tracker, parallel-apps | `dequantize` | 8 | 20.21 / 20.19 / 20.96 | 81.9 |
| `yolo26n_obb_w8a8` | `yolo26n_obb_w8a8_mpk.tar.gz` | multi-stream-obb-detector | `dequantize` | 9 | 8.24 / 8.23 / 8.86 | 344.3 |
| `yolo26n_obb_w16a16` | `yolo26n_obb_w16a16_mpk.tar.gz` | multi-stream-obb-detector | `cast` | 9 | 14.96 / 14.93 / 15.80 | 99.1 |

## Face Detection

| Model ID | Package | Used By | Postprocess | Outputs | Latency ms (mean / p50 / p99) | FPS |
| --- | --- | --- | --- | ---: | ---: | ---: |
| `retinaface_mobilenet25` | `retinaface_mobilenet25_mod_0_mpk.tar.gz` | face-detector | `detessdequant` | 9 | 3.51 / 3.50 / 4.14 | 983.0 |
| `yolov5s_face_raw_split` | `yolov5s_face_raw_split_mpk.tar.gz` | single-stream-thermal-face-detector | `detessdequant` | 6 | 7.28 / 7.27 / 7.89 | 466.3 |

## YOLO26 Detection

| Model ID | Package | Used By | Postprocess | Outputs | Latency ms (mean / p50 / p99) | FPS |
| --- | --- | --- | --- | ---: | ---: | ---: |
| `yolo26n-det-bf16-mla_tess-b1` | `yolo26n-det-bf16-mla_tess-b1.tar.gz` | single-stream-object-detector, multi-stream-object-detector, multi-stream-tracker, yolo26-object-detector, detection-to-vlm-assistant, mipi-multi-model | `cast` | 6 | 6.89 / 6.88 / 7.54 | 268.7 |
| `yolo26s-det-bf16-mla_tess-b1` | `yolo26s-det-bf16-mla_tess-b1.tar.gz` | single-stream-object-detector, multi-stream-object-detector, multi-stream-tracker, yolo26-object-detector, detection-to-vlm-assistant | `cast` | 6 | 11.41 / 11.39 / 12.03 | 121.6 |
| `yolo26m-det-bf16-mla_tess-b1` | `yolo26m-det-bf16-mla_tess-b1.tar.gz` | single-stream-object-detector, multi-stream-object-detector, multi-stream-tracker, yolo26-object-detector, detection-to-vlm-assistant | `cast` | 6 | 24.27 / 24.16 / 24.84 | 47.7 |
| `yolo26l-det-bf16-mla_tess-b1` | `yolo26l-det-bf16-mla_tess-b1.tar.gz` | single-stream-object-detector, multi-stream-object-detector, multi-stream-tracker, yolo26-object-detector, detection-to-vlm-assistant | `cast` | 6 | 29.96 / 29.84 / 30.44 | 37.7 |
| `yolo26x-det-bf16-mla_tess-b1` | `yolo26x-det-bf16-mla_tess-b1.tar.gz` | single-stream-object-detector, multi-stream-object-detector, multi-stream-tracker, yolo26-object-detector, detection-to-vlm-assistant | `cast` | 6 | 62.08 / 62.05 / 62.67 | 17.0 |
| `yolo26m-det-bf16-b1` | `yolo26m-det-bf16-b1.tar.gz` | single-stream-object-detector, multi-stream-object-detector, multi-stream-tracker, yolo26-object-detector, detection-to-vlm-assistant | `unknown` | 6 | 25.82 / 25.81 / 26.50 | 48.6 |
| `yolo26n-det-int8-b1` | `yolo26n-det-int8-b1.tar.gz` | high-density-multi-stream-object-detector, pcie-high-density-multi-stream-object-detector, mipi-multi-model | `dequantize` | 6 | 4.45 / 4.44 / 5.02 | 792.9 |
| `yolo26m-det-int8-b1` | `yolo26m-det-int8-b1.tar.gz` | single-stream-object-detector, multi-stream-object-detector, multi-stream-tracker, yolo26-object-detector, detection-to-vlm-assistant, multi-stream-blazepose3d, multi-stream-multi-model, parallel-apps | `detessdequant` | 6 | 8.51 / 8.50 / 9.11 | 203.5 |
| `yolo26m-det-int8-b1-boxdecode` | `yolo26m-det-int8-b1.tar.gz` | single-stream-object-detector, multi-stream-object-detector, multi-stream-tracker, yolo26-object-detector, detection-to-vlm-assistant, multi-stream-blazepose3d, multi-stream-multi-model, parallel-apps | `boxdecode` | 1 | 9.07 / 9.06 / 9.41 | 203.4 |

## YOLO26 Segmentation

| Model ID | Package | Used By | Postprocess | Outputs | Latency ms (mean / p50 / p99) | FPS |
| --- | --- | --- | --- | ---: | ---: | ---: |
| `yolo26n-seg-bf16-mla_tess` | `yolo26n-seg-bf16-mla_tess.tar.gz` | single-stream-instance-segmenter, mipi-multi-model | `cast` | 10 | 9.90 / 9.89 / 10.63 | 204.2 |
| `yolo26s-seg-bf16-mla_tess` | `yolo26s-seg-bf16-mla_tess.tar.gz` | single-stream-instance-segmenter | `cast` | 10 | 16.43 / 16.41 / 17.08 | 87.4 |
| `yolo26m-seg-bf16-mla_tess` | `yolo26m-seg-bf16-mla_tess.tar.gz` | single-stream-instance-segmenter | `cast` | 10 | 37.32 / 37.30 / 37.99 | 31.0 |
| `yolo26l-seg-bf16-mla_tess` | `yolo26l-seg-bf16-mla_tess.tar.gz` | single-stream-instance-segmenter | `cast` | 10 | 42.88 / 42.85 / 43.54 | 26.4 |
| `yolo26x-seg-bf16-mla_tess` | `yolo26x-seg-bf16-mla_tess.tar.gz` | single-stream-instance-segmenter | `cast` | 10 | 88.39 / 88.01 / 88.71 | 12.1 |
| `yolo26m-seg-bf16-b1` | `yolo26m-seg-bf16-b1.tar.gz` | single-stream-instance-segmenter | `unknown` | 10 | 40.05 / 39.89 / 40.63 | 31.4 |
| `yolo26m-seg-bf16-mla_tess-b1` | `yolo26m-seg-bf16-mla_tess-b1.tar.gz` | single-stream-instance-segmenter | `cast` | 10 | 36.95 / 36.94 / 37.70 | 31.4 |
| `yolo26m-seg-int8-b1` | `yolo26m-seg-int8-b1.tar.gz` | single-stream-instance-segmenter, parallel-apps | `detessdequant` | 10 | 13.24 / 13.22 / 13.89 | 134.0 |
| `yolo26m-seg-int8-b1-boxdecode` | `yolo26m-seg-int8-b1.tar.gz` | single-stream-instance-segmenter, parallel-apps | `boxdecode` | 1 | 36.45 / 36.44 / 37.67 | 73.8 |

## YOLO26 Pose

| Model ID | Package | Used By | Postprocess | Outputs | Latency ms (mean / p50 / p99) | FPS |
| --- | --- | --- | --- | ---: | ---: | ---: |
| `yolo26n-pose-bf16-mla_tess-b1` | `yolo26n-pose-bf16-mla_tess-b1.tar.gz` | multi-stream-pose-estimator, mipi-multi-model | `cast` | 9 | 7.83 / 7.82 / 8.49 | 235.6 |
| `yolo26s-pose-bf16-mla_tess-b1` | `yolo26s-pose-bf16-mla_tess-b1.tar.gz` | multi-stream-pose-estimator | `cast` | 9 | 12.64 / 12.57 / 13.24 | 110.8 |
| `yolo26m-pose-bf16-mla_tess-b1` | `yolo26m-pose-bf16-mla_tess-b1.tar.gz` | multi-stream-pose-estimator | `cast` | 9 | 25.73 / 25.69 / 26.45 | 45.2 |
| `yolo26l-pose-bf16-mla_tess-b1` | `yolo26l-pose-bf16-mla_tess-b1.tar.gz` | multi-stream-pose-estimator | `cast` | 9 | 31.27 / 31.25 / 31.92 | 36.1 |
| `yolo26x-pose-bf16-mla_tess-b1` | `yolo26x-pose-bf16-mla_tess-b1.tar.gz` | multi-stream-pose-estimator | `cast` | 9 | 64.08 / 64.05 / 64.67 | 16.6 |
| `yolo26m-pose-bf16-b1` | `yolo26m-pose-bf16-b1.tar.gz` | multi-stream-pose-estimator | `unknown` | 9 | 27.13 / 27.10 / 27.78 | 46.4 |
| `yolo26m-pose-int8-b1` | `yolo26m-pose-int8-b1.tar.gz` | multi-stream-pose-estimator, multi-stream-multi-model, parallel-apps | `detessdequant` | 9 | 9.57 / 9.43 / 10.07 | 189.9 |

## Other Segmentation

| Model ID | Package | Used By | Postprocess | Outputs | Latency ms (mean / p50 / p99) | FPS |
| --- | --- | --- | --- | ---: | ---: | ---: |
| `yolo_v8n_seg` | `yolo_v8n_seg_mpk.tar.gz` | single-stream-instance-segmenter | `detessdequant` | 10 | 7.73 / 7.62 / 8.30 | 368.1 |
| `yolo_v8s_seg` | `yolo_v8s_seg_mpk.tar.gz` | single-stream-instance-segmenter | `detessdequant` | 10 | 8.96 / 8.95 / 9.60 | 374.4 |
| `yolo_v8m_seg` | `yolo_v8m_seg_mpk.tar.gz` | single-stream-instance-segmenter | `detessdequant` | 10 | 12.12 / 12.11 / 12.79 | 174.2 |
| `yolo_v8l_seg` | `yolo_v8l_seg_mpk.tar.gz` | single-stream-instance-segmenter | `detessdequant` | 10 | 16.29 / 16.28 / 16.97 | 100.3 |
| `yolo_11s_seg` | `yolo_11s_seg_mpk.tar.gz` | multi-stream-multi-model | `detessdequant` | 10 | 9.43 / 9.40 / 10.09 | 315.7 |

## Depth, Keypoints, Pose, And Anomaly

| Model ID | Package | Used By | Postprocess | Outputs | Latency ms (mean / p50 / p99) | FPS |
| --- | --- | --- | --- | ---: | ---: | ---: |
| `depth_anything_v2_vits` | `depth_anything_v2_vits_mpk.tar.gz` | depth-estimator | `detessdequant` | 1 | 32.80 / 32.58 / 33.36 | 33.5 |
| `midas_v21_small_256` | `midas_v21_small_256_mpk.tar.gz` | mipi-multi-model | `detessdequant` | 1 | 3.34 / 3.34 / 3.49 | 816.1 |
| `superpoint` | `superpoint_mpk.tar.gz` | superpoint-feature-extractor | `detessdequant` | 2 | 8.58 / 8.49 / 8.65 | 360.4 |
| `blazepose_ghum_heavy_w16a16` | `blazepose_ghum_heavy_w16a16_mpk.tar.gz` | multi-stream-blazepose3d, parallel-apps | `cast` | 3 | 5.30 / 5.24 / 5.50 | 249.9 |
| `fastflow_demo` | `fastflow_demo_mpk.tar.gz` | fastflow-anomaly-detector | `detessdequant` | 1 | 5.04 / 4.97 / 5.13 | 346.0 |
| `patchcore_wide_resnet50_2_bf16_mla` | `patchcore_wide_resnet50_2_bf16_mla.tar.gz` | patchcore-anomaly-detector | `unknown` | 1 | 10.54 / 10.49 / 10.81 | 220.7 |

## Multi-Package Model Parts

Each of these models runs as two packages. Every part is measured alone, so no row is the cost of the whole model.

| Model ID | Package | Used By | Postprocess | Outputs | Latency ms (mean / p50 / p99) | FPS |
| --- | --- | --- | --- | ---: | ---: | ---: |
| `rf_detr_nano_w8a8_w16a16_backbone` | `rf_detr_nano_w8a8_w16a16_backbone_mpk.tar.gz` | rfdetr-detection-segmentation | `dequantize` | 3 | 7.20 / 7.19 / 7.41 | 181.3 |
| `rf_detr_nano_w8a8_w16a16_decoder` | `rf_detr_nano_w8a8_w16a16_decoder_mpk.tar.gz` | rfdetr-detection-segmentation | `cast` | 2 | 3.12 / 3.12 / 3.29 | 579.4 |
| `rf_detr_small_w8a8_w16a16_backbone` | `rf_detr_small_w8a8_w16a16_backbone_mpk.tar.gz` | rfdetr-detection-segmentation, parallel-apps | `dequantize` | 3 | 12.71 / 12.59 / 12.93 | 94.8 |
| `rf_detr_small_w8a8_w16a16_decoder` | `rf_detr_small_w8a8_w16a16_decoder_mpk.tar.gz` | rfdetr-detection-segmentation, parallel-apps | `cast` | 2 | 4.46 / 4.45 / 4.69 | 334.7 |
| `rf_detr_medium_w8a8_w16a16_backbone` | `rf_detr_medium_w8a8_w16a16_backbone_mpk.tar.gz` | rfdetr-detection-segmentation | `dequantize` | 3 | 18.31 / 18.29 / 18.72 | 62.5 |
| `rf_detr_medium_w8a8_w16a16_decoder` | `rf_detr_medium_w8a8_w16a16_decoder_mpk.tar.gz` | rfdetr-detection-segmentation | `cast` | 2 | 5.54 / 5.52 / 5.85 | 250.9 |
| `rf_detr_large_w8a8_w16a16_backbone` | `rf_detr_large_w8a8_w16a16_backbone_mpk.tar.gz` | rfdetr-detection-segmentation | `dequantize` | 3 | 36.71 / 36.68 / 37.28 | 29.5 |
| `rf_detr_large_w8a8_w16a16_decoder` | `rf_detr_large_w8a8_w16a16_decoder_mpk.tar.gz` | rfdetr-detection-segmentation | `cast` | 2 | 6.24 / 6.24 / 6.44 | 216.1 |
| `rfdetr-seg-medium-backbone` | `rfdetr-seg-medium-backbone.tar.gz` | rfdetr-detection-segmentation | `detessdequant` | 3 | 19.32 / 19.31 / 19.69 | 60.6 |
| `rfdetr-seg-medium-transformer` | `rfdetr-seg-medium-transformer.tar.gz` | rfdetr-detection-segmentation | `unknown` | 3 | 31.94 / 31.90 / 32.90 | 43.0 |
| `efficientsam3` | `efficientsam3_mpk.tar.gz` | efficientsam3-promptable-segmenter | `cast` | 2 | 136.04 / 135.96 / 138.93 | 8.1 |
| `efficientsam3_text_encoder` | `efficientsam3_text_encoder_mpk.tar.gz` | efficientsam3-promptable-segmenter | `cast` | 1 | 26.73 / 26.72 / 26.98 | 40.0 |
