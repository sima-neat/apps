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
| MLA Clock | 1000 MHz |
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
| `resnet_50` | `resnet_50_mpk.tar.gz` | image-classification-explorer, mipi-multi-model | `detessdequant` | 1 | 1.82 / 1.82 / 1.87 | 1440.4 |
| `resnet_18` | `resnet_18_mpk.tar.gz` | image-classification-explorer | `detessdequant` | 1 | 1.48 / 1.47 / 1.54 | 2983.2 |
| `efficientnet_b0` | `efficientnet_b0_mpk.tar.gz` | image-classification-explorer | `detessdequant` | 1 | 1.59 / 1.59 / 1.67 | 2516.0 |
| `densenet_121` | `densenet_121_mpk.tar.gz` | image-classification-explorer | `detessdequant` | 1 | 1.97 / 1.97 / 2.05 | 1220.6 |

## Detection

| Model ID | Package | Used By | Postprocess | Outputs | Latency ms (mean / p50 / p99) | FPS |
| --- | --- | --- | --- | ---: | ---: | ---: |
| `detr_resnet50_modified_class_embed_bbox_embed` | `detr_resnet50_modified_class_embed_bbox_embed_mpk.tar.gz` | detr-object-detector | `detessdequant` | 2 | 20.45 / 20.20 / 20.58 | 70.5 |
| `ssd_mobilenet_v2_modalix_int8_tess_mla` | `ssd_mobilenet_v2_modalix_int8_tess_mla_mpk.tar.gz` | ssd-mobilenet-object-detector, mipi-multi-model | `dequantize` | 12 | 2.85 / 2.80 / 3.44 | 1697.6 |
| `yolo_11s` | `yolo_11s_mpk.tar.gz` | multi-stream-multi-model | `detessdequant` | 6 | 5.51 / 5.43 / 6.03 | 714.7 |
| `yolo26n_plc` | `yolo26n_plc_mpk.tar.gz` | pcb-defect-detector | `cast` | 6 | 4.58 / 4.56 / 5.21 | 440.4 |
| `yolo26n_p2_tiny_drone_int8_qat_b1` | `yolo26n_p2_tiny_drone_int8_qat_b1_mpk.tar.gz` | yolo26-tiny-drone-tracker, parallel-apps | `dequantize` | 8 | 15.24 / 15.21 / 15.93 | 135.4 |
| `yolo26n_obb_w8a8` | `yolo26n_obb_w8a8_mpk.tar.gz` | multi-stream-obb-detector | `dequantize` | 9 | 7.11 / 7.09 / 7.74 | 560.9 |
| `yolo26n_obb_w16a16` | `yolo26n_obb_w16a16_mpk.tar.gz` | multi-stream-obb-detector | `cast` | 9 | 10.82 / 10.81 / 11.50 | 164.7 |

## Face Detection

| Model ID | Package | Used By | Postprocess | Outputs | Latency ms (mean / p50 / p99) | FPS |
| --- | --- | --- | --- | ---: | ---: | ---: |
| `retinaface_mobilenet25` | `retinaface_mobilenet25_mod_0_mpk.tar.gz` | face-detector | `detessdequant` | 9 | 3.27 / 3.25 / 3.88 | 993.2 |
| `yolov5s_face_raw_split` | `yolov5s_face_raw_split_mpk.tar.gz` | single-stream-thermal-face-detector | `detessdequant` | 6 | 6.41 / 6.41 / 7.03 | 523.4 |

## YOLO26 Detection

| Model ID | Package | Used By | Postprocess | Outputs | Latency ms (mean / p50 / p99) | FPS |
| --- | --- | --- | --- | ---: | ---: | ---: |
| `yolo26n-det-bf16-mla_tess-b1` | `yolo26n-det-bf16-mla_tess-b1.tar.gz` | single-stream-object-detector, multi-stream-object-detector, multi-stream-tracker, yolo26-object-detector, detection-to-vlm-assistant, mipi-multi-model | `cast` | 6 | 5.36 / 5.34 / 5.98 | 444.5 |
| `yolo26s-det-bf16-mla_tess-b1` | `yolo26s-det-bf16-mla_tess-b1.tar.gz` | single-stream-object-detector, multi-stream-object-detector, multi-stream-tracker, yolo26-object-detector, detection-to-vlm-assistant | `cast` | 6 | 8.22 / 8.23 / 8.85 | 202.1 |
| `yolo26m-det-bf16-mla_tess-b1` | `yolo26m-det-bf16-mla_tess-b1.tar.gz` | single-stream-object-detector, multi-stream-object-detector, multi-stream-tracker, yolo26-object-detector, detection-to-vlm-assistant | `cast` | 6 | 15.77 / 15.75 / 16.39 | 79.3 |
| `yolo26l-det-bf16-mla_tess-b1` | `yolo26l-det-bf16-mla_tess-b1.tar.gz` | single-stream-object-detector, multi-stream-object-detector, multi-stream-tracker, yolo26-object-detector, detection-to-vlm-assistant | `cast` | 6 | 19.23 / 19.08 / 19.72 | 62.7 |
| `yolo26x-det-bf16-mla_tess-b1` | `yolo26x-det-bf16-mla_tess-b1.tar.gz` | single-stream-object-detector, multi-stream-object-detector, multi-stream-tracker, yolo26-object-detector, detection-to-vlm-assistant | `cast` | 6 | 38.60 / 38.57 / 39.26 | 28.3 |
| `yolo26m-det-bf16-b1` | `yolo26m-det-bf16-b1.tar.gz` | single-stream-object-detector, multi-stream-object-detector, multi-stream-tracker, yolo26-object-detector, detection-to-vlm-assistant | `unknown` | 6 | 17.71 / 17.71 / 18.31 | 80.9 |
| `yolo26n-det-int8-b1` | `yolo26n-det-int8-b1.tar.gz` | high-density-multi-stream-object-detector, pcie-high-density-multi-stream-object-detector, mipi-multi-model | `dequantize` | 6 | 3.90 / 3.83 / 4.43 | 1201.6 |
| `yolo26m-det-int8-b1` | `yolo26m-det-int8-b1.tar.gz` | single-stream-object-detector, multi-stream-object-detector, multi-stream-tracker, yolo26-object-detector, detection-to-vlm-assistant, multi-stream-blazepose3d, multi-stream-multi-model, parallel-apps | `detessdequant` | 6 | 6.52 / 6.51 / 7.12 | 334.7 |
| `yolo26m-det-int8-b1-boxdecode` | `yolo26m-det-int8-b1.tar.gz` | single-stream-object-detector, multi-stream-object-detector, multi-stream-tracker, yolo26-object-detector, detection-to-vlm-assistant, multi-stream-blazepose3d, multi-stream-multi-model, parallel-apps | `boxdecode` | 1 | 7.07 / 7.05 / 7.39 | 334.8 |

## YOLO26 Segmentation

| Model ID | Package | Used By | Postprocess | Outputs | Latency ms (mean / p50 / p99) | FPS |
| --- | --- | --- | --- | ---: | ---: | ---: |
| `yolo26n-seg-bf16-mla_tess` | `yolo26n-seg-bf16-mla_tess.tar.gz` | single-stream-instance-segmenter, mipi-multi-model | `cast` | 10 | 7.90 / 7.88 / 8.57 | 338.6 |
| `yolo26s-seg-bf16-mla_tess` | `yolo26s-seg-bf16-mla_tess.tar.gz` | single-stream-instance-segmenter | `cast` | 10 | 11.93 / 11.87 / 12.56 | 144.5 |
| `yolo26m-seg-bf16-mla_tess` | `yolo26m-seg-bf16-mla_tess.tar.gz` | single-stream-instance-segmenter | `cast` | 10 | 24.41 / 24.39 / 25.11 | 51.6 |
| `yolo26l-seg-bf16-mla_tess` | `yolo26l-seg-bf16-mla_tess.tar.gz` | single-stream-instance-segmenter | `cast` | 10 | 27.80 / 27.77 / 28.53 | 44.0 |
| `yolo26x-seg-bf16-mla_tess` | `yolo26x-seg-bf16-mla_tess.tar.gz` | single-stream-instance-segmenter | `cast` | 10 | 54.78 / 54.76 / 55.46 | 20.1 |
| `yolo26m-seg-bf16-b1` | `yolo26m-seg-bf16-b1.tar.gz` | single-stream-instance-segmenter | `unknown` | 10 | 27.28 / 27.16 / 27.87 | 52.3 |
| `yolo26m-seg-bf16-mla_tess-b1` | `yolo26m-seg-bf16-mla_tess-b1.tar.gz` | single-stream-instance-segmenter | `cast` | 10 | 24.08 / 24.07 / 24.77 | 52.2 |
| `yolo26m-seg-int8-b1` | `yolo26m-seg-int8-b1.tar.gz` | single-stream-instance-segmenter, parallel-apps | `detessdequant` | 10 | 10.33 / 10.31 / 10.97 | 220.6 |
| `yolo26m-seg-int8-b1-boxdecode` | `yolo26m-seg-int8-b1.tar.gz` | single-stream-instance-segmenter, parallel-apps | `boxdecode` | 1 | 33.44 / 33.42 / 33.79 | 73.9 |

The BoxDecode row builds a 160x160 mask on the A65 for every detection it keeps, and synthetic input fills all `top_k=100` slots, so it is a worst case. With `top_k=10`, the same package measured 163.6 FPS (19.41 ms) on the same board at 1000 MHz.

## YOLO26 Pose

| Model ID | Package | Used By | Postprocess | Outputs | Latency ms (mean / p50 / p99) | FPS |
| --- | --- | --- | --- | ---: | ---: | ---: |
| `yolo26n-pose-bf16-mla_tess-b1` | `yolo26n-pose-bf16-mla_tess-b1.tar.gz` | multi-stream-pose-estimator, mipi-multi-model | `cast` | 9 | 6.04 / 6.03 / 6.68 | 391.0 |
| `yolo26s-pose-bf16-mla_tess-b1` | `yolo26s-pose-bf16-mla_tess-b1.tar.gz` | multi-stream-pose-estimator | `cast` | 9 | 8.94 / 8.89 / 9.57 | 183.8 |
| `yolo26m-pose-bf16-mla_tess-b1` | `yolo26m-pose-bf16-mla_tess-b1.tar.gz` | multi-stream-pose-estimator | `cast` | 9 | 16.86 / 16.84 / 17.54 | 75.2 |
| `yolo26l-pose-bf16-mla_tess-b1` | `yolo26l-pose-bf16-mla_tess-b1.tar.gz` | multi-stream-pose-estimator | `cast` | 9 | 20.15 / 20.13 / 20.79 | 60.1 |
| `yolo26x-pose-bf16-mla_tess-b1` | `yolo26x-pose-bf16-mla_tess-b1.tar.gz` | multi-stream-pose-estimator | `cast` | 9 | 39.88 / 39.87 / 40.55 | 27.6 |
| `yolo26m-pose-bf16-b1` | `yolo26m-pose-bf16-b1.tar.gz` | multi-stream-pose-estimator | `unknown` | 9 | 18.55 / 18.39 / 19.04 | 77.2 |
| `yolo26m-pose-int8-b1` | `yolo26m-pose-int8-b1.tar.gz` | multi-stream-pose-estimator, multi-stream-multi-model, parallel-apps | `detessdequant` | 9 | 7.56 / 7.52 / 8.15 | 311.6 |

## Other Segmentation

| Model ID | Package | Used By | Postprocess | Outputs | Latency ms (mean / p50 / p99) | FPS |
| --- | --- | --- | --- | ---: | ---: | ---: |
| `yolo_v8n_seg` | `yolo_v8n_seg_mpk.tar.gz` | single-stream-instance-segmenter | `detessdequant` | 10 | 7.16 / 7.15 / 7.82 | 369.4 |
| `yolo_v8s_seg` | `yolo_v8s_seg_mpk.tar.gz` | single-stream-instance-segmenter | `detessdequant` | 10 | 7.97 / 7.92 / 8.58 | 361.2 |
| `yolo_v8m_seg` | `yolo_v8m_seg_mpk.tar.gz` | single-stream-instance-segmenter | `detessdequant` | 10 | 9.81 / 9.79 / 10.50 | 288.2 |
| `yolo_v8l_seg` | `yolo_v8l_seg_mpk.tar.gz` | single-stream-instance-segmenter | `detessdequant` | 10 | 12.48 / 12.30 / 12.97 | 165.6 |
| `yolo_11s_seg` | `yolo_11s_seg_mpk.tar.gz` | multi-stream-multi-model | `detessdequant` | 10 | 8.07 / 8.06 / 8.70 | 353.3 |

## Depth, Keypoints, Pose, And Anomaly

| Model ID | Package | Used By | Postprocess | Outputs | Latency ms (mean / p50 / p99) | FPS |
| --- | --- | --- | --- | ---: | ---: | ---: |
| `depth_anything_v2_vits` | `depth_anything_v2_vits_mpk.tar.gz` | depth-estimator | `detessdequant` | 1 | 20.80 / 20.61 / 20.84 | 55.8 |
| `midas_v21_small_256` | `midas_v21_small_256_mpk.tar.gz` | mipi-multi-model | `detessdequant` | 1 | 2.85 / 2.85 / 2.95 | 1340.8 |
| `superpoint` | `superpoint_mpk.tar.gz` | superpoint-feature-extractor | `detessdequant` | 2 | 7.38 / 7.36 / 8.13 | 494.0 |
| `blazepose_ghum_heavy_w16a16` | `blazepose_ghum_heavy_w16a16_mpk.tar.gz` | multi-stream-blazepose3d, parallel-apps | `cast` | 3 | 3.66 / 3.66 / 3.86 | 415.9 |
| `fastflow_demo` | `fastflow_demo_mpk.tar.gz` | fastflow-anomaly-detector | `detessdequant` | 1 | 3.82 / 3.82 / 3.94 | 574.7 |
| `patchcore_wide_resnet50_2_bf16_mla` | `patchcore_wide_resnet50_2_bf16_mla.tar.gz` | patchcore-anomaly-detector | `unknown` | 1 | 8.66 / 8.65 / 8.81 | 366.3 |

## Multi-Package Model Parts

Each of these models runs as two packages. Every part is measured alone, so no row is the cost of the whole model.

| Model ID | Package | Used By | Postprocess | Outputs | Latency ms (mean / p50 / p99) | FPS |
| --- | --- | --- | --- | ---: | ---: | ---: |
| `rf_detr_nano_w8a8_w16a16_backbone` | `rf_detr_nano_w8a8_w16a16_backbone_mpk.tar.gz` | rfdetr-detection-segmentation | `dequantize` | 3 | 5.06 / 5.04 / 5.16 | 294.3 |
| `rf_detr_nano_w8a8_w16a16_decoder` | `rf_detr_nano_w8a8_w16a16_decoder_mpk.tar.gz` | rfdetr-detection-segmentation | `cast` | 2 | 2.43 / 2.42 / 2.52 | 955.0 |
| `rf_detr_small_w8a8_w16a16_backbone` | `rf_detr_small_w8a8_w16a16_backbone_mpk.tar.gz` | rfdetr-detection-segmentation, parallel-apps | `dequantize` | 3 | 8.40 / 8.39 / 8.84 | 156.4 |
| `rf_detr_small_w8a8_w16a16_decoder` | `rf_detr_small_w8a8_w16a16_decoder_mpk.tar.gz` | rfdetr-detection-segmentation, parallel-apps | `cast` | 2 | 3.26 / 3.25 / 3.36 | 547.4 |
| `rf_detr_medium_w8a8_w16a16_backbone` | `rf_detr_medium_w8a8_w16a16_backbone_mpk.tar.gz` | rfdetr-detection-segmentation | `dequantize` | 3 | 11.90 / 11.88 / 12.25 | 103.2 |
| `rf_detr_medium_w8a8_w16a16_decoder` | `rf_detr_medium_w8a8_w16a16_decoder_mpk.tar.gz` | rfdetr-detection-segmentation | `cast` | 2 | 3.94 / 3.94 / 4.06 | 411.3 |
| `rf_detr_large_w8a8_w16a16_backbone` | `rf_detr_large_w8a8_w16a16_backbone_mpk.tar.gz` | rfdetr-detection-segmentation | `dequantize` | 3 | 23.23 / 23.21 / 23.61 | 48.9 |
| `rf_detr_large_w8a8_w16a16_decoder` | `rf_detr_large_w8a8_w16a16_decoder_mpk.tar.gz` | rfdetr-detection-segmentation | `cast` | 2 | 4.44 / 4.43 / 4.60 | 352.5 |
| `rfdetr-seg-medium-backbone` | `rfdetr-seg-medium-backbone.tar.gz` | rfdetr-detection-segmentation | `detessdequant` | 3 | 12.76 / 12.74 / 13.11 | 100.5 |
| `rfdetr-seg-medium-transformer` | `rfdetr-seg-medium-transformer.tar.gz` | rfdetr-detection-segmentation | `unknown` | 3 | 23.28 / 23.25 / 23.91 | 67.3 |
| `efficientsam3` | `efficientsam3_mpk.tar.gz` | efficientsam3-promptable-segmenter | `cast` | 2 | 86.89 / 86.73 / 87.42 | 13.4 |
| `efficientsam3_text_encoder` | `efficientsam3_text_encoder_mpk.tar.gz` | efficientsam3-promptable-segmenter | `cast` | 1 | 17.77 / 17.75 / 18.36 | 62.2 |
