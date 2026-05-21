# YOLOv8 Detections to Gemma3

## Metadata
| Field | Value |
| --- | --- |
| Category | genai |
| Difficulty | Intermediate |
| Tags | object-detection, genai, yolov8, gemma3 |
| Languages | Python |
| Status | experimental |
| Binary Name | yolov8-detections-to-gemma3 |
| Model | YOLOv8 + Gemma3 |

## Concept
This example decodes an RTSP stream, runs YOLOv8 detection with internal box decode, and sends video plus object-detection metadata to Insight. When OpenAI is enabled, the highest-score detected person is cropped and sent to the configured OpenAI-compatible Gemma server from a bounded background worker, so the detection and Insight loop keeps running.

## Preview
Detection metadata visualized in Insight:

![YOLOv8 detections to Gemma3 preview](../../../assets/portal/genai/yolov8-detections-to-gemma3/image.png)

## Prerequisites
- Installed NEAT SDK.
- YOLOv8 model package available under `assets/models/`.
- RTSP stream readable from the SDK runtime.
- Insight receiver running at the configured host and ports.
- OpenAI-compatible Gemma server running when `openai.enabled` is true.

## Run
From the `apps` repository root:

```bash
python3 examples/genai/yolov8-detections-to-gemma3/python/main.py \
  --config examples/genai/yolov8-detections-to-gemma3/common/config.yaml
```

Set `openai.enabled: false` in the config to run only the detection and Insight path.

The OpenAI path checks `/v1/models` before sending a crop, waits at least `openai.interval_seconds` between attempts, and keeps at most `openai.max_pending_requests` queued or in-flight requests.

## Source Files
- Python source: `python/main.py`, `python/utils/helpers.py`, `python/utils/openai_commenter.py`
- Shared assets: `common/`
