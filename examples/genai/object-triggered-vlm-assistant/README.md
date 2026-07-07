# Object-Triggered VLM Assistant

## Metadata
| Field | Value |
| --- | --- |
| Category | genai |
| Difficulty | Advanced |
| Tags | object-detection, genai, vlm, yolo26, rtsp, insight, openai-compatible |
| Languages | Python |
| Status | experimental |
| Binary Name | object-triggered-vlm-assistant |
| Model | yolo26m-det-bf16-mla_tess-b1, gemma4-E4B-it |

## Concept
This example reads an RTSP video stream, runs a YOLO26 object detector, sends the video and detection metadata to Insight, and invokes a VLM only when a configured detector class is present.

The VLM receives a crop of the best matching detection and a strict prompt that asks for a very short response:

```text
{object_detector_class} of {c} color was just seen
```

For example:

```text
car of red color was just seen
```

## Runtime Flow
1. Insight or another RTSP source provides the input video stream.
2. PyNeat decodes frames from RTSP.
3. YOLO26 runs object detection on each frame.
4. The full frame is forwarded to Insight through `VideoSender`.
5. Detection boxes are forwarded to Insight through `MetadataSender`.
6. If `trigger.class` is detected, a small tracker updates trigger-class object tracks.
7. Every `memory.sample_interval_seconds`, sampled trigger-class metadata is linked to a JPEG-compressed frame by `frame_id`.
8. A bounded background worker can still send the current crop to the local OpenAI-compatible VLM server.
9. The `/ask` HTTP API answers current questions from the latest frame or past questions from linked frame/object memory.

## Prerequisites
- Installed Neat Development Environment + Neat Library.
- RTSP stream from Insight or a camera.
- Insight receiver running at the configured host and UDP ports.
- YOLO26 detector model package.
- VLM model directory at `/workspace/llima/models/gemma4-E4B-it`.

## Download Detector Model
Use the SDK platform version wherever `<platform-version>` appears.

```bash
mkdir -p assets/models
cd assets/models
sima-cli download https://docs.sima.ai/pkg_downloads/SDK<platform-version>/models/modalix/yolo26-detection/yolo26m-det-bf16-mla_tess-b1.tar.gz
cd ../..
```

## Configure
Edit `examples/genai/object-triggered-vlm-assistant/src/common/config.yaml`.

Set the RTSP source, detector model package, Insight destination, and trigger class:

```yaml
source:
  rtsp_url: rtsp://<rtsp-source-ip>:<port>/<stream-name>

model:
  path: assets/models/yolo26m-det-bf16-mla_tess-b1.tar.gz

trigger:
  class: car
  # classes:
  #   - car
  #   - truck

insight:
  host: <insight-host-ip>
  video_port: 9000
  metadata_port: 9100

memory:
  sample_interval_seconds: 0.5
  retention_seconds: 10

qa:
  enabled: true
  port: 8088
  crop_padding_ratio: 0.75
```

## Start The VLM Server
Run this in one terminal:

```bash
python3 examples/genai/object-triggered-vlm-assistant/src/python/deploy_vlm_server.py \
  --host 0.0.0.0 \
  --port 9998 \
  --model /workspace/llima/models/gemma4-E4B-it:gemma4-E4B-it
```

The suffix after `:` is the served model name. It must match `vlm.model` in the config.

## Run The App
Run this in another terminal from the `apps` repository root:

```bash
python3 examples/genai/object-triggered-vlm-assistant/src/python/main.py \
  --config examples/genai/object-triggered-vlm-assistant/src/common/config.yaml
```

When the configured class is detected, the app prints a VLM response such as:

```text
vlm: car of red color was just seen
```

Set `vlm.enabled: false` to run only the RTSP, detector, and Insight path.

## Ask Questions
The app exposes a small browser chat UI when `qa.enabled: true`. The top class selector supports one or more active classes. Changing the selection updates which classes are tracked going forward while retaining already sampled history until `memory.retention_seconds` expires. Prompt chips are regenerated for the selected class set, and clicking a chip sends the question immediately using the selected time dropdown:

```text
http://<target-ip>:8088/
```

When opening the UI from a host browser, use the Modalix/DevKit IP address for
`<target-ip>`, not `127.0.0.1`. The loopback address only works from commands
running on the board itself, or from the host after creating an SSH tunnel.
To find the board address, run:

```bash
hostname -I
```

To verify the question server from the board:

```bash
curl -s http://127.0.0.1:8088/health
```

The same server keeps the JSON API available. Chat answers include an evidence thumbnail so you can verify what the VLM saw. Current-frame questions use the latest frame directly, without detector metadata:

```bash
curl -s http://127.0.0.1:8088/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is visible now?"}'
```

Past questions use trigger-class object memory. The app first looks near the requested time; if no exact match exists, it falls back to the nearest retained trigger-class observation within `memory.retention_seconds`. It then follows `frame_id` to the matching frame, crops the stored bbox, and sends that crop plus metadata to the VLM:

```bash
curl -s http://127.0.0.1:8088/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"What was the brand of the bicycle 3 seconds ago?"}'
```

You can also pass `seconds_ago` explicitly:

```bash
curl -s http://127.0.0.1:8088/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"What color was it?","seconds_ago":3}'
```

Only detections matching `trigger.class` are tracked and sampled into memory.

## Source Files
- App entry: `src/python/main.py`
- VLM server helper: `src/python/deploy_vlm_server.py`
- Runtime helpers: `src/python/utils/helpers.py`
- Linked memory buffers: `src/python/utils/memory.py`
- Question API: `src/python/utils/question_server.py`
- Bounded VLM worker: `src/python/utils/vlm_commenter.py`
- Config and labels: `src/common/`
