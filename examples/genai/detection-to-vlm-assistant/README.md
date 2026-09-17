# Semantic People Tracker

## Metadata

| Field | Value |
| --- | --- |
| Category | genai |
| Difficulty | Advanced |
| Tags | object-detection, genai, yolo26, webcam, browser, vlm |
| Languages | Python |
| Status | stable |
| Binary Name | detection-to-vlm-assistant |
| Model | yolo26m-det-bf16-mla_tess-b1 |

## Concept

Detects and tracks people from the browser's live webcam with YOLO26 and presents
the result as a browser-based **Real-Time Semantic People Tracker**. The
dashboard keeps the local 30 FPS camera preview full width and annotates every
confirmed track with its session-local ID and a short appearance description.

The browser streams the newest JPEG frame to Modalix over one binary WSS
connection. The application feeds it through the public PyNeat image-input API,
returns tracks to the browser, and sends one padded crop from each newly
confirmed person to the local VLM. The description is retained for that track
through short occlusions. Slow inference never blocks the browser preview
because queued camera frames are replaced by the newest frame.

Tracking uses the same two-stage, motion-aware association algorithm as the
YOLO26 tiny-drone tracker. IDs are local to this application session; this is
not biometric identification or long-term person re-identification. A person
who leaves beyond the configured missing-frame budget receives a new ID and a
new description when returning.

The dashboard is served directly by the detector process with Python's standard
HTTP server, so the packaged application needs no Node.js build or additional
web framework. Browser updates use Server-Sent Events with periodic JSON polling
as a fallback.

## Preview

![Semantic people tracker preview](../../../portal/assets/examples/genai/detection-to-vlm-assistant/image.png)

## Prerequisites

- `sima-cli` ([documentation](https://developer.sima.ai/software/tools/sima-cli/)) on a supported Modalix or DevKit target.
- A webcam attached to the computer opening the dashboard.
- The [LLiMa model manager](https://developer.sima.ai/software/genai-llima/runtime) available on the target when GenAI is enabled.

## Install Apps

Install the latest Neat Apps runtime and enter the installed bundle:

```bash
sima-cli neat install apps
cd prebuilt-apps
APP_DIR=examples/genai/detection-to-vlm-assistant
```

Run the remaining commands from `prebuilt-apps/`.

## Prepare the Model

Supported detector packages:

| Model file | Role |
| --- | --- |
| `yolo26m-det-bf16-mla_tess-b1.tar.gz` | Default |
| `yolo26n-det-bf16-mla_tess-b1.tar.gz` | Supported |
| `yolo26s-det-bf16-mla_tess-b1.tar.gz` | Supported |
| `yolo26l-det-bf16-mla_tess-b1.tar.gz` | Supported |
| `yolo26x-det-bf16-mla_tess-b1.tar.gz` | Supported |
| `yolo26m-det-bf16-b1.tar.gz` | Supported |
| `yolo26m-det-int8-b1.tar.gz` | Supported |

Model packages come from the Model Zoo release below, which can differ from the installed platform version. Replace `<model-file>` with a file from the table.

```bash
export MODELZOO_VERSION="2.1.3"
mkdir -p models
cd models
sima-cli download "https://docs.sima.ai/pkg_downloads/SDK${MODELZOO_VERSION}/models/modalix/yolo26-detection/<model-file>"
cd ..
```

Set `model.path` to the downloaded detector package.

The default VLM is the image-capable `Gemma-4-E2B-it-GPTQ-a16w4`. Do not use
the separate `Gemma-4-E2B-it-TextOnly-GPTQ-a16w4` artifact for this demo.

Search the supported models:

```bash
llima search
```

Install the default model:

```bash
llima pull Gemma-4-E2B-it-GPTQ-a16w4
```

LLiMa stores models under `/media/nvme/llima/models/` by default. Set `LLIMA_MODELS_PATH` before `llima pull` to use another model directory, then update `genai_server.model.path` accordingly.

## Configure

Open `${APP_DIR}/src/common/config.yaml`. Set the detector model path and VLM
model. `webcam.target_fps` defaults to 30. `webcam.upload_max_width` defaults to
640 so the browser keeps its full-resolution local preview but only JPEG-encodes
the resolution needed by the detector. `webcam.max_frame_bytes` bounds each
uploaded JPEG. Configure `web.tls.cert`, `web.tls.key`, and
`web.tls.ca_cert` with the HTTPS server and local CA certificate paths.
`inference.queue_depth` bounds the realtime model queue, while
`inference.internal_queue_depth` controls asynchronous preprocess/inference
handoffs.

The `tracking` section controls confirmation, IoU and motion matching, short
occlusion retention, and the active-track bound. The supplied defaults confirm
a person after three detections and retain the track for 30 missed processed
frames. A confirmed track must then remain at least 3% inside the top, left,
and right image edges for five consecutive processed frames before its crop is
sent to the VLM. The bottom edge is allowed so a standing person's box can
reach the base of the camera view. A crop is also blocked whenever its person
box overlaps another detected person box; separation resets the five-frame
counter. Until eligible, the overlay says `waiting for clear view…`; the demo
never falls back to an overlapping, side-entry, or top-clipped crop.

The dashboard defaults to HTTPS port `5000`. The packaged config enables GenAI
and uses a fixed appearance-only prompt. Set `genai.enabled` to `false` to run
tracking without semantic descriptions.

## Run

Install the Python dependencies:

```bash
source ~/pyneat/bin/activate
pip install -r ${APP_DIR}/src/python/requirements.txt
```

When GenAI is enabled, start the server in one terminal:

```bash
APP_DIR=examples/genai/detection-to-vlm-assistant
source ~/pyneat/bin/activate
python3 ${APP_DIR}/src/python/genai_server.py \
  --config ${APP_DIR}/src/common/config.yaml
```

Start the detection pipeline in another terminal:

```bash
APP_DIR=examples/genai/detection-to-vlm-assistant
source ~/pyneat/bin/activate
python3 ${APP_DIR}/src/python/detector_app.py \
  --config ${APP_DIR}/src/common/config.yaml
```

On the computer with the webcam, browse to
`https://<modalix-ip>:5000`. Accept the local certificate warning on the first
visit, then allow camera access. To remove future warnings, use **Download local
CA** and install `modalix-webcam-ca.crt` in the computer's trusted certificate
authority store. The CA private key must remain on Modalix.

The GenAI path checks `/v1/models` and serializes new tracks through the bounded
`genai.max_pending_requests` queue. An eligible track is described once. A
failed request is retried once and then displayed as `description unavailable`.
Responses retain Gemma's complete enriched comma-separated attribute line and
are printed to stdout; sentence-style responses are rejected. The overlay wraps
longer descriptions onto multiple lines without truncating them.

Camera capture/JPEG encoding, binary WSS ingest, JPEG decode/model submission,
model result collection, and VLM analysis run as independent bounded stages.
The WebSocket stays open for the camera session, avoiding a per-frame HTTP
request/response. When a consumer falls behind, its pending slot keeps the
newest frame instead of building latency from stale webcam frames. Model
results carry the source frame ID so tracks and VLM crops remain correlated
across the async pipeline. A separate lightweight server-sent event stream
publishes every completed track immediately; semantic results also trigger an
immediate overlay update. The larger dashboard state continues at a lower rate
so metrics do not delay the overlay.
`POST /api/frame` remains available for diagnostics.

## Troubleshooting

- Verify the page was opened as `https://<modalix-ip>:5000`, including `https`.
- Confirm the browser has camera permission and no other application owns it.
- Verify `model.path` points to the detector package.
- Verify the VLM server lists `genai_server.model.name` under `/v1/models`.
- Verify `curl -k https://127.0.0.1:5000/api/health` reports `status: ok`.
- Press **Retry camera** if permission was granted after the first page load.
- Disable GenAI to isolate webcam upload and YOLO inference.

## Source Files

- Detector application: `src/python/detector_app.py`
- GenAI server: `src/python/genai_server.py`
- Browser state/server: `src/python/dashboard.py`
- Async model runner: `src/python/async_pipeline.py`
- Motion tracker: `src/python/tracker.py`
- One-shot semantic worker: `src/python/semantic_describer.py`
- Browser assets: `src/python/web/`
- Shared runtime files: `src/common/`

## Development From Source

To modify or test this example, use the [Apps contributor workflow](https://github.com/sima-neat/apps/blob/main/CONTRIBUTING.md).
