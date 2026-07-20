# Detection-to-VLM Assistant

## Metadata

| Field | Value |
| --- | --- |
| Category | genai |
| Difficulty | Advanced |
| Tags | object-detection, genai, yolo26, rtsp, insight, vlm |
| Languages | Python |
| Status | experimental |
| Binary Name | detection-to-vlm-assistant |
| Model | yolo26m-det-bf16-mla_tess-b1 |

## Concept

This example decodes an RTSP stream, runs YOLO26 detection, and sends video plus detection metadata to Insight. When GenAI is enabled, a bounded background worker crops the highest-scoring person and sends it to a configured VLM server without blocking detection or Insight output.

## Preview

![Detection-to-VLM assistant preview](../../../portal/assets/examples/genai/detection-to-vlm-assistant/image.png)

## Prerequisites

- `sima-cli` on a supported Modalix or DevKit target.
- An RTSP source and Insight receiver reachable from the target.
- A local model from the [SiMa.ai VLM collection](https://huggingface.co/collections/simaai/vision-language-models) when GenAI is enabled.

## Install Apps

1. Choose a version from the [Neat Apps releases](https://github.com/sima-neat/apps/releases).
2. Install that version and enter the installed bundle:

```bash
sima-cli neat install apps@<release-version>
cd prebuilt-apps
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

The required platform version is recorded in `manifest.json`. Replace `<model-file>` with a file from the table.

```bash
mkdir -p models
cd models
sima-cli download https://docs.sima.ai/pkg_downloads/SDK<platform-version>/models/modalix/yolo26-detection/<model-file>
cd ..
```

Set `model.path` to the downloaded detector package. VLM directories are not installed by this `sima-cli` command. Download the selected VLM from the linked collection and set `genai_server.model.path` to its local directory.

## Prepare Insight

Insight can host the input stream and render the video and detection metadata. Install its sample video assets when needed:

```bash
sima-cli install assets/multi-video-sources
```

In the Insight Web UI, start a source and copy its RTSP URL. Use the host and UDP port ranges reported by `neat` for the output settings.

## Configure

Edit `examples/genai/detection-to-vlm-assistant/src/common/config.yaml`.

```yaml
source:
  rtsp_url: <rtsp-url>

model:
  path: <model-path>

insight:
  host: <insight-host-ip>
  video_port: <videoUDP-start-port>
  metadata_port: <metadataUDP-start-port>

genai_server:
  host: 0.0.0.0
  port: 9998
  model:
    name: <served-vlm-model-name>
    path: <path-to-vlm-model-dir>

genai:
  enabled: false
  host: 127.0.0.1
  port: 9998
  system_prompt: <system-prompt>
  user_prompt: <user-prompt>
```

Set `genai.enabled: false` to run only detection and Insight output.

## Run

Install the Python dependencies:

```bash
source ~/pyneat/bin/activate
pip install -r examples/genai/detection-to-vlm-assistant/src/python/requirements.txt
```

When GenAI is enabled, start the server in one terminal:

```bash
python3 examples/genai/detection-to-vlm-assistant/src/python/genai_server.py \
  --config examples/genai/detection-to-vlm-assistant/src/common/config.yaml
```

Start the detection pipeline in another terminal:

```bash
python3 examples/genai/detection-to-vlm-assistant/src/python/detector_app.py \
  --config examples/genai/detection-to-vlm-assistant/src/common/config.yaml
```

The GenAI path checks `/v1/models`, waits at least `genai.interval_seconds` between requests, and bounds queued and in-flight work with `genai.max_pending_requests`.

## Troubleshooting

- Verify the RTSP URL and Insight ports before investigating inference.
- Verify `model.path` points to the detector package.
- Verify the VLM server lists `genai_server.model.name` under `/v1/models`.
- Disable GenAI to isolate the detector and Insight path.

## Source Files

- Detector application: `src/python/detector_app.py`
- GenAI server: `src/python/genai_server.py`
- Shared runtime files: `src/common/`

## Development From Source

To modify or test this example, use the [Apps contributor workflow](https://github.com/sima-neat/apps/blob/main/CONTRIBUTING.md).
