# Detection-to-VLM Assistant

## Metadata
| Field | Value |
| --- | --- |
| Category | genai |
| Difficulty | Advanced |
| Tags | object-detection, genai, yolo26, rtsp, insight, openai-compatible |
| Languages | Python |
| Status | experimental |
| Binary Name | detection-to-vlm-assistant |
| Model | yolo26m-det-bf16-mla_tess-b1 |

## Concept
This example decodes an RTSP stream, runs YOLO26 detection with internal box decode, and sends video plus object-detection metadata to Insight. Insight video uses `VideoSender`, which owns raw-frame caps, conversion, H.264 encoding, RTP packetization, and UDP output. When OpenAI is enabled, the highest-score detected person is cropped and sent to the configured OpenAI-compatible Gemma server from a bounded background worker, so the detection and Insight loop keeps running.

## Preview
Detection metadata visualized in Insight:

![Detection-to-VLM assistant preview](../../../assets/portal/genai/detection-to-vlm-assistant/image.png)

## Prerequisites
- Installed Neat Development Environment.
- YOLO26 detector model package available under `assets/models/`.
- RTSP stream readable from the Neat Development Environment.
- Insight receiver running at the configured host and ports.
- OpenAI-compatible Gemma server running when `openai.enabled` is true.

## Get The Apps Repo
Install the Neat Library first by following the official [Neat Library installation guide](https://developer.sima.ai/software/getting-started/installation/neat-library).

Then clone and build the apps repo:

```bash
git clone https://github.com/sima-neat/apps.git
cd apps
./build.sh --clean
```

After this setup, follow the example-specific commands below.

## Download Models
Use the platform version wherever `<platform-version>` appears.

Supported batch-1 YOLO26 detector models:
- `yolo26n-det-bf16-mla_tess-b1.tar.gz`
- `yolo26s-det-bf16-mla_tess-b1.tar.gz`
- `yolo26m-det-bf16-mla_tess-b1.tar.gz`
- `yolo26l-det-bf16-mla_tess-b1.tar.gz`
- `yolo26x-det-bf16-mla_tess-b1.tar.gz`
- `yolo26m-det-bf16-b1.tar.gz`
- `yolo26m-det-int8-b1.tar.gz`

Download one detector model:

```bash
mkdir -p assets/models
cd assets/models

PLATFORM_VERSION="<platform-version>"
MODEL=yolo26m-det-bf16-mla_tess-b1.tar.gz

sima-cli download "https://docs.sima.ai/pkg_downloads/SDK${PLATFORM_VERSION}/models/modalix/yolo26-detection/${MODEL}"

cd ../..
```

Set `PLATFORM_VERSION` to your installed SDK platform version, and replace `MODEL` with any supported model listed above.

## Run
From the `apps` repository root:

```bash
python3 examples/genai/detection-to-vlm-assistant/src/python/main.py \
  --config examples/genai/detection-to-vlm-assistant/src/common/config.yaml
```

Set `openai.enabled: false` in the config to run only the detection and Insight path.

For local model comparison, example configs are available under
`sandbox/configs/detection-to-vlm-assistant/<model-name>/config.yaml`.

The OpenAI path checks `/v1/models` before sending a crop, waits at least `openai.interval_seconds` between attempts, and keeps at most `openai.max_pending_requests` queued or in-flight requests.

## Source Files
- Python source: `src/python/main.py`, `src/python/utils/helpers.py`, `src/python/utils/openai_commenter.py`
- Shared assets: `src/common/`
