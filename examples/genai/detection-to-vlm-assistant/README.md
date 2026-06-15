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

## Insight Setup
[Neat Insight](https://developer.sima.ai/software/tools/insight/) can host an RTSP source, receive video from `VideoSender`, receive detection metadata from `MetadataSender`, and show rendered overlays plus runtime metrics in the browser.

In the Neat Development Environment, install the sample video assets:

```bash
sima-cli install assets/multi-video-sources
```

This provides 720p and 480p videos that Insight can stream as RTSP sources.

To create a reproducible RTSP input:
1. Run `neat` in the Neat Development Environment and open the reported `Insight Web UI`.
2. In Insight, open `RTSP Source`.
3. Use a sample video or upload your own video.
4. Start the stream and copy the RTSP URL.
5. Put that RTSP URL into `source.rtsp_url`.

Use the same `neat` output to set `insight.host`, `video_port`, and `metadata_port` from the reported `videoUDP` and `metadataUDP` ranges.

## Prerequisites
- Installed Neat Development Environment + Neat Library.
- Model artifacts are user-managed and should be downloaded into `assets/models/`. Download the default YOLO26 detector model, or set `model.path` to another readable model package.
- RTSP source created in Insight or provided by your camera.
- Insight receiver running at the configured host and ports.
- OpenAI-compatible Gemma server running when `openai.enabled` is true.

## Get The Apps Repo
Use the [Neat Development Environment](https://developer.sima.ai/software/getting-started/dev-environment/) with the [Neat Library](https://developer.sima.ai/software/getting-started/neat-library/) installed for setup and compilation.

Clone and build the apps repo inside the Neat Development Environment:

```bash
git clone https://github.com/sima-neat/apps.git
cd apps
./build.sh --clean
```

After building, run the example commands below on the Modalix/DevKit board.

## Download Models
Use the SDK platform version wherever `<platform-version>` appears.

Default model: `yolo26m-det-bf16-mla_tess-b1.tar.gz`.

Download the default detector model:

```bash
mkdir -p assets/models
cd assets/models

sima-cli download https://docs.sima.ai/pkg_downloads/SDK<platform-version>/models/modalix/yolo26-detection/yolo26m-det-bf16-mla_tess-b1.tar.gz

cd ../..
```

The command stores the model under `assets/models/` as a repo-local convention. `model.path` can point to any readable model package path.

## Configure
Edit `examples/genai/detection-to-vlm-assistant/src/common/config.yaml`.

```yaml
source:
  rtsp_url: <rtsp-url-copied-from-insight>              # RTSP stream URL.

model:
  path: <model-path>                                    # Path to the model package.

insight:
  host: <insight-host-ip>                                  # Host running Insight.
  video_port: <videoUDP start port from neat>              # UDP video port.
  metadata_port: <metadataUDP start port from neat>        # UDP metadata port.

openai:
  enabled: false                                           # Disable for detection plus Insight only.
```

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

## Appendix: Additional Models
Other supported batch-1 YOLO26 detector models:
- `yolo26n-det-bf16-mla_tess-b1.tar.gz`
- `yolo26s-det-bf16-mla_tess-b1.tar.gz`
- `yolo26l-det-bf16-mla_tess-b1.tar.gz`
- `yolo26x-det-bf16-mla_tess-b1.tar.gz`
- `yolo26m-det-bf16-b1.tar.gz`
- `yolo26m-det-int8-b1.tar.gz`

Replace the default filename in the download command and `model.path`.

## Source Files
- Python source: `src/python/main.py`, `src/python/utils/helpers.py`, `src/python/utils/openai_commenter.py`
- Shared assets: `src/common/`
