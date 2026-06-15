# Single-Stream Object Detector

## Metadata
| Field | Value |
| --- | --- |
| Category | object-detection |
| Difficulty | Intermediate |
| Tags | object-detection, yolo26, rtsp, insight |
| Languages | C++, Python |
| Status | experimental |
| Binary Name | single-stream-object-detector |
| Model | yolo26m-det-bf16-mla_tess-b1 |

## Concept
`single-stream-object-detector` is a focused reference example for a common deployment pattern:

- ingest one RTSP camera stream
- decode the stream into NV12 frames
- run YOLO26 object detection
- send H.264 video plus detection metadata to Insight

The example is intentionally narrow in scope. It is not a generic output-mode demo and it does not try to support multiple unrelated workflows in one binary. The code is structured to show the intended Insight path clearly.

## Preview
Snippet from a pipeline run:

![Single-stream object detector preview](../../../assets/portal/object-detection/single-stream-object-detector/image.png)

## Insight Setup
[Neat Insight](https://developer.sima.ai/software/tools/insight/) can host an RTSP source, receive video from `VideoSender`, receive detection metadata from `MetadataSender`, and show rendered overlays plus runtime metrics in the browser.

To create a reproducible RTSP input:
1. Run `neat` in the Neat Developer Environment and open the reported `Insight Web UI`.
2. In Insight, open `RTSP Source`.
3. Use a sample video or upload your own video.
4. Start the stream and copy the RTSP URL.
5. Put that RTSP URL into `source.rtsp_url`.

Use the same `neat` output to set `output.insight.host`, `video_port`, and `metadata_port` from the reported `videoUDP` and `metadataUDP` ranges.

## Prerequisites
- Installed Neat Library and Insight on the DevKit
- RTSP source created in Insight or provided by your camera
- Default model downloaded, or `model.path` set to another readable model package

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
Use the SDK platform version wherever `<platform-version>` appears.

Default model: `yolo26m-det-bf16-mla_tess-b1.tar.gz`.

Download the default model:

```bash
mkdir -p assets/models
cd assets/models

sima-cli download https://docs.sima.ai/pkg_downloads/SDK<platform-version>/models/modalix/yolo26-detection/yolo26m-det-bf16-mla_tess-b1.tar.gz

cd ../..
```

The command stores the model under `assets/models/` as a repo-local convention. `model.path` can point to any readable model package path.

## Configure
Edit `examples/object-detection/single-stream-object-detector/src/common/config.yaml`.

```yaml
model:
  path: <model-path>                                      # Path to the model package.

source:
  rtsp_url: <rtsp-url-copied-from-insight>                # RTSP stream URL.
  tcp: true                                               # Use TCP transport for RTSP.

inference:
  frames: 0                                               # Frame limit. 0 runs continuously.
  min_score: 0.30                                         # Minimum object confidence.

output:
  insight:
    host: <insight-host-ip>                               # Host running Insight.
    video_port: <videoUDP start port from neat>           # UDP video port.
    metadata_port: <metadataUDP start port from neat>     # UDP metadata port.
```

## Run
### C++
```bash
./build/examples/object-detection/single-stream-object-detector/single-stream-object-detector \
  --config examples/object-detection/single-stream-object-detector/src/common/config.yaml
```

### Python
```bash
source ~/pyneat/bin/activate
pip install -r examples/object-detection/single-stream-object-detector/src/python/requirements.txt
python3 examples/object-detection/single-stream-object-detector/src/python/main.py \
  --config examples/object-detection/single-stream-object-detector/src/common/config.yaml
```

## Debugging Notes
- If the sample times out waiting for the first RTSP frame, the problem is usually upstream stream delivery or device connectivity, not YOLO itself.
- If the RTSP source resolution changes, the startup probe is expected to adapt the decode path automatically.
- If detections are missing but video is flowing, focus on the YOLO session and bbox extraction/parse path.
- If video and detections are both missing in Insight, verify the host and UDP ports first.

## Appendix: Additional Models
Other supported batch-1 YOLO26 detection models:
- `yolo26n-det-bf16-mla_tess-b1.tar.gz`
- `yolo26s-det-bf16-mla_tess-b1.tar.gz`
- `yolo26l-det-bf16-mla_tess-b1.tar.gz`
- `yolo26x-det-bf16-mla_tess-b1.tar.gz`
- `yolo26m-det-bf16-b1.tar.gz`
- `yolo26m-det-int8-b1.tar.gz`

Replace the default filename in the download command and `model.path`.

## Source Files
- C++ source: `src/cpp/main.cpp`
- C++ tests: `tests/cpp/test_unit.cpp`, `tests/cpp/test_e2e.cpp`
- Python source: `src/python/main.py`
- Python tests: `tests/python/test_unit.py`, `tests/python/test_e2e.py`
- Insight documentation: <https://developer.sima.ai/software/tools/insight>
