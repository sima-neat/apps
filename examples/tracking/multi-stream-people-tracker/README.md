# Multi-Stream People Tracker

## Metadata
| Field | Value |
| --- | --- |
| Category | tracking |
| Difficulty | Advanced |
| Tags | object-detection, yolo26, rtsp, multistream, insight, people-tracking |
| Languages | C++, Python |
| Status | experimental |
| Binary Name | multi-stream-people-tracker |
| Model | yolo26m-det-int8-b1 |

## Concept
Multi-stream people tracking example with RTSP inputs, mixed-resolution support, Insight live video plus metadata output, and optional sampled overlay saves. The pipeline filters detector output to the configured person class and assigns stable track IDs per stream.

## Preview
Preview image from a live run:

![Multi-stream people tracker preview](../../../assets/portal/tracking/multi-stream-people-tracker/image.png)

## Insight Setup
[Neat Insight](https://developer.sima.ai/software/tools/insight/) can host RTSP streams, receive video from `VideoSender`, receive tracking metadata from `MetadataSender`, and show rendered overlays plus runtime metrics in the browser.

Run `neat` in the Neat Developer Environment and copy these values into your config:

- `Insight Web UI`: browser URL for the viewer
- `rtsp.tcp`: RTSP source port
- `videoUDP`: UDP video port range
- `metadataUDP`: UDP metadata port range

## Supported Models
Use the platform version wherever `<platform-version>` appears.

Default model: `yolo26m-det-int8-b1.tar.gz`.

Supported batch-1 YOLO26 detection models:
- `yolo26n-det-bf16-mla_tess-b1.tar.gz`
- `yolo26s-det-bf16-mla_tess-b1.tar.gz`
- `yolo26m-det-bf16-mla_tess-b1.tar.gz`
- `yolo26l-det-bf16-mla_tess-b1.tar.gz`
- `yolo26x-det-bf16-mla_tess-b1.tar.gz`
- `yolo26m-det-bf16-b1.tar.gz`
- `yolo26m-det-int8-b1.tar.gz`

Download a supported model:

```bash
mkdir -p assets/models
cd assets/models

PLATFORM_VERSION="<platform-version>"
MODEL=yolo26m-det-int8-b1.tar.gz

sima-cli download "https://docs.sima.ai/pkg_downloads/SDK${PLATFORM_VERSION}/models/modalix/yolo26-detection/${MODEL}"

cd ../..
```

Set `PLATFORM_VERSION` to your installed SDK platform version, and replace `MODEL` with any supported model listed above.

## Prerequisites
- A Neat Python environment with `pyneat`, `numpy`, and OpenCV available.
- One or more reachable RTSP camera URLs.
- A YOLO26 detector model pack downloaded into `assets/models/`.
- An Insight viewer instance reachable from the board/host running this example.

## Get The Apps Repo
Install the Neat Library first by following the official [Neat Library installation guide](https://developer.sima.ai/software/getting-started/installation/neat-library).

Then clone and build the apps repo:

```bash
git clone https://github.com/sima-neat/apps.git
cd apps
./build.sh --clean
```

After this setup, follow the example-specific commands below.

## Configure
Edit `examples/tracking/multi-stream-people-tracker/src/common/config.yaml`.

```yaml
model:
  path: assets/models/yolo26m-det-int8-b1.tar.gz       # Model package to load.

streams:
  - rtsp://<insight-host-ip>:<rtsp.tcp>/<stream-1>     # First RTSP stream.
  - rtsp://<insight-host-ip>:<rtsp.tcp>/<stream-2>     # Second RTSP stream.

inference:
  frames: 0                                            # Frame limit per stream. 0 runs continuously.
  min_score: 0.30                                      # Minimum person confidence.

tracking:
  max_missing_frames: 15                               # Frames to keep a missing track alive.

output:
  insight:
    host: <insight-host-ip>                            # Host running Insight.
    video_port_base: <videoUDP start port from neat>   # First UDP video port.
    metadata_port_base: <metadataUDP start port from neat> # First UDP metadata port.
```

## Run
### C++
```bash
./build/examples/tracking/multi-stream-people-tracker/multi-stream-people-tracker \
  --config examples/tracking/multi-stream-people-tracker/src/common/config.yaml
```

### Python
```bash
source ~/pyneat/bin/activate
pip install -r examples/tracking/multi-stream-people-tracker/src/python/requirements.txt
python3 examples/tracking/multi-stream-people-tracker/src/python/main.py \
  --config examples/tracking/multi-stream-people-tracker/src/common/config.yaml
```

## Debugging Notes
- Start with one RTSP stream and confirm the config before scaling to multiple cameras.
- Confirm the model file exists under `assets/models/`.
- Confirm each RTSP URL is reachable from the board or host running the example.
- If Insight appears idle, verify `output.insight.host`, `video_port_base`, and `metadata_port_base`.
- If you want saved overlay frames, set both `output.debug_dir` and `output.save_every`.

## Source Files
- C++ source: `src/cpp/main.cpp`
- C++ tracker helpers: `src/cpp/utils/tracker_api.cpp`, `src/cpp/utils/tracker.cpp`
- C++ tests: `tests/cpp/test_unit.cpp`, `tests/cpp/test_e2e.cpp`
- Python source: `src/python/main.py`
- Python tracker helpers: `src/python/utils/tracker.py`
- Example config: `src/common/config.yaml`
- Python tests: `tests/python/test_unit.py`, `tests/python/test_e2e.py`
- Shared example data: `src/common/`
