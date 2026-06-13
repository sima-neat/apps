# Multi-Stream Object Detector

## Metadata
| Field | Value |
| --- | --- |
| Category | object-detection |
| Difficulty | Advanced |
| Tags | object-detection, rtsp, multistream, insight, yolo26 |
| Languages | C++, Python |
| Status | experimental |
| Binary Name | multi-stream-object-detector |
| Model | yolo26m-det-int8-b1 |

## Concept
This example runs a config-driven multistream RTSP detection pipeline for YOLO26 model packs and publishes per-stream video plus detection metadata to Insight.

## Preview
Snippet from a pipeline run:

![Multi-stream object detector preview](../../../assets/portal/object-detection/multi-stream-object-detector/image.png)

Architecture:
- one complete graph per RTSP stream
- each stream branches decoded NV12 frames into video and model paths
- clean Insight video is sent through `VideoSender`
- YOLO26 detections are sent as metadata for Insight overlay
- optional debug output joins decoded frames and detections by frame id before saving

Per-stream graph:
- `RtspDecodedInput -> Branch(video, model[, debug_frame])`
- `video -> VideoSender(H.264 RTP/UDP)`
- `model -> YOLO26 model.graph() -> detections`
- debug only: `Combine(debug_frame, detections, ByFrame) -> debug_output`

## Prerequisites
- Installed Neat Development Environment.
- One or more reachable RTSP camera URLs.
- A YOLO26 model pack downloaded into `assets/models/`.
- Edit `src/common/config.yaml` before running with real streams.
- On Modalix DevKit, run `bash /usr/bin/fix_devkit_runtime.sh` before starting the example if the runtime has been used by earlier ML/video apps.

## Get The Apps Repo
Install the Neat Library first by following the official [Neat Library installation guide](https://developer.sima.ai/software/getting-started/installation/neat-library).

Then clone and build the apps repo:

```bash
git clone https://github.com/sima-neat/apps.git
cd apps
./build.sh --clean
```

After this setup, follow the example-specific commands below.

## Command-line options
- `--config <path>`: path to the YAML configuration file
- `--validate-config-only`: validate the config and exit without opening RTSP streams
- `--help`: print the CLI help text

## Download Models
Use the platform version wherever `<platform-version>` appears.

The default model is `yolo26m-det-int8-b1.tar.gz`.

Supported batch-1 YOLO26 detection models:
- `yolo26n-det-bf16-mla_tess-b1.tar.gz`
- `yolo26s-det-bf16-mla_tess-b1.tar.gz`
- `yolo26m-det-bf16-mla_tess-b1.tar.gz`
- `yolo26l-det-bf16-mla_tess-b1.tar.gz`
- `yolo26x-det-bf16-mla_tess-b1.tar.gz`
- `yolo26m-det-bf16-b1.tar.gz`
- `yolo26m-det-int8-b1.tar.gz`

Download all supported batch-1 variants:

```bash
mkdir -p assets/models
cd assets/models

sima-cli download https://docs.sima.ai/pkg_downloads/SDK<platform-version>/models/modalix/yolo26-detection/yolo26n-det-bf16-mla_tess-b1.tar.gz
sima-cli download https://docs.sima.ai/pkg_downloads/SDK<platform-version>/models/modalix/yolo26-detection/yolo26s-det-bf16-mla_tess-b1.tar.gz
sima-cli download https://docs.sima.ai/pkg_downloads/SDK<platform-version>/models/modalix/yolo26-detection/yolo26m-det-bf16-mla_tess-b1.tar.gz
sima-cli download https://docs.sima.ai/pkg_downloads/SDK<platform-version>/models/modalix/yolo26-detection/yolo26l-det-bf16-mla_tess-b1.tar.gz
sima-cli download https://docs.sima.ai/pkg_downloads/SDK<platform-version>/models/modalix/yolo26-detection/yolo26x-det-bf16-mla_tess-b1.tar.gz
sima-cli download https://docs.sima.ai/pkg_downloads/SDK<platform-version>/models/modalix/yolo26-detection/yolo26m-det-bf16-b1.tar.gz
sima-cli download https://docs.sima.ai/pkg_downloads/SDK<platform-version>/models/modalix/yolo26-detection/yolo26m-det-int8-b1.tar.gz

cd ../..
```

## Build
### Build From The Apps Repo
```bash
cd <apps-repo-root>
./build.sh
```

### Build This Example Directly With CMake
```bash
cd <apps-repo-root>
cmake -S examples/object-detection/multi-stream-object-detector/src/cpp -B build/multi-stream-object-detector
cmake --build build/multi-stream-object-detector -j
```

## Run
### Validate Config Only
This is useful for a quick smoke test without opening RTSP streams.

```bash
./build/examples/object-detection/multi-stream-object-detector/multi-stream-object-detector \
  --config examples/object-detection/multi-stream-object-detector/src/common/config.yaml \
  --validate-config-only
```

### C++
```bash
SIMA_GST_RUN_INPUT_TIMEOUT_MS=120000 ./build/examples/object-detection/multi-stream-object-detector/multi-stream-object-detector \
  --config examples/object-detection/multi-stream-object-detector/src/common/config.yaml
```

### Python
```bash
source ~/pyneat/bin/activate
pip install -r examples/object-detection/multi-stream-object-detector/src/python/requirements.txt
SIMA_GST_RUN_INPUT_TIMEOUT_MS=120000 python3 examples/object-detection/multi-stream-object-detector/src/python/main.py \
  --config examples/object-detection/multi-stream-object-detector/src/common/config.yaml
```

## Debugging notes
- The checked-in `src/common/config.yaml` includes four placeholder stream slots. Replace the RTSP URLs and Insight host before running.
- This phase supports up to four active streams.
- On Modalix DevKit, start with `bash /usr/bin/fix_devkit_runtime.sh`. If the runtime still behaves inconsistently, a full board reboot has been a more reliable reset than service restarts alone.
- `output.debug_dir` and `output.save_every` let you save periodic aligned debug frames locally without changing the Insight output contract.
- Profiling prints per-stream pull, metadata, output FPS, and detection-count summaries.

## Notes
- Point `model.path` at a YOLO26 detection pack. This example does not use a `model.family` config key.
- Both C++ and Python use the same public graph shape: `RtspDecodedInput`, `graphs::Branch`, `model.graph()`, `graphs::Combine` for debug saves, and `VideoSender`.
- The live path keeps video and metadata separate. Insight overlays metadata on the clean video stream.
- `output.video_enabled: false` disables per-stream H.264 video output. Metadata still runs.

## Source Files
- C++: `src/cpp/main.cpp`
- C++ tests: `tests/cpp/test_unit.cpp`, `tests/cpp/test_e2e.cpp`
- Python: `src/python/main.py`
- Python tests: `tests/python/test_unit.py`, `tests/python/test_e2e.py`
- Shared assets: `src/common/`
