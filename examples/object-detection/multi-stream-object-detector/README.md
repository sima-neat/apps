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

## Insight Setup
[Neat Insight](https://developer.sima.ai/software/tools/insight/) can host RTSP streams, receive video from `VideoSender`, receive detection metadata from `MetadataSender`, and show rendered overlays plus runtime metrics in the browser.

To create reproducible RTSP inputs:
1. Run `neat` in the Neat Developer Environment and open the reported `Insight Web UI`.
2. In Insight, open `RTSP Source`.
3. Use sample videos or upload your own videos.
4. Start each stream and copy the RTSP URLs.
5. Put those RTSP URLs into `streams`.

Use the same `neat` output to set `output.insight.host`, `video_port_base`, and `metadata_port_base` from the reported `videoUDP` and `metadataUDP` ranges.

## Prerequisites
- Installed Neat Development Environment.
- RTSP sources created in Insight or provided by your cameras.
- Default YOLO26 model pack downloaded into `assets/models/`.
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

## Download Models
This README is written for SDK `2.1.2`.

The default model is `yolo26m-det-int8-b1.tar.gz`.

Download the default model:

```bash
mkdir -p assets/models
cd assets/models

sima-cli download https://docs.sima.ai/pkg_downloads/SDK2.1.2/models/modalix/yolo26-detection/yolo26m-det-int8-b1.tar.gz

cd ../..
```

## Configure
Edit `examples/object-detection/multi-stream-object-detector/src/common/config.yaml`.

```yaml
model:
  path: assets/models/yolo26m-det-int8-b1.tar.gz       # Model package to load.

streams:
  - <first-rtsp-url-copied-from-insight>               # First RTSP stream.
  - <second-rtsp-url-copied-from-insight>              # Second RTSP stream.

inference:
  frames: 0                                            # Frame limit per stream. 0 runs continuously.
  min_score: 0.30                                      # Minimum object confidence.

output:
  insight:
    host: <insight-host-ip>                            # Host running Insight.
    video_port_base: <videoUDP start port from neat>   # First UDP video port.
    metadata_port_base: <metadataUDP start port from neat> # First UDP metadata port.
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

## Debugging Notes
- The checked-in `src/common/config.yaml` includes four placeholder stream slots. Replace the RTSP URLs and Insight host before running.
- This phase supports up to four active streams.
- On Modalix DevKit, start with `bash /usr/bin/fix_devkit_runtime.sh`. If the runtime still behaves inconsistently, a full board reboot has been a more reliable reset than service restarts alone.
- `output.debug_dir` and `output.save_every` let you save periodic aligned debug frames locally without changing the Insight output contract.
- Profiling prints per-stream pull, metadata, output FPS, and detection-count summaries.

## Appendix: Additional Models
Other supported batch-1 YOLO26 detection models:
- `yolo26n-det-bf16-mla_tess-b1.tar.gz`
- `yolo26s-det-bf16-mla_tess-b1.tar.gz`
- `yolo26m-det-bf16-mla_tess-b1.tar.gz`
- `yolo26l-det-bf16-mla_tess-b1.tar.gz`
- `yolo26x-det-bf16-mla_tess-b1.tar.gz`
- `yolo26m-det-bf16-b1.tar.gz`

Replace the default filename in the download command and `model.path`.

## Source Files
- C++: `src/cpp/main.cpp`
- C++ tests: `tests/cpp/test_unit.cpp`, `tests/cpp/test_e2e.cpp`
- Python: `src/python/main.py`
- Python tests: `tests/python/test_unit.py`, `tests/python/test_e2e.py`
- Shared assets: `src/common/`
