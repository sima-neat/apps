# Multi-Stream People Tracker

## Metadata
| Field | Value |
| --- | --- |
| Category | tracking |
| Difficulty | Intermediate |
| Tags | object-detection, rtsp, tracking, insight |
| Languages | C++, Python |
| Status | experimental |
| Binary Name | multi-stream-people-tracker |
| Model | yolo26m-det-bf16-mla_tess-b1 |

## Concept
Multi-stream people tracking example with RTSP inputs, mixed-resolution support, per-stream worker threads, Insight live video plus metadata output, and optional sampled overlay saves. The pipeline filters detector output to the configured person class and assigns stable track IDs per stream.

Both the Python and C++ entrypoints keep the detector graph explicit:

`RTSP decode -> model.graph() -> tracker -> VideoSender(H.264 RTP/UDP)`

Each RTSP stream gets its own source, detection, tracker, and `VideoSender` runtime so native stream resolution can be preserved per camera.

## Preview
Preview image from a live run:

![Multi-stream people tracker preview](../../../assets/portal/tracking/multi-stream-people-tracker/image.png)

## Supported Models
Default model: `yolo26m-det-bf16-mla_tess-b1.tar.gz`.

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

sima-cli download https://docs.sima.ai/pkg_downloads/SDK2.0.0/models/modalix/yolo26-detection/yolo26n-det-bf16-mla_tess-b1.tar.gz
sima-cli download https://docs.sima.ai/pkg_downloads/SDK2.0.0/models/modalix/yolo26-detection/yolo26s-det-bf16-mla_tess-b1.tar.gz
sima-cli download https://docs.sima.ai/pkg_downloads/SDK2.0.0/models/modalix/yolo26-detection/yolo26m-det-bf16-mla_tess-b1.tar.gz
sima-cli download https://docs.sima.ai/pkg_downloads/SDK2.0.0/models/modalix/yolo26-detection/yolo26l-det-bf16-mla_tess-b1.tar.gz
sima-cli download https://docs.sima.ai/pkg_downloads/SDK2.0.0/models/modalix/yolo26-detection/yolo26x-det-bf16-mla_tess-b1.tar.gz
sima-cli download https://docs.sima.ai/pkg_downloads/SDK2.0.0/models/modalix/yolo26-detection/yolo26m-det-bf16-b1.tar.gz
sima-cli download https://docs.sima.ai/pkg_downloads/SDK2.0.0/models/modalix/yolo26-detection/yolo26m-det-int8-b1.tar.gz

cd ../..
```

## Prerequisites
- A Neat Python environment with `pyneat`, `numpy`, and OpenCV available.
- One or more reachable RTSP camera URLs.
- A YOLO26 detector model pack downloaded into `assets/models/`.
- An Insight viewer instance reachable from the board/host running this example.

## Important Behavior
- The Python and C++ implementations follow the same config-driven structure: config loading, pipeline builders, tracker helpers, image helpers, sample helpers, and worker orchestration.
- The `streams:` list in `src/common/config.yaml` controls the number of cameras dynamically.
- The checked-in `src/common/config.yaml` uses placeholder RTSP and Insight values; fill them with your own camera URLs and receiver host before running.
- Stream `i` publishes video to `output.insight.video_port_base + i`.
- `output.video_mode: clean` publishes unannotated RGB frames through `VideoSender` and sends tracking metadata to `output.insight.metadata_port_base + i` for Insight-side overlay. `annotated` draws tracking boxes into RGB frames before `VideoSender` encodes them and suppresses metadata so Insight does not overlay twice.
- `inference.frames: 0` runs indefinitely.
- `output.debug_dir: null` and `output.save_every: 0` disable saved overlay frames while keeping live Insight output enabled.
- `inference.detection_threshold`, `inference.nms_iou_threshold`, and `inference.top_k` are explicit detector decode parameters in the checked-in config.
- The example defaults to person class id `0`, and tracker behavior is configurable from the config file.
- Both C++ and Python add the `VideoSender` nodegroup for video transport. They do not manually add lower-level color conversion, encoder, parser, packetizer, or UDP nodes outside `VideoSender`.
- Because this example sends raw frames, it uses the raw-frame `VideoSender` option. If an upstream pipeline already produces H.264, `VideoSender` also supports an encoded-input option that parses, packetizes, and sends without re-encoding.

## Command-Line Options
### C++
- Invocation:
  ```bash
  ./build/examples/tracking/multi-stream-people-tracker/multi-stream-people-tracker \
    --config examples/tracking/multi-stream-people-tracker/src/common/config.yaml
  ```
- Required arguments:
  None.
- Optional arguments:
  `--config <path>` to load a different YAML configuration file.

### Python
- Invocation:
  ```bash
  python3 examples/tracking/multi-stream-people-tracker/src/python/main.py \
    --config examples/tracking/multi-stream-people-tracker/src/common/config.yaml
  ```
- Required arguments:
  None.
- Optional arguments:
  `--config <path>` to load a different YAML configuration file.

## Build
### Build From The Apps Repo
```bash
cd <apps-repo-root>
./build.sh
```

Binary output:
```bash
./build/examples/tracking/multi-stream-people-tracker/multi-stream-people-tracker
```

### Build This Example Directly With CMake
```bash
cd <apps-repo-root>
cmake -S examples/tracking/multi-stream-people-tracker/src/cpp -B build/multi-stream-people-tracker
cmake --build build/multi-stream-people-tracker -j
```

Binary output:
```bash
./build/multi-stream-people-tracker/multi-stream-people-tracker
```

## Run
Before running either entrypoint, edit `examples/tracking/multi-stream-people-tracker/src/common/config.yaml` and replace the placeholder values in:

- `streams`
- `output.insight.host`

### C++
```bash
./build/examples/tracking/multi-stream-people-tracker/multi-stream-people-tracker \
  --config examples/tracking/multi-stream-people-tracker/src/common/config.yaml
```

The C++ binary follows the same config contract and worker topology as the Python example.

### Python
Install the small Python-side dependencies:

```bash
source ~/pyneat/bin/activate
pip install -r examples/tracking/multi-stream-people-tracker/src/python/requirements.txt
```

Edit the example config in `src/common/config.yaml`, especially the placeholder values in:

- `streams`
- `output.insight.host`

The `streams:` list controls the number of cameras dynamically.

Run the Python example with that config:

```bash
python3 examples/tracking/multi-stream-people-tracker/src/python/main.py \
  --config examples/tracking/multi-stream-people-tracker/src/common/config.yaml
```

Notes:

- in `output.video_mode: clean`, stream `i` feeds clean frames into `VideoSender` at
  `output.insight.video_port_base + i` and sends tracking metadata to
  `output.insight.metadata_port_base + i`
- in `output.video_mode: annotated`, stream `i` draws tracks into the frame before
  feeding it into `VideoSender` and suppresses metadata so Insight does not overlay twice
- the default config runs indefinitely and does not save frames because
  `output.debug_dir` is `null` and `output.save_every` is `0`
- set `inference.frames` for a bounded smoke run
- set `output.debug_dir` and `output.save_every` if you want sampled overlay frames
  written under `stream_<index>/`; the live Insight video stays clean
- if the app runs on a DevKit, set `output.insight.host` to the Insight host IP,
  not `127.0.0.1`
- `pyneat.Model(...)` is still used, but the detector runtime composes the
  model-owned graph fragment instead of a black-box one-call inference path
- live metadata is emitted separately from video in Insight metadata format in
  `clean` mode, with one channel per stream

## Debugging Notes
- Start with one RTSP stream and confirm the config before scaling to multiple cameras.
- Confirm the model file exists under `assets/models/`.
- Confirm each RTSP URL is reachable from the board or host running the example.
- If Insight appears idle, verify `output.insight.host`, `video_port_base`, and `metadata_port_base`.
- If you want saved overlay frames, set both `output.debug_dir` and `output.save_every`.

## Source Files
- C++ source: `src/cpp/main.cpp`
- C++ config loader: `src/cpp/utils/config_api.cpp`, `src/cpp/utils/config.cpp`
- C++ tracker helpers: `src/cpp/utils/tracker_api.cpp`, `src/cpp/utils/tracker.cpp`
- C++ sample helpers: `src/cpp/utils/sample_utils_api.cpp`, `src/cpp/utils/sample_utils.cpp`
- C++ pipeline builders: `src/cpp/utils/pipeline_api.cpp`, `src/cpp/utils/pipeline.cpp`
- C++ image helpers: `src/cpp/utils/image_utils_api.cpp`, `src/cpp/utils/image_utils.cpp`
- C++ worker orchestration: `src/cpp/utils/workers_api.cpp`, `src/cpp/utils/workers.cpp`
- C++ tests: `tests/cpp/test_unit.cpp`, `tests/cpp/test_e2e.cpp`
- Python source: `src/python/main.py`
- Example config: `src/common/config.yaml`
- Python utilities: `src/python/utils/`
- Python tests: `tests/python/test_unit.py`, `tests/python/test_e2e.py`
- Shared example data: `src/common/`
