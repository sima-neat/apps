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
| Model | yolo_v8m |

## Concept
Multi-stream people tracking example with RTSP inputs, mixed-resolution support, per-stream worker threads, Insight live video plus metadata output, and optional sampled overlay saves. The pipeline filters detector output to the configured person class and assigns stable track IDs per stream.

Both the Python and C++ entrypoints keep the detector graph explicit rather than hiding it behind a single `model.run(...)` call:

`RTSP decode -> CPU letterbox/normalize -> QuantTess -> MLA -> SimaBoxDecode -> tracker -> VideoSender(H.264 RTP/UDP)`

Each RTSP stream gets its own source, detection, tracker, and `VideoSender` runtime so native stream resolution can be preserved per camera.

## Preview
Preview image from a live run:

![Multi-stream people tracker preview](../../../assets/portal/tracking/multi-stream-people-tracker/image.png)

## Supported Models
Also works with: `yolo_v8n`, `yolo_v8s`, `yolo_v8l`, `yolo_v8x`

Download any variant into `assets/models/`:

```bash
mkdir -p assets/models
cd assets/models
sima-cli modelzoo -v 2.0.0 get yolo_v8n
sima-cli modelzoo -v 2.0.0 get yolo_v8s
sima-cli modelzoo -v 2.0.0 get yolo_v8m
sima-cli modelzoo -v 2.0.0 get yolo_v8l
sima-cli modelzoo -v 2.0.0 get yolo_v8x
cd ../..
```

## Prerequisites
- A Neat Python environment with `pyneat`, `numpy`, and OpenCV available.
- One or more reachable RTSP camera URLs.
- A YOLOv8 detector model pack downloaded into `assets/models/`.
- An Insight viewer instance reachable from the board/host running this example.

## Important Behavior
- The Python and C++ implementations follow the same config-driven structure: config loading, pipeline builders, tracker helpers, image helpers, sample helpers, and worker orchestration.
- The `streams:` list in `common/config.yaml` controls the number of cameras dynamically.
- The checked-in `common/config.yaml` uses placeholder RTSP and Insight values; fill them with your own camera URLs and receiver host before running.
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
    --config examples/tracking/multi-stream-people-tracker/common/config.yaml
  ```
- Required arguments:
  None.
- Optional arguments:
  `--config <path>` to load a different YAML configuration file.

### Python
- Invocation:
  ```bash
  python3 examples/tracking/multi-stream-people-tracker/python/main.py \
    --config examples/tracking/multi-stream-people-tracker/common/config.yaml
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
cmake -S examples/tracking/multi-stream-people-tracker/cpp -B build/multi-stream-people-tracker
cmake --build build/multi-stream-people-tracker -j
```

Binary output:
```bash
./build/multi-stream-people-tracker/multi-stream-people-tracker
```

## Run
Before running either entrypoint, edit `examples/tracking/multi-stream-people-tracker/common/config.yaml` and replace the placeholder values in:

- `streams`
- `output.insight.host`

### C++
```bash
./build/examples/tracking/multi-stream-people-tracker/multi-stream-people-tracker \
  --config examples/tracking/multi-stream-people-tracker/common/config.yaml
```

The C++ binary follows the same config contract and worker topology as the Python example.

### Python
Install the small Python-side dependencies:

```bash
source ~/pyneat/bin/activate
pip install -r examples/tracking/multi-stream-people-tracker/python/requirements.txt
```

Edit the example config in `common/config.yaml`, especially the placeholder values in:

- `streams`
- `output.insight.host`

The `streams:` list controls the number of cameras dynamically.

Run the Python example with that config:

```bash
python3 examples/tracking/multi-stream-people-tracker/python/main.py \
  --config examples/tracking/multi-stream-people-tracker/common/config.yaml
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
- `pyneat.Model(...)` is still used, but as the model-pack contract source for
  the explicit `QuantTess -> MLA -> SimaBoxDecode` session, not as a black-box
  one-call inference path
- the example uses CPU-side OpenCV letterbox + normalize on A65 and feeds the
  detector through the model's tensor-input `QuantTess` contract
- live metadata is emitted separately from video in Insight metadata format in
  `clean` mode, with one channel per stream

## Debugging Notes
- Start with one RTSP stream and confirm the config before scaling to multiple cameras.
- Confirm the model file exists under `assets/models/`.
- Confirm each RTSP URL is reachable from the board or host running the example.
- If Insight appears idle, verify `output.insight.host`, `video_port_base`, and `metadata_port_base`.
- If you want saved overlay frames, set both `output.debug_dir` and `output.save_every`.

## Source Files
- C++ source: `cpp/main.cpp`
- C++ config loader: `cpp/utils/config_api.cpp`, `cpp/utils/config.cpp`
- C++ tracker helpers: `cpp/utils/tracker_api.cpp`, `cpp/utils/tracker.cpp`
- C++ sample helpers: `cpp/utils/sample_utils_api.cpp`, `cpp/utils/sample_utils.cpp`
- C++ pipeline builders: `cpp/utils/pipeline_api.cpp`, `cpp/utils/pipeline.cpp`
- C++ image helpers: `cpp/utils/image_utils_api.cpp`, `cpp/utils/image_utils.cpp`
- C++ worker orchestration: `cpp/utils/workers_api.cpp`, `cpp/utils/workers.cpp`
- C++ tests: `cpp/tests/unit_test.cpp`, `cpp/tests/e2e_test.cpp`
- Python source: `python/main.py`
- Example config: `common/config.yaml`
- Python utilities: `python/utils/`
- Python tests: `python/tests/test_unit.py`, `python/tests/test_e2e.py`
- Shared example data: `common/`
