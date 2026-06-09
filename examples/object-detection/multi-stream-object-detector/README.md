# Multi-Stream Object Detector

## Metadata
| Field | Value |
| --- | --- |
| Category | object-detection |
| Difficulty | Advanced |
| Tags | object-detection, rtsp, multistream, insight, yolov8 |
| Languages | C++, Python |
| Status | experimental |
| Binary Name | multi-stream-object-detector |
| Model | yolo_v8m |

## Concept
This example runs a config-driven multistream RTSP detection pipeline for YOLOv8 model packs and publishes per-stream video plus detection metadata to Insight.

## Preview
Snippet from a pipeline run:

![Multi-stream object detector preview](../../../assets/portal/object-detection/multi-stream-object-detector/image.png)

Architecture:
- one decoded RGB RTSP source runtime per stream
- one lazy Insight video runtime and one Insight metadata sender per stream
- a shared detector worker pool sized by `runtime.worker_count`
- one keep-latest mailbox per stream so the example can scale to many cameras without building large per-stream backlogs

Detector graph:
- YOLOv8: `Input(RGB) -> Preprocess -> Infer/MLA -> SimaBoxDecode`

Video graph:
- `Input(RGB) -> VideoSender(H.264 RTP/UDP)`

## Prerequisites
- Installed Neat SDK.
- One or more reachable RTSP camera URLs.
- A YOLOv8 model pack downloaded into `assets/models/`.
- Edit `src/common/config.yaml` before running with real streams.
- On Modalix DevKit, run `bash /usr/bin/fix_devkit_runtime.sh` before starting the example if the runtime has been used by earlier ML/video apps.

## Command-line options
- `--config <path>`: path to the YAML configuration file
- `--validate-config-only`: validate the config and exit without opening RTSP streams
- `--help`: print the CLI help text

## Download Models
```bash
mkdir -p assets/models
cd assets/models

sima-cli modelzoo -v 2.0.0 get yolo_v8n
sima-cli modelzoo -v 2.0.0 get yolo_v8s
sima-cli modelzoo -v 2.0.0 get yolo_v8m
sima-cli modelzoo -v 2.0.0 get yolo_v8l
sima-cli modelzoo -v 2.0.0 get yolo_v8x
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
- The checked-in `src/common/config.yaml` includes 16 placeholder stream slots. Replace the RTSP URLs and Insight host before running.
- On Modalix DevKit, start with `bash /usr/bin/fix_devkit_runtime.sh`. If the runtime still behaves inconsistently, a full board reboot has been a more reliable reset than service restarts alone.
- `output.debug_dir` and `output.save_every` let you save periodic RGB debug frames locally without changing the Insight output contract.
- `output.insight.metadata_offset_ms` lets you shift metadata timestamps to better align Insight boxes with the published video stream when transport latency makes boxes appear early or late. It only applies when metadata output is enabled.
- When `inference.fps` is lower than the source FPS, the example throttles after decode and keeps only the most recent frame per stream in the mailbox.
- Profiling prints `source`, `preproc`, `detect`, `video`, `metadata`, `publish`, and `loop` timings per stream so bottlenecks are easier to isolate.

## Notes
- Point `model.path` at a YOLOv8 pack. This example infers YOLOv8 from the model path and does not use a `model.family` config key.
- Both the C++ and Python paths decode each RTSP stream to RGB in system memory, run YOLOv8 on those RGB frames, and feed the selected RGB frame into `VideoSender`. They do not manually build lower-level color conversion, encoder, parser, packetizer, or UDP nodes.
- `output.video_mode: clean` feeds unannotated RGB frames into `VideoSender` and keeps metadata enabled. `annotated` draws detection boxes into the RGB frame before feeding it into `VideoSender` and suppresses metadata so Insight does not overlay detections twice.
- Because these examples send raw frames, they use the raw-frame `VideoSender` option. If an upstream pipeline already produces H.264, `VideoSender` also supports an encoded-input option that parses, packetizes, and sends without re-encoding.
- `output.video_enabled: false` disables per-stream H264 video output. In `clean` mode the example still sends metadata detections; in `annotated` mode metadata is suppressed.
- `runtime.mailbox_depth` defaults to `1` and should usually stay small for dense multistream runs.
- The example applies the following runtime defaults for dense RTSP runs when the environment does not already override them:
  `SIMA_FORCE_MODEL_NUM_BUFFERS=3`, `SIMA_FORCE_DECODER_NUM_BUFFERS=7`, and `SIMA_FORCE_DECODER_POOL_BUFFERS=7`.
- The Python implementation mirrors the same high-level contract as C++ while staying on public `pyneat`: `RtspDecodedInput`, model preprocess, `groups.mla(model)`, `nodes.sima_box_decode(...)`, and `groups.video_sender(...)`.

## Source Files
- C++: `src/cpp/main.cpp`
- C++ runtime helpers: `src/cpp/utils/config.cpp`, `src/cpp/utils/pipeline.cpp`, `src/cpp/utils/sample_utils.cpp`, `src/cpp/utils/workers.cpp`
- C++ tests: `tests/cpp/test_unit.cpp`, `tests/cpp/test_e2e.cpp`
- Python: `src/python/main.py`
- Python runtime helpers: `src/python/utils/config.py`, `src/python/utils/model_family.py`, `src/python/utils/pipeline.py`, `src/python/utils/sample_utils.py`, `src/python/utils/image_utils.py`, `src/python/utils/workers.py`
- Python tests: `tests/python/test_unit.py`, `tests/python/test_e2e.py`
- Shared assets: `src/common/`
