# Multistream Object Detection via Insight

## Metadata
| Field | Value |
| --- | --- |
| Category | object-detection |
| Difficulty | Advanced |
| Tags | object-detection, rtsp, multistream, insight, yolov8 |
| Languages | C++, Python |
| Status | experimental |
| Binary Name | multistream-object-detection-insight |
| Model | yolo_v8m |

## Concept
This example runs a config-driven multistream RTSP detection pipeline for YOLOv8 model packs and publishes per-stream video plus detection metadata to Insight.

## Preview
Snippet from a pipeline run:

![Multistream YOLOv8 Insight preview](../../../assets/portal/object-detection/multistream-object-detection-insight/image.png)

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
- Edit `common/config.yaml` before running with real streams.
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
cmake -S examples/object-detection/multistream-object-detection-insight/cpp -B build/multistream-object-detection-insight
cmake --build build/multistream-object-detection-insight -j
```

## Run
### Validate Config Only
This is useful for a quick smoke test without opening RTSP streams.

```bash
./build/examples/object-detection/multistream-object-detection-insight/multistream-object-detection-insight \
  --config examples/object-detection/multistream-object-detection-insight/common/config.yaml \
  --validate-config-only
```

### C++
```bash
SIMA_GST_RUN_INPUT_TIMEOUT_MS=120000 ./build/examples/object-detection/multistream-object-detection-insight/multistream-object-detection-insight \
  --config examples/object-detection/multistream-object-detection-insight/common/config.yaml
```

### Python
```bash
source ~/pyneat/bin/activate
pip install -r examples/object-detection/multistream-object-detection-insight/python/requirements.txt
SIMA_GST_RUN_INPUT_TIMEOUT_MS=120000 python3 examples/object-detection/multistream-object-detection-insight/python/main.py \
  --config examples/object-detection/multistream-object-detection-insight/common/config.yaml
```

## Debugging notes
- The checked-in `common/config.yaml` includes 16 placeholder stream slots. Replace the RTSP URLs and Insight host before running.
- On Modalix DevKit, start with `bash /usr/bin/fix_devkit_runtime.sh`. If the runtime still behaves inconsistently, a full board reboot has been a more reliable reset than service restarts alone.
- `output.debug_dir` and `output.save_every` let you save periodic RGB debug frames locally without changing the Insight output contract.
- `output.insight.metadata_offset_ms` lets you shift metadata timestamps to better align Insight boxes with the published video stream when transport latency makes boxes appear early or late. It only applies when metadata output is enabled.
- When `inference.fps` is lower than the source FPS, the example throttles after decode and keeps only the most recent frame per stream in the mailbox.
- Profiling prints `source`, `preproc`, `detect`, `video`, `metadata`, `publish`, and `loop` timings per stream so bottlenecks are easier to isolate.

## Notes
- Point `model.path` at a YOLOv8 pack. This example infers YOLOv8 from the model path and does not use a `model.family` config key.
- Both the C++ and Python paths decode each RTSP stream to RGB in system memory, run YOLOv8 on those RGB frames, and re-encode RGB frames for Insight video output. They do not forward the original encoded H264 bitstream.
- `output.video_mode: clean` publishes unannotated RGB frames to Insight and keeps metadata enabled. `annotated` draws detection boxes into the RGB video stream and suppresses metadata so Insight does not overlay detections twice.
- `output.video_enabled: false` disables per-stream H264 video output. In `clean` mode the example still sends metadata detections; in `annotated` mode metadata is suppressed.
- `runtime.mailbox_depth` defaults to `1` and should usually stay small for dense multistream runs.
- The example applies the following runtime defaults for dense RTSP runs when the environment does not already override them:
  `SIMA_FORCE_MODEL_NUM_BUFFERS=3`, `SIMA_FORCE_DECODER_NUM_BUFFERS=7`, and `SIMA_FORCE_DECODER_POOL_BUFFERS=7`.
- The Python implementation mirrors the same high-level contract as C++ while staying on public `pyneat`: `RtspDecodedInput`, model preprocess, `groups.mla(model)`, `nodes.sima_box_decode(...)`, and `groups.video_sender(...)`.

## Source Files
- C++: `cpp/main.cpp`
- C++ runtime helpers: `cpp/utils/config.cpp`, `cpp/utils/pipeline.cpp`, `cpp/utils/sample_utils.cpp`, `cpp/utils/workers.cpp`
- C++ tests: `cpp/tests/unit_test.cpp`, `cpp/tests/e2e_test.cpp`
- Python: `python/main.py`
- Python runtime helpers: `python/utils/config.py`, `python/utils/model_family.py`, `python/utils/pipeline.py`, `python/utils/sample_utils.py`, `python/utils/image_utils.py`, `python/utils/workers.py`
- Python tests: `python/tests/test_unit.py`, `python/tests/test_e2e.py`
- Shared assets: `common/`
