# Multistream YOLOv8 Object Detection OptiView

## Metadata
| Field | Value |
| --- | --- |
| Category | object-detection |
| Difficulty | Intermediate |
| Tags | object-detection, rtsp, multistream, optiview, yolov8 |
| Languages | C++, Python |
| Status | experimental |
| Binary Name | multistream-yolox-yolov8-object-detection-optiview |
| Model | yolo_v8m |

## Concept
This example runs a config-driven multistream RTSP detection pipeline for YOLOv8 model packs and publishes per-stream video plus detection metadata to OptiView.

The folder and binary keep their original `multistream-yolox-yolov8-...` names for compatibility, but the implementation is currently YOLOv8-only.

## Preview
Snippet from a pipeline run:

![Multistream YOLOv8 OptiView preview](../../../assets/portal/object-detection/multistream-yolox-yolov8-object-detection-optiview/image.png)

Architecture:
- one encoded RTSP source runtime per stream
- one NV12 decode runtime per stream for detector input
- one OptiView video runtime and one OptiView JSON sender per stream
- a shared detector worker pool sized by `runtime.worker_count`
- one keep-latest mailbox per stream so the example can scale to many cameras without building large per-stream backlogs
- the C++ clean-video path now mirrors the GraphPipes strategy: forward original H264 to OptiView, decode separately to NV12 for YOLO, and avoid per-stream video re-encode on the scalable path

Detector graph:
- YOLOv8: `Input(NV12) -> Preprocess -> Infer -> SimaBoxDecode`

YOLOX model packs are not supported yet by this example. Future support is planned, but the current implementation and tests only cover YOLOv8.

## Prerequisites
- Installed NEAT SDK.
- One or more reachable RTSP camera URLs.
- A YOLOv8 model pack downloaded into `assets/models/`.
- Edit `common/config.yaml` before running with real streams.

## Download Models
```bash
mkdir -p assets/models
cd assets/models

sima-cli modelzoo get object_detection/yolo_v8n
sima-cli modelzoo get object_detection/yolo_v8s
sima-cli modelzoo get object_detection/yolo_v8m
sima-cli modelzoo get object_detection/yolo_v8l
sima-cli modelzoo get object_detection/yolo_v8x
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
cmake -S examples/object-detection/multistream-yolox-yolov8-object-detection-optiview/cpp -B build/multistream-yolox-yolov8-object-detection-optiview
cmake --build build/multistream-yolox-yolov8-object-detection-optiview -j
```

## Run
### Validate Config Only
This is useful for a quick smoke test without opening RTSP streams.

```bash
./build/examples/object-detection/multistream-yolox-yolov8-object-detection-optiview/multistream-yolox-yolov8-object-detection-optiview \
  --config examples/object-detection/multistream-yolox-yolov8-object-detection-optiview/common/config.yaml \
  --validate-config-only
```

### C++
```bash
./build/examples/object-detection/multistream-yolox-yolov8-object-detection-optiview/multistream-yolox-yolov8-object-detection-optiview \
  --config examples/object-detection/multistream-yolox-yolov8-object-detection-optiview/common/config.yaml
```

### Python
```bash
source ~/pyneat/bin/activate
pip install -r examples/object-detection/multistream-yolox-yolov8-object-detection-optiview/python/requirements.txt
python3 examples/object-detection/multistream-yolox-yolov8-object-detection-optiview/python/main.py \
  --config examples/object-detection/multistream-yolox-yolov8-object-detection-optiview/common/config.yaml
```

## Notes
- Set `model.family: yolov8` and point `model.path` at a YOLOv8 pack. If you try a YOLOX pack, the example now fails fast with a clear "not supported yet" message instead of building a mismatched detector graph.
- YOLOX support is planned for a future revision of this example, but it is not part of the current runtime or test contract.
- The checked-in `common/config.yaml` includes 16 stream slots and example RTSP/OptiView values. Replace the stream URLs and receiver host with your own camera and OptiView endpoints before running.
- `output.video_mode: clean` forwards the original encoded H264 stream to OptiView and keeps the JSON side channel enabled. `annotated` decodes for overlay, draws detection boxes and labels into the video stream, and suppresses JSON output so OptiView does not overlay detections twice.
- `output.video_enabled: false` disables per-stream H264 video output. In `clean` mode the example still sends JSON detections; in `annotated` mode JSON is suppressed as well.
- `runtime.mailbox_depth` defaults to `1` and should usually stay small for dense multistream runs.
- the example now applies GraphPipes-inspired runtime defaults for dense RTSP runs when the environment does not already override them:
  `SIMA_FORCE_MODEL_NUM_BUFFERS=3`, `SIMA_FORCE_DECODER_NUM_BUFFERS=7`, and `SIMA_FORCE_DECODER_POOL_BUFFERS=7`.
- the C++ path now uses an explicit public-node encoded RTSP source, a separate `H264Decode(..., out_format=\"NV12\", num_buffers=7)` detector branch, and encoded H264 OptiView forwarding in `clean` mode. The Python path applies the same runtime defaults but still uses the high-level `pyneat.groups.rtsp_decoded_input(...)` surface until lower-level binding parity is available.
- when `inference.fps` is lower than the source FPS, the source runtime now applies `EveryFrame(n)` decimation when it can reduce app-side work without obviously undershooting the requested rate.
- `output.optiview.json_offset_ms` lets you shift JSON timestamps to better align OptiView boxes with the published video stream when transport latency makes boxes appear early or late. It only applies when JSON output is enabled.
- profiling now prints `source`, `preproc`, `detect`, `video`, `json`, `publish`, and `loop` timings per stream so bottlenecks are easier to isolate.
- `output.debug_dir` and `output.save_every` let you save periodic RGB debug frames locally without changing the OptiView output contract.

## Source Files
- C++: `cpp/main.cpp`
- C++ tests: `cpp/tests/unit_test.cpp`, `cpp/tests/e2e_test.cpp`
- Python: `python/main.py`
- Python tests: `python/tests/test_unit.py`, `python/tests/test_e2e.py`
- Shared assets: `common/`
