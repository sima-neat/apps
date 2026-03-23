# Multistream YOLOX/YOLOv8 Object Detection OptiView

## Metadata
| Field | Value |
| --- | --- |
| Category | object-detection |
| Difficulty | Intermediate |
| Tags | object-detection, rtsp, multistream, optiview, yolox, yolov8 |
| Languages | C++, Python |
| Status | experimental |
| Binary Name | multistream-yolox-yolov8-object-detection-optiview |
| Model | yolox_s / yolo_v8m |

## Concept
This example runs a config-driven multistream RTSP detection pipeline that supports both YOLOX and YOLOv8 model packs and publishes per-stream video plus detection metadata to OptiView.

Architecture:
- one RTSP source runtime per stream
- one OptiView video runtime and one OptiView JSON sender per stream
- a shared detector worker pool sized by `runtime.worker_count`
- one keep-latest mailbox per stream so the example can scale to many cameras without building large per-stream backlogs

Detector graphs:
- YOLOv8: `QuantTess -> MLA -> SimaBoxDecode`
- YOLOX: `QuantTess -> MLA -> DetessDequant`

If a current YOLOX model pack fails during runtime construction with a schema/dependency error, switch `model.path` to a YOLOv8 pack until updated YOLOX packs are available.

## Prerequisites
- Installed NEAT SDK.
- One or more reachable RTSP camera URLs.
- YOLOX or YOLOv8 model packs downloaded into `assets/models/`.
- Edit `common/config.yaml` before running with real streams.

## Download Models
```bash
mkdir -p assets/models
cd assets/models

sima-cli modelzoo get yolox_nano
sima-cli modelzoo get yolox_tiny
sima-cli modelzoo get yolox_s
sima-cli modelzoo get yolox_m
sima-cli modelzoo get yolox_l
sima-cli modelzoo get yolox_x

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
- `output.video_mode: clean` sends the original frame to OptiView. `annotated` draws detection boxes and labels before publishing.
- `runtime.mailbox_depth` defaults to `1` and should usually stay small for dense multistream runs.
- `output.optiview.json_offset_ms` lets you shift JSON timestamps to better align OptiView boxes with the published video stream when transport latency makes boxes appear early or late.
- profiling now prints `source`, `preproc`, `detect`, `video`, `json`, `publish`, and `loop` timings per stream so bottlenecks are easier to isolate.
- `output.debug_dir` and `output.save_every` let you save periodic RGB debug frames locally without changing the OptiView output contract.

## Source Files
- C++: `cpp/main.cpp`
- C++ tests: `cpp/tests/unit_test.cpp`, `cpp/tests/e2e_test.cpp`
- Python: `python/main.py`
- Python tests: `python/tests/test_unit.py`, `python/tests/test_e2e.py`
- Shared assets: `common/`
