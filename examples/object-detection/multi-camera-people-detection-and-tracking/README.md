# multi-camera-people-detection-and-tracking

## Metadata
| Field | Value |
| --- | --- |
| Category | object-detection |
| Difficulty | Intermediate |
| Tags | object-detection, rtsp, tracking, optiview |
| Languages | C++, Python |
| Status | experimental |
| Binary Name | multi-camera-people-detection-and-tracking |
| Model | YOLOv8 person detection |

## Concept
Python-first multi-camera people detection and tracking example with RTSP inputs,
mixed-resolution support, per-stream worker threads, OptiView live video plus
JSON metadata output, and optional sampled overlay saves.

The Python entrypoint keeps the detector graph explicit rather than hiding it
behind a single `model.run(...)` call:

`RTSP decode -> CPU letterbox/normalize -> QuantTess -> MLA -> SimaBoxDecode -> tracker -> clean H264/OptiView + tracked JSON`

Each RTSP stream gets its own source, detection, tracker, encoder, and OptiView
publisher runtime so
native stream resolution can be preserved per camera.

## Prerequisites
- A NEAT Python environment with `pyneat`, `numpy`, and OpenCV available.
- One or more reachable RTSP camera URLs.
- A YOLOv8 detector model pack downloaded into `assets/models/`.

Representative model download commands:

```bash
mkdir -p assets/models
cd assets/models
sima-cli modelzoo get object_detection/yolo_v8n
sima-cli modelzoo get object_detection/yolo_v8s
sima-cli modelzoo get object_detection/yolo_v8m
sima-cli modelzoo get object_detection/yolo_v8l
sima-cli modelzoo get object_detection/yolo_v8x
cd ../..
```

- An OptiView viewer instance reachable from the board/host running this example.

## Run
Install the small Python-side dependencies:

```bash
source ~/pyneat/bin/activate
pip install -r examples/object-detection/multi-camera-people-detection-and-tracking/python/requirements.txt
```

Edit the example config in `common/config.yaml`, especially:

- `model`
- `streams`
- `output.optiview.host`

The `streams:` list controls the number of cameras dynamically.

Run the Python example with that config:

```bash
python examples/object-detection/multi-camera-people-detection-and-tracking/python/main.py \
  --config examples/object-detection/multi-camera-people-detection-and-tracking/common/config.yaml
```

Notes:

- stream `i` publishes clean video to `output.optiview.video_port_base + i` and tracked JSON to
  `output.optiview.json_port_base + i`
- the default config runs indefinitely and does not save frames because
  `output.debug_dir` is `null` and `output.save_every` is `0`
- set `inference.frames` for a bounded smoke run
- set `output.debug_dir` and `output.save_every` if you want sampled overlay frames
  written under `stream_<index>/`; the live OptiView video stays clean
- if the app runs on a DevKit, set `output.optiview.host` to the OptiView host IP,
  not `127.0.0.1`
- `pyneat.Model(...)` is still used, but as the model-pack contract source for
  the explicit `QuantTess -> MLA -> SimaBoxDecode` session, not as a black-box
  one-call inference path
- the example uses CPU-side OpenCV letterbox + normalize on A65 and feeds the
  detector through the model's tensor-input `QuantTess` contract
- live metadata is emitted separately from video in OptiView JSON format, with
  one channel per stream
- `inference.detection_threshold`, `inference.nms_iou_threshold`, and
  `inference.top_k` are optional; if omitted, `SimaBoxDecode` keeps the model-pack defaults
- the example defaults to person class id `0`, and tracker behavior remains
  configurable from the config
- C++ parity is tracked in a follow-up ticket; the C++ entrypoint currently exists
  only as a placeholder scaffold

## Source Files
- Python entrypoint: `python/main.py`
- Python utilities: `python/utils/`
- Example config: `common/config.yaml`
- Python tests: `python/tests/test_config.py`, `python/tests/test_tracker.py`,
  `python/tests/test_pipeline.py`, `python/tests/test_image_utils.py`,
  `python/tests/test_workers.py`, `python/tests/test_main.py`,
  `python/tests/test_e2e.py`
- C++ scaffold: `cpp/main.cpp`
- C++ scaffold tests: `cpp/tests/unit_test.cpp`, `cpp/tests/e2e_test.cpp`
- Shared example data: `common/`
