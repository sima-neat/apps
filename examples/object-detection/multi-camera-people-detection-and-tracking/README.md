# multi-camera-people-detection-and-tracking

## Metadata
| Field | Value |
| --- | --- |
| Category | object-detection |
| Difficulty | Intermediate |
| Tags | object-detection, rtsp, tracking, udp |
| Languages | C++, Python |
| Status | experimental |
| Binary Name | multi-camera-people-detection-and-tracking |
| Model | YOLOv8 person detection |

## Concept
Python-first multi-camera people detection and tracking example with RTSP inputs,
mixed-resolution support, per-stream worker threads, RTP/H.264 UDP preview output,
and optional sampled overlay saves.

The Python pipeline keeps the important NEAT stages visible:

`RTSP decode -> CPU letterbox/normalize -> QuantTess -> MLA -> SimaBoxDecode -> tracker -> overlay -> H264/UDP`

Each RTSP stream gets its own source, detection, tracker, and UDP writer runtime so
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

Optional UDP preview workflow:

- start the UDP relay/browser preview from `tool-mediasources`
- point this example at the same host and UDP port base
- open `preview-udp.html` to watch the overlayed streams

## Run
Install the small Python-side dependencies:

```bash
source ~/pyneat/bin/activate
pip install -r examples/object-detection/multi-camera-people-detection-and-tracking/python/requirements.txt
```

Run the Python example with one `--rtsp` per stream:

```bash
python examples/object-detection/multi-camera-people-detection-and-tracking/python/main.py \
  --model assets/models/yolo_v8m_mpk.tar.gz \
  --udp-host <host> \
  --udp-port-base 5600 \
  --profile \
  --rtsp <rtsp-url-src-0> \
  --rtsp <rtsp-url-src-1>
```

Notes:

- `--udp-port-base` maps streams to `port_base + stream_index`
- the default run is unlimited and does not save frames
- add `--frames 100` if you want a bounded smoke run
- add `--output sandbox/people-tracking --save-every 10` if you want sampled
  overlay frames written under `stream_<index>/`
- if the app runs on a DevKit and `mediasrc-udp.sh` runs on your host, set
  `--udp-host` to the host IP, not `127.0.0.1`
- the example uses CPU-side OpenCV letterbox + normalize on A65 and feeds the
  detector through the model's tensor-input `QuantTess` contract
- `--detection-threshold`, `--nms-iou-threshold`, and `--top-k` are optional;
  if omitted, `SimaBoxDecode` keeps the model-pack defaults
- the example defaults to person class id `0`, and tracker behavior remains
  configurable from the CLI
- C++ parity is tracked in a follow-up ticket; the C++ entrypoint currently exists
  only as a placeholder scaffold

## Source Files
- Python entrypoint: `python/main.py`
- Python tracker: `python/tracker.py`
- Python tests: `python/tests/test_unit.py`, `python/tests/test_e2e.py`
- C++ scaffold: `cpp/main.cpp`
- C++ scaffold tests: `cpp/tests/unit_test.cpp`, `cpp/tests/e2e_test.cpp`
- Shared example data: `common/`
