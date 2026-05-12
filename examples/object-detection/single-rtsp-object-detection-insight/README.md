# Single RTSP Object Detection Insight

## Metadata
| Field | Value |
| --- | --- |
| Category | object-detection |
| Difficulty | Intermediate |
| Tags | object-detection, rtsp, insight |
| Languages | C++, Python |
| Status | experimental |
| Binary Name | single-rtsp-object-detection-insight |
| Model | yolo_v8s |

## Concept
`single-rtsp-object-detection-insight` is a focused reference example for a common deployment pattern:

- ingest one RTSP camera stream
- decode the stream into NV12 frames
- run YOLOv8 object detection
- send H.264 video plus detection metadata to Insight

The example is intentionally narrow in scope. It is not a generic output-mode demo and it does not try to support multiple unrelated workflows in one binary. The code is structured to show the intended Insight path clearly.

## Preview
Snippet from a pipeline run:

![Single RTSP object detection Insight preview](../../../assets/portal/object-detection/single-rtsp-object-detection-insight/image.png)

## What Is Insight?
Insight is SiMa.ai's lightweight, cross-platform development and visualization tool for vision pipelines on DevKits:

- a media source manager that can host test media and stream it as RTSP
- a zero-install web viewer for real-time video and metadata visualization

For this sample, the most important part is the viewer/output contract: the application sends video on the Insight video channel and sends detection metadata as JSON on the Insight side channel. That allows the browser UI to display the live stream together with object detections without relying on external tools such as `ffplay`, VLC, or ad hoc debug viewers.

For more information regarding Insight, please refer to this [page](https://docs.sima.ai/pages/insight/main.html#).

## Architecture
The sample is split into three independent runtime stages:

1. `RTSP ingest and decode`
   The application first probes the RTSP source to learn the decoded frame size, then builds a decode session that outputs NV12 frames. This avoids hardcoding `640x480` and makes the example more robust when the source changes resolution.

2. `YOLO inference`
   Decoded NV12 frames are pushed into a dedicated YOLO pipeline:
   `Input -> Preprocess -> Infer -> SimaBoxDecode -> Output`

   The model stage is isolated from transport logic so detection behavior can be debugged separately from RTSP or Insight issues.

3. `Insight output`
   The original decoded frame is copied into a second runtime path that re-encodes to H.264, packetizes to RTP, and sends video over UDP to Insight. Detection results from the YOLO path are converted into Insight metadata and sent on the metadata side channel.

## Neat API Usage

- RTSP ingest: `RtspDecodedInputOptions` -> `Session.add(rtsp_decoded_input)` -> `Session.build(...)`
- YOLO path:
  C++ graph uses `Input -> Preprocess -> Infer -> SimaBoxDecode -> Output`
  Python path uses `Model.build(...)`/`Model.run(...)` with packed BBOX parsing first and manual decode fallback.
- Insight output:
  C++ builds a dedicated Insight video runtime and MetadataSender.
  Python builds a UDP video writer plus MetadataSender.

## Lifecycle
The example uses a producer/consumer design:

- the producer thread pulls decoded frames from the RTSP session and places them into a bounded queue
- the consumer thread pulls frames from that queue, submits them to YOLO, converts detection results to Insight objects, and publishes both video and metadata

This separation keeps the RTSP session from being tightly coupled to the inference latency of each frame and makes timing/debug output easier to interpret.

## Prerequisites
- Installed Neat framework and Insight on the DevKit
- RTSP camera source or use Insight to start RTSP source
- SiMa.ai developer portal account so the sample can download the model from modelzoo
- Model artifacts are user-managed and should be downloaded into `assets/models/`.
- Download command: `mkdir -p assets/models && cd assets/models && sima-cli modelzoo -v 2.0.0 get yolo_v8s && cd ../..`


## Important Behavior
- The sample always publishes to Insight.
- Video is sent to the Insight video UDP port 9000.
- Detection metadata is sent to the Insight metadata UDP port 9100.
- `--model` is required and must point to a valid YOLO compiled model package file.
- If `--frames` is omitted, the sample runs continuously.

## Command-Line Options
- `--rtsp <url>`
  Required. RTSP source URL.
- `--model <path>`
  Required. Path to the YOLO model pack.
- `--frames <n>`
  Optional. Number of frames to process before exiting.
- `--debug`
  Optional. Enables per-stage timing prints and additional runtime diagnostics.
- `--insight-host <host>`
  Optional. Destination host for Insight video and metadata. Default: `127.0.0.1`.
- `--insight-video-port <port>`
  Optional. UDP port for Insight video. Default: `9000`.
- `--insight-metadata-port <port>`
  Optional. UDP port for Insight metadata. Default: `9100`.

## Build
This example can be built in either of these environments:

- from an `eLxr SDK` environment
- directly on a `DevKit`

Within either environment, the C++ implementation can be built in two ways. The Python implementation does not require a compile step.

### Build From The Apps Repo
Build all C++ examples from the `apps` repo root:

```bash
cd <apps-repo-root>
./build.sh
```

The resulting binary is:

```bash
./build/examples/object-detection/single-rtsp-object-detection-insight/single-rtsp-object-detection-insight
```

### Build This Example Directly With CMake
Configure and build only this example from its own directory:

```bash
cd <apps-repo-root>/examples/object-detection/single-rtsp-object-detection-insight
cmake -S cpp -B build
cmake --build build -j
```

The resulting binary is:

```bash
./build/single-rtsp-object-detection-insight
```

Direct CMake builds use the shared example module support in the `apps` repo and link against the available Neat/core installation or local core build.

In practice:

- on `eLxr SDK`, this is typically done after sourcing the SDK environment and then building from the repo or the example folder
- on `DevKit`, this can be done directly on the target device as long as the required Neat dependencies are installed

## RTSP Source

If you want a quick RTSP source for testing, [`tool-mediasources`](https://github.com/SiMa-ai/tool-mediasources) on the host is one option:

```bash
sima-cli install gh:sima-ai/tool-mediasources
./mediasrc.sh <video-dir>
open preview.html
```

If you use host-streamed sources from a board/devkit, use the host IP in the RTSP URL instead of `127.0.0.1`. Any other RTSP source also works.

## Run
### Binary Built From The Apps Repo
```bash
./build/examples/object-detection/single-rtsp-object-detection-insight/single-rtsp-object-detection-insight \
  --rtsp <rtsp_url> \
  --model assets/models/yolo_v8s_mpk.tar.gz
```

### Binary Built Directly In The Example Folder
```bash
./build/single-rtsp-object-detection-insight \
  --rtsp <rtsp_url> \
  --model assets/models/yolo_v8s_mpk.tar.gz
```

Example with explicit Insight host:

```bash
./build/examples/object-detection/single-rtsp-object-detection-insight/single-rtsp-object-detection-insight \
  --rtsp <rtsp-url> \
  --model assets/models/yolo_v8s_mpk.tar.gz
```

### Python Implementation
Run the Python sample directly from the example folder:

```bash
cd <apps-repo-root>/examples/object-detection/single-rtsp-object-detection-insight
pip install -r python/requirements.txt
python3 python/main.py --model <path-to-yolo_v8s-mpk.tar.gz> --rtsp <rtsp_url>
```

Example workflow:

Download the `yolo_v8s` model using `sima-cli`:

```bash
sima-cli modelzoo -v 2.0.0 get yolo_v8s
```

Then start the Python app:

```bash
source ~/pyneat/bin/activate
pip install -r python/requirements.txt
python3 python/main.py --rtsp <rtsp-url> --model yolo_v8s_mpk.tar.gz
```

Python-specific notes:

- if `--model` is omitted, the Python version tries to locate `yolo_v8s` locally and then falls back to `sima-cli modelzoo -v 2.0.0 get yolo_v8s`
- it sends Insight metadata directly over UDP and streams video to the Insight UDP video port
- it expects OpenCV to be built with GStreamer support for the UDP H.264 video path

## Debugging Notes
- If the sample times out waiting for the first RTSP frame, the problem is usually upstream stream delivery or device connectivity, not YOLO itself.
- If the RTSP source resolution changes, the startup probe is expected to adapt the decode path automatically.
- If detections are missing but video is flowing, focus on the YOLO session and bbox extraction/parse path.
- If video and detections are both missing in Insight, verify the host and UDP ports first.

## Source Files
- C++ source: `cpp/main.cpp`
- C++ tests: `cpp/tests/unit_test.cpp`, `cpp/tests/e2e_test.cpp`
- Python source: `python/main.py`
- Python tests: `python/tests/test_unit.py`, `python/tests/test_e2e.py`
- Insight documentation: <https://docs.sima.ai/pages/insight/main.html>
