# YOLOv8 One RTSP OptiView

## Metadata
| Field | Value |
| --- | --- |
| Category | object-detection |
| Difficulty | Intermediate |
| Tags | object-detection, rtsp, optiview |
| Status | experimental |
| Binary Name | single-rtsp-object-detection-optiview |
| Model | yolo_v8s |

## Concept
`single-rtsp-object-detection-optiview` is a focused reference example for a common deployment pattern:

- ingest one RTSP camera stream
- decode the stream into NV12 frames
- run YOLOv8 object detection
- send H.264 video plus detection JSON to OptiView

The example is intentionally narrow in scope. It is not a generic output-mode demo and it does not try to support multiple unrelated workflows in one binary. The code is structured to show the intended OptiView path clearly.

## What Is OptiView?
OptiView is SiMa.ai's lightweight, cross-platform development and visualization tool for vision pipelines on DevKits:

- a media source manager that can host test media and stream it as RTSP
- a zero-install web viewer for real-time video and metadata visualization

For this sample, the most important part is the viewer/output contract: the application sends video on the OptiView video channel and sends detection metadata as JSON on the OptiView side channel. That allows the browser UI to display the live stream together with object detections without relying on external tools such as `ffplay`, VLC, or ad hoc debug viewers.

For more information regarding OptiView, please refer to this [page](https://docs.sima.ai/pages/optiview/main.html#).

## Design
The sample is split into three independent runtime stages:

1. `RTSP ingest and decode`
   The application first probes the RTSP source to learn the decoded frame size, then builds a decode session that outputs NV12 frames. This avoids hardcoding `640x480` and makes the example more robust when the source changes resolution.

2. `YOLO inference`
   Decoded NV12 frames are pushed into a dedicated YOLO pipeline:
   `Input -> Preprocess -> Infer -> SimaBoxDecode -> Output`

   The model stage is isolated from transport logic so detection behavior can be debugged separately from RTSP or OptiView issues.

3. `OptiView output`
   The original decoded frame is copied into a second runtime path that re-encodes to H.264, packetizes to RTP, and sends video over UDP to OptiView. Detection results from the YOLO path are converted into OptiView JSON and sent on the JSON side channel.

## Runtime Model
The example uses a producer/consumer design:

- the producer thread pulls decoded frames from the RTSP session and places them into a bounded queue
- the consumer thread pulls frames from that queue, submits them to YOLO, converts detection results to OptiView objects, and publishes both video and JSON

This separation keeps the RTSP session from being tightly coupled to the inference latency of each frame and makes timing/debug output easier to interpret.

## Prerequisites
- Installed NEAT framework and OptiView on the DevKit
- RTSP camera source or use OptiView to start RTSP source
- SiMa.ai developer portal account so the sample can download the model from modelzoo
- Model artifacts are user-managed and should be downloaded into `assets/models/`.
- Download command: `mkdir -p assets/models && cd assets/models && sima-cli modelzoo get yolo_v8s && cd ../..`


## Important Behavior
- The sample always publishes to OptiView.
- Video is sent to the OptiView video UDP port 9000.
- Detection metadata is sent to the OptiView JSON UDP port 9100.
- `--mpk` is required and must point to a valid YOLO MPK tarball.
- If `--frames` is omitted, the sample runs continuously.

## Command-Line Options
- `--rtsp <url>`
  Required. RTSP source URL.
- `--mpk <path>`
  Required. Path to the YOLO model pack.
- `--frames <n>`
  Optional. Number of frames to process before exiting.
- `--debug`
  Optional. Enables per-stage timing prints and additional runtime diagnostics.
- `--optiview-host <host>`
  Optional. Destination host for OptiView video and JSON. Default: `127.0.0.1`.
- `--optiview-video-port <port>`
  Optional. UDP port for OptiView video. Default: `9000`.
- `--optiview-json-port <port>`
  Optional. UDP port for OptiView JSON. Default: `9100`.

## Build
This example can be built in either of these environments:

- from an `eLxr SDK` environment
- directly on a `DevKit`

Within either environment, the example can be built in two ways.

### Build From The Apps Repo
Build all C++ examples from the `apps` repo root:

```bash
cd <apps-repo-root>
./build.sh
```

The resulting binary is:

```bash
./build/examples/object-detection/single-rtsp-object-detection-optiview/single-rtsp-object-detection-optiview
```

### Build This Example Directly With CMake
Configure and build only this example from its own directory:

```bash
cd <apps-repo-root>/examples/object-detection/single-rtsp-object-detection-optiview
cmake -S . -B build
cmake --build build -j
```

The resulting binary is:

```bash
./build/single-rtsp-object-detection-optiview
```

Direct CMake builds use the shared example module support in the `apps` repo and link against the available NEAT/core installation or local core build.

In practice:

- on `eLxr SDK`, this is typically done after sourcing the SDK environment and then building from the repo or the example folder
- on `DevKit`, this can be done directly on the target device as long as the required NEAT dependencies are installed

## Run
### Binary Built From The Apps Repo
```bash
./build/examples/object-detection/single-rtsp-object-detection-optiview/single-rtsp-object-detection-optiview \
  --rtsp <rtsp_url> \
  --mpk assets/models/yolo_v8s_mpk.tar.gz
```

### Binary Built Directly In The Example Folder
```bash
./build/single-rtsp-object-detection-optiview \
  --rtsp <rtsp_url> \
  --mpk assets/models/yolo_v8s_mpk.tar.gz
```

Example with explicit OptiView host:

```bash
./build/examples/object-detection/single-rtsp-object-detection-optiview/single-rtsp-object-detection-optiview \
  --rtsp rtsp://192.168.1.10:8554/src1 \
  --mpk assets/models/yolo_v8s_mpk.tar.gz
```

## Debugging Notes
- If the sample times out waiting for the first RTSP frame, the problem is usually upstream stream delivery or device connectivity, not YOLO itself.
- If the RTSP source resolution changes, the startup probe is expected to adapt the decode path automatically.
- If detections are missing but video is flowing, focus on the YOLO session and bbox extraction/parse path.
- If video and detections are both missing in OptiView, verify the host and UDP ports first.

## Reference
- OptiView documentation: <https://docs.sima.ai/pages/optiview/main.html>
