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
   The original decoded frame is pushed into `VideoSender`. The nodegroup owns the raw-frame video transport path, including conversion, H.264 encoding, RTP packetization, and UDP output. Detection results from the YOLO path are converted into Insight metadata and sent on the metadata side channel.

## Neat API Usage

- RTSP ingest: `RtspDecodedInputOptions` -> `Session.add(rtsp_decoded_input)` -> `Session.build(...)`
- YOLO path:
  C++ graph uses `Input -> Preprocess -> Infer -> SimaBoxDecode -> Output`
  Python path uses `Model.build(...)`/`Model.run(...)` with packed BBOX parsing first and manual decode fallback.
- Insight output:
  C++ and Python build a dedicated `VideoSender` runtime plus `MetadataSender`.

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
- Video is sent to the configured Insight video UDP port.
- Detection metadata is sent to the configured Insight metadata UDP port.
- The app adds only the `VideoSender` nodegroup for video output. It does not manually add lower-level color conversion, encoder, parser, packetizer, or UDP nodes.
- This example feeds raw decoded frames to `VideoSender` with the raw-frame option. If an upstream pipeline already produces H.264, `VideoSender` also supports the encoded-input option, where it parses, packetizes, and sends without re-encoding.
- `model.path` must point to a valid YOLO compiled model package file.
- `source.rtsp_url` must be set before running.
- If `runtime.frames` is empty or zero, the sample runs continuously.

## Command-Line Options
- `--config <path>`
  Optional. YAML config path. Defaults to `common/config.yaml`.

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
  --config examples/object-detection/single-rtsp-object-detection-insight/common/config.yaml
```

### Binary Built Directly In The Example Folder
```bash
./build/single-rtsp-object-detection-insight \
  --config common/config.yaml
```

Set the RTSP source and Insight destination in the config:

```yaml
source:
  rtsp_url: rtsp://<host>:8554/<stream>
model:
  path: assets/models/yolo_v8s_mpk.tar.gz
insight:
  host: <insight-host>
```

### Python Implementation
Run the Python sample directly from the example folder:

```bash
cd <apps-repo-root>/examples/object-detection/single-rtsp-object-detection-insight
pip install -r python/requirements.txt
python3 python/main.py --config common/config.yaml
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
python3 python/main.py --config common/config.yaml
```

Python-specific notes:

- if `model.path` is empty, the Python version tries to locate `yolo_v8s` locally and then falls back to `sima-cli modelzoo -v 2.0.0 get yolo_v8s`
- it sends Insight metadata directly over UDP and streams video through `VideoSender`

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
