# Single-Stream Thermal Face Detector

## Metadata
| Field | Value |
| --- | --- |
| Category | face-detection |
| Difficulty | Intermediate |
| Tags | face-detection, keypoints, yolov5s-face, thermal, rtsp, insight |
| Languages | C++, Python |
| Status | stable |
| Binary Name | single-stream-thermal-face-detector |
| Model | yolov5s_face_raw_split |

## Concept
Detects faces and five facial landmarks in a live thermal RTSP stream with YOLOv5s-Face, then sends the overlay to Insight.

The graph decodes the stream once and branches the frames to Insight's video and to the
model:

```text
RtspDecodedInput (H.264 to NV12) --> Branch --> "video": VideoSender (H.264 RTP/UDP to Insight)
                                            \--> "model": yolov5s-face (six raw heads)
                                                    --> Output "detections" --> host decode
                                                    --> MetadataSender (pose-estimation to Insight)
```

Both branch links keep only the newest frame, so a slow branch drops frames instead of holding
back the stream.

Each frame sends one `pose-estimation` message with the left eye, right eye, nose tip, and both mouth corners. Insight draws them as labeled dots. The application sends one metadata type because Insight shows only one type per channel at a time.

YOLOv5s-Face returns raw box and landmark outputs. The C++ and Python applications decode
those outputs with the same math, in `src/cpp/utils/yolov5_face.h` and
`src/python/utils/yolov5_face.py`, before sending the five points to Insight.

## Preview
![Thermal face detector reference preview](../../../portal/assets/examples/face-detection/single-stream-thermal-face-detector/image.png)

## Prerequisites

- `sima-cli` ([documentation](https://developer.sima.ai/software/tools/sima-cli/))
  on the Modalix DevKit, or in a Neat SDK container paired with it.
- A running [Neat Insight](https://developer.sima.ai/software/tools/insight/)
  instance reachable from the target, with an RTSP source containing faces.
- The model expects an 800x800 compiled canvas; its on-device preprocess accepts
  NV12, BGR, RGB, I420, or grayscale input up to 4096x4096.

## Install Apps

Install on the Modalix DevKit, or under `/workspace` in a Neat SDK container paired with it. Only the example itself must run on the DevKit. `sima-cli neat install apps` also offers to install the compatible Neat Library and Insight, and skips them when they are up to date. To keep the Neat Library and Insight you have without being asked, run the install as `NEAT_APPS_SKIP_DEPENDENCIES=1 sima-cli neat install apps`. See [Install Apps](https://github.com/sima-neat/apps/blob/main/README.md#install-apps).

Install the latest Neat Apps runtime and enter the installed bundle:

```bash
sima-cli neat install apps
cd prebuilt-apps
APP_DIR=examples/face-detection/single-stream-thermal-face-detector
```

Run the remaining commands from `prebuilt-apps/`.

## Prepare the Model

Primary model: `yolov5s_face_raw_split`

Model packages come from the Model Zoo release below, which can differ from the
installed platform version.

```bash
export MODELZOO_VERSION="3.0.0"
mkdir -p models
cd models
sima-cli download \
  "https://docs.sima.ai/pkg_downloads/SDK${MODELZOO_VERSION}/models/modalix/yolov5s_face_raw_split_mpk.tar.gz"
cd ..
```

The command stores the package under `models/`, where the shipped `model.path` points. A
relative `model.path` starts at the config file.

## Prepare Insight
[Neat Insight](https://developer.sima.ai/software/tools/insight/) can host an RTSP source, receive video from `VideoSender`, receive landmark metadata from `MetadataSender`, and show rendered overlays plus runtime metrics in the browser.

In the Palette SDK, install the sample video assets:

```bash
sima-cli install assets/multi-video-sources
```

This provides 720p and 480p videos that Insight can stream as RTSP sources. This
example expects faces in frame, so upload a thermal or visible-light face clip of
your own when the sample videos do not contain any.

To create a reproducible RTSP input:
1. Run `neat` in the Palette SDK and open the reported `Insight Web UI`.
2. In Insight, open `RTSP Source`.
3. Use a sample video or upload your own video.
4. Start the stream and copy the RTSP URL.
5. Put that RTSP URL into `input.source`.

Use the same `neat` output to set `output.insight.host`, `video_port_base`, and
`metadata_port_base` from the reported `videoUDP` and `metadataUDP` ranges.

**Ports when the app runs on a board.** Insight runs inside the SDK container and
builds its URLs from the container's own view, so the RTSP link copied from the UI
carries the container-internal port (`8554`). An app running on a DevKit is
outside that container and must use the SDK host IP plus the **published** host
port instead. Read the mapping from `exposedPorts` in `neat --json` and never
assume the defaults:

| Endpoint | Container port (shown in the UI) | Host port (use this from a board) |
| --- | --- | --- |
| RTSP | 8554 | `rtsp.tcp` → `hostPortStart` |
| Video UDP | 9000 | `videoUDP` → `hostPortStart` |
| Metadata UDP | 9100 | `metadataUDP` → `hostPortStart` |

## Configure

Open `${APP_DIR}/src/common/config.yaml`. Set `input.source` to the H.264 RTSP URL and
`output.insight.host` to the machine running Insight. If Insight uses nondefault ports, also set
`output.insight.video_port_base` and `output.insight.metadata_port_base` to the ports reported
by `neat --json`.

`input.width`, `input.height`, and `input.fps` stay at 0 to read the stream's frame size and
rate with a probe; set them only for a stream that does not report them. The shipped thresholds
are ready for a first run.

## Run
### C++
```bash
./${APP_DIR}/src/cpp/pre-built/single-stream-thermal-face-detector \
  --config ${APP_DIR}/src/common/config.yaml
```

### Python
```bash
source ~/pyneat/bin/activate
pip install -r ${APP_DIR}/src/python/requirements.txt
python3 ${APP_DIR}/src/python/main.py \
  --config ${APP_DIR}/src/common/config.yaml
```

Open the Insight video viewer for the channel (e.g. `/api/viewer-url?src=0`) to
watch the annotated stream.

## Expected Result

This application writes nothing to disk; its output is the Insight video channel
plus the landmark metadata channel. It prints the source on startup and a count when it
stops:

```text
source=rtsp://<host>:<port>/<stream> stream=1280x720@30 insight=<insight-host> video=9000 metadata=9100
processed=200 video_sender=<insight-host>:9000
```

The packaged config ships `inference.frames: 0`, which runs until Ctrl-C, SIGTERM, or a closed
ssh session stops it; the count prints then too. For a bounded check that ends by itself, set
a positive limit:

```yaml
inference:
  frames: 200
```

When the metadata socket is full, a message is dropped rather than delaying the stream, and the
count of dropped messages prints after `processed=`.

Either way, `processed` reports frames pushed through the pipeline, not faces
found, so a healthy count does not by itself confirm detections. Confirm those
in Insight, which draws the five landmarks as labeled dots on the stream.

The preview image is a reference visualization: it pairs the thermal stream
with its visible-light source and draws face, eye, and mouth boxes. The
application publishes only the five landmarks, not those boxes.

## Debugging Notes

- A missing or mistyped config key stops the application with the key's name, such as
  `input.source`. An unedited placeholder, such as `<insight-host-ip>`, fails when the
  application probes the stream or opens the sender.
- `failed to probe RTSP frame size` means the stream did not report its size from the target;
  verify the RTSP URL from the board itself, or set `input.width`, `input.height`, and
  `input.fps`.
- `method SETUP failed: 461 Unsupported Transport` was observed once at startup
  against an Insight-hosted source, after which the pipeline built and the run
  completed normally. Treat a single occurrence as noise unless the pipeline also
  fails to start.
- If the run reports no frames at all, verify the RTSP URL from the board itself
  rather than from the machine running Insight. The URL Insight displays is not
  always reachable from the target, and the ports table above explains why.
- If the viewer shows video but no overlays, confirm `output.insight.host` and the
  metadata port are reachable and match the viewer's channel. Use Insight's
  `/api/ingest/stats` to check `metadata.messages_received`.
- If the viewer shows no video, verify the video UDP port and that RTP is arriving
  (`/api/ingest/stats` -> `rtp.packets_received`, `media.seen_sps`).
- If keypoints look mis-anchored, the model may have been retrained with different
  anchors: update `STRIDES` and `ANCHORS` in `src/python/utils/yolov5_face.py` and
  `kStrides` and `kAnchors` in `src/cpp/utils/yolov5_face.h` to match.
- Use `inference.min_score` and `inference.nms_iou` to tune detection sensitivity.

## Source Files
- C++ source: `src/cpp/main.cpp`, `src/cpp/utils/yolov5_face.h`
- C++ tests: `tests/cpp/test_unit.cpp`, `tests/cpp/test_e2e.cpp`
- Python source: `src/python/main.py`, `src/python/utils/yolov5_face.py`
- Python tests: `tests/python/test_unit.py`, `tests/python/test_e2e.py`
- Shared assets: `src/common/face_label.txt`

The packaged C++ source is an implementation reference. Run the executable
under `src/cpp/pre-built/`; the installed bundle does not include CMake files.

## Development From Source

To modify, compile, or test this example, use the
[Apps contributor workflow](https://github.com/sima-neat/apps/blob/main/CONTRIBUTING.md).
