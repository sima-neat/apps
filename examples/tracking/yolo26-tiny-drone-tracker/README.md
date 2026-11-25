# YOLO26 Tiny-Drone Tracker

## Metadata

| Field | Value |
| --- | --- |
| Category | tracking |
| Difficulty | Advanced |
| Tags | object-detection, yolo26, rtsp, insight, tiny-drone, tracking |
| Languages | C++, Python |
| Status | stable |
| Binary Name | yolo26-tiny-drone-tracker |
| Model | yolo26n-p2-tiny-drone-int8-qat |

## Concept

Tracks tiny drones in up to four RTSP streams with one shared YOLO26 model, and sends each stream's video and drone tracks to Insight.

```text
How does a frame move through yolo26-tiny-drone-tracker?

╭┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╮
┆ RUN                                                                       ┆
┆  ┌──────────────────────────┐  encoded frames     ┌──────────────────┐    ┆
┆  │ RtspEncodedInput × N     │────────────────────►│ VideoSender × N  │────┼─────┐
┆  └──────────────────────────┘  latest/stream      └──────────────────┘    ┆     │
┆             │ encoded frames                                              ┆     │ video
┆             ▼                                                             ┆     │
┆  ┌──────────────────────────┐                                             ┆     │
┆  │ SimaDecode × N           │                                             ┆     │
┆  └──────────────────────────┘                                             ┆     │
┆             │ NV12, latest/stream                                         ┆     │
┆             ▼                                                             ┆     │
┆  ┌──────────────────────────┐                                             ┆     │
┆  │ YOLO26 Model (shared)    │                                             ┆     │
┆  └──────────────────────────┘                                             ┆     │
┆             │ BBOX tensor                                                 ┆     │
┆             ▼                                                             ┆     │
┆  ┌──────────────────────────┐                                             ┆     │
┆  │ Output "detections"      │                                             ┆     │
┆  └──────────────────────────┘                                             ┆     │
╰┄┄┄┄┄┄┄┄┄┄┄┄┄│┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╯     │
              ▼ pull, decode_bbox_tensor, route by stream_id                      │
   ┌──────────────────────────┐                                                   │
   │ ObjectTracker × N        │                                                   │
   └──────────────────────────┘                                                   │
              │ boxes with track IDs                                              │
              ▼                                                                   ▼
   ┌──────────────────────────┐   tracks as JSON                            ┌───────────┐
   │ MetadataSender × N       │────────────────────────────────────────────►│ Insight   │
   └──────────────────────────┘                                             └───────────┘
```

Each stream's encoded video goes to Insight as received, on its own channel, while its decoder feeds the shared model.
The link to the model keeps the latest frames of each stream.
When the model falls behind, a stream drops frames instead of slowing the others.
When a video sender falls behind, its stream drops encoded frames, and the video in Insight may break up until the next keyframe.
The application pulls each frame's boxes, and the stream's `ObjectTracker` gives every drone an ID that lasts across frames.
Each tracking message carries the frame's timestamp, so Insight draws the tracks on the frame they came from.
Insight colors each track by its ID and draws its trail.

The tracker matches boxes in two passes. First, a box at or above `app.tracking.high_score_threshold` can continue any track.
Then a box with a lower score can continue only a confirmed track.
An unmatched box at or above both thresholds starts a new track.
A track is confirmed at its `app.tracking.min_confirmed_hits`-th detection, counting the one that started it.
Only confirmed tracks go to Insight.
A tiny box rarely overlaps its previous position, so the tracker matches by the distance to the position its velocity predicts.
It counts frames by their timestamps, so frames the model's link dropped still age a track.
Each stream's track IDs start at 1.

## Preview

![YOLO26 tiny-drone tracker preview](../../../portal/assets/examples/tracking/yolo26-tiny-drone-tracker/image.jpg)

## Prerequisites

- `sima-cli` ([documentation](https://developer.sima.ai/software/tools/sima-cli/)) on the Modalix DevKit, or in a Neat SDK container paired with it.
- One to four H.264 or H.265 RTSP streams of one frame size that show small drones in the sky.
  The Insight video catalog has no drone video, so you provide one.
- An [Insight](https://developer.sima.ai/software/tools/insight/) URL reachable from the target.
- For H.265, a computer whose browser decodes HEVC in hardware to view Insight. Chromium has no software decoder for WebRTC H.265.

## Install Apps

Install on the Modalix DevKit, or under `/workspace` in a Neat SDK container paired with it. Only the example itself must run on the DevKit. `sima-cli neat install apps` also offers to install the compatible Neat Library and Insight, and skips them when they are up to date. To keep the Neat Library and Insight you have without being asked, run the install as `NEAT_APPS_SKIP_DEPENDENCIES=1 sima-cli neat install apps`. See [Install Apps](https://github.com/sima-neat/apps/blob/main/README.md#install-apps).

Install the latest Neat Apps runtime and enter the installed bundle:

```bash
sima-cli neat install apps
cd prebuilt-apps
APP_DIR=examples/tracking/yolo26-tiny-drone-tracker
```

Run the remaining commands from `prebuilt-apps/`.

## Prepare the Model

The drone detector is not in the model browser yet. Download it from the Model Zoo release below:

```bash
export MODELZOO_VERSION="3.0.0"
mkdir -p models
cd models
sima-cli download \
  "https://docs.sima.ai/pkg_downloads/SDK${MODELZOO_VERSION}/models/modalix/yolo26n_p2_tiny_drone_int8_qat_b1_mpk.tar.gz"
cd ..
```

The command stores `models/yolo26n_p2_tiny_drone_int8_qat_b1_mpk.tar.gz`.
The packaged `model.path` already points to it: a relative `model.path` starts at the config file.
`../../../../../models/` from `${APP_DIR}/src/common/` is `prebuilt-apps/models/`.

## Prepare Insight

In the Insight Web UI, open Media Sources and upload a drone video.
Insight's YouTube import also works.
Under Streaming Sources, assign the video to a source slot for each stream, start them, and copy their URLs into `input.source`.
Several slots may play the same video.
Use a host and port that the target can reach, not `localhost`.

## Configure

Open `${APP_DIR}/src/common/config.yaml` and set:

- `input.source`: one to four RTSP URLs, one per line. Stream `i` goes to Insight channel `i`.
- `input.codec`: `h264` or `h265`, for every stream.
- `output.insight.host`, and the Insight ports of stream 0, `output.insight.video_port_base` and `output.insight.metadata_port_base`. Stream `i` uses each port plus `i`.

Leave `input.width`, `input.height`, and `input.fps` at 0 to read them from each stream.
Keep `inference.frames` at `0` to run until you stop the application.
The `app.tracking` keys tune the tracker, as Concept describes.
`app.tracking.max_missing_frames` counts frames at `input.fps`, or at the probed rate when `input.fps` is 0.
A wrong rate changes how fast tracks age.

## Run

### C++

```bash
./${APP_DIR}/src/cpp/pre-built/yolo26-tiny-drone-tracker \
  --config ${APP_DIR}/src/common/config.yaml
```

### Python

```bash
source ~/pyneat/bin/activate
pip install -r ${APP_DIR}/src/python/requirements.txt
python3 ${APP_DIR}/src/python/main.py \
  --config ${APP_DIR}/src/common/config.yaml
```

## Expected Result

The application prints each source on startup.
When it stops, it prints the frames each stream processed and the tracks it started:

```text
[stream 0] rtsp=rtsp://<rtsp-source-ip>:<port>/<stream-name> stream=640x512@30 insight=<insight-host-ip> video=9000 metadata=9100
[stream 1] rtsp=rtsp://<rtsp-source-ip>:<port>/<stream-name> stream=640x512@30 insight=<insight-host-ip> video=9001 metadata=9101
[stream 0] processed=<frames> tracks_started=<tracks>
[stream 1] processed=<frames> tracks_started=<tracks>
```

The packaged config runs until Ctrl-C, and the `processed=` lines print when you stop it.
For a run that ends by itself, set `inference.frames` to a positive number, such as `200`.
A full network buffer drops tracking messages.
A stream's last line then counts them, such as `[stream 0] 3 metadata messages dropped`.

Insight shows each stream on its own channel, with a box, an ID, and a trail on every confirmed drone.

## Troubleshooting

- `[ERR] failed to probe RTSP frame size: <url>; set input.width and input.height` or `[ERR] failed to probe RTSP frame rate: <url>; set input.fps`: the board cannot read that value.
  The stream is unreachable from the board, or it does not report the value.
  Verify the URL from the board. For a reachable stream, set the keys the message names.
- `[ERR] every input.source stream needs the same frame size, because the streams share one model: ...`: the streams differ in size.
  Use streams of one size, or run one application per size.
- `[ERR] detections output closed unexpectedly: <reason>`: the run stopped, for example because a stream ended or dropped.
  The reason names the cause. Verify the streams from the board and run again.
- `[ERR] metadata send failed: <reason>`: a metadata send failed for a reason other than a full socket buffer.
  Verify `output.insight.host` and `output.insight.metadata_port_base`.
- `[ERR] input.codec: unknown value '<value>' (expected h264, h265)`: set `input.codec` to the streams' codec.
- `[ERR] <config> is missing <key>` or `[ERR] <key> must be <type>`: the config lacks a key or has a value of the wrong type. Compare it with the packaged config.
- Video without tracks: the model finds only small drones in the sky. Check that the video shows them, and check `inference.min_score`.
- No video or tracks on a channel: verify `output.insight.host` and the ports of that stream.

## Source Files

- C++ reference source: `src/cpp/main.cpp`
- C++ tracker: `src/cpp/utils/tracker.h` and `src/cpp/utils/tracker.cpp`
- Python source: `src/python/main.py`
- Python tracker: `src/python/utils/tracker.py`
- Shared config: `src/common/config.yaml`
- Helpers the examples share: `support/python/sima_examples/` at the root of the installed bundle.
  `main.py` finds that folder above itself.
  Their C++ reference source is in `support/cpp/sima_examples/`.

The packaged C++ source is an implementation reference. Run the executable under `src/cpp/pre-built/`. The installed bundle does not include CMake files.

## Development From Source

To modify, compile, or test this example, use the [Apps contributor workflow](https://github.com/sima-neat/apps/blob/main/CONTRIBUTING.md).
