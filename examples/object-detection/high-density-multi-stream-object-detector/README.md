# High-Density Multi-Stream Object Detector

## Metadata

| Field | Value |
| --- | --- |
| Category | object-detection |
| Difficulty | Advanced |
| Tags | object-detection, rtsp, multistream, high-density, insight, yolo26 |
| Languages | C++, Python |
| Status | stable |
| Binary Name | high-density-multi-stream-object-detector |
| Model | yolo26n-det-int8-b1 |

## Concept

This example runs one shared YOLO26 detector across fixed 16-, 24-, and
48-stream RTSP profiles and publishes encoded video with synchronized detection
metadata to Insight.

It complements `multi-stream-object-detector`. That example demonstrates the
general multi-stream API; this example demonstrates the tuned high-density
pipeline.

Each RTSP source is depacketized once, then Core fuses two branches into the
same source pipeline:

```text
RTSP H.264
  ├─ latest encoded edge ─> VideoSender ─> Insight video channel N
  └─ decode ─> shared YOLO26 detector ─> timestamped metadata ─> channel N
```

The application expresses this fan-out with ordinary `Graph::connect()` and
starts it with ordinary `Graph::build()`. Core fuses the eligible topology
internally. `VideoSender` consumes the read-only H.264 access unit before the
decoder, so the application does not open a second RTSP session, copy a decoded
EV buffer, run an encoder, or shuttle encoded frames through appsink/appsrc.
The UDP sender uses `async=false` so its sink cannot hold the shared live
pipeline in `PAUSED` while waiting for preroll from every stream.
The encoded edge uses `RealtimeLatestByStream`; a congested Insight channel can
drop stale encoded work without blocking that stream's decoder branch.

Detection metadata includes the source `rtp_timestamp` and is sent with
nonblocking UDP. A compatible Insight receiver holds complete encoded RTP
frames for 400 ms and matches metadata to that source timestamp before WebRTC
forwarding. Keep one active viewer while validating metadata, as described
below.

The three checked-in profiles use the same application and model:

| Config | Streams | Source resolution and FPS | Expected FPS per channel |
| --- | ---: | --- | ---: |
| `config.yaml` | 16 | 1280×720 at 25 FPS | 25 FPS |
| `config-24x720p20fps.yaml` | 24 | 1280×720 at 20 FPS | 20 FPS |
| `config-48x720p10fps.yaml` | 48 | 1280×720 at 10 FPS | 10 FPS |

The expected rate applies to both Insight video and detection metadata after
startup and model warmup.

## Preview

The 48-stream profile running in Insight:

![High-density multi-stream object detector in Insight](../../../assets/portal/object-detection/high-density-multi-stream-object-detector/image.png)

## Prerequisites

- Neat Apps, Core, and Internals from the same manifest.
- A Modalix DevKit with the decoder service running.
- Insight reachable from the DevKit.
- 16, 24, or 48 H.264 RTSP sources matching the selected profile.
- Constant source frame rate, 1280×720 resolution, no H.264 B-frames, and a
  short, regular IDR interval. The validated sources use one IDR per second.
- The `yolo26n-det-int8-b1.tar.gz` model pack.

The application starts all source graphs together. Start every RTSP publisher
before starting the application.

## Get The Apps Repo

Install the Neat Library by following the official Neat installation guide.
Then clone and build Apps:

```bash
git clone https://github.com/sima-neat/apps.git
cd apps
./build.sh --clean
```

Compile C++ examples in the Neat Development Environment. Run the application
on Modalix.

## Download The Model

From the Apps repository root, replace `<platform-version>` with the version in
`deps/manifest.json`:

```bash
sima-cli download https://docs.sima.ai/pkg_downloads/SDK<platform-version>/models/modalix/yolo26-detection/yolo26n-det-int8-b1.tar.gz
```

The checked-in configs resolve the model relative to their own directory, so no
absolute model path is required.

## Prepare The RTSP Sources

For quick RTSP sources, install `tool-mediasources`:

```bash
sima-cli install gh:sima-ai/tool-mediasources
./mediasrc.sh <video-dir>
```

Prepare one reachable URL per configured stream. Verify the files or publishers
before starting the application:

```bash
ffprobe -v error \
  -select_streams v:0 \
  -show_entries stream=codec_name,width,height,avg_frame_rate,has_b_frames \
  -of default=noprint_wrappers=1 \
  <rtsp-url>
```

The result must report H.264, `1280x720`, the selected profile FPS, and no
B-frames. Using a different source FPS changes the output rate and is not the
documented profile. The encoded Insight edge retains the latest complete access
unit under congestion, so a one-second-or-shorter IDR interval bounds receiver
recovery if an older access unit is replaced.

## Configure

Choose one config under `src/common/` and edit:

- every entry under `streams`
- `output.insight.host`

`inference.max_inflight_per_stream` and `inference.max_inflight_total`
bound raw decoder-backed frames admitted to the shared detector. The 16- and
48-stream profiles use a total limit of eight; the 24-stream profile uses 24
so one aggregate frame interval can be admitted without an unbounded queue.
The realtime mux retains only the latest pending frame for each stream.

Do not add the removed `inference.fan_in_policy` setting. Ordinary `connect()`
and `build()` select the eligible realtime fan-in lowering automatically.
Video/metadata synchronization is performed by Insight from each payload's
source RTP timestamp; there is no application-side video-delay setting.

Do not change `input.width`, `input.height`, or `input.fps` unless the RTSP
sources also change. `input.skip_rtsp_probe` is enabled, so those values are the
source contract.

Insight channel and port mapping is deterministic:

```text
channel index: 0 .. stream_count - 1
video port:    9000 + channel index
metadata port: 9100 + channel index
```

## Run

Set the example and selected profile from the Apps repository root:

```bash
APP_DIR=examples/object-detection/high-density-multi-stream-object-detector
APP=./build/examples/object-detection/high-density-multi-stream-object-detector/high-density-multi-stream-object-detector
CONFIG="$APP_DIR/src/common/config.yaml"
```

Use one of the named configs for the larger profiles:

```bash
CONFIG="$APP_DIR/src/common/config-24x720p20fps.yaml"
CONFIG="$APP_DIR/src/common/config-48x720p10fps.yaml"
```

Validate the selected config without starting RTSP or Insight:

```bash
"$APP" --config "$CONFIG" --validate-config-only
```

Run the C++ application:

```bash
"$APP" --config "$CONFIG"
```

Run the Python implementation with the same config:

```bash
python3 -m pip install -r "$APP_DIR/src/python/requirements.txt"
python3 "$APP_DIR/src/python/main.py" --config "$CONFIG"
```

Stop the application with `Ctrl-C`.

Use one active Insight viewer while validating metadata. Insight currently has
a single-viewer metadata rendering limitation; multiple simultaneous viewers
can make box delivery appear intermittent even when the application is
advancing normally.

## Expected Result

- Every configured Insight channel receives live video.
- Detection boxes appear on the matching channel.
- Video and metadata advance at 25 FPS for 16 streams, 20 FPS for 24 streams,
  or 10 FPS for 48 streams.
- Final per-stream counters show every stream advancing; the application fails
  with the missing channel IDs if a stream never starts or later stops.
- No detection timeout or stalled channel is reported. Metadata sender counters
  expose any nonblocking UDP drops without stalling inference.

If channels do not start, confirm that every publisher was already reachable
and that the configured source caps match the selected profile. Restart the
application after restarting the publishers.

## Tests

Run the configured unit coverage through the Apps test entrypoint:

```bash
./tests/test.sh --unit
```

The 16-, 24-, and 48-stream runtime profiles require Modalix, live RTSP
publishers, and Insight. Host unit tests do not prove their runtime FPS.

## Source Files

- C++ implementation: `src/cpp/main.cpp`
- Python implementation: `src/python/main.py`
- Default 16-stream profile: `src/common/config.yaml`
- 24-stream profile: `src/common/config-24x720p20fps.yaml`
- 48-stream profile: `src/common/config-48x720p10fps.yaml`
- COCO labels: `src/common/coco_label.txt`
- Test scope: `tests/test-scope.yaml`
