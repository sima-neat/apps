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

This example runs one shared YOLO26 detector across fixed 16-, 24-, and 48-stream RTSP profiles and publishes encoded video with synchronized detection metadata to Insight.

It complements `multi-stream-object-detector`. That example demonstrates the general multi-stream API; this example demonstrates the tuned high-density pipeline.

Each RTSP source is depacketized once, then Core fuses two branches into the same source pipeline:

```text
RTSP H.264/H.265
  ├─ latest encoded edge ─> VideoSender ─> Insight video channel N
  └─ decode ─> shared YOLO26 detector ─> timestamped metadata ─> channel N
```

The application expresses this fan-out with ordinary `Graph::connect()` and starts it with ordinary `Graph::build()`. Core fuses the eligible topology internally. `VideoSender` consumes the read-only encoded access unit before the decoder, so the application does not open a second RTSP session, copy a decoded EV buffer, run an encoder, or shuttle encoded frames through appsink/appsrc. The UDP sender uses `async=false` so its sink cannot hold the shared live pipeline in `PAUSED` while waiting for preroll from every stream. The encoded edge uses `RealtimeLatestByStream`; a congested Insight channel can drop stale encoded work without blocking that stream's decoder branch.

Detection metadata includes the source `rtp_timestamp` and is sent with nonblocking UDP. A compatible Insight receiver holds complete encoded RTP frames for 400 ms and matches metadata to that source timestamp before WebRTC forwarding. Keep one active viewer while validating metadata, as described below.

The three checked-in profiles use the same application and model:

| Config | Streams | Source resolution and FPS | Expected FPS per channel |
| --- | ---: | --- | ---: |
| `config.yaml` | 16 | 1280×720 at 30 FPS | 30 FPS |
| `config-24x720p20fps.yaml` | 24 | 1280×720 at 20 FPS | 20 FPS |
| `config-48x720p10fps.yaml` | 48 | 1280×720 at 10 FPS | 10 FPS |

The expected rate applies to both Insight video and detection metadata after startup and model warmup.

## Preview

The 48-stream profile running in Insight:

![High-density multi-stream object detector in Insight](../../../portal/assets/examples/object-detection/high-density-multi-stream-object-detector/image.png)

## Prerequisites

- A Modalix DevKit compatible with the selected Apps release, with the decoder service running.
- An [Insight](https://developer.sima.ai/software/tools/insight/) URL reachable from the DevKit.
- 16, 24, or 48 H.264 or H.265 RTSP sources matching `input.codec` and the selected profile.
- For H.265, the computer running the Insight viewer must support hardware HEVC decoding; Chromium does not provide a software decoder fallback for WebRTC H.265.
- H.265 playback in Chrome on macOS renders, but is not stable. Chrome's WebRTC HEVC decoder can stop producing frames mid-stream and fall back to a null decoder that discards what follows, at which point the tile stalls until the viewer reconnects it.
- Constant source frame rate, 1280×720 resolution, no B-frames in the selected codec, and a short, regular IDR interval. The validated sources use one IDR per second.
- The `yolo26n-det-int8-b1.tar.gz` model pack.

The application starts all source graphs together. Start every RTSP publisher before starting the application.

## Install Apps

Install the latest Neat Apps runtime and enter the installed bundle:

```bash
sima-cli neat install apps
cd prebuilt-apps
```

Run the remaining commands from `prebuilt-apps/`.

## Prepare the Model

| Model file | Role | Source |
| --- | --- | --- |
| `yolo26n-det-int8-b1.tar.gz` | Default | Direct artifact |

Model packages come from the Model Zoo release below, which can differ from the installed platform version.

```bash
export MODELZOO_VERSION="2.1.2"
mkdir -p models
cd models
sima-cli download "https://docs.sima.ai/pkg_downloads/SDK${MODELZOO_VERSION}/models/modalix/yolo26-detection/yolo26n-det-int8-b1.tar.gz"
cd ..
```

Set `model.path` in the selected config to the downloaded package. Relative paths resolve from the config file; absolute paths are also supported.

## Prepare Insight

[Insight](https://developer.sima.ai/software/tools/insight/) can host the input streams and render each output channel. Install videos directly from the Insight catalog or through YouTube support. In the Insight Web UI, start the required streams and copy their RTSP URLs into `streams`. Use the host and UDP port ranges reported by `neat` for the output settings.

Verify each source before starting the application:

```bash
ffprobe -v error \
  -select_streams v:0 \
  -show_entries stream=codec_name,width,height,avg_frame_rate,has_b_frames \
  -of default=noprint_wrappers=1 \
  <rtsp-url>
```

The result must report the codec selected by `input.codec`, `1280x720`, the selected profile FPS, and no B-frames. Using a different source FPS changes the output rate and is not the documented profile. The encoded Insight edge retains the latest complete access unit under congestion, so a one-second-or-shorter IDR interval bounds receiver recovery if an older access unit is replaced.

## Configure

Choose one config under `src/common/` and edit:

- `model.path`
- every entry under `streams`
- `output.insight.host`

Set `input.codec` to `h264`/`avc` or `h265`/`hevc`. H.264 is the default in all checked-in density profiles and remains the validated high-density configuration.

`inference.max_inflight_per_stream` and `inference.max_inflight_total` bound raw decoder-backed frames admitted to the shared detector. The 16- and 48-stream profiles use a total limit of eight; the 24-stream profile uses 24 so one aggregate frame interval can be admitted without an unbounded queue. The realtime mux retains only the latest pending frame for each stream.

Do not add the removed `inference.fan_in_policy` setting. Ordinary `connect()` and `build()` select the eligible realtime fan-in lowering automatically. Video/metadata synchronization is performed by Insight from each payload's source RTP timestamp; there is no application-side video-delay setting.

Do not change `input.width`, `input.height`, or `input.fps` unless the RTSP sources also change. `input.skip_rtsp_probe` is enabled, so those values are the source contract.

Insight channel and port mapping is deterministic:

```text
channel index: 0 .. stream_count - 1
video port:    9000 + channel index
metadata port: 9100 + channel index
```

## Run

Set the example and selected profile from the `prebuilt-apps/` root:

```bash
APP_DIR=examples/object-detection/high-density-multi-stream-object-detector
APP=./examples/object-detection/high-density-multi-stream-object-detector/src/cpp/pre-built/high-density-multi-stream-object-detector
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
source ~/pyneat/bin/activate
python3 -m pip install -r "$APP_DIR/src/python/requirements.txt"
python3 "$APP_DIR/src/python/main.py" --config "$CONFIG"
```

Stop the application with `Ctrl-C`.

Use one active Insight viewer while validating metadata. Insight currently has a single-viewer metadata rendering limitation; multiple simultaneous viewers can make box delivery appear intermittent even when the application is advancing normally.

## Expected Result

- Every configured Insight channel receives live video.
- Detection boxes appear on the matching channel.
- Video and metadata advance at 30 FPS for 16 streams, 20 FPS for 24 streams, or 10 FPS for 48 streams.
- Final per-stream counters show every stream advancing; the application fails with the missing channel IDs if a stream never starts or later stops.
- No detection timeout or stalled channel is reported. Metadata sender counters expose any nonblocking UDP drops without stalling inference.

If channels do not start, confirm that every publisher was already reachable and that the configured source caps match the selected profile. Restart the application after restarting the publishers.

## Source Files

- C++ reference source: `src/cpp/main.cpp`
- Python implementation: `src/python/main.py`
- Default 16-stream profile: `src/common/config.yaml`
- 24-stream profile: `src/common/config-24x720p20fps.yaml`
- 48-stream profile: `src/common/config-48x720p10fps.yaml`
- COCO labels: `src/common/coco_label.txt`

The packaged C++ source is an implementation reference. Run the executable under `src/cpp/pre-built/`; the installed bundle does not include CMake files.

## Development From Source

To modify, compile, or test this example, use the [Apps contributor workflow](https://github.com/sima-neat/apps/blob/main/CONTRIBUTING.md). The 16-, 24-, and 48-stream profiles still require Modalix, live RTSP publishers, and Insight for runtime validation.
