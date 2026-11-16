# High-Density Multi-Stream Object Detector — ROS 2

ROS nodes for the [High-Density Multi-Stream Object Detector native example](../../../object-detection/high-density-multi-stream-object-detector/README.md).
The native implementation remains unchanged; this entrypoint imports its model
and decoding helpers. ROS configuration and launch instructions are local to this folder.

## Prepare the model and inputs

Follow the native [model preparation](../../../object-detection/high-density-multi-stream-object-detector/README.md#prepare-the-model)
and [input configuration](../../../object-detection/high-density-multi-stream-object-detector/README.md#configure), then set the same
model, labels, source and thresholds in [src/common/config.yaml](src/common/config.yaml).
Run commands from the Apps root. Shared labels, assets and native helper modules
are reused from their original locations.

## Build on Modalix

Run from the Apps checkout or installed `prebuilt-apps/` root on Modalix.
Install ROS 2 once with `sima-cli neat install ros2@v3.0.0` and use the Neat
Python environment required by the native example. If `colcon` is installed:

```bash
source /usr/local/ros2/setup.bash
source ~/pyneat/bin/activate
python -c "import rclpy, pyneat"

colcon --log-base sandbox/ros2-colcon-log build \
  --base-paths support/ros2 --packages-select neat_apps --merge-install \
  --build-base sandbox/ros2-colcon-build \
  --install-base sandbox/ros2-colcon-install \
  --cmake-args -DPython3_EXECUTABLE="$(command -v python)"
source sandbox/ros2-colcon-install/local_setup.bash
```

This builds the shared `neat_apps` package for all ROS examples, so build it once,
not once per application. If colcon is unavailable, use the
[direct CMake build](../../../../ROS2.md#build-with-cmake-without-colcon).
In each new terminal, source ROS, the Neat environment and your chosen package
installation before running commands. Restart after editing this app's Python
or YAML; rebuild after changing shared nodes or launch files in `support/ros2/`.

## Run

Install ROS 2 with `sima-cli neat install ros2@v3.0.0`, then follow the shared
[ROS setup](../../../../ROS2.md#install-and-build) once. Edit this ROS example’s
`src/common/config.yaml` with the model, RTSP URLs and decoder settings.

For live RTSP cameras, one ROS node runs the native Neat graph, keeping camera
decoding and inference together. It uses the native per-stream admission limits
and latest-frame policy, and publishes every pulled detection result on
`/neat/results`. Images stay inside Neat in this mode. Optional Insight video
still follows `output.video_enabled`; detection metadata is published through ROS
and can be forwarded to Insight by the sink described below.

From the Apps root after sourcing the ROS package:

```bash
APP_DIR=examples/ros2/object-detection/high-density-multi-stream-object-detector
ros2 launch neat_apps app.launch.py \
  app:=object-detection/high-density-multi-stream-object-detector apps_root:="$PWD" \
  config:="$PWD/$APP_DIR/src/common/config.yaml" source:=live_graph
```

Live-graph result headers identify `camera_<index>_optical_frame`. Their timestamps
are the decoder's presentation timestamps, relative to the stream timeline,
not ROS wall-clock time. `source_frame_id` and `stream_id` preserve the native
frame and stream identities. `sequence` counts publications across all streams.
The `data.objects` values use the unchanged native detection decoder.

For an external ROS camera or rosbag, use `source:=none` and publish `nv12` images
on `/neat/image`. Use `source:=live` to run the separate common source and pass
configured RTSP inputs through that image topic. These image-topic modes retain
the incoming image headers and use the `ros` queue/window settings. Those
settings do not configure admission in `live_graph`; use
`inference.max_inflight_per_stream` and `inference.max_inflight_total` instead.
`live_graph` uses all configured streams and runs until stopped with Ctrl+C.

Inspect `/neat/results`, or use the shared `result_sink` to record results; see
the [message, queue and validation contracts](../../../../ROS2.md).

## View results and Insight

Insight video and detection overlays are implemented for `source:=live_graph`.
For `source:=live` or `none`, the shared visual output stage also supports
saved PNGs and rendered Insight video. Configure `ros.visualization.save` and
`ros.visualization.insight` as described in the
[output reference](../../../../support/ros2/README.md#7-saved-images-and-insight).
Saving decoded images requires one of those image-topic modes; `live_graph`
keeps its existing encoded-video path and rejects image saving.

In `live_graph`, set `ros.visualization.insight.enabled: true` to start the
metadata sink from YAML, or keep the `insight:=true` launch override below.
Host, ports and visible stream count in this mode still come from
`output.insight`, and `output.video_enabled` must be true.

### View video and detections in Insight

Start Insight and use the host and exposed UDP ports reported by
`neat status --offline --json`. Set `output.video_enabled: true` and configure
the same destination for video and metadata. For example, when four channels
are exposed:

```yaml
output:
  video_enabled: true
  insight:
    host: <insight-host-ip>
    video_port_base: 9000
    metadata_port_base: 9100
    max_visible_streams: 4
```

Use the actual exposed port bases if they differ. `max_visible_streams` limits
visualization only; inference and ROS results still cover every configured stream.
Start the complete application with one command:

```bash
APP_DIR=examples/ros2/object-detection/high-density-multi-stream-object-detector
ros2 launch neat_apps app.launch.py \
  app:=object-detection/high-density-multi-stream-object-detector apps_root:="$PWD" \
  config:="$PWD/$APP_DIR/src/common/config.yaml" source:=live_graph insight:=true
```

The application path is:

```text
RTSP -> native decode/inference graph -> ROS results -> insight_sink -> Insight boxes
   \-> native encoded-video sender --------------------------------> Insight video
```

Open Insight's video viewer and select channels 0 through
`max_visible_streams - 1`. Boxes, labels and confidence scores appear over the
video. Keep one active viewer while checking overlays, as in the native example.
Ctrl+C stops the graph and sink together.

The sink uses the public Neat `MetadataSender` API and preserves the source
presentation timestamp, frame ID and numerical detections. Insight matches the
metadata's 90 kHz RTP timestamp to the video; ROS reception time is never used
for synchronization. Invisible streams are filtered before sending metadata.
This separate metadata sink supports high-density `live_graph` results only: arbitrary
ROS image timestamps cannot be assumed to match encoded video timestamps.

If video appears without boxes, check that `insight_sink` is running and that
the metadata UDP base and visible channel count match Insight. Its ingest
statistics at `/api/ingest/stats` report received and matched metadata.
The viewer's video FPS and metadata message rate are separate measurements.

The ROS entrypoint is `src/python/ros_node.py`. On Modalix, twenty 640x480,
30 FPS replay streams produced 720/720 sampled detections exactly matching
native Python across three alternating comparisons. Live-graph ROS throughput
was 495.40–508.30 aggregate FPS versus native 526.60–530.23 FPS: a remaining
3.5–6.6% slowdown in paired trials. The installed launch also passed 1,000 exact
outputs across all twenty streams and clean shutdown.

The separate image-topic mode previously measured 180.53–187.07 FPS with all
240 tested outputs matching native. The combined graph removes most of that
throughput gap, but **has not met the no-measurable-slowdown target**. These
measurements use the default Fast DDS middleware and do not validate the default
720p profile or physical cameras. They were measured with video disabled and
without the Insight sink; visual-mode throughput parity has not been established.

For node responsibilities, settings, topics and extension examples, see the
[ROS node development guide](../../../../support/ros2/README.md). Its standard
image-node settings apply where this example uses the shared `ImageNode`;
custom-node differences are listed separately.

## Source files

- [ROS entrypoint](src/python/ros_node.py)
- [ROS configuration](src/common/config.yaml)
- [Shared source, sink and launch package](../../../../support/ros2/)
- [ROS-to-Insight detection sink](../../../../support/ros2/neat_apps/insight_sink.py)
