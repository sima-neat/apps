# Multi-Stream BlazePose 3D — ROS 2

ROS nodes for the [Multi-Stream BlazePose 3D native example](../../../pose-estimation/multi-stream-blazepose3d/README.md).
The native implementation remains unchanged; this entrypoint imports its model
and decoding helpers. ROS configuration and launch instructions are local to this folder.

## Prepare the model and inputs

Follow the native [model preparation](../../../pose-estimation/multi-stream-blazepose3d/README.md#prepare-the-model)
and [input configuration](../../../pose-estimation/multi-stream-blazepose3d/README.md#configure), then set the same
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
[ROS setup](../../../../ROS2.md#install-and-build) once. Edit this ROS example’s `src/common/config.yaml`; its `ros` section controls
image topics and bounded delivery.

The common live source publishes the native decoder’s `nv12` images. The inference
node reuses the unchanged native BlazePose scheduler, model construction and decoding.
Its detector, pose and publication stages each retain the newest waiting job per
camera. Already submitted model work completes; newer images may replace older
waiting work. Cameras share the models using the native round-robin dispatch.
ROI tracking is updated on pose completion, and smoothing runs only on published
frames. Stale publications are discarded using the native sequence check.

Keep `ros.input_policy: latest`. This example intentionally does not promise one
result per input image, just like the native live application. Generic
`pending_frames`, `inflight_frames` and `source_window` settings do not configure
its native scheduler. Statistics on `/neat/multi_stream_blazepose3d/statistics`
report received, admitted, published and replaced/dropped counts per camera.
A camera that exceeds `source_timeout_seconds` is marked closed; other cameras
continue, as in the native app. Restart the node to reconnect a closed camera.
If all cameras stop unexpectedly, the node exits with an error.

From the Apps root after sourcing the ROS package:

```bash
APP_DIR=examples/ros2/pose-estimation/multi-stream-blazepose3d
ros2 launch neat_apps app.launch.py \
  app:=pose-estimation/multi-stream-blazepose3d apps_root:="$PWD" \
  config:="$PWD/$APP_DIR/src/common/config.yaml" source:=live
```

Use `source:=none` with an external camera or rosbag on `/neat/image`. Publish
`nv12` images at the configured dimensions with `header.frame_id` equal to the
camera's `streams[].id`, and strictly increasing, nonnegative timestamps per
camera. Every published result preserves that exact input header.
Inspect `/neat/results`; see the [message and validation contracts](../../../../ROS2.md).
Do not set the recording sink's expected count to the source frame count:
replaced frames legitimately have no result. For finite source runs, leave
`runtime.frames: 0` and set `ros.source_frames` to the input count. When that
source finishes, the inference node drains remaining native work before closing.

Numerical comparisons should use the same admitted frame history and tracking
state. Independent live runs can replace different frames due to timing;
matching their scheduling policy does not guarantee identical output sequences.

The ROS entrypoint is `src/python/ros_node.py`. On Modalix, twelve paced positive
pose results matched the unchanged native Python app exactly, including 2D and
3D landmarks. A single 640x480 stream completed 600/600 inputs at approximately
30 FPS on both paths. A paired run at the default 1920x1080 resolution also
maintained 30 FPS, with 599 published outputs from 600 inputs on each path.
Three longer four-camera comparisons at 640x480 measured native
29.97–30.01 FPS per camera versus ROS 20.19–22.55, excluding startup outputs.
The scheduler behavior is aligned, but four-camera FPS parity has not been met.
These are replay measurements, not a full accuracy dataset or physical-camera
validation.

For node responsibilities, settings, topics and extension examples, see the
[ROS node development guide](../../../../support/ros2/README.md). Its standard
image-node settings apply where this example uses the shared `ImageNode`;
custom-node differences are listed separately.

## View results and Insight

Both outputs are configured in this app's existing `ros.visualization` mapping
in [config.yaml](src/common/config.yaml); both default to disabled. Update the
keys in that mapping (keep its other settings):

```yaml
ros:
  visualization:
    save:
      enabled: true
      directory: sandbox/ros2/pose-estimation/multi-stream-blazepose3d/visuals
      every_n_frames: 1
    insight:
      enabled: true
      host: <insight-host-ip>
      video_port_base: 9000
      fps: 30
```

Enable either output or both. Use the reachable host and exposed UDP ports of
your running Insight service. Rebuild the shared ROS package after updating its
code, then use the **same launch command above**: no separate sink command is
needed. `insight:=true` or `insight:=false` can override the YAML switch.

Saved PNGs go into a new `run-*` subfolder on every invocation. `frames.jsonl`
maps each filename to its exact input header and result sequence. Existing runs
are preserved. Native `io.output_dir`/`output.overlay` keys do not control these
ROS files; use the explicit ROS settings above.

Open Insight's video viewer and select channel 0 by default. This output draws
the decoded result on its matching source image and sends H.264 video through
Neat's hardware encoder and VideoSender. Overlays are part of the video, so
there is no separate 2D metadata channel to configure. The 3D panel uses
the auxiliary metadata channel described below. ROS result topics retain
the original numerical data. For multiple cameras, stable channel mappings,
performance limits and troubleshooting, see the
[output configuration reference](../../../../support/ros2/README.md#7-saved-images-and-insight).

BlazePose retains at most `ros.visualization.frame_cache` images per camera
(default 8). If its native latest-frame scheduler finishes a result after that
image has been evicted, the visual output is skipped and counted in
`visualization_missing_frames`; an unrelated newer image is never substituted.
The video shows 2D landmarks. With `ros.visualization.insight.auxiliary: true`
(default), native world landmarks also appear in Insight's **3D Pose** panel.
Set `metadata_port_base` (default 9100) to the exposed metadata UDP base; the
sender adds the same channel index used for video. Insight needs its
`blazepose-3d` renderer. Set `auxiliary: false` for video only. The sender uses
one local output clock for encoded video and auxiliary metadata; original
ROS headers and world coordinates remain unchanged.

Start a live source or a sufficiently long folder run before connecting the
viewer. A short dataset can finish before the browser connects; saved PNGs are
the reliable way to inspect every result from a finite run. Rendering, file I/O
and encoding add work: disable both outputs for inference-only comparisons.
The shared output path is tested on Modalix; this is not an app-specific visual
quality or visual-mode FPS parity claim.

## Source files

- [ROS entrypoint](src/python/ros_node.py)
- [ROS configuration](src/common/config.yaml)
- [Shared source, sink and launch package](../../../../support/ros2/)
