# Single-Stream Thermal Face Detector — ROS 2

ROS nodes for the [Single-Stream Thermal Face Detector native example](../../../face-detection/single-stream-thermal-face-detector/README.md).
The native implementation remains unchanged; this entrypoint imports its model
and decoding helpers. ROS configuration and launch instructions are local to this folder.

## Prepare the model and inputs

Follow the native [model preparation](../../../face-detection/single-stream-thermal-face-detector/README.md#prepare-the-model)
and [input configuration](../../../face-detection/single-stream-thermal-face-detector/README.md#configure), then set the same
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

The common live source publishes the native decoder’s `nv12` images. The inference node preserves the model settings and
native decoding and publishes timestamped results.

From the Apps root after sourcing the ROS package:

```bash
APP_DIR=examples/ros2/face-detection/single-stream-thermal-face-detector
ros2 launch neat_apps app.launch.py \
  app:=face-detection/single-stream-thermal-face-detector apps_root:="$PWD" \
  config:="$PWD/$APP_DIR/src/common/config.yaml" source:=live
```

Use `source:=none` with an external camera or rosbag on `/neat/image`.
Inspect `/neat/results`, or use the shared `result_sink` to record and check
finite runs. Input headers accompany results; see the [message, queue and
validation contracts](../../../../ROS2.md).

The ROS entrypoint is `src/python/ros_node.py`. Model accuracy and throughput
validation are in progress; an available entrypoint is not a parity claim.

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
      directory: sandbox/ros2/face-detection/single-stream-thermal-face-detector/visuals
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
there is no separate 2D metadata channel to configure. ROS result topics retain
the original numerical data. For multiple cameras, stable channel mappings,
performance limits and troubleshooting, see the
[output configuration reference](../../../../support/ros2/README.md#7-saved-images-and-insight).

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
