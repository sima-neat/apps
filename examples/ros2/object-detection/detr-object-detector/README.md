# DETR Object Detector — ROS 2

ROS nodes for the [DETR Object Detector native example](../../../object-detection/detr-object-detector/README.md).
The native implementation remains unchanged; this entrypoint imports its model
and decoding helpers. ROS configuration and launch instructions are local to this folder.

## Prepare the model and inputs

Follow the native [model preparation](../../../object-detection/detr-object-detector/README.md#prepare-the-model)
and [input configuration](../../../object-detection/detr-object-detector/README.md#configure), then set the same
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

The common folder source publishes original-size `bgr8` images. The app publishes DETR boxes and class scores.

From the Apps root after sourcing the ROS package:

```bash
APP_DIR=examples/ros2/object-detection/detr-object-detector
ros2 launch neat_apps app.launch.py \
  app:=object-detection/detr-object-detector apps_root:="$PWD" \
  config:="$PWD/$APP_DIR/src/common/config.yaml" source:=folder
```

Use `source:=none` with an external camera or rosbag on `/neat/image`.
Inspect `/neat/results`, or use the shared `result_sink` to record and check
finite runs. Input headers accompany results; see the [message, queue and
validation contracts](../../../../ROS2.md).

The ROS entrypoint is `src/python/ros_node.py`. Model accuracy and throughput
validation are in progress; an available entrypoint is not a parity claim.

## Nodes, settings and results

This example uses the shared image-node implementation. The
[DETR walkthrough](../../../../support/ros2/README.md#4-detr-walkthrough) follows
one frame through the actual code; the same guide explains how to build your
own pipeline.

| Node in the default namespace | Included work | ROS interface |
| --- | --- | --- |
| `/neat/neat_folder_source` | Discover/read images, publish original BGR pixels, wait for matching result headers before reusing window slots | Publishes `/neat/image`; subscribes to `/neat/results` |
| `/neat/detr_object_detector` | Native preprocessing, tensor graph inference, native DETR decoding and original-image coordinate mapping; shared workers own asynchronous admission and result association | Subscribes to `/neat/image`; publishes `/neat/results` and `/neat/detr_object_detector/statistics` |
| `/neat_result_sink` (optional, started separately with the command below) | Record JSON and verify consecutive result sequences | Subscribes to `/neat/results` |

Images use `sensor_msgs/msg/Image`; results and statistics use
`std_msgs/msg/String` containing JSON. With `source:=none`, your external BGR
publisher replaces the folder source. It must offer reliable QoS compatible
with the shared subscriber. The shared launch does not start a recorder.

The ROS pipeline reads these application settings:

| YAML setting | Role |
| --- | --- |
| `model.path` | Required DETR model package. Relative paths resolve from the Apps root in this ROS pipeline. |
| `io.input_dir` | Folder source input directory (or one image file). |
| `decode.confidence_threshold` | Detection score threshold; default `0.70`. |
| `decode.person_only` | Retain only the native COCO person class; default `false`. |
| `ros.*` | Topic names, admission bounds, source window and timeouts; see the [shared settings reference](../../../../support/ros2/README.md#standard-image-node-yaml-settings). |

`decode.max_draw` now limits the native DETR visual renderer when outputs are enabled.
The inherited `io.output_dir`, `runtime.profile`,
`runtime.num_runs`, `runtime.verbose` and `output.overlay` settings are not used
by this ROS inference path. Configure saved images under `ros.visualization.save`. Use
`ros.source_repeats` to repeat the folder dataset and `result_sink` to record
results. These YAML values are read at startup; changing ROS parameters at
runtime does not update the model or decoding thresholds.

Each result preserves the input header and adds a publication `sequence`.
`data` contains `width`, `height`, and `objects`. Each object has `box` as
`[x1, y1, x2, y2]` in original-image pixels, `score`, and `class_id`; an empty
`objects` list is valid. Statistics report receipts, admissions, completions,
drops, outstanding work and completion FPS. DETR uses the standard image-topic
mode, with optional rendered Insight video controlled by `ros.visualization.insight`.

To record a finite dataset, set `ros.result_subscribers: 2`, start the recorder
before launching inference, and use the number of dataset images multiplied by
`ros.source_repeats` (default 1) as `N`:

```bash
ros2 run neat_apps result_sink --topic /neat/results \
  --count N --output sandbox/detr-results.jsonl --timeout 120
```

Choose a new output filename for each run. See the
[development guide](../../../../support/ros2/README.md#6-inspect-and-validate)
for topic inspection, remapping and validation commands.

## View results and Insight

Both outputs are configured in this app's existing `ros.visualization` mapping
in [config.yaml](src/common/config.yaml); both default to disabled. Update the
keys in that mapping (keep its other settings):

```yaml
ros:
  visualization:
    save:
      enabled: true
      directory: sandbox/ros2/object-detection/detr-object-detector/visuals
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
