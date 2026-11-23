# MIPI Camera Capture — ROS 2

This is an independent ROS 2 package, `neat_mipi_camera_capture`. Build and run it from
this directory. Nodes, launch, configuration, labels and implementation helpers
are local; no `neat_apps` package or original example source is imported.

## Structure and node roles

```text
mipi-camera-capture/
├── package.xml
├── CMakeLists.txt
├── launch/app.launch.py
└── src/
    ├── common/config.yaml
    └── python/
        ├── capture_node.py
        ├── input_node.py
        ├── result_sink.py
        ├── image_transport.py
        ├── visual_output.py
        └── model_support.py
```

[capture_node.py](src/python/capture_node.py) declares its ROS interfaces and execution flow directly.
`main()` declares the image subscription and result publisher; `receive()`
selects timestamped captures. `input_node.py` owns the physical camera.
This app measures capture and has no model inference.

## Prepare and configure

Install ROS 2 with:

```bash
sima-cli neat install ros2@v3.0.0
```

Use the matching Neat Python environment and installed colcon build tools.
No model is required. Edit [src/common/config.yaml](src/common/config.yaml).

| Configuration | Meaning |
| --- | --- |
| `camera` | Camera selection, NV12 geometry, rate and capture buffers |
| `capture.duration_seconds`, `.sample_times_seconds` | Capture duration and snapshot times |
| `capture.pull_timeout_ms` | Fail if an active image source stalls |
| `output.directory` | Saved NV12 snapshots and capture summary |
| `ros.image_topic`, `ros.results_topic` | Input images and matching capture acknowledgments |
| `ros.input_policy`, `ros.source_window` | Input-node delivery policy and outstanding frames |
| `ros.source_timeout_seconds` | Initial wait for the source |
| `ros.visualization` | Optional saved previews and Insight output |

These are startup YAML settings, not dynamic ROS parameters. Restart after edits.

## Build on Modalix

From this example directory:

```bash
source /usr/local/ros2/setup.bash
source ~/pyneat/bin/activate
colcon build
source install/setup.bash
```

Only this package is built. `build/`, `install/` and `log/` stay in this folder.
Rebuild after Python or launch edits. In every new terminal source ROS, activate
Neat and source this installation. Configuration edits only require a restart
when using the source config below.

## Run

```bash
ros2 launch neat_mipi_camera_capture app.launch.py \
  config:="$PWD/src/common/config.yaml"
```

The default `source:=live` starts this app’s configured input node.
Use `source:=none` to start capture without an input node and wait for an
external publisher on `/neat/image`. Use packed `nv12` images matching the configured `camera.width` and `camera.height`.
`stream_index:=all` starts the configured cameras; an integer selects one.
`namespace:=demo` changes the namespace. Ctrl+C stops the launched nodes.
A component failure stops the launch. Omit `config` to use the installed config copy.

## Results and Insight

The capture node writes the requested `.nv12` snapshots and `summary.json` to
`output.directory`, then exits after `capture.duration_seconds`. The summary
contains received FPS, source timestamp FPS and interarrival statistics.

```bash
ros2 topic echo /neat/results
```

| Topic | Type / role |
| --- | --- |
| `image` | `sensor_msgs/msg/Image`, source pixels and timestamp |
| `results` | `std_msgs/msg/String`, JSON numerical results and matching source header |

Topics are relative to `/neat` by default. Input and result QoS is reliable;
match the offered camera QoS. JSON results retain the source `header`; the
recorder verifies sequence/header continuity.

For visualization set `ros.visualization.insight.enabled: true` and configure
its `host`. Open the video viewer in your running Insight service.
`channel_base` selects the first channel, `max_visible_streams` bounds visible
cameras, and `channels` can map camera frame IDs to explicit channels.
Overlays use the matching source image. No separate sink command is needed.
For a file result instead, enable `ros.visualization.save.enabled`.
Insight uses an even-sized canvas based on the first frame and letterboxes
later frames to fit. Saved PNGs use the original image dimensions.
Disable saving/Insight when measuring inference throughput.

## Extend and validate

Edit [capture_node.py](src/python/capture_node.py) for capture behavior and
[launch/app.launch.py](launch/app.launch.py) for wiring. Add executable scripts
to this folder’s `CMakeLists.txt` and dependencies to `package.xml`.
Change the input node to support another camera while retaining the documented
image encoding, dimensions, headers and QoS.

Capture retains the native geometry checks, snapshot selection and timing
calculations. Package build/import checks passed; physical MIPI capture and
sustained camera FPS remain unverified until a camera is connected. See the
[validation status](../../README.md#validation-status).

See the [node development guide](../../NODE_DEVELOPMENT.md) for creating
your own app, and the [ROS overview](../../../../ROS2.md) for validation limits.
