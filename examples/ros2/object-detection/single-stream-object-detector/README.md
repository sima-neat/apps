# Single-Stream Object Detector — ROS 2

This is an independent ROS 2 package, `neat_single_stream_object_detector`. Build and run it from
this directory. Nodes, launch, configuration, labels and implementation helpers
are local; no `neat_apps` package or original example source is imported.

## Structure and node roles

```text
single-stream-object-detector/
├── package.xml
├── CMakeLists.txt
├── launch/app.launch.py
└── src/
    ├── common/config.yaml
    └── python/
        ├── detection_node.py
        ├── input_node.py
        ├── result_sink.py
        ├── image_transport.py
        ├── visual_output.py
        └── model_support.py
```

| File / method | Responsibility |
| --- | --- |
| [detection_node.py](src/python/detection_node.py) — `__init__()` | Declares image subscription, results/statistics publishers and bounded inference workers |
| `initialize_model()` and `build_runner()` | Configure models and build the Neat graph; geometry-dependent setup is in `prepare_input()` |
| `prepare_input()` | Prepare the same pixels/tensors as the native Python app |
| `submit_images()` / `receive_results()` | Call `runner.push()` / `runner.pull()`, retain inputs and verify result IDs/timestamps |
| `decode_result()` | Decode numerical output; publish matching JSON and, where applicable, dense images |
| [input_node.py](src/python/input_node.py) | Read the configured input and publish timestamped images; bound outstanding work using result headers |
| [result_sink.py](src/python/result_sink.py) | Optional JSON/dense-image recorder; not a visualization node |
| [visual_output.py](src/python/visual_output.py) | Render the matching source/result, optionally save it or send Insight video |
| [model_support.py](src/python/model_support.py) | Local model configuration, numerical decoding and supporting algorithms |

Inference runs outside ROS callbacks. Submission and completion overlap; pending
and in-flight work remain bounded. There is no generic dispatcher, `Pipeline`
factory or shared `ImageNode` base class.

## Prepare and configure

Install ROS 2 with:

```bash
sima-cli neat install ros2@v3.0.0
```

Use the matching Neat Python environment and installed colcon build tools.
Use the [native model preparation guide](../../../object-detection/single-stream-object-detector/README.md#prepare-the-model)
for the same model artifacts and precision. Edit [src/common/config.yaml](src/common/config.yaml).
Model and dataset files are user-provided; use absolute paths, or paths relative
to this example directory. Shipped labels are under `src/common/`.

| Configuration | Meaning |
| --- | --- |
| `model` | Model package(s), labels and app-specific model options |
| `source` | Input URL, file, transport or camera choice |
| `inference` | Thresholds, output limits and app-specific inference settings |
| `runtime` | Frame limits, warmup and runtime controls used by this app |
| `ros.image_topic`, `ros.results_topic` | Image input and matching JSON result topics (defaults `image`, `results`) |
| `ros.input_policy` | `complete`: backpressure; `latest`: discard unadmitted older images |
| `ros.source_window`, `pending_frames`, `inflight_frames` | Bounded outstanding, queued and admitted work |
| `ros.visualization.save.enabled`, `.directory` | Save matched overlays in a new run directory |
| `ros.visualization.insight.enabled`, `.host` | Enable Insight video and choose its host |

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
ros2 launch neat_single_stream_object_detector app.launch.py \
  config:="$PWD/src/common/config.yaml"
```

The default `source:=live` starts this app’s configured input node.
Use `source:=none` to start inference without an input node and wait for an
external publisher on `/neat/image`. Use packed `nv12` or `nv16` images; keep the format fixed during a run.
`stream_index:=all` starts the configured cameras; an integer selects one.
`namespace:=demo` changes the namespace. Ctrl+C stops the launched nodes.
A component failure stops the launch. Omit `config` to use the installed config copy.

## Results and Insight

```bash
ros2 topic echo /neat/results
```

| Topic | Type / role |
| --- | --- |
| `image` | `sensor_msgs/msg/Image`, source pixels and timestamp |
| `results` | `std_msgs/msg/String`, JSON numerical results and matching source header |
| `single_stream_object_detector/statistics` | `std_msgs/msg/String`, completed outputs/FPS and queue/drop accounting |

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

Edit [detection_node.py](src/python/detection_node.py) for model behavior and
[launch/app.launch.py](launch/app.launch.py) for wiring. Add executable scripts
to this folder’s `CMakeLists.txt` and dependencies to `package.xml`.
Change the input node to support another camera while retaining the documented
image encoding, dimensions, headers and QoS.

The migration preserves model routes, preprocessing, decoding, queue bounds and
worker topology. Build/import checks alone do not establish accuracy or FPS
parity. See the [validation status](../../README.md#validation-status) for measured
coverage and remaining runtime/hardware limitations. Physical MIPI/USB and
Docker execution remain TODO where applicable.

See the [node development guide](../../NODE_DEVELOPMENT.md) for creating
your own app, and the [ROS overview](../../../../ROS2.md) for validation limits.
