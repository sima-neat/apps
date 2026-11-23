# DETR Object Detector — ROS 2

This is an independent ROS 2 package, `neat_detr_object_detector`. Build and run it from
this directory. Nodes, launch, configuration, labels and implementation helpers
are local; no `neat_apps` package or original example source is imported.

## Structure and node roles

```text
detr-object-detector/
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
Use the [native model preparation guide](../../../object-detection/detr-object-detector/README.md#prepare-the-model)
for the same model artifacts and precision. Edit [src/common/config.yaml](src/common/config.yaml).
Model and dataset files are user-provided; use absolute paths, or paths relative
to this example directory. Shipped labels are under `src/common/`.

| Configuration | Meaning |
| --- | --- |
| `model` | Model package(s), labels and app-specific model options |
| `io` | Input image directory/file and output paths |
| `decode` | Confidence/NMS thresholds and decoding options |
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
ros2 launch neat_detr_object_detector app.launch.py \
  config:="$PWD/src/common/config.yaml"
```

The default `source:=folder` starts the folder input node.
Use `source:=none` to start inference without an input node and wait for an
external publisher on `/neat/image`. The image encoding must be `bgr8`.
`namespace:=demo` changes the namespace. Ctrl+C stops the launched nodes.
A component failure stops the launch; finite folder input exits after every
image has a matching result. Omit `config` to use the installed config copy.

## Results and Insight

For a finite folder run, start the topic echo below in a second sourced terminal
before launching the app. Enable saving or Insight in the config before launch
to inspect the output visually. `ros.source_repeats` can repeat the image folder.

```bash
ros2 topic echo /neat/results
```

| Topic | Type / role |
| --- | --- |
| `image` | `sensor_msgs/msg/Image`, source pixels and timestamp |
| `results` | `std_msgs/msg/String`, JSON numerical results and matching source header |
| `detr_object_detector/statistics` | `std_msgs/msg/String`, completed outputs/FPS and queue/drop accounting |

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

DETR requires Neat's `BilinearAntialiased` resize, ImageNet preprocessing and
built-in DETR decoder, matching the native example. Runtimes without that resize
option fail clearly; no interpolation fallback is used.

Retested on Modalix with Neat **0.6.2** and board image **3.0.1_master_B2796**:
the standalone `colcon build`, installed ROS launch and native Python command
passed with the registry `detr_resnet_50` **w16a16** package. All 21 test images
(640×640) produced exactly matching boxes, scores and class IDs. Both native
and ROS runs saved all 21 overlays.

Three alternating native/ROS timing pairs each processed 903 images (43 passes
through those 21 images), excluding the first 21 outputs from timing. Saving and
Insight were disabled during timing; model initialization was excluded.

| Completed-output throughput | Observed FPS range |
| --- | --- |
| Native Python image-read/detection loop | 35.29–35.78 |
| ROS input node + asynchronous detection node | 70.99–77.84 |

ROS used the default four in-flight requests; native Python calls `detect()`
sequentially. Every recorded result matched native output exactly, with no ROS
drops, sequence gaps, header mismatches or pending work at shutdown. No slowdown
was observed for this workload. This verifies equivalence to native Python,
not a labelled mAP evaluation; other model variants and camera inputs were not
retested in this check.

For an external image publisher, `ros.input_max_width`/`input_max_height` set the
maximum accepted geometry (defaults 1920×1080). Folder mode scans the configured
images for the same maximum geometry used by the native example.

See the [node development guide](../../NODE_DEVELOPMENT.md) for creating
your own app, and the [ROS overview](../../../../ROS2.md) for validation limits.
