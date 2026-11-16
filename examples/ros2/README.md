# ROS 2 examples

Each ROS example lives in `<category>/<example>/`, with its own README,
`src/common/config.yaml` and `src/python/ros_node.py`. The Docker example contains
its launch script instead. Original examples remain in `examples/<category>/<example>/`
and remain independently runnable. Common source, sink and launch code lives
in [support/ros2](../../support/ros2/).

## Build on Modalix

Run commands from the Apps checkout or installed `prebuilt-apps/` root, not from
this `examples/ros2/` directory. Install ROS 2 once:

```bash
sima-cli neat install ros2@v3.0.0
```

Use the Neat Python environment required by the native applications. Build the
shared package on Modalix with colcon, if installed:

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

If colcon is unavailable, use the [direct CMake workflow](../../ROS2.md#build-with-cmake-without-colcon).
Both workflows install the same package. Source ROS, the Python environment and
the selected package installation in every participating terminal.

| Change | Required action |
| --- | --- |
| Build the examples for the first time | Build `neat_apps` once for all apps. |
| Edit an example's Python/YAML or add an example folder | Restart/launch the app; no package rebuild. |
| Edit shared nodes, launch files or dependencies in `support/ros2/` | Rebuild the shared package and source the installation. |

## Configure and run an app

Choose an app below. Its README includes model preparation, a local YAML config,
a build command and its specific launch command. Models are prepared separately
from the ROS package build. For example, after setting DETR's `model.path` and
`io.input_dir`:

```bash
APP_DIR=examples/ros2/object-detection/detr-object-detector
ros2 launch neat_apps app.launch.py \
  app:=object-detection/detr-object-detector apps_root:="$PWD" \
  config:="$PWD/$APP_DIR/src/common/config.yaml" source:=folder
```

Use the source mode documented by the selected app; camera, request/response and
benchmark examples have different input requirements. PCIe integration is deferred.

### Choose who supplies the images

The launch argument `source:=...` selects whether the launch starts a source
node or leaves image publication to your existing ROS system:

| Mode | Nodes started / input behavior |
| --- | --- |
| `source:=folder` (default) | Starts the folder source and application node. The source reads configured images and publishes them on the image topic. |
| `source:=live` | Starts the common live source and application node(s). The source reads configured RTSP/HTTP/USB/MIPI or supported video-file inputs and publishes images. |
| `source:=none` | Starts only the application node. For image-based apps, it subscribes to the image topic and waits for your camera node, rosbag or other publisher. |
| `source:=live_graph` | High-density detector only: decoding and inference run inside one Neat graph in the application node. No separate source node or ROS image input. |

For example, run the face detector against an existing ROS image publisher:

```bash
APP_DIR=examples/ros2/face-detection/face-detector
ros2 launch neat_apps app.launch.py \
  app:=face-detection/face-detector apps_root:="$PWD" \
  config:="$PWD/$APP_DIR/src/common/config.yaml" source:=none
```

By default it subscribes to `/neat/image` (`namespace:=neat`, `ros.image_topic:
image`). To use `/camera/image_raw`, add or update this key in the config's
existing `ros:` mapping:

```yaml
ros:
  image_topic: /camera/image_raw
```

Keep the other `ros:` settings in that mapping. The face detector requires
`sensor_msgs/msg/Image` with `bgr8` encoding and a reliable publisher. Other
apps can require different pixel formats; see their READMEs. An external
publisher controls its own rate: `source:=none` does not provide source pacing.
Choose the documented `complete` or `latest` queue policy for your input.

`source` is a launch argument, not a dynamic ROS parameter. Studio and the model
benchmark also use `source:=none`, but follow their request/benchmark workflow
rather than waiting for a per-frame inference input. See the
[launch and topic reference](../../support/ros2/README.md#2-configuration-and-launch-arguments)
for the full contracts.

## View results and Insight

All image-based ROS examples now have optional saved visual images and Insight
video controlled by their `ros.visualization` config. Both default to off.
For example, update these keys in the face detector's existing YAML:

```yaml
ros:
  visualization:
    save:
      enabled: true
      directory: sandbox/ros2/face-detector/visuals
      every_n_frames: 1
    insight:
      enabled: true
      host: <insight-host-ip>
      video_port_base: 9000
      fps: 30
```

Use the same application launch command; `insight` defaults to `auto` and reads
the config. Enable either output independently. Saved PNGs and a header/sequence
manifest go into a unique `run-*` subfolder. The image-topic output draws the
result on its exact source frame and sends that rendered video to Insight using
Neat's encoder/VideoSender. Open your running Insight viewer on channel 0 by
default. The launch does not start the Insight service itself.

The high-density `source:=live_graph` mode retains its efficient encoded-video
and separate metadata path. Its YAML enable switch is the same, but host/ports
still use `output.insight`, with `output.video_enabled: true`. To save decoded
images for that app, use `source:=live` or `none` instead. See its
[visual launch instructions](object-detection/high-density-multi-stream-object-detector/README.md#view-video-and-detections-in-insight).

For channel mapping, sampling, resource limits, custom-node differences and
what has been verified, read the
[visual output reference](../../support/ros2/README.md#7-saved-images-and-insight).
Studio text/audio requests and synthetic benchmark reports keep their existing
non-video output contracts. Docker uses each selected app's configuration and
needs a writable path for saved files.

## Understand or extend an app

Read the [node development guide](../../support/ros2/README.md): node roles,
launch flow, configuration defaults, topics/QoS, a DETR walkthrough and a new-app
skeleton.

## Examples

| Example | Documentation |
| --- | --- |
| `anomaly-detection/fastflow-anomaly-detector` | [Run and configure](anomaly-detection/fastflow-anomaly-detector/README.md) |
| `anomaly-detection/patchcore-anomaly-detector` | [Run and configure](anomaly-detection/patchcore-anomaly-detector/README.md) |
| `benchmarking/mipi-camera-capture` | [Run and configure](benchmarking/mipi-camera-capture/README.md) |
| `benchmarking/mipi-multi-model` | [Run and configure](benchmarking/mipi-multi-model/README.md) |
| `benchmarking/model-benchmark` | [Run and configure](benchmarking/model-benchmark/README.md) |
| `classification/image-classification-explorer` | [Run and configure](classification/image-classification-explorer/README.md) |
| `depth-estimation/depth-estimator` | [Run and configure](depth-estimation/depth-estimator/README.md) |
| `docker/parallel-apps` | [Run and configure](docker/parallel-apps/README.md) |
| `face-detection/face-detector` | [Run and configure](face-detection/face-detector/README.md) |
| `face-detection/single-stream-thermal-face-detector` | [Run and configure](face-detection/single-stream-thermal-face-detector/README.md) |
| `face-recognition/face-recognizer` | [Run and configure](face-recognition/face-recognizer/README.md) |
| `feature-extraction/superpoint-feature-extractor` | [Run and configure](feature-extraction/superpoint-feature-extractor/README.md) |
| `genai/detection-to-vlm-assistant` | [Run and configure](genai/detection-to-vlm-assistant/README.md) |
| `genai/neat-genai-studio` | [Run and configure](genai/neat-genai-studio/README.md) |
| `multi-model/multi-stream-multi-model` | [Run and configure](multi-model/multi-stream-multi-model/README.md) |
| `object-detection/detr-object-detector` | [Run and configure](object-detection/detr-object-detector/README.md) |
| `object-detection/high-density-multi-stream-object-detector` | [Run and configure](object-detection/high-density-multi-stream-object-detector/README.md) |
| `object-detection/multi-stream-obb-detector` | [Run and configure](object-detection/multi-stream-obb-detector/README.md) |
| `object-detection/multi-stream-object-detector` | [Run and configure](object-detection/multi-stream-object-detector/README.md) |
| `object-detection/pcb-defect-detector` | [Run and configure](object-detection/pcb-defect-detector/README.md) |
| `object-detection/rfdetr-detection-segmentation` | [Run and configure](object-detection/rfdetr-detection-segmentation/README.md) |
| `object-detection/single-stream-object-detector` | [Run and configure](object-detection/single-stream-object-detector/README.md) |
| `object-detection/ssd-mobilenet-object-detector` | [Run and configure](object-detection/ssd-mobilenet-object-detector/README.md) |
| `object-detection/yolo26-object-detector` | [Run and configure](object-detection/yolo26-object-detector/README.md) |
| `pose-estimation/multi-stream-blazepose3d` | [Run and configure](pose-estimation/multi-stream-blazepose3d/README.md) |
| `pose-estimation/multi-stream-pose-estimator` | [Run and configure](pose-estimation/multi-stream-pose-estimator/README.md) |
| `segmentation/efficientsam3-promptable-segmenter` | [Run and configure](segmentation/efficientsam3-promptable-segmenter/README.md) |
| `segmentation/single-stream-instance-segmenter` | [Run and configure](segmentation/single-stream-instance-segmenter/README.md) |
| `tracking/multi-stream-tracker` | [Run and configure](tracking/multi-stream-tracker/README.md) |
| `tracking/yolo26-tiny-drone-tracker` | [Run and configure](tracking/yolo26-tiny-drone-tracker/README.md) |
