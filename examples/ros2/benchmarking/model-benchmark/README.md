# Model Benchmark — ROS 2

ROS nodes for the [Model Benchmark native example](../../../benchmarking/model-benchmark/README.md).
The native implementation remains unchanged; this entrypoint imports its model
and decoding helpers. ROS configuration and launch instructions are local to this folder.

## Prepare the model and inputs

Follow the native [model preparation](../../../benchmarking/model-benchmark/README.md#prepare-the-model)
and [input configuration](../../../benchmarking/model-benchmark/README.md#configure), then set the same
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

The node runs the same synthetic `Model.benchmark()` operation and publishes its report on `/neat/report`. Call `/neat/model_benchmark/run` (`std_srvs/srv/Trigger`) to repeat it. Synthetic FPS is not image-topic throughput.

From the Apps root after sourcing the ROS package:

```bash
APP_DIR=examples/ros2/benchmarking/model-benchmark
ros2 launch neat_apps app.launch.py \
  app:=benchmarking/model-benchmark apps_root:="$PWD" \
  config:="$PWD/$APP_DIR/src/common/config.yaml" source:=none
```

The ROS entrypoint is `src/python/ros_node.py`. Model accuracy and throughput
validation are in progress; an available entrypoint is not a parity claim.

For node responsibilities, settings, topics and extension examples, see the
[ROS node development guide](../../../../support/ros2/README.md). Its standard
image-node settings apply where this example uses the shared `ImageNode`;
custom-node differences are listed separately.

## View results and Insight

This synthetic benchmark produces a report, not camera video or overlays;
Insight visualization does not apply. Read its retained report after launch:

```bash
ros2 topic echo /neat/report --qos-durability transient_local --once
```

For the currently supported complete ROS + Insight example, see
[high-density video and detections](../../object-detection/high-density-multi-stream-object-detector/README.md#view-video-and-detections-in-insight).

## Source files

- [ROS entrypoint](src/python/ros_node.py)
- [ROS configuration](src/common/config.yaml)
- [Shared source, sink and launch package](../../../../support/ros2/)
