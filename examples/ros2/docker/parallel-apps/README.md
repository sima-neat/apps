# Parallel ROS 2 applications in Docker

The ROS variant runs the same six application choices with image source and
inference nodes in each container. Install ROS with
`sima-cli neat install ros2@v3.0.0` and follow the shared
[ROS package setup](../../../../ROS2.md#install-and-build) on Modalix first.
Configure and test each application's ROS launch individually before starting
all six. Model paths in these configs are relative to the Apps root, including
any `models/<container>/` subdirectory.

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

Run on the Modalix board, from the Apps root:

```bash
source /usr/local/ros2/setup.bash
source ~/pyneat/bin/activate
export ROS_PACKAGE_PREFIX="$(ros2 pkg prefix neat_apps)"
examples/ros2/docker/parallel-apps/src/common/ros-parallel-apps.sh start
docker logs neat-ros-detection
examples/ros2/docker/parallel-apps/src/common/ros-parallel-apps.sh stop
```

This builds one local image and mounts the Apps bundle, Neat Python environment,
ROS package and system libraries read-only. Namespaces are distinct, for example
`/neat_ros_detection/results` and `/neat_ros_pose/results`. Source and inference
still communicate over image topics. The containers share host networking and
IPC so Fast DDS can communicate through shared memory across containers and
the host, as described in the [Fast DDS Docker guide](https://fast-dds.docs.eprosima.com/en/v2.14.7/docker/shm_docker.html).
Shared-accelerator throughput must be
compared with all six native applications active under the same source rates.
Container execution is not yet validated; Docker is required on the board.

For node responsibilities, settings, topics and extension examples, see the
[ROS node development guide](../../../../support/ros2/README.md). Each container
runs the selected application's nodes; Docker adds no inference node of its own.
Use that application's configuration and topic contract inside its namespace.

## View results and Insight

Enable `ros.visualization.insight.enabled` in each selected app's config and set
its Insight host, video port base and a distinct `channel_base`. The containers
share the host's network, so no separate visualization process is needed.
Use non-overlapping channel ranges for multi-stream containers. Only enable as
many encoders as the board and Insight service can sustain.

The Apps bundle is mounted read-only in these containers. Saved visual files
therefore require `ros.visualization.save.directory` to point to a writable
container path such as `/tmp/visuals` (temporary), or an explicitly mounted
writable host output directory. Container execution remains unverified.

See the [output configuration reference](../../../../support/ros2/README.md#7-saved-images-and-insight).

## Source files

- [Container lifecycle](src/common/ros-parallel-apps.sh)
- [Container image](src/common/Dockerfile.ros2)
