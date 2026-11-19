# Parallel apps in Docker — ROS 2

This directory is the independent `neat_parallel_apps` composition package.
It launches the configured ROS examples in separate containers, using their
own installed packages. It contains no inference node or `neat_apps` dependency.
Docker runtime validation remains TODO; Docker is not installed on the test board.

## Configure

Install ROS 2 with `sima-cli neat install ros2@v3.0.0` and prepare Docker on Modalix.
Edit [src/common/config.yaml](src/common/config.yaml): set the absolute `apps_root`
and choose app paths/package names/container names. Each listed app keeps its
own configuration and model/input paths. Its native accuracy/FPS contract still
applies; running several apps simultaneously introduces resource contention.

First enter each listed app directory, configure its `src/common/config.yaml`
and run `colcon build` there. The launcher checks those local installations.
There is no shared package to build. Model/dataset paths must be available at the
same path inside the containers; the Apps directory and installed runtimes are
mounted. Add explicit mounts to the script if your configured data is elsewhere.

## Build and run

From this composition directory:

```bash
source /usr/local/ros2/setup.bash
source ~/pyneat/bin/activate
colcon build
source install/setup.bash
ros2 launch neat_parallel_apps app.launch.py \
  config:="$PWD/src/common/config.yaml"
```

Ctrl+C stops and removes only the containers started by this launch. Existing
containers with the same names cause an error. To manage detached runs instead:

```bash
bash src/common/ros-parallel-apps.sh start "$PWD/src/common/config.yaml"
bash src/common/ros-parallel-apps.sh stop "$PWD/src/common/config.yaml"
```

## Structure and results

- `launch/app.launch.py` starts the local composition script.
- `src/common/ros-parallel-apps.sh` checks local installations and starts/stops containers.
- `src/common/Dockerfile.ros2` sources each app's own installation and launches it.
- `src/common/config.yaml` selects the apps.

Containers share host networking/IPC for DDS. Each app uses the container name
(with underscores) as its ROS namespace. Inspect `ros2 topic list -t` and the
app-specific results topics. Enable Insight in each app's YAML and assign distinct
channels/ports to prevent collisions. Read that app's README for node roles,
source modes, topics and visualization settings.

## Pending validation

- [ ] Docker build, start, result delivery and Ctrl+C cleanup on Modalix.
- [ ] Shared-memory DDS across containers and simultaneous throughput.
- [ ] Physical MIPI/USB camera tests where those sources are selected.

See the [node development guide](../../NODE_DEVELOPMENT.md) for creating
your own app, and the [ROS overview](../../../../ROS2.md) for validation limits.
