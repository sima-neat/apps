# Model Benchmark — ROS 2

This is an independent ROS 2 package, `neat_model_benchmark`. Build and run it from
this directory. Nodes, launch, configuration, labels and implementation helpers
are local; no `neat_apps` package or original example source is imported.

## Structure and node roles

```text
model-benchmark/
├── package.xml
├── CMakeLists.txt
├── launch/app.launch.py
└── src/
    ├── common/config.yaml
    └── python/
        ├── benchmark_node.py
        └── model_support.py
```

[benchmark_node.py](src/python/benchmark_node.py) declares its ROS interfaces and execution flow directly.
`main()` constructs the Neat model and declares `/neat/report` and the
`/neat/model_benchmark/run` Trigger service. `benchmark_model()` calls
`model.benchmark()` on a background worker. There is no image source.

## Prepare and configure

Install ROS 2 with:

```bash
sima-cli neat install ros2@v3.0.0
```

Use the matching Neat Python environment and installed colcon build tools.
Use the [native model preparation guide](../../../benchmarking/model-benchmark/README.md#prepare-the-model)
for the same model artifacts and precision. Edit [src/common/config.yaml](src/common/config.yaml).
Provide the model package path. This benchmark creates synthetic inputs; no
dataset or label file is required.

| Configuration | Meaning |
| --- | --- |
| `model.path` | Compiled model package |
| `benchmark.frames` | Minimum synthetic samples per pass |
| `benchmark.min_duration_ms` | Minimum duration of each pass |
| `output.report_json` | JSON report file, also published on `report` |
| `ros.decode_type` | Optional native BoxDecode route: `yolo26-det` or `yolo26-seg` |

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
ros2 launch neat_model_benchmark app.launch.py \
  config:="$PWD/src/common/config.yaml"
```

`namespace:=demo` changes the namespace. Ctrl+C stops the launched nodes.
A component failure stops the launch. Omit `config` to use the installed config copy.

## Results and Insight

There is no image or Insight video for a synthetic model report. Read it with
`ros2 topic echo /neat/report --qos-durability transient_local`. Run again with:

```bash
ros2 service call /neat/model_benchmark/run std_srvs/srv/Trigger "{}"
```

## Extend and validate

Edit [benchmark_node.py](src/python/benchmark_node.py) for model behavior and
[launch/app.launch.py](launch/app.launch.py) for wiring. Add executable scripts
to this folder’s `CMakeLists.txt` and dependencies to `package.xml`.

The migration preserves model routes, preprocessing, decoding, queue bounds and
worker topology. Build/import checks alone do not establish accuracy or FPS
parity. See the [validation status](../../README.md#validation-status) for measured
coverage and remaining runtime/hardware limitations. Physical MIPI/USB and
Docker execution remain TODO where applicable.

See the [node development guide](../../NODE_DEVELOPMENT.md) for creating
your own app, and the [ROS overview](../../../../ROS2.md) for validation limits.
