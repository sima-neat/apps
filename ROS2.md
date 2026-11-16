# ROS 2 examples

Most ROS examples use shared source nodes to publish images, an application-specific
inference node, and ROS result topics. Each ROS example has its own directory in
`examples/ros2/<category>/<example>/`, containing a README, ROS configuration
and `src/python/ros_node.py`. Browse the [ROS example index](examples/ros2/README.md).
The original examples stay in `examples/<category>/<example>/` and remain
independently runnable. The native VLM crop correction is described below.
Shared transport, launch, and recording code lives in `support/ros2/`.
Use the existing `category/example` name to select an application; each example's
README describes its ROS inputs and outputs. The ROS entrypoints are Python and
reuse the native Python model setup and decoding. Compare these ROS examples
with their native Python counterparts for accuracy and FPS acceptance. Native
C++ examples can provide additional diagnostic measurements. PCIe integration is deferred;
the parallel Docker example has a separate ROS launch script.

For node responsibilities, launch/configuration settings, topic contracts and
an implementation walkthrough, read the
[ROS node development guide](support/ros2/README.md). It follows a DETR image
through the source, inference workers and result recorder, then shows how to
create another pipeline or a custom ROS node.

**Development status:** integration and Modalix validation are in progress.
Do not assume every application or source has passed accuracy or FPS parity.
The existing native examples remain independently runnable.

The high-density live-graph mode now reaches 495.40–508.30 aggregate FPS on
twenty 640x480 streams, with exact sampled detections, but remains 3.5–6.6%
below native in paired trials. The no-measurable-slowdown target remains open. BlazePose
now uses the native latest-frame scheduler and matches controlled, paced inputs.
Its tested single 640x480 and 1920x1080 streams maintain 30 FPS, but four-camera throughput
remains below native. Independent live runs can retain different frame histories
and therefore produce different smoothed results. The installed Core's
decoder stream-ID error blocks the default native OBB path. A full native OBB
numerical comparison passed with its debug-frame branch enabled; this does not
establish parity for the default path. Studio preserves the tested speech
outputs, but its request transport adds latency and sustained speech throughput
has not yet matched native. Physical MIPI/USB capture and Docker
execution also remain unverified.
Detection-to-VLM now includes the native crop correction for padded decoder
planes. NV12/I420 pixel regression tests pass; earlier matching captions used
this correction, but full caption generation has not been rerun after restoring it.

## Install and build

Install ROS 2 using:

```bash
sima-cli neat install ros2@v3.0.0
```

Use the Neat Library and Python environment required by the application's
existing README. Source both environments in every participating terminal:

```bash
source /usr/local/ros2/setup.bash
source ~/pyneat/bin/activate
```

Build on Modalix from the Apps bundle (`prebuilt-apps/`) or checkout root.
Choose one of the two workflows below. Both install the same shared Python
`neat_apps` package; individual examples do not have separate build targets.
Use the board's Python interpreter that can import both dependencies:

```bash
python -c "import rclpy, pyneat"
```

### Build with colcon

If colcon is installed, use:

```bash
colcon --log-base sandbox/ros2-colcon-log build \
  --base-paths support/ros2 --packages-select neat_apps --merge-install \
  --build-base sandbox/ros2-colcon-build \
  --install-base sandbox/ros2-colcon-install \
  --cmake-args -DPython3_EXECUTABLE="$(command -v python)"
source sandbox/ros2-colcon-install/local_setup.bash
```

### Build with CMake without colcon

The board's ament CMake installation can install this Python package directly;
no C++ compilation or colcon installation is needed:

```bash
cmake -S support/ros2 -B sandbox/ros2-build \
  -DCMAKE_INSTALL_PREFIX="$PWD/sandbox/ros2-install" \
  -DPython3_EXECUTABLE="$(command -v python)"
cmake --build sandbox/ros2-build --target install
source sandbox/ros2-install/share/neat_apps/local_setup.bash
```

In each new terminal, source ROS, the Neat Python environment, and the setup
file from the build workflow you selected. The two workflows use separate build
and installation directories; use one installation consistently during a run.

Rebuild after changing shared nodes, launch files or dependencies under
`support/ros2/`. Edits to an example's `ros_node.py` or YAML only need an app
restart: the launcher loads them from `apps_root`. Model preparation/download
is a separate step described by each example, not part of this package build.

## Run an image-topic example

Prepare the model using the native README, then edit the ROS example's own
`src/common/config.yaml`. Model paths, thresholds, labels and decoding are shared with the
native implementation. Run from the Apps root:

```bash
APP_DIR=examples/ros2/object-detection/yolo26-object-detector
ros2 launch neat_apps app.launch.py \
  app:=object-detection/yolo26-object-detector \
  apps_root:="$PWD" config:="$PWD/$APP_DIR/src/common/config.yaml" \
  source:=folder
```

The folder source sends the original BGR image pixels without resizing. A
small window of outstanding frames overlaps inference; matching result headers
release the next source frames. One background reader decodes the next file
while the current image is published, preserving file order and bounded memory.
The finite launch stops after every image has
produced a result. Use `source:=none` to feed `/neat/image` from another ROS
camera or rosbag, or `source:=live` for the configured RTSP/HTTP/USB/MIPI source.
The high-density detector also supports `source:=live_graph`: its native RTSP
decoder and inference share one graph inside a ROS node, with detections
published on ROS topics. See its [mode and timestamp contracts](examples/ros2/object-detection/high-density-multi-stream-object-detector/README.md#run).
For that mode, add `insight:=true` and enable/configure `output.video_enabled`
and `output.insight` to view video with detection overlays. The launch starts
the shared `insight_sink`, which forwards ROS detections through Neat's metadata
API with the original source timestamps. Image-topic apps now also support saved visual PNGs and rendered Insight video
through `ros.visualization`; see the [output configuration reference](support/ros2/README.md#7-saved-images-and-insight).
The high-density mode keeps its existing separate video/metadata path; see its
[visual run instructions](examples/ros2/object-detection/high-density-multi-stream-object-detector/README.md#view-video-and-detections-in-insight).

Results are `std_msgs/msg/String` messages on `/neat/results` containing JSON:

```json
{
  "schema_version": 1,
  "sequence": 0,
  "header": {
    "stamp": {"sec": 123, "nanosec": 456},
    "frame_id": "camera_optical_frame"
  },
  "data": {
    "width": 640,
    "height": 480,
    "objects": []
  }
}
```

The complete input header is preserved. `sequence` counts published results;
it is not an optical frame name. Each application's `data` uses its native
decoded result schema and coordinate conventions. Dense depth/anomaly outputs
use `sensor_msgs/msg/Image` topics with the matching input header. Relative
depth is not a metric distance in metres.

Start subscribers before a finite run. The recording sink checks sequence gaps
and fails if the expected number of results does not arrive:

```bash
ros2 run neat_apps result_sink --topic /neat/results \
  --count 21 --output sandbox/ros-results.jsonl --timeout 120
```

For dense output, append `--image-topic /neat/depth` (or `/neat/anomaly_map`).
The recorder saves NumPy arrays and header manifests beside the JSONL file and
requires every image header to match a result. Set `ros.image_subscribers: 1`
for this test so the first image waits for DDS discovery.

## Studio requests

The Studio ROS node uses its existing backend, including the same loaded models
and generation settings. Launch with `source:=none`. Publish JSON in
`std_msgs/msg/String` on `/neat/requests`; replies arrive on `/neat/responses`
with the same request `id`. There is one pending or active request by default;
excess requests receive an explicit error.
Use distinct IDs for outstanding requests. Increasing `ros.pending_requests`
allows that many concurrent backend requests, with one worker per active
request. The backend retains control of model scheduling. This overlaps request
transport for multiple clients while keeping the total admitted work bounded.

```json
{
  "id": "question-1",
  "operation": "chat",
  "use_image": false,
  "body": {
    "model": "your-loaded-model",
    "messages": [{"role": "user", "content": "Describe the scene."}],
    "stream": true,
    "temperature": 0
  }
}
```

Set `use_image: true` to snapshot the latest `bgr8` or `rgb8` image from
`/neat/image`; replies include that image's header. Streamed chat replies carry
unchanged backend chunks in `data`, followed by `done: true`.
For `operation: speech`, `body` is the native `/v1/audio/speech` JSON request;
the reply contains `audio_base64`, `content_type` and `metadata` preserving the
backend's timing, engine, voice, language and speed response headers as strings.
For `operation: transcribe`
or `operation: translate` (speech to English),
provide a WAV file as `audio_base64` alongside the request and put the native
transcription fields in `body`. The reply's `data` is the backend JSON object
for `response_format: json` or `verbose_json`, or a string for `response_format: text`.
Start the native Studio backend first and set
`ros.studio_url` if its address differs from the native configuration.
For a continuous camera feed from the common `live_source`, set
`ros.input_policy: latest`: Studio keeps the latest image for the next request
and does not acknowledge each camera frame with an inference result.

Text generation is evaluated in tokens per second and request latency, rather
than image FPS. Matching prompts, images, sampling parameters and model state
is necessary for a meaningful native/ROS comparison.
For speech synthesis, record completed requests, generated audio duration and
wall time with the same text, engine, voice and speed. Piper can generate
different waveforms and durations for repeated identical requests; compare
audio seconds per wall-clock second as well as request rate. Backend timing
headers exclude ROS transport and must not be presented as end-to-end latency.

## ROS settings

The ROS example's own configuration includes an optional `ros` mapping:

These values are application YAML loaded at startup, not dynamically declared
ROS parameters. Restart the nodes after editing them. The
[settings reference](support/ros2/README.md#2-configuration-and-launch-arguments)
lists defaults, consumers and exceptions for custom nodes.

```yaml
ros:
  image_topic: image
  results_topic: results
  input_policy: complete
  pending_frames: 8
  inflight_frames: 4
  source_window: 4
  result_timeout_seconds: 30
  result_subscribers: 0
```

- `complete` fails on ingress overflow. The folder source uses acknowledged
  delivery so it can preserve every image without blocking subscription callbacks.
- `latest` replaces the oldest unadmitted image when the pending queue fills.
  Admitted frames retain their buffers and headers until their outputs return.
- `source_encoding: bgr8` converts a hardware-decoded live frame for applications
  whose native flow uses BGR. Otherwise live sources retain NV12/NV16.
- `source_repeats` explicitly repeats a folder dataset for a throughput run.
  YOLO26 and PCB also honor their native `runtime.num_runs` folder-loop setting.
- `warmup_results` excludes initial completions from reported throughput; it
  does not discard predictions or change the model's warmup policy.
- `result_subscribers: 2` waits for both the common source acknowledgment
  subscriber and a recorder before starting inference. Use it for finite
  recordings; merely starting a recorder process does not complete DDS discovery.
- Live multi-stream launches use one shared source node for the configured cameras. Set
  `pending_frames` at least to the sum of their `source_window` values in
  complete mode. DDS image/result histories are bounded and sized to cover
  the configured pending and in-flight window. `stream_index:=0` selects a single source.
- Multi-stream multi-model launches use `/neat/stream_0/image`,
  `/neat/stream_0/results`, and corresponding namespaces for subsequent streams,
  with a source and inference node for each model.

All queues are bounded. An inference output must carry the submitted Neat
frame ID and timestamp; a missing, duplicate or inconsistent identity fails
the run instead of attaching a guessed ROS header. Subscriber callbacks only
enqueue images. Separate submission and result workers overlap the native
model pipeline.

## Image formats and clocks

`bgr8`/`rgb8` images follow the usual `sensor_msgs/Image` row-stride contract.
The planar adapter uses visible width and height, `step = width`, and an
`nv12` payload of `width * height * 3 / 2` bytes or `nv16` payload of
`width * height * 2` bytes. Dimensions must be even. This planar convention is
for compatible Neat nodes; it is not accepted by every ROS image consumer.
Use BGR for standard consumers. Hardware planes are copied according to their
reported offsets and strides, including padded decoder surfaces.

Folder timestamps are ROS time at publication. Live-source timestamps are ROS
time after decode, not sensor exposure time. External ROS inputs retain their
own timestamp semantics. Processing-rate measurements use a monotonic clock.

## Validation

Compare identical model archives, image pixels, preprocessing, thresholds,
output settings and source rates. Accuracy comparisons use received ROS results
and native decoded values, matched by source image and input header. An empty
scene alone is insufficient evidence. Dense outputs require numerical tensor
comparisons in addition to metadata checks.

Throughput is completed outputs over wall time, excluding model construction
and documented warmup windows. It is not reciprocal latency or synthetic model
benchmark FPS. Record source publications, admissions, completions, drops and
subscriber receipts separately. Repeat native and ROS runs in alternating order.
The acceptance target is no measurable slowdown; no 5% allowance applies.
Use the native Python implementation as the baseline for each Python ROS example.
For live inputs, allow startup buffers to drain before measuring steady throughput;
report source-limited rates separately from maximum processing capacity.

Runtime evidence and disposable model/cache files belong outside this repository.
See [CONTRIBUTING.md](CONTRIBUTING.md) for native build and validation requirements.

The shared transport tests run on a development host with NumPy and pytest:

```bash
python -m pytest -q tests/ros2/test_transport.py
```

After sourcing ROS and the Neat Python environment on Modalix, run the lifecycle
tests against real ROS messaging and Neat tensor allocations:

```bash
python -m pytest -q tests/ros2
```

These tests cover image layout, queue overflow, frame identity, missing results,
and shutdown. Their controlled inference runner does not establish model
accuracy or FPS parity; those require the model comparisons described above.
