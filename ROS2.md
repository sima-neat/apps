# ROS 2 examples

ROS examples are independent packages under `examples/ros2/<category>/<example>`.
Each owns its node implementations, local helpers, package metadata, configuration
and launch file. Build only the example you need; none imports `neat_apps` or
loads another example's Python source at runtime.

Browse the [example index](examples/ros2/README.md) and the
[node development guide](examples/ros2/NODE_DEVELOPMENT.md).

## Install and build

Use an Apps repository checkout on Modalix. ROS packages build independently
with colcon; the native `build.sh` runtime bundle does not provide their complete
build/launch files.

Install ROS with `sima-cli neat install ros2@v3.0.0`. Use the matching Neat Python
environment and colcon tools installed on Modalix. From the desired example:

```bash
source /usr/local/ros2/setup.bash
source ~/pyneat/bin/activate
python -c "import rclpy, pyneat"
colcon build
source install/setup.bash
```

This builds one package, including all of that example's local implementation.
Rebuild after code/launch changes, and source its installation in every terminal.
Model and dataset preparation is separate from the package build.

### Build with CMake without colcon

If only CMake is available, from the same example directory:

```bash
cmake -S . -B build -DCMAKE_INSTALL_PREFIX="$PWD/install" \
  -DPython3_EXECUTABLE="$(command -v python)"
cmake --build build --target install
source install/local_setup.bash
```

## Run an image-topic example

Edit `src/common/config.yaml` inside the example, then use its package name:

```bash
ros2 launch neat_face_detector app.launch.py \
  config:="$PWD/src/common/config.yaml"
```

The local launch declares the supported `source` choices. `none` waits for an
external image publisher; `folder` or `live` starts a local source node.
High-density defaults to `live_graph`, preserving its complete native camera graph.
All app-specific launch choices, topic names and output settings are documented
in that app's README. Use absolute model/input paths or paths relative to the
example's working directory. Shipped labels and assets are local.

## Image formats and clocks

Messages use source geometry and retain the source header through inference.
Folder inputs use BGR8. Camera models commonly require packed NV12 with visible
width/height, `step=width`, then Y followed by interleaved UV bytes; this planar
convention is not supported by every ROS image viewer. Use each app's documented
input format and compatible reliable QoS. Exact result IDs/timestamps are checked
before publication. Queue bounds and warmup policies match the previous flow.

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
Start the Studio backend from the local example’s `run.sh` first and set
`ros.studio_url` if its address differs from the native configuration.
For a continuous external camera feed, use a source in `latest` mode: Studio
keeps the latest image for the next request and does not acknowledge each
camera frame with an inference result. A local input node from a camera example
can be reused with `ros.input_policy: latest` and its image topic set to `/neat/image`.

Text generation is evaluated in tokens per second and request latency, rather
than image FPS. Matching prompts, images, sampling parameters and model state
is necessary for a meaningful native/ROS comparison.
For speech synthesis, record completed requests, generated audio duration and
wall time with the same text, engine, voice and speed. Piper can generate
different waveforms and durations for repeated identical requests; compare
audio seconds per wall-clock second as well as request rate. Backend timing
headers exclude ROS transport and must not be presented as end-to-end latency.

## Validation

Compare ROS Python with native Python using identical model artifacts, settings,
inputs and output workloads. Measure completed outputs per steady wall-clock
interval, repeat comparisons and exclude startup. The target is no measurable
FPS slowdown and matching numerical output, not a fixed tolerance concession.

Migration validation is recorded in the [example index](examples/ros2/README.md#validation-status).
Earlier measurements are not new measurements of the standalone packages.
Known earlier gaps remain explicit: high-density 20-stream graph mode was 3.5–6.6%
below native (495.40–508.30 aggregate FPS); BlazePose matched single-camera 30 FPS
but four-camera throughput was lower. Its latest-frame smoothing depends on
which frames were admitted. The Core used in those earlier tests rejected the default native OBB
path; only its debug-frame branch has numerical/throughput evidence. Studio
speech quality/output matched, with request throughput tracked separately.

Physical MIPI/USB, Docker, PCIe and dedicated ROS CI are pending. No runtime or
firmware upgrade is part of the package migration. Validation uses repository
checkouts; ROS installation through an Apps runtime artifact is not supported.
