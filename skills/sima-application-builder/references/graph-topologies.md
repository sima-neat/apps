# Graph Topologies

How to start a `Graph` at its input, combine it with models and outputs, and drive the `Run`.
The installed headers under `nodes/`, `nodes/groups/`, and `pipeline/` define every node and
option and show each default. Set only the options whose default does not fit. Core's
`docs/develop-apps/development-workflow/graph.mdx` and `pipeline.mdx` explain them.

## Pick the input

| The frames come from | Start the graph with |
|---|---|
| The application, which pushes images or tensors | `nodes::Input("name", options)` |
| A live RTSP stream that Neat decodes | `nodes::groups::RtspDecodedInput` |
| A live RTSP stream that is also forwarded still encoded | `nodes::groups::RtspEncodedInput`, then `nodes::SimaDecode` |
| A video file | `nodes::groups::VideoInputGroup` |
| Still images | `nodes::groups::ImageInputGroup` or `nodes::StillImageInput` |
| An HTTP MJPEG camera | `nodes::groups::HttpMjpegDecodedInput` |
| A MIPI camera on the board | `nodes::CameraInput` ([cameras](cameras.md)) |
| A USB webcam | `nodes::UsbCameraInput` ([cameras](cameras.md)) |

## RTSP and decoder options

`RtspEncodedInputOptions` (C++ `nodes::groups`, Python `pyneat.RtspEncodedInputOptions`):

- `url` and `codec` (`RtspCodec::H264`, `H265`, or `MJPEG`) for every stream.
- `tcp` (default `true`) and `latency_ms`, the jitter buffer (default `200`). The high-density
  example sets `latency_ms = 100`.
- `drop_on_latency` (default `true`) drops late packets. Turn it off only when every frame
  matters more than freshness.
- `buffer_mode = "none"` keeps the RTP timestamps as sent. The RF-DETR example sets it for TCP
  sources, because arrival-time corrections drift during high-rate replay.
- Frame rate and caps for H.264:
  - With `auto_caps_from_stream = true` (the default), Neat reads the size and rate from the
    stream and uses `fallback_h264_width` and `fallback_h264_height` when the stream lacks them.
    `fallback_h264_fps` has no effect when `source_fps` or `h264_fps` is set.
  - With `auto_caps_from_stream = false`, Neat takes the rate from `source_fps` or `h264_fps`,
    then `fallback_h264_fps`, and the size from `h264_width` and `h264_height`, then the
    fallback fields. The build fails when the rate or a dimension is still missing.
  - The high-density example sets the fallback fields to the stream's size and rate, turns
    auto caps off when its configuration gives the rate, and leaves them on when it does not.
- `source_fps` declares the rate for H.265 and MJPEG too. Set it when the stream does not
  announce one, as some MJPEG cameras do not.

`SimaDecodeOptions` (Python `pyneat.SimaDecodeOptions`) for `nodes::SimaDecode`:

- `type` is a `SimaDecodeType` (`H264`, `H265`, `MJPEG`, `JPEG`), not an `RtspCodec`. The
  default is `H264`, so set it for every other codec.
- `dec_width`, `dec_height`, and `dec_fps` default to the upstream values. Both examples set
  them to the stream's own size and rate. The model's preprocessing resizes later.
- `next_element` selects the decoder's downstream consumer, `"CVU"` or `"APU"`. The
  high-density example sets `"CVU"`, where its frames go to the model's EV74 preprocessing.
  `decoder_name` names the element and its output buffer. The model's preprocessing reads the
  buffer named by `Model::Options::upstream_name`, `"decoder"` by default.
- `num_buffers`, `input_buffers`, `decoder_tuning`, and `memory_opt` trade memory for
  throughput. The high-density example sets all four for many streams, and the RF-DETR example
  raises only `num_buffers` (to 32) for MJPEG. The `sima-application-optimizer` skill's tuning
  reference covers them.

The high-density decoder subgraph ends with `nodes::CapsRaw("NV12", width, height, fps,
CapsMemory::Any)` before its `Output`, which fixes the format handed to the model.

## Build the graph

- `add(...)` appends a stage to a linear chain.
- `connect(a, b, link_options)` joins two graphs. Use it to send one producer to several
  consumers, join several producers into one consumer, or join subgraphs built separately.
  A plain `connect(a, b)` is lossless, except that Core promotes live fan-in with the default
  policy to latest-by-stream.
- `model.graph()` returns the model's route as one stage. Add or connect it. A default
  `nodes::Input` wired straight to one model image input takes that input's image format. Give
  the `Input` explicit options when it feeds several models.
- End every result the application reads with `nodes::Output("name", OutputOptions)`, and pull
  it by that name.
- Name inputs and outputs for what they carry (`detector_frame`, `detections`). The
  `Graph("name")` label is diagnostic only.
- Create a subgraph once in a variable and connect the variable. A builder function called in
  each `connect` creates two copies.
- **A subgraph that starts at `nodes::Input`:** put the `Input` alone in its own graph and
  `connect` that graph to the first stage. Adding the `Input` and the next stage to one graph
  with `add` fails at build with `internal runtime-port edge is missing ports`. This applies
  only where a graph starts at `Input`, such as the shared detector below. A subgraph that starts
  at another node, such as the decoder subgraph, is built with `add`.
- For a model with several inputs, name each upstream `Output` after the model input it feeds.
  `model.graph().inputs()` lists those names.

## Keep every frame or the newest

Each of these drops stale frames or results in favor of newer ones:

- `GraphLinkPolicy::RealtimeLatestByStream` on a link
- a `nodes::Queue` with `QueueOptions::max_buffers` and `OverflowPolicy::KeepLatest`
- `OutputOptions::Latest()` on an output
- `RunOptions::preset = RunPreset::Realtime`, which sets small queues with `KeepLatest`
  overflow, and which both examples use for live input

Keep plain links and `OutputOptions::EveryFrame(n)`, which holds up to `n` results, on paths
that must process every frame. That includes the encoded link to a `VideoSender` when metadata
pairs with its frames: a realtime link there drops frames of a burst, and their metadata has no
frame in Insight.

Keep a plain link on encoded H.264 or H.265 that goes into a decoder whose frames the
application processes. A realtime link drops encoded frames there, and every frame after a
dropped one decodes corrupted until the next keyframe.

`GraphLinkOptions::queue_depth` has no effect: a realtime link keeps one pending sample per
stream. Do not raise it to add buffering. Use `OutputOptions` on the `Output`, or a
`nodes::Queue` with `max_buffers`. See `GraphLinkOptions` in `include/pipeline/GraphOptions.h`.

## Combine inputs with models

- **One stream, one model:** `RtspEncodedInput → decoder → model.graph() → Output`. To give the
  model the newest frame when inference lags, the RF-DETR example puts a `KeepLatest`
  `nodes::Queue` with `max_buffers = 4` before the model.
- **One stream, model and Insight video:** connect the `RtspEncodedInput` graph both to the
  decoder and to a `VideoSender`. Neat reads the stream once and feeds both.
  [Application output](application-output.md) owns the sender options and the link to it.
- **Many streams, one model:** see [Many streams, one model](#many-streams-one-model).
- **Results joined by frame:** before a `graphs::Combine` with `CombinePolicy::ByFrame` or
  `ByPts`, use plain links. A latest-by-stream link drops frames, and the join then waits for
  frames that never arrive.
- **Two models in sequence:** connect the first model's graph to the second model's graph.
  Check each model's `input_specs()` so the formats match.
- **Two models with host work between them:** the RF-DETR example shows the pattern.
  - The first model runs in the `Graph` and ends at `Output("backbone")`. The second runs as a
    `Model::Runner`, built with a seed of zero tensors shaped by `input_specs()` in
    `TensorMemory::EV74` and with `RunOptions::output_memory = OutputMemory::Owned`.
  - A feeder thread pulls each first-model result from the `Run`, computes on the host, and
    pushes a new `Sample` of kind `TensorSet` to the runner. Copy the frame identity from the
    pulled `Sample` to the pushed one: `frame_id`, `stream_id`, `stream_label`, `input_seq`,
    `orig_input_seq`, `pts_ns`, `dts_ns`, `duration_ns`, and `attributes`.
  - A tensor passed on from the first model may sit in CPU storage (`StorageKind::CpuOwned` or
    `CpuExternal`). Move it with `cvu()` before pushing it to a model that reads device memory.
    A device tensor passes without a copy.
  - The main thread pulls the runner's results. To stop, set the stop flag, call the runner's
    `close_input()` to release a push that is waiting, join the feeder, then close the runner
    and the `Run`.

## Many streams, one model

All streams share one model inside one parent graph with one `Run`. Graphs built with separate
`build()` calls cannot share a model. The high-density example wires it like this (Python, and
`main.cpp` makes the same calls):

```python
# The shared detector: three graphs, joined with connect.
input_options = model.input_appsrc_options(False)
input_options.max_bytes = width * height * 3 // 2 * input_options.pool_max_buffers
frames = pyneat.Graph()
frames.add(pyneat.nodes.input("detector_frame", input_options))
results = pyneat.Graph()
results.add(pyneat.nodes.output("detections", pyneat.OutputOptions.every_frame(queue_depth)))
model_graph = model.graph()
detector = pyneat.Graph()
detector.connect(frames, model_graph)
detector.connect(model_graph, results)

graph = pyneat.Graph(options=graph_options)  # advanced_execution.internal_queue_depth
for index, url in enumerate(urls):
    rtsp = pyneat.groups.rtsp_encoded_input(rtsp_options(url))
    decoder = pyneat.Graph("decoder")  # built with add
    decoder.add(pyneat.nodes.sima_decode(decode_options))
    decoder.add(pyneat.nodes.caps_raw("NV12", width, height, fps, pyneat.CapsMemory.Any))
    decoder.add(pyneat.nodes.output("detector_frame"))
    graph.connect(rtsp, decoder)
    graph.connect(decoder, detector, detector_link(index))  # the same detector every time
    # Insight video: application-output.md owns these options and this link.
    graph.connect(rtsp, pyneat.groups.video_sender(video_options(index)), video_link)
run = graph.build(run_options)
```

- Each decoder subgraph ends at `Output("detector_frame")`, the name of the detector's `Input`.
- Every stream sharing the model needs the same frame size. A stream of another size fails
  `build()` with `input spec mismatch for port 'in' (width)`. Read each stream's size before
  building, as the example does with its RTSP probe, and stop with an error that names the
  streams whose sizes differ.
- `model.input_appsrc_options(False)` has `block = true`, so frames wait at the model's input
  and the dropping happens on the links. `max_bytes` bounds the bytes queued there.
- `detector_link(index)` is a `GraphLinkOptions` with
  `policy = GraphLinkPolicy::RealtimeLatestByStream`, `stream_id = f"stream{index}"`,
  `max_inflight_per_stream`, and `max_inflight_total`. The pulled `Sample` carries the
  `stream_id`, so one pull loop on `detections` routes each result to its stream. With one
  source, the sample carries the decoder name instead, so route it to stream 0.
- `RunOptions` sets `preset = RunPreset::Realtime`, `queue_depth`, and
  `output_memory = OutputMemory::ZeroCopy`.

Starting values from the example's profiles. The example sets `OutputOptions::EveryFrame(n)` on
`detections` and `RunOptions::queue_depth` to the same value, and sets both in-flight caps.
Start from the row closest to your stream count and rate.

| Streams and rate | `max_inflight_per_stream` | `max_inflight_total` | `EveryFrame(n)` and `queue_depth` | `internal_queue_depth` | Decoder `num_buffers` |
|---|---|---|---|---|---|
| 20 at 30 FPS | 1 | 8 | 16 | 1 | 8 |
| 30 at 20 FPS | 4 | 30 | 4 | 2 | 16 |
| 60 at 10 FPS | 1 | 8 | 1 | 2 | 4 |

Changing these values for speed belongs to the `sima-application-optimizer` skill.

## Drive the run

1. `graph.build(run_options)` returns a `Run`. A graph that starts at a source group produces
   its own frames, so the application only pulls. A graph that starts at `nodes::Input` waits
   for `push()`. Building it with a first input, `graph.build(first_input, run_options)`, lets
   Neat derive the frame format from real data.
2. Push with `run.push("input name", inputs)`. Unnamed `push` and `pull` work only when the
   graph has one input or one output.
3. Pull in a loop with a short timeout. Both reference examples use 100 ms.
   - C++: `run.pull("name", timeout_ms, sample, &error)` returns a `PullStatus`.
   - Python: `run.pull("name", timeout_ms)` returns `None` on a timeout and on a closed run.
     After `None`, stop when `run.last_error()` is set or `run.running()` is false, and
     otherwise pull again.

Pushing from one thread and pulling from another is safe. Several threads pushing into the same
`Run` must take turns behind one lock.

### Output memory

Choose `RunOptions::output_memory` by how long the application keeps each result:

- It reads each result and releases it before the next pull: `ZeroCopy`, as both reference
  examples use for their graphs. A held result or open `Mapping` keeps its runtime buffer, so
  holding several stalls the producer.
- It keeps results (frame history, a hand-off to another thread): `Owned`.
  Each pull copies the result into system memory that holds no runtime buffer. The RF-DETR
  transformer runner uses it so that reading results holds none of the transformer's buffers.

### Decoded frames read on the host

When application code reads a decoded frame on the host (a crop for a VLM, an annotated
snapshot), expect decoder padding. A 1080p NV12 frame has 2048-byte rows, and its UV plane
starts after 1088 rows. Pack it with `tensor.contiguous().copy_payload_bytes()`, or read each
plane at its `byte_offset` and `strides_bytes`. A slice of `copy_payload_bytes()` to
`width * height * 3 / 2` bytes corrupts the image. Test host-side frames at 1080p, because
720p rows are unpadded and only the colors shift there.
