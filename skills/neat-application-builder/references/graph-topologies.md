# Graph Topologies

How to pick the inputs, combine them with models in a `Graph`, build the `Run`, and choose
the options each step takes. The installed headers under `nodes/`, `nodes/groups/`, and
`pipeline/` define every node and option, and Core's `docs/develop-apps/development-workflow/`
pages explain them.

## Pick the input

| The frames come from | Start the graph with |
|---|---|
| The application, which pushes images or tensors | `nodes::Input("name", options)`, then `graph.build(first_input, run_options)` |
| A live RTSP stream, decoded by Neat | `nodes::groups::RtspDecodedInput` |
| A live RTSP stream you also forward still encoded | `nodes::groups::RtspEncodedInput`, then `nodes::SimaDecode` and `nodes::CapsRaw` |
| A video file | `nodes::groups::VideoInputGroup` |
| Still images | `nodes::groups::ImageInputGroup` or `nodes::StillImageInput` |
| An HTTP MJPEG camera | `nodes::groups::HttpMjpegDecodedInput` |
| A camera on the board | `nodes::CameraInput` or `nodes::UsbCameraInput` |

A graph that starts at a source group produces its own frames: build it with
`graph.build(run_options)` and only pull. A graph that starts at `nodes::Input` waits for
`push()`. Building it with a first input lets Neat derive the frame format from real data.

- **Frame rate:** `source_fps` on a stream input declares the rate the source sends. Set it
  when the stream does not announce one, as some MJPEG cameras do not. To change the rate,
  set `use_videorate` and `video_rate_fps`.
- **Camera queue:** `CameraInputOptions::leaky_queue` defaults to `true` with a depth of 2, so
  a graph slower than the camera drops frames without an error. Set it to `false` on a path
  that must process every frame.
- **Camera into a model:** make the camera the graph's first node and add the model with
  `Model::RouteOptions`: `include_input = false`, and `upstream_name` and `buffer_name` set
  to the camera's `buffer_name`. The camera's frames then reach the model without a copy or
  a CPU conversion.

## Build the graph

- `add(...)` appends a stage to a linear chain: input, model, output.
- `connect(a, b)` joins two graphs. Use it to branch one producer to several consumers, join
  several producers into one consumer, or join subgraphs you built separately.
- `model.graph()` returns the model's own route: preprocessing, inference, and decoding. Add or
  connect it as one stage. Do not rebuild that route by hand.
- End every result the application reads with `nodes::Output("name", OutputOptions)`, and pull
  it by that name.
- Create a subgraph once in a variable and connect the variable. Calling a builder function in
  each `connect` creates two copies.
- A subgraph that another graph connects into starts with
  `connect(nodes::Input("name"), first_node)`. Starting it with `add(Input)` then
  `add(first_node)` fails at build with `internal runtime-port edge is missing ports`.
- End a subgraph that feeds a model with several inputs with an `Output` named after the model
  input it feeds.

## Combine inputs with models

- **One input, one model:** `source → model.graph() → Output`.
- **One input, two consumers:** connect the same source to the model and to a
  `nodes::groups::VideoSender`. Neat reads the stream once and feeds both.
- **Many streams, one model:** build the model graph once and connect each stream's decoder to
  it, all inside one parent graph with one `Run`. Graphs built with separate `build()` calls
  cannot share a model. Give the model's `nodes::Input` the options from
  `model.input_appsrc_options(false)` with `block = true`, so frames wait for the model and
  the dropping happens on the links. Pass `GraphLinkOptions` on each stream's link:
  - `policy = GraphLinkPolicy::RealtimeLatestByStream`, so a busy model receives each
    stream's newest frame and drops stale ones;
  - `stream_id`, such as `stream3`. The pulled `Sample` carries it, so the application knows
    which stream a result belongs to;
  - `max_inflight_per_stream` and `max_inflight_total`, so one stream cannot fill the model.
- **Results joined by frame:** before a `graphs::Combine` with `CombinePolicy::ByFrame` or
  `ByPts`, use plain links. A latest-by-stream link drops frames, and the join then waits
  for frames that never arrive.
- **Two models in sequence:** connect the first model's output graph to the second model's
  input. Check each model's `input_specs()` so the formats match.

## Build and drive the run

1. `graph.build(run_options)` or `graph.build(first_input, run_options)` returns a `Run`.
2. Push images or tensors with `run.push("input name", inputs)` when the graph starts at
   `nodes::Input`.
3. Pull with `run.pull("output name", timeout_ms, ...)`. C++ returns a `PullStatus`; Python
   returns `None` on a timeout or a closed run. Check `run.running()` and `run.last_error()`
   to tell them apart.
4. Close the run with `run.close()` on every exit path, including `SIGINT`.

Pushing from one thread and pulling from another is safe. Several threads pushing into the
same `Run` must take turns, behind one lock.

A detection model's pulled result is a `TensorSet` whose first tensor is the `BBOX` payload,
not a single `Tensor`. Pass that first tensor to `decode_bbox_tensor`.

## Options to pass

| Option struct | Passed to | Controls |
|---|---|---|
| `Model::Options` | `Model` | Preprocessing input format and preset, box decoder, score and NMS thresholds, `top_k` |
| Input group options, such as `RtspEncodedInputOptions` | The input group | URL or path, codec, transport, jitter buffer, frame size and rate |
| `SimaDecodeOptions` | `nodes::SimaDecode` | Codec, output size and rate, decoder buffers |
| `GraphLinkOptions` | `connect` | Link policy, `stream_id`, frames in flight |
| `OutputOptions` | `nodes::Output` | How many results wait to be pulled |
| `GraphOptions` | `Graph` | Execution settings such as `advanced_execution.internal_queue_depth` |
| `RunOptions` | `graph.build` | `preset` (`Realtime` for live input), `queue_depth`, `output_memory` (`ZeroCopy` or `Owned`) |
| `VideoSenderOptions`, `MetadataSenderOptions` | Insight senders | Host, channel, ports; `VideoSenderOptions::Passthrough(codec)` forwards encoded video |

Set only the options the application needs to change. The installed header shows each default.

## Keep the topology readable

- Build the whole graph in one place, so every `add` and `connect` is visible together.
- Return each option struct or small subgraph from a function named for what it returns, such
  as `rtsp_options` or `decoder_graph`.
