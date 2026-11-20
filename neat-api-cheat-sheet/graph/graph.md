# Graph

## What it is

A `Graph` is where an application assembles its pipeline. You add nodes (the
building blocks: an input, a decoder, a model, an output), reusable graph
fragments (small graphs that you insert into a bigger one as one piece), and
models. `build()` turns the graph into a running pipeline and returns a `Run`,
the handle you push inputs to and pull outputs from (see the Run page of this
cheat sheet).

Core runs the pipeline on GStreamer, the open-source media framework that
moves frames between the stages. While it builds, GStreamer agrees on the
frame format between neighbouring stages. This is called caps negotiation.

A `Model` is itself a small graph, so a model can be added to a graph like any
node. `GraphOptions` holds the settings of one graph and is passed to its
constructor.

## When to use it

Use a `Graph` when the application needs more than the model alone: a live
source such as RTSP or a camera, a branch that sends video to Insight next to the
model, several models, or named inputs and outputs.

When the application only runs a model on tensors it already holds, skip the
graph. Use `Model::run()` for one call or `Model::build()` for a reusable runner,
as [depth-estimator](../../examples/depth-estimation/depth-estimator/) does. See
[Model](../model/model.md).

Two ways to compose a graph:

- `add()` appends nodes in a straight line: input, model, output.
- `connect()` builds an explicit topology: branches, joins, and links between
  named endpoints. See [links](links.md) and
  [Branch and Combine](branch-and-combine.md).

Four ways to start it:

- `build(run_options)` when the graph has its own source, such as RTSP or a
  camera, and you only pull.
- `build(inputs, run_options)` when you push inputs. The first input sets the
  formats the pipeline negotiates.
- `run(inputs)` for one synchronous push and pull, in tests and offline tools.
- `run_rtsp(options)` serves a still image as an RTSP stream. It works only on
  a linear graph (one built with `add()`) that contains a `StillImageInput` and
  ends in H.264 encoding with an `h264parse` and `rtph264pay` stage, so it is
  for demos and for testing an RTSP client without a camera. It does not serve
  the output of an arbitrary graph.

Three calls help when a graph does not behave:

- `inputs()` and `outputs()` list the endpoint names that `Run::push()` and
  `Run::pull()` accept.
- `describe()` prints the nodes as you composed them. For a linear graph,
  `describe_backend()` returns the GStreamer launch string, which you can run
  with `gst-launch-1.0` to tell a Neat issue from a GStreamer issue. For a
  graph built with `connect()`, it returns a summary of the planned pipeline
  segments instead, which `gst-launch-1.0` cannot run.
- `validate()` checks the graph's structure without starting it, for unit tests
  and CI. In C++, a linear graph that starts with a pushed `Input` needs
  `validate(options, input)` with a sample input. Python has only
  `validate(options)`, so it cannot validate such a graph.

`save(path)` and `Graph::load(path)` write the graph to JSON and read it back.
`custom(fragment)` splices raw GStreamer launch text into the pipeline for a
GStreamer element that Neat has no node for.

## Minimal example

A linear model graph with one named input and one named output.

C++

```cpp
simaai::neat::Graph graph("model_" + std::to_string(stream.index));
graph.add(simaai::neat::nodes::Input(
    kModelInput, make_nv12_input_options(stream.frame_w, stream.frame_h, stream.output_fps)));
graph.add(*stream.model);
graph.add(simaai::neat::nodes::Output(
    kModelOutput,
    simaai::neat::OutputOptions::EveryFrame(std::max(1, cfg.max_inflight_per_stream))));
```

Python

```python
graph = pyneat.Graph(f"model_{stream.index}")
graph.add(pyneat.nodes.input(
    MODEL_INPUT, make_nv12_input_options(stream.frame_w, stream.frame_h, stream.output_fps)))
graph.add(stream.model)
graph.add(pyneat.nodes.output(
    MODEL_OUTPUT, pyneat.OutputOptions.every_frame(max(1, cfg.max_inflight_per_stream))))
```

Complete flow: [multi-stream-multi-model](../../examples/multi-model/multi-stream-multi-model/)

## Options

`GraphOptions`, passed to the constructor.

| Option | C++ | Python | Default | What it does | Why and when to change it |
|---|---|---|---|---|---|
| Callback timeout | `callback_timeout_ms` | `callback_timeout_ms` | `1000` | How long the source-mode `run()` waits for each output before it tries again. Source-mode `run()` is the form that hands every output tensor to the callback set with `set_tensor_callback()`. It does not limit how long your callback may take | Rarely. A timeout only makes `run()` wait again, so the value changes how often it checks for output, not whether it stops |
| Element name prefix | `element_name_prefix` | `element_name_prefix` | `""` | Prepended to the name of every GStreamer element the graph creates | Several graphs run in one process and their element names collide |
| Element name suffix | `element_name_suffix` | `element_name_suffix` | `""` | Appended to the name of every GStreamer element the graph creates | Same as the prefix |
| Verbosity | `verbose` | `verbose` | `VerbosityLevel::Production`, progress on, topic flags off | Diagnostic output of this graph | Turn topics on while you debug how the graph is planned and built. Keep production output in a shipped app |
| A65 workers | `kernel_runner_a65_workers` | `kernel_runner_a65_workers` | `2` | Number of worker threads for the numerical kernels that run on the A65, the board's Arm CPU cores. 1 to 64. Created only when the graph has such kernels | Numerical stages on the A65 are the bottleneck and the CPU has spare cores |
| Execution controls | `advanced_execution` | `advanced_execution` | every field unset: no effect | Where and how the model stages of the graph run: preprocessing and postprocessing placement, asynchronous submission, the model output buffer pool, and the depth of the queues Neat inserts. Its fields and their defaults are listed on the [Model](../model/model.md#options) page. Set here, a field applies to every model in the graph | The [thermal face detector](../../examples/face-detection/single-stream-thermal-face-detector/) moves postprocessing off the CPU with `postprocess_target = "EV74"`. The EV74 is the board's vision processor. The [high-density detector](../../examples/object-detection/high-density-multi-stream-object-detector/) sets `internal_queue_depth`, which adds bounded queues between the stages of the graph, including between the nodes of a linear graph |

C++ also keeps older execution fields for existing code: `processcvu`,
`processmla`, `async_queue_depth`, `processcvu_requested_run_target`, and
`prepared_runner`. They still take effect, but `advanced_execution` replaces
them in new code and overrides any field it sets. Python has only
`advanced_execution`. `prepared_runner` is a retired experiment: a non-empty
value fails when the graph is built.

## Language differences

- Python has extra names for the same calls: `add_node()`, `add_graph()`,
  `add_model()`, `build_source()`, `run_source()`, and `custom_with_role()`.
- C++ `connect(node, "src_N", graph)` for named pads is not available in Python.
- C++ `validate(options, input)` is not available in Python, so Python cannot
  validate a linear graph that starts with a pushed `Input`.
- C++ `build()` and `run()` take `std::vector<cv::Mat>`, `TensorList`, or
  `Sample`. Python takes a list of NumPy arrays, tensors, or samples, plus
  `copy`, `layout`, and `image_format` arguments, and rejects a single tensor or
  sample that is not in a list.
- `GraphOptions` in Python has only `callback_timeout_ms`,
  `element_name_prefix`, `element_name_suffix`, `verbose`,
  `kernel_runner_a65_workers`, and `advanced_execution`.

## Pitfalls

- **The graph name is not an endpoint.** `pyneat.Graph("detections")` does not
  create a `detections` output, so `run.pull("detections")` fails. Declare the
  endpoint with `nodes::Output("detections")`.
- **A node object can be added once.** Adding the same node twice is rejected.
  Create a new node for a new vertex. Passing an existing node to `connect()`
  reuses it, which is how one output feeds several places.
- **A connected fragment cannot be appended.** `add(fragment)` with a fragment
  that uses `connect()` works only on an empty graph. Compose such fragments
  with `connect()`.
- **Some errors appear on the first push or pull.** A graph whose segments
  depend on the input may build without error and report the build failure on
  the first `push()` or `pull()`.
- **Element names collide across graphs.** Several graphs in one process can
  generate the same GStreamer element names. Set `element_name_prefix` or
  `element_name_suffix`.
- **A graph is not thread-safe.** Build it on one thread, then hand the `Run`
  to the thread that uses it.

## Used in

- [single-stream-object-detector](../../examples/object-detection/single-stream-object-detector/)
- [single-stream-thermal-face-detector](../../examples/face-detection/single-stream-thermal-face-detector/) (`GraphOptions.advanced_execution.postprocess_target`)
- [high-density-multi-stream-object-detector](../../examples/object-detection/high-density-multi-stream-object-detector/) (`advanced_execution.internal_queue_depth`)
- [multi-stream-multi-model](../../examples/multi-model/multi-stream-multi-model/)
- [mipi-multi-model](../../examples/benchmarking/mipi-multi-model/) (`describe_backend`)
- [face-recognizer](../../examples/face-recognition/face-recognizer/)

Most other live examples build a `Graph` as well.

## Source

- C++ header: `include/pipeline/Graph.h`, `include/pipeline/GraphOptions.h`
- Python binding: `python/src/module.cpp`, `Graph`, `GraphOptions`, `AdvancedExecutionOptions`
- Core commit: `c68a5241`
