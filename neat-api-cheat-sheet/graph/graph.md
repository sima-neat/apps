# Graph

## What it is

A `Graph` is where an application assembles its pipeline. You add nodes (the
building blocks: an input, a decoder, a model, an output), reusable graph
fragments, and models, and `build()` turns them into a running pipeline. Under
the hood Neat compiles the graph into a GStreamer pipeline, lets GStreamer agree
on formats between neighbouring elements ("caps negotiation"), and returns a
`Run`, the handle you push inputs to and pull outputs from (see
[Run](../run/run.md)).

A `Model` is itself a small graph, so a model can be added to a graph like any
node. `GraphOptions` holds the settings of one graph and is passed to its
constructor.

## When to use it

Use a `Graph` when the application needs more than the model alone: a live
source such as RTSP or a camera, a branch that sends video to Insight next to the
model, several models, or named inputs and outputs.

When the application only runs a model on tensors it already holds, skip the
graph. Use `Model::run()` for one call or `Model::build()` for a reusable runner,
as [depth-estimator](../../examples/depth-estimation/depth-estimator/) does.

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
- `run_rtsp(options)` when the graph ends in H.264 and should be served to RTSP
  clients.

Three calls help when a graph does not behave:

- `inputs()` and `outputs()` list the endpoint names that `Run::push()` and
  `Run::pull()` accept.
- `describe()` prints the nodes as you composed them. `describe_backend()`
  returns the GStreamer launch string the graph compiles to, which you can run
  with `gst-launch-1.0` to tell a Neat issue from a GStreamer issue.
- `validate()` checks the graph contracts without starting GStreamer, for unit
  tests and CI.

`save(path)` and `Graph::load(path)` write the graph to JSON and read it back.
`custom(fragment)` splices a raw GStreamer launch string into the pipeline for a
plugin that Neat does not wrap.

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
| Callback timeout | `callback_timeout_ms` | `callback_timeout_ms` | `1000` | Longest time a user callback, such as the source-mode tensor callback, may take | Your callback does slow work per tensor and the framework intervenes too early |
| Element name prefix | `element_name_prefix` | `element_name_prefix` | `""` | Prepended to every generated GStreamer element name | Several graphs run in one process and their element names collide |
| Element name suffix | `element_name_suffix` | `element_name_suffix` | `""` | Appended to every generated GStreamer element name | Same as the prefix |
| Verbosity | `verbose` | `verbose` | `VerbosityLevel::Production`, progress on, topic flags off | Diagnostic output of this graph | Turn topics on while you debug planning or GStreamer, keep production output in a shipped app |
| CVU target (legacy) | `processcvu_requested_run_target` | not available | `"AUTO"` | Coarse backend for model-managed CVU stages: `"AUTO"`, `"EV74"`, or `"A65"` | Leave it. Use `advanced_execution` |
| CVU placement (legacy) | `processcvu` | not available | `pre_run_target = "AUTO"`, `post_run_target = "AUTO"`, `async = true` | Pre and post stage placement and the async submit path | Leave it. Use `advanced_execution` |
| MLA options (legacy) | `processmla` | not available | `async = true`, `output_pool_buffers = 0` (runtime default 4), `defer_output_invalidate = true` | MLA submit path and output pool size | Leave it. Use `advanced_execution` |
| Prepared runner (retired) | `prepared_runner` | not available | empty | Retired experiment kept so old code compiles | Never. Non-empty retired values fail when the graph is built |
| Internal queue depth (legacy) | `async_queue_depth` | not available | `0` | Depth of queues the framework inserts between stages | Leave it. Use `advanced_execution.internal_queue_depth` |
| A65 workers | `kernel_runner_a65_workers` | `kernel_runner_a65_workers` | `2` | Number of A65 worker threads for numerical kernels, 1 to 64. Created only when the graph has A65 kernels | Numerical stages on A65 are the bottleneck and the CPU has spare cores |
| Advanced execution | `advanced_execution` | `advanced_execution` | every field unset | The preferred way to control execution. An unset field changes nothing. A set field overrides the legacy fields above | Any of the cases in the next table |

`AdvancedExecutionOptions`, the fields of `advanced_execution`. Every field is
optional and unset by default.

| Option | C++ | Python | Default | What it does | Why and when to change it |
|---|---|---|---|---|---|
| Preprocess target | `preprocess_target` | `preprocess_target` | unset (`"AUTO"`) | Backend for model-managed preprocessing: `"AUTO"`, `"A65"`, or `"EV74"`. An unsupported explicit placement is an error | Measure both when the CPU or EV74 is saturated |
| Postprocess target | `postprocess_target` | `postprocess_target` | unset (`"AUTO"`) | Backend for model-managed post adapters. It does not move box decoding, which always runs on A65 | The thermal face detector sets `"EV74"` to keep post adapters off the CPU |
| Preprocess async | `preprocess_async` | `preprocess_async` | unset (`true`) | Asynchronous submit path for CVU stages | Set `false` to force the synchronous path while you isolate a problem |
| Inference async | `inference_async` | `inference_async` | unset (`true`) | Asynchronous submit path for MLA stages | The high-density detector sets it explicitly for throughput. Set `false` to isolate a problem |
| Inference output buffers | `inference_output_buffers` | `inference_output_buffers` | unset (runtime default 4) | Size of the MLA output buffer pool | Downstream stages keep outputs longer and the MLA stalls waiting for a free buffer |
| Defer output cache sync | `defer_output_cache_sync` | `defer_output_cache_sync` | unset (`true`) | Compatibility field | Never. `false` is rejected |
| Prepared runner | `prepared_runner` | `prepared_runner` | unset | Retired experiment | Never |
| Internal queue depth | `internal_queue_depth` | `internal_queue_depth` | unset (`0`) | Depth of the bounded queues the framework inserts before CVU, MLA, and decode stages in live fan-in. `0` keeps the single-chain path | Many live streams share one model and need buffering between stages. The high-density detector sets it |

## Language differences

- Python has extra names for the same calls: `add_node()`, `add_graph()`,
  `add_model()`, `build_source()`, `run_source()`, and `custom_with_role()`.
- C++ `connect(node, "src_N", graph)` for named pads is not available in Python.
- C++ `validate(opt, cv::Mat)` is not available in Python.
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
- [high-density-multi-stream-object-detector](../../examples/object-detection/high-density-multi-stream-object-detector/) (`advanced_execution.internal_queue_depth`, `inference_async`)
- [multi-stream-multi-model](../../examples/multi-model/multi-stream-multi-model/)
- [mipi-multi-model](../../examples/benchmarking/mipi-multi-model/) (`describe_backend`)
- [face-recognizer](../../examples/face-recognition/face-recognizer/)

Most other live examples build a `Graph` as well.

## Source

- C++ header: `include/pipeline/Graph.h`, `include/pipeline/GraphOptions.h`
- Python binding: `python/src/module.cpp`, `Graph`, `GraphOptions`, `AdvancedExecutionOptions`
- Core commit: `8c9ffead`
