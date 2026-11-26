# Model::RouteOptions

## What it is

A model's route is the chain of stages a `Model` builds from its compiled
package (an MPK, `*_mpk.tar.gz`; see [model.md](model.md)). A route has up to
three portions: preprocessing (resize, color conversion and normalization on
the EV74, the board's vision processor, then quantization to the INT8 numbers
the MLA reads and, for some models, tessellation into the MLA's tile layout),
MLA inference on the board's machine-learning accelerator, and
postprocessing (dequantization back to float, or box decoding on the A65, the
board's Arm CPU). `Model::run` and `Model::build` hide the route from you.

`Model::graph()` hands you the route as a `Graph` fragment: a piece of a Neat
pipeline you place inside your own `Graph`, next to a camera or video input,
other models, or outputs to Insight, SiMa's live viewer. `Model::RouteOptions`
(Python `pyneat.ModelRouteOptions`) controls what that fragment includes and
exposes: boundary nodes, its endpoints (the named connection points that
`Graph::connect` binds to its neighbors), the names of the elements it
generates (each stage runs as a named element of GStreamer, the media
framework Neat runs on), and per-route execution settings. `Model::fragment(Stage)` returns one portion of the
route, and `groups::MLA`, `QuantTess` and `DetessDequant` are building blocks
for an application that assembles the stages by hand.

## When to use it

| Situation | Use |
|---|---|
| Your code feeds the model tensors or images and reads the results | `model.run(...)` or `model.build(...)` ([model.md](model.md)). `build` takes `RouteOptions` only for names and execution settings |
| The model is one stage of a larger `Graph`: a camera, RTSP or video input, a branch to Insight, other models | `model.graph()`, or `model.graph(route)` when you need names or execution settings. `Graph::add(model)` adds the same route as `model.graph()` |
| You want a `Graph` that holds only the model and runs on its own | `model.graph(route)` with `include_input` and `include_output` |
| You want to see which stages the route has and where each one runs | `model.execution_plan(route)` |
| You debug or replace one portion of the route | `model.fragment(Stage)` |
| You build the stages yourself, one node at a time | `groups::MLA(model)` with `QuantTess` and `DetessDequant`. Rare: `model.graph()` contains the same stages |

How a `Graph` connects fragments and nodes, with `add` and `connect`, is on
the `graph/` pages. To decode raw detection heads outside the
route, C++ has `stages::BoxDecode`; see [decoding.md](decoding.md).

## Minimal example

C++, from
[`patchcore-anomaly-detector/src/cpp/main.cpp`](../../../examples/anomaly-detection/patchcore-anomaly-detector/src/cpp/main.cpp)
(lines 795–797 and 814–817, de-indented). A branch splits the RTSP source
into `"model"` and `"frame"`. The model's route sits behind a named `Input`
endpoint, and its output feeds the `"embedding"` graph:

```cpp
simaai::neat::Graph model_graph("model");
model_graph.connect(simaai::neat::nodes::Input("model"),
                    model.graph(simaai::neat::Model::RouteOptions{}));
rt.graph = simaai::neat::Graph("patchcore");
rt.graph.connect(source, branch);
rt.graph.connect(branch, model_graph);
rt.graph.connect(model_graph, embedding_graph);
```

Python, from
[`mipi-multi-model/src/python/main.py`](../../../examples/benchmarking/mipi-multi-model/src/python/main.py)
(lines 172–178 and 188–189, de-indented). `CAMERA_BUFFER` is `"camera0"`,
the `buffer_name` of the MIPI camera node. With Insight on, the frames reach
the model through a branch output named `"images"` instead:

```python
route = pyneat.ModelRouteOptions()
route.include_input = False
route.include_output = False
route.upstream_name = "images" if connected_insight else CAMERA_BUFFER
route.buffer_name = CAMERA_BUFFER
route.name_suffix = "_camera0"
route.advanced_execution.preprocess_target = "EV74"
model_graph = model.graph(route)
model_graph.add(pyneat.nodes.output(OUTPUT_NAME))
```

Complete flow: [MIPI Multi-Model](../../../examples/benchmarking/mipi-multi-model/),
[PatchCore Anomaly Detector](../../../examples/anomaly-detection/patchcore-anomaly-detector/)

## Options

`Model::RouteOptions` (Python `pyneat.ModelRouteOptions`) holds every option.
Pass it to `model.graph(route)`, to `build`, or to `execution_plan`. The
execution fields override `Model::Options` for this route only.

| Option | C++ | Python | Default | What it does | Why and when to change it |
|---|---|---|---|---|---|
| Input node | `include_input` | `include_input` | `false` / `False` | Adds a public `nodes::Input` node at the head of the fragment, set up for the model's input. | Set it, with `include_output`, when the model's `Graph` must run on its own, for example to build and test the route without the rest of your pipeline. Leave it off when the model sits behind a camera, decoder or named `Input` in a larger graph. `build` always adds both nodes, whatever you set. |
| Output node | `include_output` | `include_output` | `false` / `False` | Adds a public `nodes::Output` node at the tail. | Same cases as `include_input`. Leave it off and add your own named output, as the MIPI Multi-Model example adds `output("results")`. |
| Separate outputs | `expose_all_outputs` | `expose_all_outputs` | `false` / `False` | Exposes one output endpoint per physical model output buffer, instead of one aggregate output that carries all tensors in one sample. Ignored when the route has one physical output buffer. | Advanced and debug code that must connect to the separate output buffers of a model by name. On a DevKit it changed nothing on YOLO26n and ResNet-50, which pack their outputs into one buffer. |
| Upstream element name | `upstream_name` | `upstream_name` | Empty: `"decoder"`, or the route's own `Input` when `include_input` is set | The name of the element whose buffer the route's first stage reads. The route writes it into that stage's configuration. | A `CameraInput` feeds the route directly: set it to the camera's `buffer_name`, as the Core MIPI camera guide and the MIPI Multi-Model example do. Leave it empty behind a named `Input` endpoint or a decoder, as most examples do. |
| Name suffix | `name_suffix` | `name_suffix` | `""` | Appended to every element name the route generates: `quant_1` became `quant_cam0_2` on a DevKit. | Two routes of one model in one `Graph`, such as one per camera: give each its own suffix so element names, logs and diagnostics tell them apart. `Model::Options::name_suffix` does the same for every route of the model. |
| Output and input name | `buffer_name` | `buffer_name` | `""`: the output endpoint is named after the model's unpacked folder | Names the route's aggregate output endpoint. With `include_input`, it also names the buffer of the added `Input` node. | Set a fixed name when another fragment connects to the model's output by name: the default name changes every run (see Pitfalls). The Core MIPI camera guide sets it to the camera's `buffer_name`, together with `upstream_name`. |
| Build messages | `verbose` | not available | `VerboseOptions{}`: level `Production`, progress messages on | Which messages building the route prints, such as `[1/4] Initializing runtime...` to `[4/4] Graph ready`. It replaces the model's `verbose` for this route. | `VerboseOptions::quiet()` for a headless service whose log should hold only your own messages. In Python these lines cannot be switched off. |
| Pre- and postprocess placement (coarse) | `processcvu_requested_run_target` | not available | `"AUTO"` | Where the model's pre- and postprocessing kernels run: `"AUTO"`, `"EV74"` or `"A65"`. `AUTO` runs a stage on the EV74 whenever the EV74 can run it; only the conversion steps after the MLA, such as `Dequant` and `DetessDequant`, go to the A65 when the A65 can run them. | Prefer `advanced_execution.preprocess_target` and `postprocess_target`, which override it. Keep it only in older C++ code. |

The other execution fields are the same structures as on `Model::Options`:
`processcvu`, `processmla`, `prepared_runner` and `async_queue_depth` (C++
only) and `advanced_execution` (both languages). Their meaning and defaults
are on [model.md](model.md#options). Set here, they change this route only: a
field set in `advanced_execution` wins over the model's value, a non-`AUTO`
`processcvu` target and a positive `processmla.output_pool_buffers` override
the model's, and `async_queue_depth = 0` keeps the model's. On a DevKit,
`preprocess_async = False` on the model and `True` on the route gave an
asynchronous route. The MIPI Multi-Model example sets
`advanced_execution.preprocess_target = "EV74"` on its camera route.

### How an app gets a route

| C++ | Python | Returns |
|---|---|---|
| `model.graph()` | `model.graph()` | The full route with default `RouteOptions`: no `Input` or `Output` node, open named endpoints |
| `model.graph(route)` | `model.graph(options)` or `model.graph_with_options(options)` | The full route, shaped by `route` |
| `graph.add(model)`, `graph.connect(..., model)` | `graph.add(model)`, `graph.add_model(model)`, `graph.connect(..., model)` | Adds the model's route to your `Graph`; `add` uses default `RouteOptions`. See the `graph/` pages |
| `model.fragment(Stage)`, or `preprocess()`, `inference()`, `postprocess()` | `model.fragment(pyneat.ModelStage.*)`, or the same three methods | One portion of the route; see [`Stage` values](#stage-values) |
| `model.backend_fragment(Stage)` | `model.backend_fragment(stage)` | The GStreamer text of one portion, for debugging |
| `model.build(route, ...)` | `model.build_with_route_options(options)`, or `model.build(input, route_options=...)` | A `Model::Runner` with an `Input` and `Output` added; see [model.md](model.md) |
| `model.execution_plan(route, ...)` | `model.execution_plan(route_options=..., run_options=...)`, or with an `input` first | An `ExecutionPlan` of what the matching `build` runs: every node, the unit it runs on, and the tensors between nodes. Python reads it with `to_json()` |

`execution_plan` builds the pipeline, reads the plan and closes the pipeline
again. On a DevKit it took a few hundred milliseconds and printed the
`[1/4]` to `[4/4]` build messages.

### What a route contains

The default route of the two packages on a DevKit, read with
`model.execution_plan()`. Each node's `runs_on` names the unit:

| Model and options | Preprocess | Inference | Postprocess |
|---|---|---|---|
| YOLO26n, default `Model::Options` | `Quant` on the EV74 | MLA | `Dequant` on the A65: six float32 head tensors |
| YOLO26n, BGR image preprocessing with `COCO_YOLO` ([preprocessing.md](preprocessing.md)) and `decode_type = YoloV26` ([decoding.md](decoding.md)) | `Preproc` on the EV74: resize, color conversion, normalization and quantization in one kernel | MLA | `SimaBoxDecode` on the A65 |
| ResNet-50, default `Model::Options` | `Quant` on the EV74 | MLA | `Dequant` on the A65: one `[1, 1000]` float32 tensor |

`describe()` on the fragment lists the same nodes. `include_input` added
`Input` before them and `include_output` added `Output` after them; nothing
else changed. The input endpoint is named after the model's input tensor:
`images` for YOLO26n, `pixel_values` for ResNet-50. Neither package needed
tessellation, so neither route used `QuantTess` or `DetessDequant`.

### `Stage` values

`Model::Stage` (Python `pyneat.ModelStage`) selects the portion that
`fragment(stage)` and `backend_fragment(stage)` return. A fragment carries no
named endpoints, so put it after an `Input` with `add` instead of binding it
by name.

| C++ | Python | Contains | On YOLO26n with default options |
|---|---|---|---|
| `Stage::Preprocess` | `ModelStage.Preprocess` | Everything before the MLA: resize, color conversion, normalization, quantization, tessellation | `Quant` |
| `Stage::Inference` | `ModelStage.Inference` | The MLA stage only. Same as `model.inference()` and `groups::MLA(model)` | `ModelFragment` (the MLA) |
| `Stage::Postprocess` | `ModelStage.Postprocess` | Everything after the MLA: detessellation, dequantization, casts, box decoding | `Dequant`; `SimaBoxDecode` with `decode_type` set |
| `Stage::Full` | `ModelStage.Full` | All three, the same nodes as `model.graph()` | `Quant`, MLA, `Dequant` |

On a DevKit, `Input`, then the `Preprocess`, `Inference` and `Postprocess`
fragments, then `Output` in one `Graph` returned the same six tensors as
`model.run`.

### Hand-built routes: `groups::MLA`, `QuantTess` and `DetessDequant`

Some models want their INT8 input tessellated: rearranged into the MLA's tile
layout. When the compiled MLA program does not tessellate itself, the route
planner puts `QuantTess` before the MLA: one kernel pass that quantizes float
input to INT8 and tessellates it. After the MLA it puts `DetessDequant`: one
pass that undoes the tile layout and converts INT8 back to float. Both are
`processcvu` kernels, the model-managed pre- and postprocessing kernels that
Neat places on the EV74 or the A65 (see `advanced_execution` on
[model.md](model.md#options)).

The [Face Detector](../../../examples/face-detection/face-detector/) loads
its RetinaFace model with `preprocess.kind = InputKind::Tensor` and builds
this chain by hand
([`main.cpp`](../../../examples/face-detection/face-detector/src/cpp/main.cpp)
lines 587–591, [`main.py`](../../../examples/face-detection/face-detector/src/python/main.py)
lines 404–408):

| Node | C++ | Python |
|---|---|---|
| Input | `nodes::Input(model.input_appsrc_options(true))` | `pyneat.nodes.input(model.input_appsrc_options(True))` |
| Quantize and tessellate | `nodes::QuantTess(QuantTessOptions(model))` | `pyneat.nodes.quant_tess(pyneat.QuantTessOptions(model))` |
| MLA | `nodes::groups::MLA(model)` | `pyneat.groups.mla(model)` |
| Detessellate and dequantize | `nodes::DetessDequant(DetessDequantOptions(model))` | `pyneat.nodes.detess_dequant(pyneat.DetessDequantOptions(model))` |
| Output | `nodes::Output()` | `pyneat.nodes.output()` |

The kernels need the model's quantization scale and zero point, its tile
geometry and its buffer counts. The constructors that take a `Model`
(`QuantTessOptions(model)`, `DetessDequantOptions(model)`) copy them from the
stage the model's own route already planned and lock the buffer count to the
model's. They throw when the route has no such stage, so you can only
hand-build stages the planner chose, and `model.graph()` already contains
them. Prefer `model.graph()` in new code.

`groups::MLA(model)` alone is the bare MLA stage. It takes the MLA's raw INT8
input and returns raw INT8 output, so your application must quantize (and
tessellate, if the model needs it) before it and convert the output after it.

The option structs, `QuantTessOptions` and `DetessDequantOptions`
(Python `pyneat.QuantTessOptions` and `pyneat.DetessDequantOptions`), have
these fields. An application takes them from `XOptions(model)` and does not
change them.

| Option | C++ | Python | Default | What it does | Why and when to change it |
|---|---|---|---|---|---|
| Config file | `config_path` | `config_path` | `""` | Path of a kernel configuration JSON file. No effect on `DetessDequant`, which runs only from `compiled_contract`. | Only for a kernel configured outside a `Model`. Use `XOptions(model)` instead. |
| Inline config | `config_json` | `config_json` (a `dict`, or `None`) | Not set (`None`) | Kernel configuration given inline. Wins over `config_path`. No effect on `DetessDequant`. | Same as `config_path`. |
| Upstream element (DetessDequant only) | `upstream_name` | `upstream_name` | `""`; `DetessDequantOptions(model)` sets the model's MLA stage | The MLA element whose output it reads. It has no effect. | Leave what `DetessDequantOptions(model)` sets. |
| Element name | `element_name` | `element_name` | `""`: generated, such as `n2_quanttess`; `DetessDequantOptions(model)` sets the model's stage name | The GStreamer element name. | A fixed name you want to find in `describe_backend()` output and error messages. |
| Compiled contract | `compiled_contract` | not available | Empty; `XOptions(model)` fills it | The model's compiled kernel description: scale, zero point, tile geometry. It replaces the config fields. | Never set it yourself. It is the reason to use `XOptions(model)`. |
| Buffer count | `num_buffers` | `num_buffers` | `0`: the default; `XOptions(model)` sets the model's count | Size of the element's buffer pool. | Leave it. When locked, it must equal `num_buffers_model` and be 4 (asynchronous) or 1 (synchronous), or the node constructor throws. |
| Model buffer count | `num_buffers_model` | `num_buffers_model` | `0`; `XOptions(model)` sets it | The buffer count the model's route uses. | Leave it. |
| Buffer count locked | `num_buffers_locked` | `num_buffers_locked` | `false` / `False`; `XOptions(model)` sets `true` | `true` stops the planner from changing `num_buffers`. `DetessDequant` requires `true`. | Leave it. |
| Model link (QuantTess only, internal) | `model_lineage` | not available | Empty | Internal bookkeeping. | Never. |

## Language differences

- Python's `ModelRouteOptions` has 7 of the 13 fields. `verbose`,
  `processcvu_requested_run_target`, `processcvu`, `processmla`,
  `prepared_runner` and `async_queue_depth` exist only in C++. Python sets
  execution through `advanced_execution`, and cannot silence the build
  messages of a route.
- Python spells `graph(RouteOptions)` also as `graph_with_options(options)`,
  and `build(RouteOptions)` as `build_with_route_options(options)`. Python's
  `Graph` also has `add_model(model)`.
- Python's `execution_plan` has two forms, `execution_plan(route_options=...,
  run_options=...)` and `execution_plan(input, route_options=...,
  run_options=..., copy=False)`. It releases the GIL while it builds. Read the
  result with `to_json()`.
- Python binds only `pyneat.groups.mla(model)` from the model groups. C++ also
  has `groups::Preprocess(model)`, `groups::Postprocess(model)` and
  `groups::Infer(model)` (the same as `MLA`), and `preprocessing`,
  `simple_infer`, `infer` and `postprocessing`, which take an archive path
  and `groups::InferOptions`. In Python, use `model.preprocess()`,
  `model.postprocess()` or `model.fragment(...)`.
- Python's `QuantTessOptions` and `DetessDequantOptions` have no
  `compiled_contract` or `model_lineage`; `XOptions(model)` still fills the
  contract inside. `config_json` takes and returns a `dict`, or `None`.
- A from-Model constructor on a route without that stage throws
  `std::runtime_error` in C++ and `RuntimeError` in Python.

## Pitfalls

- **A default fragment cannot run on its own.** `model.graph().build([x])`
  throws `NeatError` `[misconfig.pipeline_shape] Graph::build(input): missing
  Input; use run() or add Input.` Fix: set `include_input` and
  `include_output`, or put the fragment behind an input in a larger `Graph`.
  With both set, the YOLO26n fragment built and returned its six tensors.
- **The default output name changes every run.** Without `buffer_name`, the
  output endpoint is named after the folder the package was unpacked into,
  for example `tmp_simaai_coprocessing_models_proc_run_zumre3_pkg_482c34f2748b3e24_p_etc`
  on a DevKit. The folder name is random per process. Fix: set `buffer_name`,
  or add your own named `Output`, as the examples do.
- **`groups::MLA` alone wants raw INT8 bytes.** `Input`, `groups.mla(model)`
  and `Output` with a float32 `[1, 640, 640, 3]` YOLO26n tensor failed with
  `NeatError` `[misconfig.caps] ... inference-only expects ... byte-stream
  input for 'quantize_0'; expected 1228800 bytes, got 4915200 bytes`, with
  the hint to insert the model's preprocessing first. Fix: put
  `model.fragment(Stage::Preprocess)` before it and
  `fragment(Stage::Postprocess)` after it, or use `model.graph()`.
- **The from-Model constructors need the stage in the route.** On YOLO26n,
  `QuantTessOptions(model)` threw `RuntimeError: QuantTessOptions(Model):
  resolved model route does not contain QuantTess. Route summary:
  pre=[quant] post=[dequantize].` `DetessDequantOptions(model)` failed the
  same way. Check `model.execution_plan()` first.
- **Default-constructed `DetessDequant` and `QuantTess` cannot run.**
  `nodes.detess_dequant()` throws at once: `DetessDequant: num_buffers must
  be model-managed (use DetessDequantOptions(Model))`. `QuantTess()` is
  created, but the graph build fails because it has no configuration
  ("graph-owned contract compilation requires config_json or
  compiled_contract"). Fix: pass `XOptions(model)`.
- **`build` ignores the boundary flags.** `Model::build` always sets
  `include_input` and `include_output` to `true` before it builds, so setting
  them for `build`, as the Neat GenAI Studio's Supertonic engine does, has no
  effect.

## Used in

- `model.graph()` inside a `Graph`:
  [PatchCore Anomaly Detector](../../../examples/anomaly-detection/patchcore-anomaly-detector/),
  [MIPI Multi-Model](../../../examples/benchmarking/mipi-multi-model/) (Python,
  with route names and placement),
  [High-Density Multi-Stream Object Detector](../../../examples/object-detection/high-density-multi-stream-object-detector/)
  (Python),
  [Multi-Stream BlazePose 3D](../../../examples/pose-estimation/multi-stream-blazepose3d/)
  (Python),
  [RF-DETR Detection and Segmentation](../../../examples/object-detection/rfdetr-detection-segmentation/)
- A `Model` passed to `Graph::add` or `connect`:
  [Face Recognizer](../../../examples/face-recognition/face-recognizer/),
  [Single-Stream Thermal Face Detector](../../../examples/face-detection/single-stream-thermal-face-detector/)
- `RouteOptions` passed to `build`:
  [Depth Estimator](../../../examples/depth-estimation/depth-estimator/),
  [PCB Defect Detector](../../../examples/object-detection/pcb-defect-detector/),
  [YOLO26 Object Detector](../../../examples/object-detection/yolo26-object-detector/),
  [PatchCore Anomaly Detector](../../../examples/anomaly-detection/patchcore-anomaly-detector/),
  [RF-DETR Detection and Segmentation](../../../examples/object-detection/rfdetr-detection-segmentation/),
  [EfficientSAM3 Promptable Segmenter](../../../examples/segmentation/efficientsam3-promptable-segmenter/)
  (C++),
  [Neat GenAI Studio](../../../examples/genai/neat-genai-studio/) (Python)
- `groups::MLA`, `QuantTess` and `DetessDequant`:
  [Face Detector](../../../examples/face-detection/face-detector/)

## Source

- C++ header: `include/model/Model.h`, `Model::RouteOptions`, `Model::Stage`,
  `Model::graph`, `Model::fragment`, `Model::backend_fragment`,
  `Model::build` and `Model::execution_plan`;
  `include/nodes/groups/ModelGroups.h`, `groups::MLA` and its siblings;
  `include/nodes/sima/QuantTess.h`, `QuantTessOptions` and `QuantTess`;
  `include/nodes/sima/DetessDequant.h`, `DetessDequantOptions` and
  `DetessDequant`; `include/pipeline/Graph.h`, `Graph::add(const Model&)`
- Python binding: `python/src/module.cpp`, `ModelRouteOptions`,
  `ModelStage`, `Model.graph`, `Model.graph_with_options`,
  `Model.fragment`, `Model.backend_fragment`, `Model.build_with_route_options`,
  `Model.execution_plan`, `groups.mla`, `QuantTessOptions`,
  `DetessDequantOptions`, `nodes.quant_tess` and `nodes.detess_dequant`
- Core commit: `6ed16525`
