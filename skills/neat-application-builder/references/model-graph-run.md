# Model, Graph, And Run

Use this reference for classic compiled model applications. These are models
packaged for Neat as compiled archives, commonly `.tar.gz` MPK artifacts.

## Model

`Model` loads and validates a compiled model archive. Use its direct execution
methods for caller-owned input, or embed its model route in a larger `Graph`
when the graph owns the source or surrounding stages.

Common application steps:

1. Construct `simaai::neat::Model` or `pyneat.Model` from the model path.
2. Inspect `input_specs()`, `output_specs()`, `metadata()`, and `info()` when the input/output contract is not obvious.
3. Use `model.run(inputs, timeout_ms)` for the shortest synchronous path.
4. Use `model.build(...)` for repeated push/pull workloads that still need only the model route.
5. Use `model.graph()` when the model is one fragment inside a larger `Graph`.

Do not manually recreate the model route unless the public API requires it. Let
`Model` assemble the model-owned preprocess, inference, and postprocess stages.

Read outputs by the contract `output_specs()` reports:

- Outputs carry the model package's logical shape (for example `[1, 1000]` for
  a classifier), and a Float32 output's format tag may be `EVXX_FLOAT32`
  instead of `FP32`. Decode by dtype and shape rather than by format tag.
- A default `nodes::Input` wired straight to one model image input inherits
  that input's image format. Give explicit options when one input feeds
  several models.
- Detection routes take `score_threshold`, `nms_iou_threshold`, and `top_k`
  from `Model::Options`. Set model-appropriate values (for example `0.25`,
  `0.45`, `100`): a zero score threshold keeps zero-score boxes, and a zero
  `top_k` builds only when the model package supplies a cap. Core's
  `docs/reference/boxdecode_decode_types.md` defines each value.

## Graph

`Graph` is the application assembly boundary.

Use `add(...)` for linear chains:

```cpp
simaai::neat::Graph graph("classifier");
graph.add(simaai::neat::nodes::Input("image"));
graph.add(model);
graph.add(simaai::neat::nodes::Output("classes"));
```

Use `connect(...)` for explicit topology:

- connecting reusable fragments
- branching one input to multiple paths
- combining multiple inputs into one output
- wiring named endpoints

Names on `nodes::Input("name")` and `nodes::Output("name")` define public graph
endpoints. A `Graph("name")` constructor label is diagnostic metadata, not a
runtime endpoint.

Prefer application-meaning names:

- `image`
- `left_camera`
- `classes`
- `detections`
- `preview`

Avoid runtime plumbing names such as `appsrc0`, `sink1`, or `out` in customer
application code.

## Run

`Run` is the live handle returned by `Graph::build(...)`. Application code uses
it to push input samples and pull output samples.

Use named endpoints when the graph has more than one public input or output:

```cpp
run.push("image", simaai::neat::TensorList{image_tensor});
auto detections = run.pull("detections", 1000);
```

Unnamed `push(...)` and `pull(...)` are appropriate only when the graph has one
public input or output. If more than one endpoint exists, use the names.

Use `Sample` when the application needs frame IDs, timestamps, bundled values,
or other metadata. Use `TensorList` when raw tensor payloads are enough.

Outputs pulled with the default output memory share runtime buffers. When the
application keeps outputs after the next pull, choose output memory as
[inspect, measure, and tune](inspect-measure-tune.md) describes.
