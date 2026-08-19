# Model, Graph, And Run

Use this reference for classic compiled model applications. These are models
packaged for Neat as compiled archives, commonly `.tar.gz` MPK artifacts.

## Model

`Model` loads and validates a compiled model archive. Use it when application
code supplies decoded images or tensors and the package-owned route is the
complete pipeline. Classification, detection, segmentation, depth, or embedding
describes the model task; it does not decide whether the application needs a
`Model` or an explicit `Graph`.

Common application steps:

1. Construct `simaai::neat::Model` or `pyneat.Model` from the model path.
2. Inspect `input_specs()`, `output_specs()`, `metadata()`, and `info()` when the input/output contract is not obvious.
3. Use `model.run(inputs, timeout_ms)` for the shortest synchronous path.
4. Use `model.build(...)` for repeated push/pull workloads that still need only the model route.
5. Use `model.graph()` when the model is one fragment inside a larger `Graph`.

Do not manually recreate the model route unless the public API requires it. Let
`Model` assemble the model-owned preprocess, inference, and postprocess stages.

## Graph

`Graph` is the application assembly boundary.

Use it when the application owns a source such as a camera, file, or RTSP
stream, or when it explicitly composes decode, preprocessing, multiple model
stages, branching, or output side paths. Keep package-owned stages inside the
model route rather than recreating them as application nodes.

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
