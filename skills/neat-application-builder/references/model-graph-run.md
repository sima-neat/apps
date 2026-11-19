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
3. Use `model.run(inputs)` for the shortest synchronous path.
4. Use `model.build(...)` for repeated push/pull workloads that still need only the model route.
5. Use `model.graph()` when the model is one fragment inside a larger `Graph`.

Do not manually recreate the model route unless the public API requires it. Let
`Model` assemble the model-owned preprocess, inference, and postprocess stages.

For tensors matching `model.input_specs()`, leave preprocessing options unset.
Required MPK Cast/Quant/Tess adapters remain automatic; `preprocess.enable = Off`
does not select MLA-only execution. See Core's
[Model guide](https://github.com/sima-neat/core/blob/develop/docs/develop-apps/development-workflow/model.mdx#raw-tensor-input).

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
- NMS-free decoders such as `YoloV26Obb` require `nms_iou_threshold = 0`. Do
  not carry that zero to a decoder that runs NMS: `YoloV8Obb`, used for YOLOv8
  and YOLO11 oriented boxes, then treats any overlap as a duplicate and drops
  most boxes. `examples/object-detection/multi-stream-obb-detector` switches
  both values with its `model.family` setting.

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

In Python, a timed `Graph` Run pull can return `None`, while `ModelRunner.pull`
can return an empty, false-valued `Sample`. Check for an available payload before
decoding or validating its identity; a timeout is not a completed prediction.

Outputs pulled with the default output memory share runtime buffers. When the
application keeps outputs after the next pull, choose output memory as
[inspect, measure, and tune](inspect-measure-tune.md) describes.

## Images received from ROS

Keep ROS adapters in `examples/ros2/<category>/<example>/`, with their own
configuration, README, package metadata, launch file and local implementation.
Build with `colcon build` inside the example. Keep needed helpers local without
modifying or importing the original example's source or configuration.

The face-detector's local `src/python/image_transport.py` shows the public Python tensor
contract for packed NV12/NV16 messages. Set the visible geometry, plane roles,
offsets and strides on the CPU tensor before calling `tensor.cvu()`. Constructing
an EV74 NV12 tensor before supplying its Y/UV descriptors can fail during the
initial device copy. Check the installed Tensor API and the native BlazePose
example's `nv12_seed` for the matching version.

Keep each image's storage and ROS header until inference completes. Set explicit
`Sample.frame_id` and `pts_ns`, and verify them on output before publishing the
header. Arbitrary `Sample.attributes` are not preserved by every model route;
do not use them as an unverified frame-association mechanism.

The app's local `visual_output.py` can save PNGs and send rendered Insight video
through `ros.visualization.save` and `ros.visualization.insight`. Both default to off.
Keep any required drawing routine local; the output stage receives an owned image
matched to that result before the node publishes its acknowledgment.
Do not mutate the numerical data or draw on a newer frame.
The high-density `live_graph` mode keeps its separate encoded-video/metadata path.
See `examples/ros2/NODE_DEVELOPMENT.md` for channel mapping and custom-node limits, and compare
throughput with identical visualization settings.

For a multi-stage model, device-visible intermediate storage matters separately
from lifetime. The RF-DETR example retains its `ZeroCopy` backbone output through
transformer completion. An `Owned` CPU output is independent storage, but is not
automatically suitable for a downstream EV74 input. Inspect the installed
`OutputMemory` contract and the example's explicit stage bridge before changing
the output memory policy.
