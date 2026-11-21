# Branch and Combine

## What it is

`graphs::Branch` and `graphs::Combine` build the two most common shapes of a
graph. `Branch` sends one input to several outputs (fan-out). `Combine` brings
several inputs into one output (fan-in). Both return an ordinary `Graph`
fragment that you connect like any other, see [links](links.md).

`CombinePolicy` tells a combined output how to match samples from its inputs:
pair them by frame ID or by timestamp into one bundle, forward them one at a
time, or reject more than one producer. A sample is one frame, tensor, or set
of tensors with its frame ID and timestamp. A bundle is a sample whose fields
are the matched samples, see the Run page of this reference.

## When to use it

Use `Branch` when one source feeds more than one consumer: video to Insight and
frames to a model, or frames to a model and to a frame saver.

Use `Combine` with `ByFrame` or `ByPts` when a consumer needs results that
belong together, such as a frame and the detections computed from it. Use
`RoundRobin` when independent live streams share one downstream path and must
not be paired.

For a one-way chain, neither is needed: use `Graph::add()`. You can also build
the same shapes by hand with `nodes::Input`, `nodes::Output`, and
`connect("from", "to")`. Do that only when the output needs options the helpers
do not set, such as a deeper queue on a combined output.

## Minimal example

Split the source, then join the frame and its detections by frame ID. This is
the detector's flow with frame saving turned on.

C++

```cpp
auto branch = simaai::neat::graphs::Branch("source", {"video", "model", "frame"});
runtime.graph.connect(source, branch);
runtime.graph.connect(branch, video_graph);
runtime.graph.connect(branch, model_graph);
runtime.graph.connect(model_graph, detections_graph);
auto joined = simaai::neat::graphs::Combine({"frame", "detections"}, "detector_output",
                                            simaai::neat::CombinePolicy::ByFrame);
runtime.graph.connect(branch, frame_graph);
runtime.graph.connect(frame_graph, joined);
runtime.graph.connect(detections_graph, joined);
```

Python

```python
branch = pyneat.graphs.branch("source", ["video", "model", "frame"])
graph.connect(source, branch)
graph.connect(branch, video_graph)
graph.connect(branch, model_graph)
graph.connect(model_graph, detections_graph)
joined = pyneat.graphs.combine(["frame", "detections"], "detector_output",
                               pyneat.CombinePolicy.ByFrame)
graph.connect(branch, frame_graph)
graph.connect(frame_graph, joined)
graph.connect(detections_graph, joined)
```

Complete flow: [single-stream-object-detector](https://github.com/sima-neat/apps/tree/main/examples/object-detection/single-stream-object-detector)

## Options

| Option | C++ | Python | Default | What it does | Why and when to change it |
|---|---|---|---|---|---|
| Branch input | `Branch(input, ...)` | `branch(input, ...)` | required | Name of the single input endpoint. It also becomes the name of the fragment | Name it after what flows in, such as `source` or `decoded` |
| Branch outputs | `Branch(..., outputs)` | `branch(..., outputs)` | required, at least one | Names of the output endpoints. `connect()` matches them to inputs of the same name downstream | Add one name per consumer |
| Combine inputs | `Combine(inputs, ...)` | `combine(inputs, ...)` | required, at least one | Names of the input endpoints. Upstream outputs with these names connect to them | One name per producer whose samples you combine |
| Combine output | `Combine(..., output, ...)` | `combine(..., output, ...)` | required | Name of the combined output endpoint, the name you pass to `Run::pull()` | Name it after the joined result, such as `detector_output` |
| Combine policy | `Combine(..., policy)` | `combine(..., policy)` | `CombinePolicy::ByFrame` | How samples from the inputs are matched. See the next table | `ByPts` when the producers share timestamps but not frame IDs, `RoundRobin` for independent live streams |

A hand-built fan-in on a plain `nodes::Output` takes the same values through
`OutputOptions::combine_policy`, see [Output](../outputs/output-node.md).

`CombinePolicy`

| Value | C++ | Python | What it does | When to use it |
|---|---|---|---|---|
| None | `CombinePolicy::None` | `pyneat.CombinePolicy.None_` | Accepts one producer. More than one producer fails when the graph is built, with the exception in Pitfalls | Default of a plain `Output`, so an accidental fan-in does not mix samples |
| By frame | `CombinePolicy::ByFrame` | `pyneat.CombinePolicy.ByFrame` | Waits for one sample from every input with the same `Sample::frame_id` and emits them as one bundle | A frame and the results computed from it, as in the detector above |
| By PTS | `CombinePolicy::ByPts` | `pyneat.CombinePolicy.ByPts` | Same, matched on `Sample::pts_ns`, the presentation timestamp | Producers that keep media timestamps but do not carry frame IDs |
| Round robin | `CombinePolicy::RoundRobin` | `pyneat.CombinePolicy.RoundRobin` | Forwards samples from all inputs fairly, one at a time, without bundling | Several independent live streams that share one model path |

## Language differences

- The no-combine value is `CombinePolicy.None_` in Python, because `None` is a
  reserved word.
- Otherwise none. Python has both helpers and all policy values.

## Pitfalls

- **Branches share one buffer.** `Branch` does not copy the frame, so every
  branch sees the same frame buffer. A branch that changes the frame in place
  changes it for the others. Copy the frame before you change it.
- **No silent fallback.** `ByFrame` fails when a sample has no frame ID and
  `ByPts` fails when it has no timestamp. Neither falls back to the other. Keep
  frame IDs when samples pass through your own code.
- **Pairing policies wait for every input.** `ByFrame` and `ByPts` emit only
  when each input delivered a matching sample. Independent cameras never match,
  so pairing them stalls. Use `RoundRobin` for independent streams.
- **Two different defaults.** `Combine()` defaults to `ByFrame`, but
  `OutputOptions::combine_policy` defaults to `None`. A hand-built fan-in
  without a policy fails when the graph is built with
  `public endpoint has multiple producers`, unless every producer connects
  through a `RealtimeLatestByStream` link.
- **A batch pushed before the first pull can stall.** `Combine()` gives its
  output the default `OutputOptions`: four waiting samples, and the graph waits
  when they are full. The input and internal queues hold more samples, so more
  than four inputs can go in before the first pull, but how many depends on the
  graph. Pull while you push, or build the fan-in by hand with
  `OutputOptions::EveryFrame(n)` sized for the batch, see
  [Output](../outputs/output-node.md).
- **Endpoint names must be non-empty and unique.** An empty name fails
  immediately with `endpoint name must not be empty`. A repeated name fails with
  `duplicate endpoint name`.

## Used in

- [single-stream-object-detector](https://github.com/sima-neat/apps/tree/main/examples/object-detection/single-stream-object-detector) (`Branch`, `Combine` with `ByFrame`)
- [single-stream-instance-segmenter](https://github.com/sima-neat/apps/tree/main/examples/segmentation/single-stream-instance-segmenter) (`Branch`, `Combine` with `ByFrame`)
- [face-recognizer](https://github.com/sima-neat/apps/tree/main/examples/face-recognition/face-recognizer) (`Branch`, `Combine` with `ByFrame`)
- [single-stream-thermal-face-detector](https://github.com/sima-neat/apps/tree/main/examples/face-detection/single-stream-thermal-face-detector) (`Branch`)
- [multi-stream-multi-model](https://github.com/sima-neat/apps/tree/main/examples/multi-model/multi-stream-multi-model) (`Branch`)
- [patchcore-anomaly-detector](https://github.com/sima-neat/apps/tree/main/examples/anomaly-detection/patchcore-anomaly-detector) (`Branch`)

`RoundRobin` and `ByPts` are not used in Apps.

## Source

- C++ header: `include/graphs/Fragments.h` (`Branch`, `Combine`), `include/nodes/common/Output.h` (`CombinePolicy`)
- Python binding: `python/src/module.cpp`, `pyneat.graphs.branch`, `pyneat.graphs.combine`, `CombinePolicy`
- Core commit: `c68a5241`
