# GraphLinkOptions

## What it is

Every `Graph::connect(from, to)` call creates a link between two graph
fragments. `GraphLinkOptions` sets the link policy, which decides what happens
when the receiving side is slower than the sending side: the sender waits, or
the link keeps only the newest frame and drops the older one.

`GraphLinkPolicy` names the two policies. The other fields name the stream
that travels over the link and limit how many decoded frames it may hold.

`connect(from, to)` between two fragments links an output endpoint of `from` to
an input endpoint of `to`. Endpoints are declared with `nodes::Output("name")`
and `nodes::Input("name")`, or by the outputs of `graphs::Branch`. When `from`
has one output and `to` has one input, they are linked whatever their names are.
Otherwise the output and the input with the same name are linked. When neither
rule finds a pair, `connect()` fails with
`cannot infer endpoints between Graph fragments` and lists the names on both
sides. So a branch with the outputs `video` and `model` connects its `model`
output to a fragment that declares `nodes::Input("model")`.

## When to use it

Pass `GraphLinkOptions` as the third argument of `connect()` when the link
carries live video and a late frame is worthless: the branch that feeds a model
from a camera, or the branch that sends video to Insight. With the default
policy a slow branch holds back the source, and through the source every other
branch.

Keep the default policy on a link that carries encoded H.264 or H.265 video
into a decoder whose frames the application processes. Each encoded frame
depends on the frames before it, so a frame dropped on such a link corrupts the
decoded picture until the next keyframe (a frame that decodes on its own). The
[multi-stream-multi-model](../../../examples/multi-model/multi-stream-multi-model/)
example keeps that link lossless for this reason, and still uses a realtime
link for the encoded video it passes to Insight, where a dropped frame affects
only what the viewer shows.

Leave the options out when every sample must arrive: offline processing of
files, or the link from a model to its `Output`.

You do not need it when several live sources connect to the same input (live
fan-in). Core then switches those links to `RealtimeLatestByStream` by itself.

## Minimal example

One live source split into a video branch and a model branch.

C++

```cpp
simaai::neat::GraphLinkOptions live_link_options;
live_link_options.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
runtime.graph.connect(source, branch);
runtime.graph.connect(branch, video_graph, live_link_options);
runtime.graph.connect(branch, model_graph, live_link_options);
runtime.graph.connect(model_graph, detections_graph);
```

Python

```python
live_link_options = pyneat.GraphLinkOptions()
live_link_options.policy = pyneat.GraphLinkPolicy.RealtimeLatestByStream
graph.connect(source, branch)
graph.connect(branch, video_graph, live_link_options)
graph.connect(branch, model_graph, live_link_options)
graph.connect(model_graph, detections_graph)
```

Complete flow: [single-stream-thermal-face-detector](../../../examples/face-detection/single-stream-thermal-face-detector/)

## Options

| Option | C++ | Python | Default | What it does | Why and when to change it |
|---|---|---|---|---|---|
| Policy | `policy` | `policy` | `GraphLinkPolicy::Default` | Chooses how the link behaves when the receiver is slow. See the next table | Set `RealtimeLatestByStream` on live video branches |
| Stream ID | `stream_id` | `stream_id` | `""` | Stable stream identity stamped on samples before realtime scheduling. Empty derives the identity from the source link | Several cameras share a model and you need to tell their results apart, count drops per camera, or keep IDs stable across restarts |
| Inflight per stream | `max_inflight_per_stream` | `max_inflight_per_stream` | `-1` (the framework default, 4) | Maximum decoded frames one stream may have in the graph at once. Applies only to realtime links that carry raw decoder frames | A stream must keep more frames in flight to reach its frame rate, or fewer to leave decoder buffers for others |
| Inflight total | `max_inflight_total` | `max_inflight_total` | `-1` (the per-stream limits added up, at most 8) | Maximum decoded frames all streams together may have in the graph. Applies only to realtime links that carry raw decoder frames | More than two streams share one model: the default total of 8 caps them below their per-stream limits. The pose estimator sets 16 for four streams |

`GraphLinkOptions` also has `queue_depth`, which is kept so that older code
still compiles. It has no effect, see Pitfalls.

`GraphLinkPolicy`

| Value | C++ | Python | What it does | When to use it |
|---|---|---|---|---|
| Default | `GraphLinkPolicy::Default` | `pyneat.GraphLinkPolicy.Default` | Lossless. A slow receiver makes the sender wait (backpressure) | Offline work where no sample may be lost, and links after the model |
| Realtime latest by stream | `GraphLinkPolicy::RealtimeLatestByStream` | `pyneat.GraphLinkPolicy.RealtimeLatestByStream` | The sender never waits. The link keeps only the newest sample per stream, replaces older ones, and serves ready streams fairly | Live video where a fresh frame matters more than every frame: model branches and video to Insight, including encoded video passed through to `VideoSender`. Not on a link into a decoder whose frames you process |

## Language differences

None. Python has every field and both policy values.

## Pitfalls

- **A slow video branch slows the model.** With `Default` on the branch that
  sends video to Insight, a slow network makes the sender wait, which holds back
  the shared source and the model branch with it. Symptom: model frame rate drops
  whenever the viewer or the network lags. Fix: set `RealtimeLatestByStream` on
  that branch.
- **One-to-one links ignore names.** A fragment with one output connects to a
  fragment with one input even when the names differ. Adding a second output
  later turns the same `connect()` into a `cannot infer endpoints` error. Give
  both ends the same name from the start.
- **A realtime link into a decoder corrupts video.** Symptom: smeared or
  blocky frames after every drop, until the next keyframe. Fix: keep the
  default policy on links that carry encoded video into a decoder.
- **`queue_depth` does nothing.** Several examples still set it, but a realtime
  link always keeps one pending sample per stream. Raising it does not add
  buffering. Use [`OutputOptions`](../outputs/output-node.md) on the `Output` or
  `nodes::Queue` for buffering, see [queues and overflow](queues-and-overflow.md).
- **The total limit is lower than it looks.** With the default total of 8, eight
  streams at 4 frames each still share 8 frames in flight. Raise
  `max_inflight_total` together with the stream count.

## Used in

- [single-stream-thermal-face-detector](../../../examples/face-detection/single-stream-thermal-face-detector/)
- [single-stream-object-detector](../../../examples/object-detection/single-stream-object-detector/)
- [multi-stream-pose-estimator](../../../examples/pose-estimation/multi-stream-pose-estimator/) (`stream_id`, `max_inflight_per_stream`, `max_inflight_total`)
- [multi-stream-blazepose3d](../../../examples/pose-estimation/multi-stream-blazepose3d/)
- [multi-stream-multi-model](../../../examples/multi-model/multi-stream-multi-model/) (`stream_id`)
- [yolo26-tiny-drone-tracker](../../../examples/tracking/yolo26-tiny-drone-tracker/)

## Source

- C++ header: `include/pipeline/GraphOptions.h` (`GraphLinkPolicy`, `GraphLinkOptions`), `include/pipeline/Graph.h` (`Graph::connect`)
- Python binding: `python/src/module.cpp`, `GraphLinkPolicy`, `GraphLinkOptions`, `Graph.connect`
