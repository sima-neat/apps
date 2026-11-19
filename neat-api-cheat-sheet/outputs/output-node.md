# Output

## What it is

`nodes::Output` ends a graph when the application wants to read the results
itself. A graph is the chain of nodes that Core runs. Each result is a
sample: one frame, tensor or set of tensors, with its presentation timestamp
(PTS, the time on the video timeline that the frame belongs to) and its frame
ID. The application reads samples with `Run::pull()`, see
[run (#746)](https://github.com/sima-neat/apps/issues/746).

`OutputOptions` sets how many samples may wait for the application, what
happens when the application pulls slower than the graph produces, and
whether samples are released at video speed.

`nodes::Output("name")` gives the output a name. A graph with several named
outputs is read one output at a time with `run.pull("name", ...)`.

## When to use it

Use `Output` when application code needs the result: it parses model tensors
on the board's CPU, scores an anomaly, counts objects, or builds the metadata
that [MetadataSender](metadata-sender.md) sends to Insight. Insight is the
viewer that shows the video and overlays from the board.

Do not use it for data that only goes to the network. A branch that sends
video to Insight ends in [VideoSender](video-sender.md) and needs no
`Output`. The application does not see those frames.

To end a graph in a GStreamer element that has no Neat node, use
`nodes::Custom(fragment)` (`pyneat.nodes.custom(fragment)`). GStreamer is the
media framework Core builds its pipelines on, and `fragment` is raw GStreamer
launch text, for example `"fakesink sync=false"`. It is an escape hatch:
Core does not check that the elements in the text exist, so a typo fails only
when the graph is built. An empty fragment becomes a pass-through element.
Prefer a typed node when one exists. The only Apps use of `Custom` adds an
extra decode step in
[yolo26-tiny-drone-tracker](../../examples/tracking/yolo26-tiny-drone-tracker/),
not an end node.

## Minimal example

The model branch ends in a named output. The application pulls each
detection sample from it, with a 20 second timeout. The video branch of the
same graph ends in `VideoSender`.

C++

```cpp
simaai::neat::Graph detections_graph("detections");
detections_graph.add(
    simaai::neat::nodes::Output("detections", simaai::neat::OutputOptions::EveryFrame(4)));
runtime.graph.connect(model_graph, detections_graph);

simaai::neat::Sample sample;
simaai::neat::PullError pull_error;
const auto status = runtime.run.pull("detections", 20000, sample, &pull_error);
```

Python

```python
detections_graph = pyneat.Graph("detections")
detections_graph.add(
    pyneat.nodes.output("detections", pyneat.OutputOptions.every_frame(4))
)
graph.connect(model_graph, detections_graph)

sample = runtime.run.pull("detections", 20000)
```

Complete flow: [single-stream-thermal-face-detector](../../examples/face-detection/single-stream-thermal-face-detector/)

## Options

`nodes::Output` has these forms:

| Form | C++ | Python | What it does |
|---|---|---|---|
| Unnamed | `nodes::Output(opt)` | `pyneat.nodes.output(opt)` | Output read with `pull()` without a name. Options default to `OutputOptions{}` |
| Named | `nodes::Output("name", opt)` | `pyneat.nodes.output("name", opt)` | Output read with `pull("name", ...)`. Other fragments can connect to it by name, see [links (#745)](https://github.com/sima-neat/apps/issues/745) |

`OutputOptions` presets. Pick one and change a field only when the preset is
close but not right.

| Preset | C++ | Python | `max_buffers` | `drop` | `sync` | When to use it |
|---|---|---|---|---|---|---|
| Latest | `OutputOptions::Latest()` | `pyneat.OutputOptions.latest()` | `1` | `true` | `false` | Live view. Only the newest result matters |
| Every frame | `OutputOptions::EveryFrame(n)` | `pyneat.OutputOptions.every_frame(max_buffers=n)` | `n`, default `30` | `false` | `false` | The application must see every result, for example to count or record |
| Clocked | `OutputOptions::Clocked(n)` | `pyneat.OutputOptions.clocked(max_buffers=n)` | `n`, default `1` | `true` | `true` | Results must come out at the speed of the video, not as fast as the board can work |

`OutputOptions`

| Option | C++ | Python | Default | What it does | Why and when to change it |
|---|---|---|---|---|---|
| Max buffers | `max_buffers` | `max_buffers` | `4` | Most samples that can wait for `pull()`. `0` means no limit. A negative value counts as `0` | Raise it when the pull loop pauses now and then, for example to write a file, and frames must not be lost. Use `1` for the lowest latency |
| Drop | `drop` | `drop` | `false` | What happens when `max_buffers` samples are waiting. `false`: the graph waits until the application pulls. `true`: the oldest waiting sample is thrown away | Set `true` for a live camera or RTSP stream. With `false`, an application that is slower than the stream sees the delay to live video grow until the stream stalls |
| Sync | `sync` | `sync` | `false` | `true` holds each sample until the pipeline clock reaches its PTS. The pipeline clock is the running time Core uses to schedule media | A video file that should play at its recorded speed. Leave it `false` for live sources, which already arrive at video speed |
| Combine policy | `combine_policy` | `combine_policy` | `CombinePolicy::None` (`pyneat.CombinePolicy.None_`) | How a named output merges samples when two or more fragments connect to it. See the `CombinePolicy` table | Set it only when several producers connect to one output. Otherwise the build fails with the error below |

`CombinePolicy`

| Value | C++ | Python | What it does | When to use it |
|---|---|---|---|---|
| None | `CombinePolicy::None` | `pyneat.CombinePolicy.None_` | One producer only. A second producer fails the build | Default. Keeps two streams from being mixed by accident |
| By frame | `CombinePolicy::ByFrame` | `pyneat.CombinePolicy.ByFrame` | Waits for one sample from each producer with the same frame ID and delivers them together. A sample without a frame ID is an error | A frame and its model result that must stay paired, for example to draw boxes on the right frame |
| By PTS | `CombinePolicy::ByPts` | `pyneat.CombinePolicy.ByPts` | Same as by frame, but matches on PTS | The producers keep PTS but do not carry frame IDs |
| Round robin | `CombinePolicy::RoundRobin` | `pyneat.CombinePolicy.RoundRobin` | Forwards samples from each producer in turn, one at a time, without pairing them | Several independent streams share one output and the application handles them one by one |

[patchcore-anomaly-detector](../../examples/anomaly-detection/patchcore-anomaly-detector/)
sets `combine_policy` on `OutputOptions` directly. Most examples use
`graphs::Combine`, see [Branch and Combine (#745)](https://github.com/sima-neat/apps/issues/745).

## Language differences

- The Python spelling of `CombinePolicy::None` is `CombinePolicy.None_`,
  because `None` is a reserved word in Python.
- `pull()` returns a status and fills a `Sample` in C++. In Python it returns
  the sample, returns `None` on a timeout or when the run is closed, and
  raises on an error. See [run (#746)](https://github.com/sima-neat/apps/issues/746).

## Pitfalls

- **Two producers on one output without a policy.** Symptom: the build fails
  with `public endpoint has multiple producers; set an explicit
  CombinePolicy::ByFrame, CombinePolicy::ByPts, or CombinePolicy::RoundRobin`.
  Fix: set `combine_policy`, or join the producers with `graphs::Combine`.
- **`OutputOptions` replace the run's queue settings for the last node.** When
  the graph ends in an `Output`, its `max_buffers`, `drop` and `sync` decide
  the final queue. `RunOptions::queue_depth` and `overflow_policy` do not. A
  graph with an RTSP input and no `Output` drops old results by default.
  Adding a plain `nodes::Output()` turns that into `max_buffers = 4` and
  `drop = false`, so a slow pull loop now holds up the stream. Fix: use
  `OutputOptions::Latest()`, or set `drop = true`, for live sources.
- **`Clocked()` drops samples.** It sets `drop = true` as well as `sync =
  true`. The header comment names only `sync`. Use
  `EveryFrame(n)` and set `sync = true` yourself when no sample may be lost.
- **A negative `max_buffers` means no limit.** Core turns it into `0`, which
  is an unbounded queue. Memory then grows while the application is slower
  than the graph.

## Used in

- [single-stream-thermal-face-detector](../../examples/face-detection/single-stream-thermal-face-detector/): one named output, `EveryFrame(4)`
- [patchcore-anomaly-detector](../../examples/anomaly-detection/patchcore-anomaly-detector/): `Latest()` with `CombinePolicy::ByFrame`
- [face-recognizer](../../examples/face-recognition/face-recognizer/): two named outputs joined with `graphs::Combine`
- [multi-stream-multi-model](../../examples/multi-model/multi-stream-multi-model/): `drop = true` on a frame output
- [superpoint-feature-extractor](../../examples/feature-extraction/superpoint-feature-extractor/): `max_buffers` set by hand

## Source

- C++ headers: `include/nodes/common/Output.h`, `include/nodes/common/Caps.h` (`nodes::Custom`)
- Python binding: `python/src/module.cpp`, `OutputOptions`, `CombinePolicy`, `nodes.output`, `nodes.custom`
- Core commit: `7e5317be`
