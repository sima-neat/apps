# Queue and OverflowPolicy

## What it is

`nodes::Queue` puts an explicit queue between two stages of a graph. The stage
before the queue and the stage after it then run on separate threads, so a slow
consumer does not stop the producer at once. `QueueOptions` sets how many items
the queue holds and what it does when it is full.

`OverflowPolicy` names the three answers to "the queue is full": wait, drop the
oldest item, or drop the new one. The same policy is used by other bounded
queues in Neat, such as the `Run` input queue.

## When to use it

Add a `Queue` when a fast producer feeds a slower stage and the producer must not
wait for it. The common case is a decoder that feeds a model: with `KeepLatest`
the queue replaces the oldest decoded frame when the model lags, instead of
slowing the decoder and everything that shares it.

Neat has other queues, each set in its own place. Use the matching option
instead of adding a `Queue`:

| Queue | Set it with | Page |
|---|---|---|
| `Run` input and internal runtime queues | `RunOptions::queue_depth` (default `4`) and `RunOptions::overflow_policy` (default `Block`) | [Run](../run/run.md) |
| A public `Output` | `OutputOptions`, or the presets `Latest()`, `EveryFrame(n)`, and `Clocked()` | [Output](../outputs/output-node.md) |
| A link between fragments | `GraphLinkPolicy::RealtimeLatestByStream`, one sample per stream | [links](links.md) |

## Minimal example

A bounded queue in front of a model backbone that keeps the newest frames.

C++

```cpp
neat::QueueOptions opt;
opt.max_buffers = 4;
opt.overflow_policy = neat::OverflowPolicy::KeepLatest;
neat::Graph inference("inference");
inference.add(neat::nodes::Queue(opt));
inference.add(backbone.graph());
```

Python

```python
opt = pyneat.QueueOptions()
opt.max_buffers = 4
opt.overflow_policy = pyneat.OverflowPolicy.KeepLatest
inference = pyneat.Graph("inference")
inference.add(pyneat.nodes.queue(opt))
inference.add(backbone.graph())
```

Complete flow: [rfdetr-detection-segmentation](../../examples/object-detection/rfdetr-detection-segmentation/)

## Options

`QueueOptions`, passed to `nodes::Queue(options)`. `nodes::Queue()` without
options uses the defaults.

| Option | C++ | Python | Default | What it does | Why and when to change it |
|---|---|---|---|---|---|
| Max buffers | `max_buffers` | `max_buffers` | unset: GStreamer's own `queue` limits (200 buffers, 10 MB, 1 s) | When set, the queue holds at most this many items and the buffer count is its only limit. It must be positive | Always set it for decoded frames, so the queue holds a few frames instead of up to a second of video |
| Overflow policy | `overflow_policy` | `overflow_policy` | `OverflowPolicy::Block` | What the queue does when it is full. See the next table | `KeepLatest` for live video where a late frame is worthless |

`OverflowPolicy`

| Value | C++ | Python | What it does | When to use it |
|---|---|---|---|---|
| Block | `OverflowPolicy::Block` | `pyneat.OverflowPolicy.Block` | Waits for space and keeps every item | Files and batches where no item may be lost |
| Keep latest | `OverflowPolicy::KeepLatest` | `pyneat.OverflowPolicy.KeepLatest` | Drops the oldest queued item to make room for the new one. In a `Queue` this is GStreamer `leaky=downstream` | Live cameras and streams, where the newest frame matters most |
| Drop incoming | `OverflowPolicy::DropIncoming` | `pyneat.OverflowPolicy.DropIncoming` | Drops the new item and keeps what is already queued. In a `Queue` this is GStreamer `leaky=upstream` | Items already accepted must be finished, and new ones may be skipped while the stage is busy |

## Language differences

None. Python has `pyneat.nodes.queue()` with and without options, both
`QueueOptions` fields, and all three policy values.

## Pitfalls

- **A queue without options is large.** `nodes::Queue()` keeps GStreamer's
  default limits, up to 200 buffers or one second of data. On a live source a
  full queue with `Block` adds that much delay before the model sees a frame.
  Set `max_buffers` to a small number.
- **`max_buffers` must be positive.** `0` or a negative value fails when the node
  is created with `QueueOptions.max_buffers must be positive when set`. There is
  no unbounded setting. Leave the field unset for GStreamer's defaults.
- **A blocked `Output` is not a `Queue` problem.** When `pull()` stalls after a
  few samples, the full queue is usually the `Output` queue, which holds four
  samples and blocks by default. Change `OutputOptions`, not `QueueOptions`.
- **`GraphLinkOptions::queue_depth` is not a queue.** It is reserved and does
  nothing, see [links](links.md).

## Used in

- [rfdetr-detection-segmentation](../../examples/object-detection/rfdetr-detection-segmentation/) (`Queue` with `KeepLatest` in front of the backbone)

`OverflowPolicy` is also set through `RunOptions::overflow_policy` in most live
examples, for example
[single-stream-object-detector](../../examples/object-detection/single-stream-object-detector/).

## Source

- C++ header: `include/nodes/common/Queue.h` (`Queue`, `QueueOptions`), `include/policy/OverflowPolicy.h` (`OverflowPolicy`)
- Python binding: `python/src/module.cpp`, `pyneat.nodes.queue`, `QueueOptions`, `OverflowPolicy`
- Implementation checked: `src/nodes/common/Queue.cpp` (GStreamer `leaky` mapping, positive `max_buffers`)
- Core commit: `8c9ffead`
