# Queue and OverflowPolicy

## What it is

`nodes::Queue` puts an explicit queue between two stages of a graph. The stage
before the queue and the stage after it then run on separate threads, so a slow
consumer does not stop the producer at once. `QueueOptions` sets how many items
the queue holds and what it does when it is full.

`OverflowPolicy` names the answers to "the queue is full": wait, drop the
oldest item, or drop the new one. A fourth value, `Auto`, lets the run preset
choose. The same enum is used by other bounded queues in Neat, such as the
`Run` input queue.

## When to use it

Add a `Queue` when a fast producer feeds a slower stage and the producer must not
wait for it. The common case is a decoder that feeds a model: with `KeepLatest`
the queue replaces the oldest decoded frame when the model lags, instead of
slowing the decoder and everything that shares it.

Neat has other queues, each set in its own place. Use the matching option
instead of adding a `Queue`:

- The `Run` input queue and the queues Neat inserts are set with `RunOptions`,
  and their defaults depend on the run preset. See the Run page of this cheat
  sheet.
- The queue in front of `pull()` is set with `OutputOptions`, see
  [Output](../outputs/output-node.md).
- A link between fragments can keep only the newest frame, see
  [links](links.md).

## Minimal example

A bounded queue in front of a model backbone that keeps the newest frames.

C++

```cpp
neat::QueueOptions newest_frames;
newest_frames.max_buffers = 4;
newest_frames.overflow_policy = neat::OverflowPolicy::KeepLatest;
neat::Graph graph("inference");
graph.add(neat::nodes::Queue(newest_frames));
graph.add(backbone.graph());
```

Python

```python
newest_frames = pyneat.QueueOptions()
newest_frames.max_buffers = 4
newest_frames.overflow_policy = pyneat.OverflowPolicy.KeepLatest
graph = pyneat.Graph("inference")
graph.add(pyneat.nodes.queue(newest_frames))
graph.add(backbone.graph())
```

Complete flow: [rfdetr-detection-segmentation](../../examples/object-detection/rfdetr-detection-segmentation/)

## Options

`QueueOptions`, passed to `nodes::Queue(options)`. `nodes::Queue()` without
options uses the defaults.

| Option | C++ | Python | Default | What it does | Why and when to change it |
|---|---|---|---|---|---|
| Max buffers | `max_buffers` | `max_buffers` | unset: the queue holds up to 200 items, 10 MB, or 1 second of data, whichever comes first | When set, the queue holds at most this many items and the item count is its only limit. It must be positive | Always set it for decoded frames, so the queue holds a few frames instead of up to a second of video |
| Overflow policy | `overflow_policy` | `overflow_policy` | `OverflowPolicy::Block` | What the queue does when it is full. See the next table | `KeepLatest` for live video where a late frame is worthless |

`OverflowPolicy`

| Value | C++ | Python | What it does | When to use it |
|---|---|---|---|---|
| Block | `OverflowPolicy::Block` | `pyneat.OverflowPolicy.Block` | Waits for space and keeps every item | Files and batches where no item may be lost |
| Keep latest | `OverflowPolicy::KeepLatest` | `pyneat.OverflowPolicy.KeepLatest` | Drops the oldest queued item to make room for the new one | Live cameras and streams, where the newest frame matters most |
| Drop incoming | `OverflowPolicy::DropIncoming` | `pyneat.OverflowPolicy.DropIncoming` | Drops the new item and keeps what is already queued | Items already accepted must be finished, and new ones may be skipped while the stage is busy |
| Auto | `OverflowPolicy::Auto` | `pyneat.OverflowPolicy.Auto` | Lets the run preset decide: `KeepLatest` with the Realtime preset, `Block` otherwise. It is the default of `RunOptions::overflow_policy`, where it decides what `Run::try_push()` does on a full input queue and whether the run's own output queue drops results. `push()` always waits, and an `Output` node uses its own `OutputOptions`. A `Queue` set to `Auto` blocks | Leave it on `RunOptions`. On a `Queue`, pick an explicit value |

## Language differences

None. Python has `pyneat.nodes.queue()` with and without options, both
`QueueOptions` fields, and all four policy values.

## Pitfalls

- **A queue without options is large.** `nodes::Queue()` holds up to 200
  items or one second of data. On a live source a
  full queue with `Block` adds that much delay before the model sees a frame.
  Set `max_buffers` to a small number.
- **`max_buffers` must be positive.** `0` or a negative value fails when the node
  is created with `QueueOptions.max_buffers must be positive when set`. There is
  no unbounded setting. Leave the field unset for the default limits.
- **A blocked `Output` is not a `Queue` problem.** When `pull()` stalls after a
  few samples, the full queue is usually the `Output` queue, which holds four
  samples and waits by default. Change `OutputOptions`, not `QueueOptions`.
- **`Auto` on a `Queue` is `Block`.** Only `RunOptions` maps `Auto` to the
  run preset. Set `KeepLatest` explicitly on a live-video `Queue`.
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
- Core commit: `c68a5241`
