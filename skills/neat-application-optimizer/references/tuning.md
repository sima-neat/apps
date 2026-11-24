# Tuning

Each item names what to change and what it affects. The installed headers define every option
and its default, so read the header before setting one. Core's
`docs/develop-apps/development-workflow/graph.mdx` and `pipeline.mdx` explain them. The
`neat-application-builder` skill's graph topologies reference shows how the graphs named here
are built and gives the high-density example's starting values.

## Latency or throughput

The latest-only controls in the builder's graph topologies reference ("Keep every frame or the
newest") lower latency by dropping stale frames. None of them raises the model's throughput.
Use them for live views, and keep every-frame paths where every frame must be processed.

## Remove copies

- Feed decoder or camera output to the model as NV12 frames with the model's NV12
  preprocessing, as the builder's model reference ("Construct and configure") sets it up. That
  moves color conversion and resizing from the CPU to the EV74. A gray-input model is the
  exception that reference names.
- Send video to Insight with `VideoSenderOptions::Passthrough(codec)` from the encoded stream,
  which skips decoding and re-encoding for the display path. The builder's application output
  reference owns the sender's link.
- Connect an encoded source to its decoder with a plain `connect`, so Neat runs the source and
  decoder as one pipeline. A `graphs::Branch`, a decoder fed through `nodes::Input`, or a
  realtime link between source and decoder splits them into separate pipelines with a hand-off
  between them.
- `RunOptions::output_memory = OutputMemory::Owned` copies every result. The builder's graph
  topologies reference ("Output memory") says when a result needs it, and its model reference
  ("Read outputs") says how to read a result once.
- A `cv::Mat` or NumPy input is copied into device memory when pushed. Create the tensor in the
  `TensorMemory` its first stage runs on: `EV74` for a model with preprocessing, `MLA` for
  direct model input, `CPU` for CPU work.

## Keep metadata small

A `send_metadata` call costs more the larger the message: on a DevKit about 9 ms for 37 KB and
3 ms for 10 KB. For many points per frame, send only the fields Insight draws, in whole pixels.

## Share the model across streams

These `GraphLinkOptions` on each stream's `RealtimeLatestByStream` link control how streams
share one model:

- `max_inflight_per_stream`: frames of one stream the model holds at once. `-1` uses Core's
  default of 4. Lower it to share the model more evenly and keep each stream fresh. The
  high-density example uses `1` for 20 and 60 streams.
- `max_inflight_total`: frames of all streams the model holds at once. `-1` uses
  `min(max_inflight_per_stream * stream_count, 8)`. Set it explicitly, as the example does, so
  the cap does not depend on the default. Raise it above 8 when `max_inflight_per_stream *
  stream_count` exceeds 8 and the per-stream rate stays below target while the model has spare
  throughput. The example's 30-stream profile uses `30` with `max_inflight_per_stream = 4`, so
  one frame interval of all 30 streams can be admitted at once.
- `queue_depth` on the link is reserved. A latest-by-stream link always keeps one pending frame
  per stream.

## Queues, buffers, and placement

Change these only after a measurement points at them, one at a time:

| Option | What it controls |
|---|---|
| `OutputOptions::EveryFrame(n)`, `max_buffers` | Results waiting on a named `Output` for the application to pull |
| `RunOptions::queue_depth` | The run's input and internal queues. A named `Output` uses its own `max_buffers` instead. The high-density example keeps it equal to its `EveryFrame(n)` |
| `GraphOptions::advanced_execution.internal_queue_depth` | A positive value adds bounded queues before the CVU, MLA, and decode stages, so model stages overlap. Unset keeps one chain |
| `nodes::Queue` with `QueueOptions::max_buffers` and `overflow_policy` | Frames waiting before the stage that follows the queue |
| `InputOptions::max_bytes` on the shared model's `Input` | Bytes queued at the model's input |
| `SimaDecodeOptions::num_buffers`, `input_buffers` | Decoded frames and encoded input buffers each decoder keeps. The high-density example uses `input_buffers = 2` |
| `SimaDecodeOptions::decoder_tuning`, `memory_opt` | Decoder profile and its lower-memory path. The high-density example uses `throughput-low-latency` with `memory_opt = true` |
| `Model::Options::processcvu.pre_run_target`, `post_run_target` | Where the model's pre and post stages run: `AUTO`, `EV74`, or `A65`. Box decoding always runs on the A65 |
| `RunOptions::startup_preflight` | Whether a build with a first input pushes and pulls it once at startup |

`RtspDecodedInputOptions` carries the same decoder controls as `num_buffers`,
`decoder_input_buffers`, and `decoder_tuning`.

## CPU threads in Python

When per-thread CPU use shows OpenCV or BLAS worker threads competing with the pipeline, set
`OPENBLAS_NUM_THREADS=1` before `import cv2` and call `cv2.setNumThreads(1)`.
