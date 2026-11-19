# Tuning

Each item names what to change and what it affects. The installed headers define every option
and its default; read the header before setting one. Core's
`docs/develop-apps/development-workflow/` pages explain them.

## Find the limit first

- A source sending fewer frames per second than the target caps the result. Check the source
  rate before tuning the application.
- Keeping up with the source is not a capacity measurement. To find capacity, feed more than
  the application can process and count what arrives.
- The same URL repeated does not prove capacity for that many distinct streams. State which
  one you measured.
- One model has one set of hardware lanes. More model graphs or more processes add queues,
  not capacity. Divide the measured total rate by the stream count to size the application,
  and split streams across boards when the total is too low.
- Count results where they arrive. A count of results the application sent overstates the
  rate when a nonblocking sender drops messages.

## Latency or throughput

These lower latency by dropping stale frames. None raises the model's throughput:

- `RunOptions::preset = RunPreset::Realtime`;
- `GraphLinkPolicy::RealtimeLatestByStream` on a stream's link;
- `OutputOptions::Latest()` on an output.

Use them for live views. Keep `OutputOptions::EveryFrame(n)` and plain links on paths that
must process every frame.

## Remove copies

- Feed decoder or camera output straight into the model, as NV12 frames its preprocessing
  accepts, instead of converting frames on the CPU.
- Send video to Insight with `VideoSenderOptions::Passthrough(codec)` from the encoded
  stream. It skips decoding and re-encoding for the display path. Leave
  `VideoSenderOptions::async` at `false`, so the sender stays in the source's pipeline.
- Connect one encoded source to its decoder and to its video sender with plain `connect`
  calls. Neat then runs source, decoder, and sender as one pipeline. A `graphs::Branch`, a
  decoder or sender fed through `nodes::Input`, or a realtime link between source and decoder
  splits it into several pipelines with a hand-off between them.
- Choose `RunOptions::output_memory` as the builder skill's inspect, measure, and tune
  reference describes. An `Owned` result is already a copy: read it in place with
  `map_read()` or `data_ptr<T>()` in C++ and `to_numpy(copy=False)` in Python.
  `copy_payload_bytes()` copies it a second time.
- A `cv::Mat` or NumPy input is copied into device memory when pushed. Create the tensor with
  the `TensorMemory` its first stage runs on: `EV74` for a model with preprocessing, `MLA`
  for direct model input, `CPU` for CPU work.
- Decode detections with `decode_bbox_tensor` rather than parsing the payload bytes yourself.

## Share the model across streams

- Run one model for many streams: connect every stream's decoder to one model graph, in one
  parent graph with one `Run`.
- Use `GraphLinkPolicy::RealtimeLatestByStream` on each stream's link, with a `stream_id`.
- `max_inflight_per_stream` on that link is the live control: `-1` keeps the default of 4.
  Lower it to share the model more evenly between streams.
- Leave `max_inflight_total` at `-1` unless the application needs a hard memory limit.
- `GraphLinkOptions::queue_depth` is reserved. A latest-by-stream link always keeps one
  pending frame per stream.

## Queues, buffers, and placement

Change these only after a measurement points at them, one at a time:

| Option | What it controls |
|---|---|
| `OutputOptions::Latest()`, `EveryFrame(n)`, `max_buffers` | Results waiting on a named `Output` for the application to pull |
| `RunOptions::queue_depth` | The run's input and internal queues. A named `Output` uses its own `max_buffers` instead |
| `GraphOptions::advanced_execution.internal_queue_depth` | Queues between the model's stages; a positive value also adds queues before decoding |
| `SimaDecodeOptions::num_buffers`, `input_buffers` | Decoded frames and encoded input buffers each decoder keeps |
| `SimaDecodeOptions::decoder_tuning`, `memory_opt` | Decoder profile, such as `low-memory` or `throughput-low-latency`, and its lower-memory path |
| `advanced_execution.preprocess_target`, `postprocess_target` | Where the model's pre and post stages run: `AUTO`, `A65`, or `EV74` |
| `RunOptions::startup_preflight` | Whether the run pushes and pulls one test input at startup |

`RtspDecodedInputOptions` carries the same decoder controls under `num_buffers`,
`decoder_input_buffers`, and `decoder_tuning`.

## CPU threads in Python

When per-thread CPU use shows OpenCV or BLAS worker threads competing with the pipeline, set
`OPENBLAS_NUM_THREADS=1` before `import cv2` and call `cv2.setNumThreads(1)`. Read per-thread
CPU use first, with `top -H`.
