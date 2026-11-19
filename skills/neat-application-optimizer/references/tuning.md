# Tuning

Each item names what to change and what to measure. Core's
`docs/develop-apps/development-workflow/pipeline.mdx` and the installed headers define every
option.

## Remove copies

- Feed decoder output straight into the model, as NV12 frames the model's preprocessing
  accepts, instead of converting frames on the CPU.
- Send video to Insight with `VideoSenderOptions::Passthrough(codec)` from the encoded
  stream. It skips decoding and re-encoding for the display path.
- Pull results with `RunOptions::output_memory = OutputMemory::ZeroCopy` when the application
  reads each result and releases it before the next pull. Holding several zero-copy results
  stalls the producer; use `OutputMemory::Owned` when the application keeps them.
- Decode detections with `decode_bbox_tensor` rather than parsing the payload bytes yourself.

## Share the model across streams

- Run one model for many streams: connect every stream's decoder to one detector graph.
- Use `GraphLinkPolicy::RealtimeLatestByStream` on each stream's link, with a `stream_id`, so
  a busy model receives each stream's latest frame and drops stale ones.
- Bound the frames in the model with `max_inflight_per_stream` and `max_inflight_total` on
  that link. Raise them when the model idles between frames; lower them when one stream
  starves the others.

## Queues and buffers

Change these only after a measurement points at them, one at a time:

| Option | What it bounds |
|---|---|
| `RunOptions::queue_depth` and `OutputOptions::EveryFrame(n)` | Results waiting for the application to pull them |
| `GraphOptions::advanced_execution.internal_queue_depth` | Frames queued between the model's stages |
| `SimaDecodeOptions::num_buffers` | Decoded frames each stream's decoder keeps |

## Pitfalls

- A source sending fewer frames per second than the target caps the result; check the source
  rate before tuning the application.
- More streams than the model's throughput allows lower every stream's rate together.
  Divide the measured total by the stream count before adding streams.
- A test that counts results the application sent, not results received, overstates the
  rate when a nonblocking sender drops messages.
