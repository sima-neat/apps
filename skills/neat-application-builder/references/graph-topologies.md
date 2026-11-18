# Graph Topologies

How to choose nodes, connect them, and keep the topology readable. Core's
`docs/develop-apps/development-workflow/` pages and the installed headers define each node
and option.

## Choose the nodes

| Need | Use |
|---|---|
| A compiled model as one stage | `model.graph()`, or `graph.add(model)` in a linear chain |
| A named entry point the application pushes to, or another graph feeds | `nodes::Input("name", options)` |
| A named result the application pulls | `nodes::Output("name", OutputOptions)` |
| A live RTSP stream, still encoded | `nodes::groups::RtspEncodedInput` |
| Hardware decoding of an encoded stream | `nodes::SimaDecode`, followed by `nodes::CapsRaw` to fix the frame format and size |
| Video in Insight | `nodes::groups::VideoSender`. `VideoSenderOptions::Passthrough(codec)` forwards the encoded stream without decoding or re-encoding it |
| Detection boxes in Insight | `MetadataSender`, called by the application for each pulled result |

## Connect them

- Use `add(...)` for a linear chain inside one graph, and `connect(a, b)` to join graphs,
  branch one producer to several consumers, or join several producers into one consumer.
- One producer can feed several consumers: connect the same RTSP input to the decoder and to
  the video sender. Neat reads the stream once.
- Pass `GraphLinkOptions` to `connect` when a link needs a policy. For live streams, use
  `GraphLinkPolicy::RealtimeLatestByStream`, so a slow consumer receives each stream's latest
  frame instead of a growing backlog.

## Shared model across many streams

`examples/object-detection/high-density-multi-stream-object-detector` runs one model for 20 to
60 RTSP streams:

```text
per stream:  RtspEncodedInput ──► decoder graph ──► detector graph (shared) ──► "detections"
                    └──────────► VideoSender (passthrough)
```

- Build the detector graph once: `Input("detector_frame") → model → Output("detections")`.
  Connect every stream's decoder graph to that same detector graph.
- Give each stream's link to the detector a `stream_id`, such as `stream3`. The pulled sample
  carries it, so the application knows which stream a result belongs to.
- Bound the frames the model holds with `max_inflight_per_stream` and `max_inflight_total` on
  that link, so one stream cannot starve the others.
- A detection model's pulled result is a `TensorSet` whose first tensor is the `BBOX` payload,
  not a single `Tensor`. Pass that first tensor to `decode_bbox_tensor`.
- Send each result to the stream's own Insight channel: `channel = stream index`, with video
  and metadata on `port base + channel`.

## Keep the topology readable

- Build the whole graph in one place, the application's `main`, so every `connect` is
  visible together.
- Return each option struct or small subgraph from a function named for it, such as
  `rtsp_options` or `decoder_graph`, and connect the returned value in `main`.
- Create a subgraph once in a variable and connect the variable. Calling the function in
  each `connect` builds two copies.
- Set only the options whose values differ from Core's defaults; the installed header shows
  each default.
