# Application Output

Live results go to Insight, which shows them in a browser: video through the `VideoSender`
graph fragment and results through `MetadataSender`. Offline applications can save results or
show annotated images in a local window. `include/nodes/groups/VideoSender.h`,
`include/nodes/io/MetadataSender.h`, and Core's `video_sender.md` and `metadata_sender.md`
under `docs/develop-apps/advanced-concepts/application-design/` define both senders. Viewer
setup belongs to the Insight documentation or the installed Insight skill.

## Video to Insight

- `VideoSenderOptions::Passthrough(codec)` forwards encoded H.264, H.265, or MJPEG without
  re-encoding. Pass the `RtspCodec` the source uses, the same one as
  `RtspEncodedInputOptions::codec`. Insight plays MJPEG in releases after v0.0.8.
- MJPEG goes out as RTP/JPEG, so each side must be a multiple of 8, at most 2040 pixels, such
  as 1280x720. Another size fails the sender, with `Passthrough` and `FromRaw` alike. When the
  sender's input has a size at build, as decoded frames do, the build throws
  `VideoSender MJPEG: RTP/JPEG carries widths and heights in multiples of 8 up to 2040`. RTSP
  MJPEG passthrough has no size at build, so a caps filter rejects the frames while running.
- `VideoSenderOptions::FromRaw(SimaEncodeOptions{...})` encodes decoded frames at their input
  resolution, as H.264 by default. It replaces the deprecated `H264RtpUdpFromRaw`.
- Set `host`, `channel`, and `video_port_base`. The sender uses port `video_port_base + channel`,
  so give each stream its own channel.
- **Encoded passthrough beside inference:** connect the `RtspEncodedInput` graph to both the
  decoder and the `VideoSender`. With `VideoSenderOptions::async` left `false`, Core runs the
  sender behind a `tee` in the source's pipeline. Choose the link to the sender:
  - A `GraphLinkOptions` with `policy = GraphLinkPolicy::RealtimeLatestByStream` keeps at most
    one pending encoded frame for the sender and replaces it when the network is slow, so the
    preview never holds back the decoder. The high-density example uses it.
  - A plain `connect` sends every encoded frame, and a slow sender can hold back the decoder.
    The RF-DETR example uses it for H.264, H.265, and MJPEG.
- **Decoded frames re-encoded:** connect the decoder to a `FromRaw` sender with a
  `RealtimeLatestByStream` link and small `max_inflight_per_stream` and `max_inflight_total`,
  so the encoder cannot hold the frames the decoder needs. Use it only when the source codec is
  one Insight cannot play.

## Metadata to Insight

- `MetadataSender(MetadataSenderOptions)` is an object the application calls, not a graph node.
  Set `host`, `channel`, and `metadata_port_base`, and create one sender per stream channel.
  Insight pairs metadata channel `N` with video channel `N`.
- **Construction:** in C++, pass an error string to the constructor and check `ok()`. In
  Python, the constructor raises when the sender cannot open.
- **Sending:** `send_metadata(type, data_json, timestamp_ms, frame_id)` builds the envelope
  `{"type", "timestamp", "frame_id", "data"}` from a JSON object string. `send_raw_json(payload)`
  sends an envelope the application built itself, as the high-density example does.
- **Drops and errors:** sends are nonblocking by default. When the socket buffer is full, the
  message is dropped and `stats().send_failures` counts it.
  - C++: the call returns `false`. A drop leaves the error string empty, and a real failure
    fills it. Treat a filled error string as a failure.
  - Python: the call returns `False` only for a drop and raises on a real failure.
  - Report the drop count when the run ends, as both reference examples do.
- A message larger than 1200 bytes is split into chunks that the receiving Insight version
  must reassemble. One message holds at most 65507 bytes.

### Box overlay payload

```json
{"type": "object-detection", "timestamp": 1234567, "frame_id": "42",
 "data": {"objects": [{"id": "obj_1", "label": "car", "confidence": 0.92,
                       "bbox": [120, 80, 96, 64]}]}}
```

- `type` is `object-detection`. Each entry of `data.objects` has a string `id`, the class
  `label`, the `confidence` score, and `bbox` as `[x, y, width, height]` in pixels of the video
  Insight shows. From a decoded `Box`, `bbox` is `[x1, y1, x2 - x1, y2 - y1]`. The boxes match
  the passthrough video only when the decoder outputs frames at the stream's own size.
- `timestamp` lines the boxes up with their video frame. Set it to the pulled `Sample`'s
  `pts_ns / 1000000` (milliseconds). `VideoSender` sends RTP timestamps of `pts` on the 90 kHz
  clock with no offset, and Insight multiplies `timestamp` by 90 and matches it to a video frame
  within 1 ms. A negative or missing `timestamp` sends the boxes without pairing.
- `frame_id` is the `Sample`'s `frame_id` as a string. Insight reports it in its statistics.

## Oriented boxes

Oriented-box decoders emit `BBOX_OBB_V1`, which the axis-aligned `SimaRender` rejects. Draw
oriented boxes in application code or send them as metadata.

## Result lines on stdout

- Once Core starts GStreamer, it forwards stdout and stderr through a detached thread, unless
  `SIMA_GST_SUPPRESS_DEVICE_LOGS=0` is set. When stdout is a pipe or a file, a line still in the
  stdout buffer at exit can be lost, and the exit code is still 0.
- Flush each result line when you print it: `std::endl` in C++, `print(..., flush=True)` in
  Python. This narrows the race but does not close it, so a test that reads the last lines can
  still miss them. `src/gst/GstInit.cpp` (`start_stdio_filter`) holds the forwarding.

## Local display

For a local window, check the selected GStreamer sink or OpenCV backend on the target. The
installed platform documentation covers the display session, DRM, connector, and driver
requirements. For an OpenCV window, handle the close button as well as keyboard exit: check
`getWindowProperty(name, WND_PROP_VISIBLE) < 1` before the next `imshow` call, which would
otherwise recreate a closed window.
