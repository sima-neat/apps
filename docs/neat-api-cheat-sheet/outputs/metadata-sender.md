# MetadataSender

## What it is

`MetadataSender` sends JSON messages over UDP, so Insight can draw results
such as boxes, masks and poses on top of the video. Insight is the viewer that
shows the board's video and overlays. UDP is a network protocol that does not
resend lost packets and does not wait for the receiver.

It is not a graph node. The application creates one sender per stream and
calls `send_metadata()` from its own code, usually right after it pulls a
model result from an [Output](output-node.md). The video of the same stream
goes to Insight through [VideoSender](video-sender.md).

Each message has this shape:

```json
{"type": "object-detection", "data": {...}, "timestamp": 1234, "frame_id": "42"}
```

`type` tells Insight which overlay to draw, and `data` holds the result. The
fields that `data` needs for each type are defined by Insight, see
[Insight: the video viewer](https://github.com/sima-neat/insight/blob/main/docs/user-interface.md).

## When to use it

Use `MetadataSender` when a stream's results should appear in Insight next to
its video. Types that Insight draws include `object-detection`,
`segmentation`, `classification`, `tracking` and `pose-estimation`.
`auxiliary-visualization` fills a separate panel, see
[Insight: auxiliary visualizations](https://github.com/sima-neat/insight/blob/main/docs/auxiliary-visualizations.md).

Do not use it to send video, and do not use it for results the application
keeps for itself. For an offline application that writes files, write the
results to disk instead.

## Minimal example

Create the sender for channel 0 once, then send each pose result stamped with
the frame's timestamp. `sima_examples::require` is a helper in the example
that throws when the condition is false.

C++

```cpp
simaai::neat::MetadataSenderOptions metadata_options;
metadata_options.host = cfg.insight_host;
metadata_options.channel = 0;
metadata_options.metadata_port_base = cfg.metadata_port;
std::string metadata_err;
runtime.metadata_sender =
    std::make_unique<simaai::neat::MetadataSender>(metadata_options, &metadata_err);
sima_examples::require(runtime.metadata_sender->ok(), metadata_err);
runtime.metadata_sender->send_metadata("pose-estimation", poses_json, ts_ms, frame_id_str,
                                       &err);
```

Python

```python
metadata_options = pyneat.MetadataSenderOptions()
metadata_options.host = cfg.insight_host
metadata_options.channel = 0
metadata_options.metadata_port_base = cfg.metadata_port
metadata_sender = pyneat.MetadataSender(metadata_options)
ts_ms = int(sample.pts_ns // 1_000_000) if sample.pts_ns >= 0 else -1
metadata_sender.send_metadata(
    "pose-estimation", build_poses_json(landmarks, scores, label), ts_ms, str(frame_id)
)
```

Complete flow: [single-stream-thermal-face-detector](../../../examples/face-detection/single-stream-thermal-face-detector/)

## Options

`MetadataSenderOptions`

| Option | C++ | Python | Default | What it does | Why and when to change it |
|---|---|---|---|---|---|
| Host | `host` | `host` | `"127.0.0.1"` | Address that receives the messages | Insight runs on another machine. Symptom: video shows in Insight but no overlays |
| Channel | `channel` | `channel` | `0` | Stream number. The UDP port is `metadata_port_base + channel` | Use the same channel as the stream's `VideoSender`, so the overlays land on that stream's tile |
| Metadata port base | `metadata_port_base` | `metadata_port_base` | `9100` | UDP port of channel 0 | Insight listens on other ports, for example the mapped `metadataUDP` range that `neat --json` reports when the sender runs on a DevKit |

`MetadataSenderSendOptions`, the optional second constructor argument

| Option | C++ | Python | Default | What it does | Why and when to change it |
|---|---|---|---|---|---|
| Non-blocking | `nonblocking` | `nonblocking` | `true` | A send that finds the network buffer full fails at once instead of waiting. The message is lost | Keep `true` so a slow network never holds up video or inference. Set `false` only when every message must be tried, even if that slows the loop |

Methods

| Method | C++ | Python | What it does |
|---|---|---|---|
| Send a result | `send_metadata(type, data_json, timestamp_ms, frame_id, &err)` | `send_metadata(type, data_json, timestamp_ms, frame_id)` | Wraps `data_json` in the message shape above and sends it. `timestamp_ms < 0` leaves out `timestamp`; an empty `frame_id` leaves out `frame_id` |
| Send your own JSON | `send_raw_json(payload, &err)` | `send_raw_json(payload)` | Sends `payload` as it is. Use it when you already built the whole message |
| Check the sender | `ok()` | `ok()` | `true` when the address resolved and the socket opened |
| Read the address | `host()`, `metadata_port()`, `nonblocking()` | same names | The values in use |
| Read counters | `stats()` | `stats()` | Returns `MetadataSenderStats`, see below |

`MetadataSenderStats`, a snapshot of counters that is safe to read while
other threads send

| Field | C++ and Python | What it counts |
|---|---|---|
| Attempts | `send_attempts` | Every UDP send call, including failed ones. A message split into several packets counts once per packet |
| Sent | `datagrams_sent` | UDP packets the system accepted |
| Failures | `send_failures` | Failed or partial sends |
| Would block | `would_block` | Failures because the send buffer was full |
| No buffer space | `no_buffer_space` | Failures because the system was out of buffer memory |
| Last duration | `last_send_duration_ns` | Time of the last send, in nanoseconds |
| Longest duration | `max_send_duration_ns` | Longest send so far, in nanoseconds |
| Last error | `last_errno` | System error number of the last failure, `0` after a success |

## Language differences

- **Creating the sender.** C++ does not throw. Check `ok()` and read the
  error string you passed in. Python raises `RuntimeError` with the same
  message.
- **Sending.** C++ returns `false` and fills `err`. Python raises
  `RuntimeError` on an error. A message dropped because the buffer was full is
  not an error: both languages return `false` with no error text.
- `send_metadata()` has no default arguments in Python. Pass `-1` and `""` to
  leave out the timestamp and the frame ID.

## Pitfalls

- **Wall-clock timestamps.** Insight lines up overlays with video frames by
  timestamp, in milliseconds of the source PTS (the time on the video
  timeline that the frame belongs to). A wall-clock time makes the overlays
  drift away from the video. Fix: use `sample.pts_ns / 1000000` of the pulled
  sample, as in the example.
- **Overlays that stay on screen.** When a frame has no message of a type,
  Insight keeps the last overlay of that type for up to 200 ms. Send a message
  with an empty result when nothing was found, so old boxes disappear.
- **Dropped messages are silent.** In non-blocking mode a full buffer makes
  `send_metadata()` return `false` without an error text. Symptom: overlays
  flicker or go missing under load. Check `stats().would_block` and
  `stats().no_buffer_space`.
- **`data_json` must be a JSON object.** A string that does not parse, or an
  array, fails with `MetadataSender data_json parse failed` or
  `MetadataSender data_json must be a JSON object`. An empty `type` fails with
  `MetadataSender metadata type must not be empty`.
- **Size limit.** A message larger than 65507 bytes fails with
  `MetadataSender payload exceeds 65507 bytes`. A message larger than 1200
  bytes is split into several UDP packets with a 12-byte header each. Insight
  joins them again. Another receiver has to do the same.

## Used in

`MetadataSender` is used by most streaming examples, for example:

- [single-stream-thermal-face-detector](../../../examples/face-detection/single-stream-thermal-face-detector/): `pose-estimation`
- [multi-stream-object-detector](../../../examples/object-detection/multi-stream-object-detector/): `object-detection`, one sender per stream

## Source

- C++ header: `include/nodes/io/MetadataSender.h`
- Python binding: `python/src/module.cpp`, `MetadataSenderOptions`, `MetadataSenderSendOptions`, `MetadataSenderStats`, `MetadataSender`
- Core commit: `3f8b8d04`
