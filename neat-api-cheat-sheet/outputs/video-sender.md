# VideoSender

## What it is

`nodes::groups::VideoSender` ends a graph branch and sends its frames as video
to Insight, or to any other RTP receiver. A node group is a ready-made chain
of nodes that Core builds from one set of options. RTP is the standard packet
format for live video, and it travels over UDP, a network protocol that does
not resend lost packets. Insight is the viewer that shows the board's video
and overlays.

`VideoSenderOptions` has two starting points. `FromRaw` takes raw frames and
encodes them on the board's hardware video encoder. `Passthrough` takes frames
that are already encoded, for example the H.264 frames of an RTSP camera, and
forwards them without decoding and encoding again.

Two nodes from this page can also be used on their own:

- `nodes::SimaEncode` is the hardware encoder. It turns raw frames into H.264,
  H.265 or MJPEG frames and does not send them anywhere. `FromRaw` uses it.
- `nodes::UdpOutput` writes each buffer to a UDP address. An H.264 or H.265
  `VideoSender` ends with it. An MJPEG sender may end in an internal sink
  (`JpegRtpUdpSink`) that packetizes and sends in one step instead.

## When to use it

Use `VideoSender` to show live video in Insight:

- `Passthrough` when the frames reach the sender unchanged from an encoded
  source, such as an RTSP camera. It costs no decode or encode work.
- `FromRaw` when the frames are raw: from a MIPI or USB camera, from a
  decoder, or pushed by the application after it drew on them.

Use `SimaEncode` alone when the application needs the encoded frames itself,
for example to pull them from an [Output](output-node.md). Use `UdpOutput`
alone only when you build your own RTP chain. Metadata such as boxes and
labels goes to Insight through [MetadataSender](metadata-sender.md), not
through `VideoSender`.

## Minimal example

Encode raw frames and send them to Insight on channel 0. In the example a
`Branch` fills `Input("video")` with the frames decoded from the RTSP source,
so the application does not push them itself. `Input` is explained on the
Input page of this cheat sheet.

C++

```cpp
auto video_options =
    simaai::neat::nodes::groups::VideoSenderOptions::FromRaw({.fps = runtime.output_fps});
video_options.host = cfg.insight_host;
video_options.channel = 0;
video_options.video_port_base = cfg.video_port;
simaai::neat::Graph video_graph("video");
video_graph.connect(simaai::neat::nodes::Input("video"),
                    simaai::neat::nodes::groups::VideoSender(video_options));
```

Python

```python
encode = pyneat.SimaEncodeOptions()
encode.fps = fps
video_options = pyneat.VideoSenderOptions.from_raw(encode)
video_options.host = cfg.insight_host
video_options.channel = 0
video_options.video_port_base = cfg.video_port
video_graph = pyneat.Graph("video")
video_graph.connect(
    pyneat.nodes.input("video"), pyneat.groups.video_sender(video_options)
)
```

Complete flow: [single-stream-thermal-face-detector](../../examples/face-detection/single-stream-thermal-face-detector/).
For `Passthrough` from RTSP cameras, see
[multi-stream-object-detector](../../examples/object-detection/multi-stream-object-detector/).

## Options

`VideoSenderOptions` is created by a factory. Its constructor is private.

| Factory | C++ | Python | What it does |
|---|---|---|---|
| From raw | `VideoSenderOptions::FromRaw(encode)` | `pyneat.VideoSenderOptions.from_raw(encode)` | Encodes raw frames with `SimaEncode` at their input size, then sends them. `encode.type` picks the codec. See `SimaEncodeOptions` |
| Passthrough | `VideoSenderOptions::Passthrough(codec)` | `pyneat.VideoSenderOptions.passthrough(codec)` | Sends encoded frames without re-encoding. `codec` is `RtspCodec::H264`, `H265` or `MJPEG`. `AVC` and `HEVC` are aliases |
| Deprecated | `H264RtpUdpFromRaw(width, height, fps)`, `H264RtpUdpFromEncoded()` | `h264_rtp_udp_from_raw(...)`, `h264_rtp_udp_from_encoded()` | Old H.264-only factories. Use `FromRaw` with `SimaEncodeType::H264` or `Passthrough(RtspCodec::H264)` |

`VideoSenderOptions`

| Option | C++ | Python | Default | What it does | Why and when to change it |
|---|---|---|---|---|---|
| Host | `host` | `host` | `"127.0.0.1"` | Address that receives the video | Insight runs on another machine. Symptom: the Insight tile stays empty |
| Channel | `channel` | `channel` | `0` | Stream number. The UDP port is `video_port_base + channel` | Several streams go to one Insight. Give each stream its own channel; Insight shows each channel in its own tile |
| Video port base | `video_port_base` | `video_port_base` | `9000` | UDP port of channel 0 | Insight listens on other ports, for example the mapped `videoUDP` range that `neat --json` reports when the sender runs on a DevKit |
| Sync | `sync` | `sync` | `false` | Holds each packet until the pipeline clock reaches its timestamp | A video file should play at its recorded speed. Live sources already arrive at video speed |
| Async | `async` | `async_` | `false` | `false` lets Core run a passthrough sender inside its source's pipeline and keeps it out of that pipeline's start-up wait. `true` keeps the sender in its own pipeline | Leave it `false`. With `true`, a passthrough sender runs as a separate pipeline after the source instead of sharing the source's pipeline |
| RTP payload type | `rtp.payload_type` | `rtp.payload_type` | `96` H.264, `98` H.265, `26` MJPEG | Number in each RTP packet that names the codec. Must be `0` to `127` | A receiver other than Insight expects another number |
| Config interval | `rtp.config_interval` | `rtp.config_interval` | `1` | Seconds between repeats of the codec setup data (H.264 and H.265) | Rarely. With `1`, a receiver that joins late can start decoding within a second |
| Encoder bitrate | `encoder.bitrate_kbps` | `encoder.bitrate_kbps` | `4000` | Target bitrate in kbit/s for `FromRaw` with H.264 or H.265 | Raise it when Insight shows blocky video on busy or large frames. Lower it when a slow network makes the video stutter |
| Encoder profile | `encoder.profile` | `encoder.profile` | `"baseline"` H.264, `"main"` H.265 | H.264 profile: `baseline`, `main` or `high`. H.265 accepts only `main` | A receiver needs a given profile |
| Encoder level | `encoder.level` | `encoder.level` | `"4.0"` | Codec level | A receiver needs a given level |

`FromRaw` copies bitrate, profile and level from `SimaEncodeOptions` into
`encoder`, and `VideoSender` reads them from `encoder`. Setting
`encoder.bitrate_kbps` after `FromRaw` therefore replaces the value given to
`FromRaw`.

Read-only values: `video_port()` (`video_port` in Python), `fps()`,
`is_raw_input()`, `is_encoded_input()`.

`SimaEncodeOptions`, used by `FromRaw` and by `nodes::SimaEncode`. Fields
marked "not set" are `std::optional` in C++ and `None` in Python.

| Option | C++ | Python | Default | What it does | Why and when to change it |
|---|---|---|---|---|---|
| Codec | `type` | `type` | `SimaEncodeType::H264` | `H264` (alias `AVC`), `H265` (alias `HEVC`) or `MJPEG` | Match the codec the receiver decodes |
| Frame rate | `fps` | `fps` | `30` | Frame rate written into the stream. It does not limit how fast frames are sent | Always set it to the real rate of the frames you send. The encoder uses it for rate control |
| Bitrate | `bitrate_kbps` | `bitrate_kbps` | Not set, which means `4000` | Target bitrate in kbit/s. H.264 and H.265 only | See `encoder.bitrate_kbps` above |
| Rate control | `rate_control` | `rate_control` | Not set, the encoder's default | `"vbr"` (variable bitrate) or `"cbr"` (constant bitrate) | `"cbr"` for a network link with a fixed budget |
| Profile | `profile` | `profile` | Not set, `baseline` for H.264, `main` for H.265 | See `encoder.profile` above | A receiver needs a given profile |
| Level | `level` | `level` | Not set, `"4.0"` | H.264: `4.0`, `4.1`, `4.2`, `5.0`, `5.1`, `5.2`. H.265: the same without `4.2` | A receiver needs a given level |
| GOP length | `gop_length` | `gop_length` | Not set, the encoder's default | Frames between two key frames, `0` to `1000`. `1` makes every frame a key frame | Lower it when the picture takes long to recover after lost packets. A key frame is a frame that decodes on its own, so more of them cost bitrate |
| IDR interval | `idr_interval` | `idr_interval` | Not set, the encoder's default | Frames between two IDR frames, the key frames a decoder can start from. `0` or more | Lower it when a viewer that connects late waits too long for the first picture |
| Quality | `quality` | `quality` | Not set, `80` | JPEG quality, `1` to `100`. MJPEG only | Raise it for sharper frames, lower it to save bandwidth |
| Output buffers | `num_buffers` | `num_buffers` | `-1`, the encoder's default | Encoded frames the encoder can hold, `1` to `20` | Leave it at `-1`. Core passes it to the encoder only when it is set, and no shipped example sets it |

`UdpOutputOptions`, for `nodes::UdpOutput`

| Option | C++ | Python | Default | What it does | Why and when to change it |
|---|---|---|---|---|---|
| Host | `host` | `host` | `"127.0.0.1"` | Destination address | The receiver is on another machine |
| Port | `port` | `port` | `5000` | Destination UDP port | Always set it to the receiver's port |
| Sync | `sync` | `sync` | `false` | Holds each buffer until the pipeline clock reaches its timestamp | Same as `VideoSenderOptions::sync` |
| Async | `async` | `async_` | `false` | Same as `VideoSenderOptions::async` | Same as `VideoSenderOptions::async` |

## Language differences

- C++ `async` is `async_` in Python, because `async` is a reserved word.
- C++ can fill `SimaEncodeOptions` inline: `FromRaw({.fps = 30})`. Python
  creates `pyneat.SimaEncodeOptions()` first and sets its fields.
- Invalid options throw `std::invalid_argument` in C++. Python raises
  `ValueError` with the same message.
- In C++ only `H264RtpUdpFromEncoded()` warns at compile time.
  `H264RtpUdpFromRaw()` is deprecated without a warning. Python gives no
  warning for either.

## Pitfalls

- **Encoder settings on MJPEG.** `FromRaw` throws `SimaEncode: MJPEG does not
  accept bitrate, rate control, profile, level, GOP or IDR settings` when one
  of them is set. Changing `encoder.*` after `FromRaw` with MJPEG throws
  `VideoSender: MJPEG does not accept legacy video encoder overrides`. Use
  `quality` for MJPEG.
- **An option outside its range throws.** `SimaEncodeOptions` values are
  checked when `FromRaw()` or `SimaEncode()` takes them, for example
  `SimaEncode: gop_length must be within 0..1000`. The `rtp.*` and `encoder.*`
  values are checked later, when `VideoSender()` builds the group. Fix the
  value named in the message.
- **The resolution is fixed by the first frame.** `SimaEncode` takes its
  size from the first frames and keeps it for the whole run. It does not
  resize. If frame sizes change, scale them to one size before the sender.
- **Wrong port or host.** The video arrives nowhere and nothing fails, since
  UDP has no connection. Symptom: an empty Insight tile while the run reports
  no error. Check that `video_port_base + channel` and `host` match what
  Insight listens on.

## Used in

`VideoSender` is used by most streaming examples, for example:

- [single-stream-thermal-face-detector](../../examples/face-detection/single-stream-thermal-face-detector/): `FromRaw`
- [multi-stream-object-detector](../../examples/object-detection/multi-stream-object-detector/): `Passthrough`, one channel per stream
- [high-density-multi-stream-object-detector](../../examples/object-detection/high-density-multi-stream-object-detector/): `Passthrough` fused into the decoder pipeline

`nodes::UdpOutput` is not used directly in Apps.

## Source

- C++ headers: `include/nodes/groups/VideoSender.h`, `include/nodes/sima/SimaEncode.h`, `include/nodes/io/UdpOutput.h`, `include/nodes/groups/RtspCodec.h`
- Python binding: `python/src/module.cpp`, `VideoSenderOptions`, `VideoSenderRtpOptions`, `VideoSenderEncoderOptions`, `SimaEncodeOptions`, `SimaEncodeType`, `RtspCodec`, `UdpOutputOptions`, `groups.video_sender`, `nodes.sima_encode`, `nodes.udp_output`
- Core commit: `3f8b8d04`
