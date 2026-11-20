# RtspDecodedInput and RtspEncodedInput

## What it is

`RtspDecodedInput` and `RtspEncodedInput` are groups that read a live stream
from an `rtsp://` URL, for example an IP camera. A group is a ready-made graph
of several nodes that you add to your own graph as one piece. Both groups
contain three steps. The receiver connects to the camera and receives the RTP
packets that carry the video. The depacketizer takes the video data out of
those packets. The parser splits that data into whole encoded frames.
`RtspDecodedInput` also decodes the frames in hardware and outputs raw
frames. `RtspEncodedInput` stops before decoding and outputs encoded frames.

`RtspCodec` selects the stream codec for both groups: H.264, H.265 or MJPEG.

The groups announce caps to the next node. Caps are the format, size and frame
rate that a node announces to the next node. When the camera does not announce
its size or frame rate, the options below supply them.

## When to use it

Use `RtspDecodedInput` when an RTSP camera feeds a model or anything else
that needs pixels. It outputs NV12 frames by default.

Use `RtspEncodedInput` when the encoded stream itself is needed: to forward it
to Insight with `VideoSender` without decoding, or to put your own `SimaDecode`
after it, for example to decode once and send the encoded stream and the
decoded frames to different nodes.

Do not use either for frames that the application produces itself. Use
[Input](input-node.md) and `Run::push()` instead. For a MIPI or USB camera use
[CameraInput or UsbCameraInput](cameras.md).

## Minimal example

Read an H.264 camera and decode it to NV12 frames. When the camera does not
announce its size in the stream, the fallback size is used.

C++

```cpp
simaai::neat::nodes::groups::RtspDecodedInputOptions opt;
opt.url = cfg.source_url;
opt.latency_ms = cfg.latency_ms;
opt.tcp = cfg.tcp;
opt.source_fps = geometry.fps;
opt.codec = simaai::neat::nodes::groups::RtspCodec::H264;
opt.fallback_h264_width = geometry.width;
opt.fallback_h264_height = geometry.height;
graph.add(simaai::neat::nodes::groups::RtspDecodedInput(opt));
```

Python

```python
opt = pyneat.RtspDecodedInputOptions()
opt.url = cfg.source_url
opt.latency_ms = cfg.latency_ms
opt.tcp = cfg.tcp
opt.source_fps = fps
opt.codec = pyneat.RtspCodec.H264
opt.fallback_h264_width = width
opt.fallback_h264_height = height
graph.add(pyneat.groups.rtsp_decoded_input(opt))
```

Complete flow: [single-stream-object-detector](../../examples/object-detection/single-stream-object-detector/)

For `RtspEncodedInput` followed by `SimaDecode`, and the encoded stream also
sent to Insight, see
[multi-stream-object-detector](../../examples/object-detection/multi-stream-object-detector/).

## Options

`RtspCodec`

| Value | C++ | Python | What it does | When to use it |
|---|---|---|---|---|
| H.264 | `RtspCodec::H264`, alias `RtspCodec::AVC` | `pyneat.RtspCodec.H264`, alias `pyneat.RtspCodec.AVC` | H.264 path. Default RTP payload type `96` | Default. Most IP cameras |
| MJPEG | `RtspCodec::MJPEG` | `pyneat.RtspCodec.MJPEG` | Motion JPEG path. Default RTP payload type `26` | The camera streams JPEG frames |
| H.265 | `RtspCodec::H265`, alias `RtspCodec::HEVC` | `pyneat.RtspCodec.H265`, alias `pyneat.RtspCodec.HEVC` | H.265 path. Default RTP payload type `96` | The camera streams H.265 |

`RtspEncodedInputOptions`

An RTP payload type is the number in each RTP packet that names its codec.
SPS and PPS are the H.264 headers that a decoder needs before the first frame.

| Option | C++ | Python | Default | What it does | Why and when to change it |
|---|---|---|---|---|---|
| URL | `url` | `url` | `""` | The `rtsp://` URL to read | Always set it |
| Codec | `codec` | `codec` | `RtspCodec::H264` | Which codec path to build. See the `RtspCodec` table | The camera streams H.265 or MJPEG |
| Latency | `latency_ms` | `latency_ms` | `200` | Size of the receiver's jitter buffer, in milliseconds | Shipped examples set `100`, for example `single-stream-object-detector` and `high-density-multi-stream-object-detector` in their `src/common/config.yaml` |
| TCP | `tcp` | `tcp` | `true` | `true` requests RTSP over TCP, `false` over UDP | Not changed in any shipped example; keep the default |
| Drop on latency | `drop_on_latency` | `drop_on_latency` | `true` | Asks the receiver to drop late buffers instead of stalling | `high-density-multi-stream-object-detector` sets it from its config. Otherwise keep the default |
| Buffer mode | `buffer_mode` | `buffer_mode` | `""` (receiver default) | Buffering mode of the RTSP receiver, for example `none` | The rfdetr example sets `none` for TCP streams. Otherwise leave it |
| Insert queue | `insert_queue` | `insert_queue` | `true` | Adds a queue after the receiver and one after the depacketizer or parser, so each runs in its own thread | Not changed in any shipped example; keep the default |
| Sync mode | `sync_mode` | `sync_mode` | `false` | `true` removes the queues that `insert_queue` adds, the same as `insert_queue = false` | Not changed in any shipped example; keep the default |
| H.264 payload type (deprecated) | `h264_payload_type` | `h264_payload_type` | `96` | Old H.264 payload filter. Used only when `payload_type` is `-1` | Never in new code. Use `payload_type` |
| MJPEG payload type (deprecated) | `mjpeg_payload_type` | `mjpeg_payload_type` | `26` | Old MJPEG payload filter. Used only when `payload_type` is `-1` | Never in new code. Use `payload_type` |
| SPS/PPS interval | `h264_parse_config_interval` | `h264_parse_config_interval` | `-1` (parser default) | How often the H.264 parser repeats SPS and PPS | A decoder that joins the stream late and waits for the next keyframe to start |
| H.264 FPS | `h264_fps` | `h264_fps` | `-1` | Frame rate written into the H.264 caps | Use `source_fps` instead. Both together must be equal |
| H.264 width | `h264_width` | `h264_width` | `-1` | Width written into the H.264 caps. With width, height and a frame rate all set, these caps are used even when `auto_caps_from_stream` is on | You know the stream size |
| H.264 height | `h264_height` | `h264_height` | `-1` | Height written into the H.264 caps | Same as width |
| Auto caps | `auto_caps_from_stream` | `auto_caps_from_stream` | `true` | Reads the H.264 size and frame rate from the stream when any of them is not set. For MJPEG, reads the frame rate from the RTSP session description when `source_fps` is not set | Set `false` only with width, height and frame rate all set, see the pitfalls |
| Fallback FPS | `fallback_h264_fps` | `fallback_h264_fps` | `-1` | H.264 frame rate used when the stream does not carry one | The camera does not announce its frame rate and a later node needs one, for example an encoder |
| Fallback width | `fallback_h264_width` | `fallback_h264_width` | `-1` | H.264 width used when the stream does not carry one | The camera does not announce its size |
| Fallback height | `fallback_h264_height` | `fallback_h264_height` | `-1` | H.264 height used when the stream does not carry one | Same as fallback width |
| Source FPS | `source_fps` | `source_fps` | `-1` | The frame rate the camera sends. Written into the caps for every codec | The camera does not announce its frame rate. Set the real rate, it does not change the rate |
| Payload type | `payload_type` | `payload_type` | `-1` | Only RTP packets with this payload type are accepted. `-1` means the codec default: `96` for H.264 and H.265, `26` for MJPEG. `0` accepts any | Keep the default, which already matches the codec |

`RtspDecodedInputOptions`

`RtspDecodedInputOptions` has the `RtspEncodedInputOptions` fields above
except `h264_payload_type`, with the same meaning and defaults. In addition,
its `codec` also selects the decoder, and its `source_fps` also sets the
decoder frame rate. The fields below are its own, in header order.

SiMa memory is buffer memory that the board's processors can read directly.
Zero-copy output means the next node reads the decoder's frame memory
directly, without a copy. CVU, also called EV74, is the board's vision
processor that runs preprocessing. APU is the application processor, the
board's general-purpose CPU.

| Option | C++ | Python | Default | What it does | Why and when to change it |
|---|---|---|---|---|---|
| Allocator type | `sima_allocator_type` | `sima_allocator_type` | `2` | Memory allocator for the decoded frames | Not changed in any shipped example; keep the default |
| Output format | `out_format` | `out_format` | `NV12` | Pixel format the decoder outputs. With `decoder_raw_output = true` it must be `NV12`, `I420`, or `NV16` for MJPEG; with `false` any raw format works. `I420` turns off zero-copy output, so each frame is copied before the next node gets it | The next node needs `I420`, or a 4:2:2 MJPEG camera should output `NV16` |
| Decoder name | `decoder_name` | `decoder_name` | `""` | Name of the decoder node, also used as the name of its output buffer. Empty gives `n<index>_decoder` | You look up the decoder by name, for example in logs or in a graph description |
| Raw output | `decoder_raw_output` | `decoder_raw_output` | `true` | `true` keeps decoded frames in SiMa memory. `false` outputs them in system memory | Not changed in any shipped example; keep the default |
| Decoder next element | `decoder_next_element` | `decoder_next_element` | `""` | Tells the decoder which processor reads its output: `CVU` or `APU` | `high-density-multi-stream-object-detector` sets `CVU` on its own `SimaDecode`. Otherwise keep the default |
| Videoconvert | `use_videoconvert` | `use_videoconvert` | `false` | Adds a format converter after the decoder | Not changed in any shipped example; keep the default |
| Videoscale | `use_videoscale` | `use_videoscale` | `false` | Adds a scaler after the decoder | Not changed in any shipped example; keep the default |
| Output caps | `output_caps` | `output_caps` | Off | Fixes the caps at the end of the group. See the `OutputCaps` table | The next node needs a fixed format, size and frame rate, for example a model input |
| Extra fragment | `extra_fragment` | `extra_fragment` | `""` | A GStreamer text fragment added at the end of the group | Advanced use only. A step that no Neat node provides |
| Decode width | `dec_width` | `dec_width` | `-1` | Width of the decoded frames. For MJPEG, `output_caps.width` is used when this is `-1` and `use_videoscale` is off | H.265 and MJPEG cameras whose size the decoder must know. The examples set it for those codecs |
| Decode height | `dec_height` | `dec_height` | `-1` | Height of the decoded frames | Same as decode width |
| Decode FPS | `dec_fps` | `dec_fps` | `-1` | Frame rate of the decoded frames. Also a fallback source frame rate | Use `source_fps` instead. Both together must be equal |
| Decoder buffers | `num_buffers` | `num_buffers` | `-1` (decoder default) | How many decoded frames the decoder can hold | The model and preview keep frames while the decoder needs free ones, for example the rfdetr example sets `32` on its own `SimaDecode` for MJPEG |
| Videorate | `use_videorate` | `use_videorate` | `false` | Adds a node that drops frames to lower the rate to `video_rate_fps`. It never repeats frames | The next node needs a lower frame rate than the camera sends |
| Videorate FPS | `video_rate_fps` | `video_rate_fps` | `-1` | Frame rate that videorate outputs. `-1` means the source frame rate | Set the rate you need. Requires `use_videorate = true` |
| Decoder input buffers | `decoder_input_buffers` | `decoder_input_buffers` | `-1` (decoder default) | How many encoded frames the decoder can queue | Many streams on one board. The high-density example sets the matching `SimaDecode` option from its config |
| Decoder tuning | `decoder_tuning` | `decoder_tuning` | `""` | Decoder tuning preset, for example `low-memory` or `throughput-low-latency` | Many streams on one board run out of decoder memory or fall behind. The high-density example sets the matching `SimaDecode` option from its config |
| Decoder memory optimization | `decoder_memory_opt` | `decoder_memory_opt` | `false` | Turns on the decoder's memory-optimized path | The high-density example turns on the matching `SimaDecode` option with the `low-memory` or `throughput-low-latency` tuning |

`OutputCaps` (`RtspDecodedInputOptions::output_caps`, Python `pyneat.RtspDecodedInputOutputCaps`)

| Option | C++ | Python | Default | What it does | Why and when to change it |
|---|---|---|---|---|---|
| Enable | `enable` | `enable` | `false` | Adds the caps filter. Without it, `fps` is still checked against `source_fps` and, for MJPEG, still serves as the fallback frame rate, and `width` and `height` still set the decode size when `dec_width` and `dec_height` are `-1` | Set it with the fields you need |
| Format | `format` | `format` | `NV12` | Pixel format the group announces | Match `out_format`, or the converter output when `use_videoconvert` is on |
| Width | `width` | `width` | `-1` (not fixed) | Width the group announces | The next node needs a fixed size |
| Height | `height` | `height` | `-1` (not fixed) | Height the group announces | Same as width |
| FPS | `fps` | `fps` | `-1` (not fixed) | Frame rate the group announces. For MJPEG also the last fallback frame rate | A later node needs a frame rate in the caps |
| Memory | `memory` | `memory` | `CapsMemory::Any` | `Any` leaves the memory open. `SystemMemory` makes the caps announce system memory | Not changed in any shipped example; keep the default |

The header comments are out of date for two options. The header says
`sync_mode` makes the sink elements run in sync mode, but Core only removes
the queues that `insert_queue` adds. The `RtspDecodedInputOptions` header
says `insert_queue` adds one queue between the depacketizer and the parser,
but Core adds one after the receiver and one after the depacketizer or
parser. The tables above describe what Core does.

## Language differences

- C++ `out_format` and `output_caps.format` accept a string such as `"NV12"`
  or a `FormatTag`. Python accepts only `pyneat.Format`, for example
  `pyneat.Format.NV12`.

## Pitfalls

- **H.264 with `auto_caps_from_stream = false` and missing size or frame
  rate.** When auto caps are off, width, height and frame rate must all
  resolve. Width and height come from `h264_width`/`h264_height`, then the
  fallback fields. The frame rate comes from `source_fps`, then `h264_fps`,
  then `fallback_h264_fps`. Otherwise both groups fail with
  `H.264 explicit caps require width, height, and fps when auto_caps_from_stream is false`,
  because `RtspDecodedInput` builds an `RtspEncodedInput` inside. `dec_fps`,
  `dec_width` and `dec_height` do not count for this check. They only satisfy
  the earlier `decoder_raw_output requires h264 width/height/fps` check of
  `RtspDecodedInput`. Fix: set all three, or keep
  `auto_caps_from_stream = true`.
- **Two frame-rate options with different values.** `source_fps` together
  with `h264_fps` or `dec_fps` of another value fails with
  `source_fps conflicts with h264_fps` or `... with dec_fps`. For H.264 in
  `RtspDecodedInput`, the next set option is checked the same way: `h264_fps`
  against `dec_fps`, then `dec_fps` against `fallback_h264_fps`. Fix: set only
  `source_fps`.
- **Changing the frame rate with `output_caps.fps`.** With `source_fps` set and
  `use_videorate` off, an `output_caps.fps` of another value fails with
  `output_caps.fps conflicts with source_fps; use video_rate_fps for rate conversion`.
  Fix: set `use_videorate = true` and `video_rate_fps`.
- **Videorate options without each other.** `video_rate_fps` without
  `use_videorate = true` fails with `video_rate_fps requires use_videorate=true`.
  `use_videorate = true` with neither `video_rate_fps` nor a source frame rate
  fails with `use_videorate requires video_rate_fps or source_fps`. With
  `output_caps.enable`, an `output_caps.fps` that differs from the videorate
  rate fails with `output_caps.fps conflicts with video_rate_fps`.
- **An output format the decoder cannot produce, or frames that do not match
  the model.** See the pitfalls in [decode-and-caps](decode-and-caps.md).
- **The deprecated payload fields print a warning.** Setting
  `h264_payload_type` or `mjpeg_payload_type` to a value other than the codec
  default, with `payload_type` still `-1`, prints a deprecation warning once.
  Fix: set `payload_type`.

## Used in

- [single-stream-object-detector](../../examples/object-detection/single-stream-object-detector/) (`RtspDecodedInput`, H.264, H.265 or MJPEG, `output_caps`)
- [multi-stream-object-detector](../../examples/object-detection/multi-stream-object-detector/) (`RtspEncodedInput` with `SimaDecode`, encoded stream to Insight)
- [rfdetr-detection-segmentation](../../examples/object-detection/rfdetr-detection-segmentation/) (`RtspEncodedInput`, `buffer_mode = "none"` with TCP)
- [high-density-multi-stream-object-detector](../../examples/object-detection/high-density-multi-stream-object-detector/) (`RtspEncodedInput`, `drop_on_latency`, decoder tuning)
- [multi-stream-multi-model](../../examples/multi-model/multi-stream-multi-model/) (`RtspEncodedInput`)
- [face-recognizer](../../examples/face-recognition/face-recognizer/) (`RtspEncodedInput`)
- [patchcore-anomaly-detector](../../examples/anomaly-detection/patchcore-anomaly-detector/) (`RtspDecodedInput`)
- [single-stream-thermal-face-detector](../../examples/face-detection/single-stream-thermal-face-detector/) (`RtspDecodedInput`, `output_caps`)

## Source

- C++ header: `include/nodes/groups/RtspDecodedInput.h` (`RtspDecodedInputOptions`, `RtspDecodedInput`), `include/nodes/groups/RtspEncodedInput.h` (`RtspEncodedInputOptions`, `RtspEncodedInput`), `include/nodes/groups/RtspCodec.h` (`RtspCodec`)
- Python binding: `python/src/module.cpp`, `RtspCodec`, `RtspEncodedInputOptions`, `RtspDecodedInputOptions`, `RtspDecodedInputOutputCaps`, `groups.rtsp_encoded_input`, `groups.rtsp_decoded_input`
- Core commit: `b1f9556c`
