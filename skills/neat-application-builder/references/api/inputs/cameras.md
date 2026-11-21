# CameraInput, UsbCameraInput and HttpMjpegDecodedInput

## What it is

These three APIs read live frames from a camera that is not an RTSP stream.

- `CameraInput` is a node that reads a MIPI camera through libcamera and
  outputs raw frames, NV12 by default.
- `UsbCameraInput` is a node that reads a USB (UVC) camera in one of its MJPEG
  modes and outputs the camera's JPEG frames. A decoder after it turns them
  into raw frames.
- `HttpMjpegDecodedInput` is a group that reads an HTTP or HTTPS multipart
  MJPEG stream and decodes it. A group is a ready-made graph of several nodes
  that you add to your own graph as one piece.

Each of them announces caps to the next node. Caps are the format, size and
frame rate that a node announces to the next node.

## When to use it

Use `CameraInput` for a camera on the board's MIPI connector. Use
`UsbCameraInput` for a USB camera, followed by `SimaDecode` with the format
that `usb_camera_info()` reports. Use `HttpMjpegDecodedInput` for a camera or
server that streams MJPEG over HTTP.

Do not use them for an RTSP camera. Use
[RtspDecodedInput or RtspEncodedInput](rtsp.md). For frames that the
application produces itself, use [Input](input-node.md).

## Minimal example

Open a USB camera once with `usb_camera_info()`, then pass the result to
`UsbCameraInput` so the camera is not opened a second time.
`usb_camera.decoded_format` is the format to give `SimaDecode`.

C++

```cpp
simaai::neat::UsbCameraInputOptions opt;
opt.camera_name = cfg.camera;
opt.width = static_cast<std::uint32_t>(cfg.source_width);
opt.height = static_cast<std::uint32_t>(cfg.source_height);
opt.framerate_num = static_cast<std::uint32_t>(cfg.source_fps);
const auto usb_camera = simaai::neat::usb_camera_info(opt);
simaai::neat::Graph camera("usb_camera");
camera.add(simaai::neat::nodes::UsbCameraInput(opt, usb_camera));
```

Python

```python
opt = pyneat.UsbCameraInputOptions()
opt.camera_name = cfg.camera
opt.width = cfg.source_width
opt.height = cfg.source_height
opt.framerate_num = cfg.source_fps
usb_camera = pyneat.usb_camera_info(opt)
camera = pyneat.Graph("usb_camera")
camera.add(pyneat.nodes.usb_camera_input(opt, usb_camera))
```

Complete flow, including `HttpMjpegDecodedInput`:
[single-stream-object-detector](https://github.com/sima-neat/apps/tree/main/examples/object-detection/single-stream-object-detector)

`CameraInput` is used only in Python examples. A MIPI camera that feeds a
model:

```python
camera = pyneat.CameraInputOptions()
camera.width = CAMERA_WIDTH
camera.height = CAMERA_HEIGHT
camera.framerate_num = CAMERA_FPS
camera.format = "NV12"
camera.buffer_name = CAMERA_BUFFER
camera.allow_cpu_fallback = False
camera_graph = pyneat.Graph(f"mipi_{profile.name}_camera")
camera_graph.add(pyneat.nodes.camera_input(camera))
```

Complete flow: [mipi-multi-model](https://github.com/sima-neat/apps/tree/main/examples/benchmarking/mipi-multi-model)

## Options

`CameraInput` has two forms:

| Form | C++ | Python | What it does |
|---|---|---|---|
| Options only | `nodes::CameraInput(opt)` | `pyneat.nodes.camera_input(opt)` | Camera with the camera pipeline's default number of capture buffers |
| Options and capture buffers | `nodes::CameraInputWithCaptureBuffers(opt, count)` | `pyneat.nodes.camera_input(opt, capture_buffer_count=count)` | At least `count` capture buffers. `0` keeps the default. At most `128` |

`CameraInputOptions`

Zero-copy means the next node reads the camera's frame memory directly,
without a copy. SiMa memory is buffer memory that the board's processors can
read directly. The MLA (Machine Learning Accelerator) runs the model. CVU,
also called EV74, is the board's vision processor that runs preprocessing.

| Option | C++ | Python | Default | What it does | Why and when to change it |
|---|---|---|---|---|---|
| Camera name | `camera_name` | `camera_name` | Not set (`std::nullopt`, `None`) | libcamera name of the camera, for example `"imx477 5-001a"` from `cam -l`. Not set means libcamera's default camera | The board has more than one camera. `mipi-camera-capture` reads it from its config |
| Width | `width` | `width` | `1920` | Frame width in pixels | The camera mode you need. `mipi-multi-model` and `mipi-camera-capture` set it |
| Height | `height` | `height` | `1080` | Frame height in pixels | Same as width |
| Frame rate numerator | `framerate_num` | `framerate_num` | `30` | Frame rate is `framerate_num / framerate_den` | The camera mode you need. `mipi-multi-model` and `mipi-camera-capture` set it |
| Frame rate denominator | `framerate_den` | `framerate_den` | `1` | Denominator of the frame rate. `0` becomes `1` | A fractional frame rate. `mipi-camera-capture` reads it from its config |
| Format | `format` | `format` | `"NV12"` | Pixel format the node announces. Empty becomes `NV12`. The output description covers `RGB`, `BGR`, `GRAY8`, `NV12`, `NV16` and `I420` | The next node needs another format. `mipi-multi-model` sets `NV12` |
| Buffer name | `buffer_name` | `buffer_name` | `"camera"` | Name that the CVU and MLA stages use to find this camera's frames. Empty becomes `"camera"` | A model route reads the camera by name: `mipi-multi-model` sets the same name in `ModelRouteOptions.buffer_name` |
| Insert queue | `insert_queue` | `insert_queue` | `true` | Adds a small queue after the camera so the camera is not held back without limit | Not changed in any shipped example; keep the default |
| Leaky queue | `leaky_queue` | `leaky_queue` | `true` | The queue drops frames when it is full instead of holding the camera back | Not changed in any shipped example; keep the default |
| Queue depth | `queue_depth` | `queue_depth` | `2` | Buffer limit of the queue. `0` drops Neat's limit, so the queue keeps its GStreamer default | `mipi-camera-capture` reads it from its config |
| Allow CPU fallback | `allow_cpu_fallback` | `allow_cpu_fallback` | `false` | `false` is strict zero-copy: a frame that would need a copy fails. `true` copies such a frame on the board's CPU instead | A copy is acceptable when zero-copy fails. `mipi-camera-capture` sets it from its `strict_zero_copy` config |

`UsbCameraInputOptions`

The fields have `CameraInputOptions`' names and defaults, but `camera_name` is
required and there is no `format`.

| Option | C++ | Python | Default | What it does | Why and when to change it |
|---|---|---|---|---|---|
| Camera name | `camera_name` | `camera_name` | `""` | Stable USB camera ID from `usb_camera_ids()`, for example `"/base/pci@7000000000-3.1:1.0-046d:08e5"`. The ID changes when the camera moves to another USB port | Always set it. Empty fails, see the pitfalls |
| Width | `width` | `width` | `1920` | Width of an exact MJPEG mode of the camera | Your camera has no 1920x1080 MJPEG mode. `single-stream-object-detector` sets `1280` in its config |
| Height | `height` | `height` | `1080` | Height of that MJPEG mode | Same as width. `single-stream-object-detector` sets `720` |
| Frame rate numerator | `framerate_num` | `framerate_num` | `30` | Frame rate of that MJPEG mode, `framerate_num / framerate_den` | Match the camera's mode. `single-stream-object-detector` sets it from its config `fps` |
| Frame rate denominator | `framerate_den` | `framerate_den` | `1` | Denominator of the frame rate | Not changed in any shipped example; keep the default |
| Buffer name | `buffer_name` | `buffer_name` | `"camera"` | Name that the CVU and MLA stages use to find this camera's frames. Empty fails | Not changed in any shipped example; keep the default |
| Insert queue | `insert_queue` | `insert_queue` | `true` | Adds a small queue after the camera | Not changed in any shipped example; keep the default |
| Leaky queue | `leaky_queue` | `leaky_queue` | `true` | The queue drops frames when it is full instead of holding the camera back | Not changed in any shipped example; keep the default |
| Queue depth | `queue_depth` | `queue_depth` | `2` | Buffer limit of the queue. With `insert_queue`, the capture pool holds `queue_depth + 6` frames, at most `64`, so the largest value is `58` | Not changed in any shipped example; keep the default |

`UsbCameraInfo`, returned by `usb_camera_info(opt)`

`usb_camera_info()` checks the options, finds the camera, checks the mode and
reads one frame.

| Field | C++ | Python | What it holds |
|---|---|---|---|
| Device | `device` | `device` | The `/dev/videoN` device that captures the MJPEG frames |
| Decoded format | `decoded_format` | `decoded_format` | `NV12` for 4:2:0 JPEGs, `NV16` for 4:2:2 JPEGs. Give it to `SimaDecode` as `out_format` |
| Node index | `node_index` | not available | sysfs index of `device` in its USB interface. `-1` when unknown |

`HttpMjpegDecodedInputOptions`

APU is the application processor, the board's general-purpose CPU.

| Option | C++ | Python | Default | What it does | Why and when to change it |
|---|---|---|---|---|---|
| URL | `url` | `url` | `""` | HTTP or HTTPS multipart MJPEG URL to read | Always set it |
| Timeout | `timeout_seconds` | `timeout_seconds` | `15` | HTTP timeout in seconds. `0` turns the timeout off | Not changed in any shipped example; keep the default |
| Retries | `retries` | `retries` | `3` | HTTP retries before the source fails. `-1` retries without limit | Not changed in any shipped example; keep the default |
| Live source | `is_live` | `is_live` | `true` | Marks the HTTP source as live | Not changed in any shipped example; keep the default |
| Timestamp | `do_timestamp` | `do_timestamp` | `true` | Stamps outgoing source buffers with stream time | Not changed in any shipped example; keep the default |
| User agent | `user_agent` | `user_agent` | `""` | HTTP User-Agent header. Empty keeps the source's default | Not changed in any shipped example; keep the default |
| Multipart boundary | `multipart_boundary` | `multipart_boundary` | `""` | Multipart boundary string. Empty means detect it from the stream | Not changed in any shipped example; keep the default |
| Single stream | `multipart_single_stream` | `multipart_single_stream` | `false` | `true` assumes the multipart content type does not change | Not changed in any shipped example; keep the default |
| Insert queue | `insert_queue` | `insert_queue` | `true` | Adds a queue after the HTTP source and one after the JPEG parser, or after the multipart demux when `header_capture` is set | Not changed in any shipped example; keep the default |
| Sync mode | `sync_mode` | `sync_mode` | `false` | `true` removes the queues that `insert_queue` adds | Not changed in any shipped example; keep the default |
| Allocator type | `sima_allocator_type` | `sima_allocator_type` | `2` | Memory allocator for the decoded frames | Not changed in any shipped example; keep the default |
| Output format | `out_format` | `out_format` | `NV12` | Pixel format the decoder outputs. With `decoder_raw_output = true` it must be `NV12`, `I420` or `NV16`; see [decode-and-caps](decode-and-caps.md) | Not changed in any shipped example; keep the default |
| Decoder name | `decoder_name` | `decoder_name` | `""` | Name of the decoder node, also used as the name of its output buffer. Empty gives `n<index>_decoder` | `single-stream-object-detector` sets `"decoder"` |
| Raw output | `decoder_raw_output` | `decoder_raw_output` | `true` | `true` keeps decoded frames in SiMa memory | Not changed in any shipped example; keep the default |
| Decoder next element | `decoder_next_element` | `decoder_next_element` | `""` | Tells the decoder which processor reads its output: `CVU` or `APU` | Not changed in any shipped example; keep the default |
| Decode width | `dec_width` | `dec_width` | `-1` | Width of the decoded frames. `-1` takes it from the stream | Not changed in any shipped example; keep the default |
| Decode height | `dec_height` | `dec_height` | `-1` | Height of the decoded frames. `-1` takes it from the stream | Not changed in any shipped example; keep the default |
| Decode FPS | `dec_fps` | `dec_fps` | `-1` | Frame rate of the decoded frames. Used when `source_fps` is not set | Use `source_fps` instead. Both together must be equal |
| Decoder buffers | `num_buffers` | `num_buffers` | `-1` | How many decoded frames the decoder can hold. `-1` keeps the decoder default | Not changed in any shipped example; keep the default |
| Videoconvert | `use_videoconvert` | `use_videoconvert` | `false` | Adds a format converter after the decoder | Not changed in any shipped example; keep the default |
| Videoscale | `use_videoscale` | `use_videoscale` | `false` | Adds a scaler after the decoder | Not changed in any shipped example; keep the default |
| Output caps | `output_caps` | `output_caps` | Off | Fixes the caps at the end of the group. Same fields as `RtspDecodedInputOptions::output_caps`, see [rtsp](rtsp.md). Python class `pyneat.HttpMjpegDecodedInputOutputCaps` | The next node needs a fixed format, size and frame rate. `single-stream-object-detector` sets NV12, the frame size and the frame rate |
| Extra fragment | `extra_fragment` | `extra_fragment` | `""` | A GStreamer text fragment added at the end of the group | Not changed in any shipped example; keep the default |
| Strict TLS | `ssl_strict` | `ssl_strict` | `true` | `false` turns off strict TLS certificate checks | The server uses a self-signed HTTPS certificate: `single-stream-object-detector` documents `ssl_strict: false` for that in its config |
| Source FPS | `source_fps` | `source_fps` | `-1` | Frame rate the server sends. Written into the caps and used as the decoder frame rate | The stream does not report a frame rate. `single-stream-object-detector` sets it from its config `fps`, where `0` detects it from the stream |
| Videorate | `use_videorate` | `use_videorate` | `false` | Adds a node that drops frames to lower the rate to `video_rate_fps`. It never repeats frames | Not changed in any shipped example; keep the default |
| Videorate FPS | `video_rate_fps` | `video_rate_fps` | `-1` | Frame rate that videorate outputs. `-1` means the source frame rate | Not changed in any shipped example; keep the default |
| Header capture | `header_capture` | `header_capture` | Empty | Names of multipart part headers to copy into `Sample::attributes` of the decoded frame they arrived with. Empty turns capture off. At most 64 names of at most 128 bytes | Not changed in any shipped example; keep the default |

The `HttpMjpegDecodedInputOptions` header says `sync_mode` makes the sink
elements run in sync mode, but Core only removes the queues that
`insert_queue` adds. The table above describes what Core does.

## Language differences

- C++ has two `CameraInput` functions, `nodes::CameraInput(opt)` and
  `nodes::CameraInputWithCaptureBuffers(opt, count)`. Python has one,
  `pyneat.nodes.camera_input(opt, capture_buffer_count=0)`.
- `UsbCameraInfo.node_index` is not available in Python.
- C++ `out_format` and `output_caps.format` of `HttpMjpegDecodedInputOptions`
  accept a string such as `"NV12"` or a `FormatTag`. Python accepts only
  `pyneat.Format`, for example `pyneat.Format.NV12`.

## Pitfalls

- **Empty values are filled in for `CameraInput` but rejected by
  `UsbCameraInput`.** `CameraInput` turns an empty `format` into `NV12`, a
  `framerate_den` of `0` into `1` and an empty `buffer_name` into `"camera"`.
  `UsbCameraInput` fails on an empty `camera_name` with
  `UsbCameraInput requires camera_name: a USB camera ID from usb_camera_ids()`,
  on a zero `width`, `height`, `framerate_num` or `framerate_den` with
  `UsbCameraInput requires a nonzero width, height and frame rate`, and on an
  empty `buffer_name` with `UsbCameraInput requires a buffer_name`. Fix: take
  the ID from `usb_camera_ids()` and set an exact MJPEG mode of the camera.
- **`UsbCameraInput` queue too deep.** With `insert_queue = true`, a
  `queue_depth` above `58` fails with
  `UsbCameraInput queue_depth exceeds 58: the capture pool (queue_depth + 6) would exceed 64 buffers`.
  Fix: keep `queue_depth` at `58` or less.
- **The USB camera is missing, busy or lacks the mode.** `UsbCameraInput` and
  `usb_camera_info()` throw when the camera is not connected, is in use, lacks
  the mode or sends no decodable frame. The message names the camera ID and
  lists the connected IDs or the camera's MJPEG modes. Building a graph again
  after the camera was unplugged, moved or renumbered also throws. Fix: call
  `usb_camera_info()` and `UsbCameraInput` again. Read in the source, not
  verified on a device.
- **The USB camera decodes to NV16.** A 4:2:2 JPEG camera decodes to NV16, not
  NV12. Fix: give `usb_camera.decoded_format` to `SimaDecode` as
  `out_format`, and set the model's preprocessing input format to match
  (`PreprocessColorFormat::NV16`). Read in the source, not verified on a
  device.
- **Too many `CameraInput` capture buffers.** A `capture_buffer_count` above
  `128` fails with
  `CameraInput capture_buffer_count exceeds Neat's 128-buffer provider limit`.
  A nonzero count on a camera stack without that setting fails with
  `CameraInput capture_buffer_count requires a libcamerasrc with the buffer-count property`.
  The `mipi-camera-capture` config notes that the camera stack in Platform
  3.0 has no such setting, so only `0` works there. Fix: use `0`.
- **Strict zero-copy rejects a frame.** With `allow_cpu_fallback = false`,
  `CameraInput` fails on any frame that it would have to copy. Fix: set
  `allow_cpu_fallback = true` for that camera. Read in the source, not
  verified on a device.
- **Header capture with a transform.** With `header_capture` set,
  `use_videoconvert`, `use_videoscale`, `use_videorate` or `extra_fragment`
  fails with
  `HttpMjpegDecodedInput: header_capture does not support use_videoconvert; per-frame attributes are only guaranteed through the default decode path, Queue/Branch, and materialized Core boundaries`
  (with the option's name). Fix: leave those options off, or turn header
  capture off.
- **HTTP frame-rate options with different values.** `source_fps` with another
  `dec_fps` fails with `HttpMjpegDecodedInput: source_fps conflicts with dec_fps`.
  `video_rate_fps` without `use_videorate = true` fails with
  `HttpMjpegDecodedInput: video_rate_fps requires use_videorate=true`.
  `use_videorate = true` with neither `video_rate_fps` nor a source frame rate
  fails with `HttpMjpegDecodedInput: use_videorate requires video_rate_fps or source_fps`.
  With `output_caps.enable = true`, an `output_caps.fps` that differs from
  `source_fps` (without videorate) or from the videorate rate fails with
  `output_caps.fps conflicts with source_fps; use video_rate_fps for rate conversion`
  or `output_caps.fps conflicts with video_rate_fps`. Fix: set only
  `source_fps`, and use videorate to change the rate.
- **An output format the decoder cannot produce, or frames that do not match
  the model.** See the pitfalls in [decode-and-caps](decode-and-caps.md).

## Used in

- [single-stream-object-detector](https://github.com/sima-neat/apps/tree/main/examples/object-detection/single-stream-object-detector) (`UsbCameraInput` with `usb_camera_info()`, `HttpMjpegDecodedInput` with `output_caps`)
- [single-stream-instance-segmenter](https://github.com/sima-neat/apps/tree/main/examples/segmentation/single-stream-instance-segmenter) (`UsbCameraInput`, `HttpMjpegDecodedInput`)
- [face-recognizer](https://github.com/sima-neat/apps/tree/main/examples/face-recognition/face-recognizer) (`UsbCameraInput`)
- [mipi-camera-capture](https://github.com/sima-neat/apps/tree/main/examples/benchmarking/mipi-camera-capture) (`CameraInput` with `capture_buffer_count`, Python only)
- [mipi-multi-model](https://github.com/sima-neat/apps/tree/main/examples/benchmarking/mipi-multi-model) (`CameraInput` with `buffer_name` shared with the model route, Python only)

## Source

- C++ header: `include/nodes/io/CameraInput.h` (`CameraInputOptions`, `CameraInput`, `CameraInputWithCaptureBuffers`), `include/nodes/io/UsbCameraInput.h` (`UsbCameraInputOptions`, `UsbCameraInfo`, `usb_camera_ids`, `usb_camera_info`, `UsbCameraInput`), `include/nodes/groups/HttpMjpegDecodedInput.h` (`HttpMjpegDecodedInputOptions`, `HttpMjpegDecodedInput`), `include/nodes/common/MultipartJpegDemux.h` (`MultipartHeaderCaptureOptions`), `include/builder/NextCpuPolicy.h` (APU)
- Python binding: `python/src/module.cpp`, `CameraInputOptions`, `UsbCameraInputOptions`, `UsbCameraInfo`, `usb_camera_ids`, `usb_camera_info`, `MultipartHeaderCaptureOptions`, `HttpMjpegDecodedInputOptions`, `HttpMjpegDecodedInputOutputCaps`, `nodes.camera_input`, `nodes.usb_camera_input`, `groups.http_mjpeg_decoded_input`
- Core commit: `b1f9556c`
