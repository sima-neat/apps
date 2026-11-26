# SimaDecode, CapsRaw and video transforms

## What it is

`SimaDecode` is a node that decodes encoded video (H.264, H.265, JPEG or
MJPEG) with the board's hardware decoder and outputs raw frames, NV12 by
default. A node is one processing step in a graph.

`CapsRaw` is a node that fixes the caps at one point of the graph. Caps are
the format, size and frame rate that a node announces to the next node.
`CapsMemory` says which memory the caps announce.

`VideoConvert`, `VideoScale` and `VideoRate` are nodes that change the pixel
format, the frame size or the frame rate. Each one takes its target from a
`CapsRaw` placed after it.

## When to use it

Use `SimaDecode` after a source that outputs encoded frames:
[RtspEncodedInput](rtsp.md), [UsbCameraInput](cameras.md), or
[Input](input-node.md) with an encoded payload. When the source can decode by
itself, use [RtspDecodedInput](rtsp.md) or
[HttpMjpegDecodedInput](cameras.md) instead. They contain a `SimaDecode`.

Use `CapsRaw` after `SimaDecode` or a transform when the next node needs a
fixed format, size or frame rate.

Use `VideoScale` only for resizes outside a model. For a model input, use the
model's preprocessing resize (`PreprocessOptions::resize`), which runs on the
CVU. CVU, also called EV74, is the board's vision processor that runs
preprocessing. Use `VideoConvert` when the next node needs a pixel format that
the decoder cannot output. Use `VideoRate` to lower the frame rate.

## Minimal example

Decode H.264 to NV12 and fix the caps for the model input.

C++

```cpp
simaai::neat::SimaDecodeOptions opt;
opt.type = simaai::neat::SimaDecodeType::H264;
opt.out_format = "NV12";
opt.decoder_name = "decoder";
opt.dec_width = stream.frame_w;
opt.dec_height = stream.frame_h;
opt.dec_fps = stream.source_fps;
decoder.add(simaai::neat::nodes::SimaDecode(opt));
decoder.add(simaai::neat::nodes::CapsRaw("NV12", stream.frame_w, stream.frame_h,
                                         stream.source_fps, simaai::neat::CapsMemory::Any));
```

Python

```python
opt = pyneat.SimaDecodeOptions()
opt.type = pyneat.SimaDecodeType.H264
opt.out_format = pyneat.Format.NV12
opt.decoder_name = "decoder"
opt.dec_width = stream.frame_w
opt.dec_height = stream.frame_h
opt.dec_fps = stream.source_fps
decoder.add(pyneat.nodes.sima_decode(opt))
decoder.add(pyneat.nodes.caps_raw("NV12", stream.frame_w, stream.frame_h,
                                  stream.source_fps, pyneat.CapsMemory.Any))
```

Complete flow: [multi-stream-object-detector](../../../examples/object-detection/multi-stream-object-detector/)

`VideoRate` followed by `CapsRaw`, added when the configured frame rate differs from the source:
[yolo26-tiny-drone-tracker](../../../examples/tracking/yolo26-tiny-drone-tracker/)

## Options

`SimaDecodeType`

| Value | C++ | Python | What it does |
|---|---|---|---|
| H.264 | `SimaDecodeType::H264`, alias `SimaDecodeType::AVC` | `pyneat.SimaDecodeType.H264`, alias `pyneat.SimaDecodeType.AVC` | H.264 stream |
| JPEG | `SimaDecodeType::JPEG` | `pyneat.SimaDecodeType.JPEG` | A single JPEG frame |
| MJPEG | `SimaDecodeType::MJPEG` | `pyneat.SimaDecodeType.MJPEG` | A stream of JPEG frames, for example from `UsbCameraInput` |
| H.265 | `SimaDecodeType::H265`, alias `SimaDecodeType::HEVC` | `pyneat.SimaDecodeType.H265`, alias `pyneat.SimaDecodeType.HEVC` | H.265 Main profile, 8-bit 4:2:0 |

`SimaDecodeOptions`

SiMa memory is buffer memory that the board's processors can read directly.
The MLA (Machine Learning Accelerator) runs the model. APU is the application
processor, the board's general-purpose CPU.

| Option | C++ | Python | Default | What it does | Why and when to change it |
|---|---|---|---|---|---|
| Type | `type` | `type` | `SimaDecodeType::H264` | Encoded input type. See the `SimaDecodeType` table | The stream is H.265, JPEG or MJPEG. Every shipped example sets it |
| Allocator type | `sima_allocator_type` | `sima_allocator_type` | `2` | Memory allocator for the decoded frames | Not changed in any shipped example; keep the default |
| Output format | `out_format` | `out_format` | `NV12` | Pixel format of the decoded frames. With `raw_output = true`: `NV12`, `I420`, or `NV16` for JPEG and MJPEG. With `false`, any raw format. `I420` turns off zero-copy output | A 4:2:2 USB camera: `single-stream-object-detector` sets `usb_camera_info().decoded_format`, which can be `NV16` |
| Decoder name | `decoder_name` | `decoder_name` | `""` | Name of the decoder, also used as the name of its output buffer. Empty names it `n<index>_decoder` | The examples set `"decoder"`, because a `Model` reads the buffer named by `Model::Options::upstream_name`, which is `"decoder"` by default |
| Raw output | `raw_output` | `raw_output` | `true` | `true` outputs the decoder's frames in SiMa memory. `false` adds a format converter and outputs `out_format` in system memory | Not changed in any shipped example; keep the default |
| Next element | `next_element` | `next_element` | `""` | Tells the decoder which processor reads its output: `CVU` or `APU` | `high-density-multi-stream-object-detector` sets `CVU` for decoded frames that go to the model |
| Decode width | `dec_width` | `dec_width` | `-1` | Width of the decoded frames. `-1` takes it from the stream | The examples set the stream width they probed |
| Decode height | `dec_height` | `dec_height` | `-1` | Height of the decoded frames. `-1` takes it from the stream | Same as decode width |
| Decode FPS | `dec_fps` | `dec_fps` | `-1` | Frame rate of the decoded frames. `-1` takes it from the stream | The examples set the stream frame rate they probed |
| Output buffers | `num_buffers` | `num_buffers` | `-1` | How many decoded frames the decoder can hold. `-1` keeps the decoder default | Later nodes keep frames while the decoder needs free ones. `rfdetr-detection-segmentation` sets `32` for MJPEG "while inference and preview retain frames" |
| Input buffers | `input_buffers` | `input_buffers` | `-1` | How many encoded frames the decoder can queue. `-1` keeps the decoder default | `high-density-multi-stream-object-detector` sets `2` from its config |
| Decoder tuning | `decoder_tuning` | `decoder_tuning` | `""` | Decoder tuning preset. The high-density example accepts `auto`, `default`, `low-memory` and `throughput-low-latency` | Many streams on one board. `high-density-multi-stream-object-detector` sets `throughput-low-latency` in its config |
| Memory optimization | `memory_opt` | `memory_opt` | `false` | Turns on the decoder's memory-optimized path | `high-density-multi-stream-object-detector` sets `true` together with the `low-memory` or `throughput-low-latency` tuning |

`CapsRaw(format, width, height, fps, memory)`

| Parameter | C++ | Python | Default | What it does | Why and when to change it |
|---|---|---|---|---|---|
| Format | `format` | `format` | Required | Pixel format, for example `"NV12"` or `"RGB"`. Empty leaves the format open | Set the format the next node needs |
| Width | `width` | `width` | `-1` | Width in pixels. `-1` leaves it open | The next node needs a fixed size, for example a model input |
| Height | `height` | `height` | `-1` | Height in pixels. `-1` leaves it open | Same as width |
| FPS | `fps` | `fps` | `-1` | Whole frames per second, written as `fps/1`. `-1` leaves it open | A later node needs a frame rate, or `VideoRate` needs a target |
| Memory | `memory` | `memory` | `CapsMemory::Any` | Memory the caps announce. See the `CapsMemory` table | Not changed in any shipped example; keep the default |

`CapsRaw` has two shortcuts: `CapsNV12SysMem(w, h, fps)` is `CapsRaw("NV12", w, h, fps, CapsMemory::SystemMemory)`, and `CapsI420(w, h, fps, memory)` is `CapsRaw("I420", ...)`.

`CapsMemory`

| Value | C++ | Python | What it does | When to use it |
|---|---|---|---|---|
| Any | `CapsMemory::Any` | `pyneat.CapsMemory.Any` | Leaves the memory open | Default. The examples set `Any` |
| System memory | `CapsMemory::SystemMemory` | `pyneat.CapsMemory.SystemMemory` | The caps require system memory. Nothing is copied, so the node before it must already output system memory, for example `VideoConvert` | Not changed in any shipped example; keep the default |

`VideoConvert()`, `VideoScale()`, `VideoRate()`

These nodes have no options. Each takes its target from a `CapsRaw` after it.

| Node | C++ | Python | What it does |
|---|---|---|---|
| VideoConvert | `nodes::VideoConvert()` | `pyneat.nodes.video_convert()` | Converts the pixel format, for example NV12 to RGB |
| VideoScale | `nodes::VideoScale()` | `pyneat.nodes.video_scale()` | Resizes the frames in software |
| VideoRate | `nodes::VideoRate()` | `pyneat.nodes.video_rate()` | Lowers the frame rate by dropping frames. It never repeats frames and keeps the input timestamps |

The `CapsRaw` header lists `SysMem` and `MlaShm` as memory examples, but
`CapsMemory` has only `Any` and `SystemMemory`.

## Language differences

- C++ `SimaDecodeOptions::out_format` accepts a string such as `"NV12"` or a
  `FormatTag`. Python accepts only `pyneat.Format`, for example
  `pyneat.Format.NV12`.
- Python `caps_raw` takes the format as a string such as `"NV12"`, not as
  `pyneat.Format`.
- `CapsI420` is not available in Python.
- Python also has `pyneat.nodes.format_filter(...)`, another name for
  `caps_raw`.

## Pitfalls

- **An output format the decoder cannot produce.** With `raw_output = true`,
  an `out_format` other than `NV12`, `I420`, or `NV16` for JPEG and MJPEG
  fails with
  `SimaDecode: raw_output supports only NV12 or I420 (or NV16 for JPEG/MJPEG)`.
  Fix: decode to `NV12` and add `VideoConvert` and `CapsRaw` with the format
  you need, for a consumer other than a model; see the next pitfall. A name
  Core does not know, such as `"RGBA"`, does not fail: it becomes `NV12`.
- **`raw_output = false` moves frames to system memory.** `SimaDecode` then
  adds a format converter after the decoder and announces `out_format` in
  system memory, so the frames leave SiMa memory. Fix: keep
  `raw_output = true` for frames that go to a model.
- **A converter in front of a model loses the frame metadata.** Frames from
  `VideoConvert`, or from `raw_output = false`, failed at the model on a
  DevKit with `input.meta: GstSimaMeta needs nonnegative frame-id and
  timestamp`. Fix: convert on the host and push the image to a
  `Model::Runner`.
- **A fractional frame rate in `CapsRaw`.** `fps` is a whole number written
  as `fps/1`, so a rate such as `30000/1001` cannot be set.
- **The decoded format does not match the model.** A model's preprocessing is
  prepared for one input format. When frames arrive in another format, or larger than the prepared size, the
  preprocessing rejects them at run time with
  `video caps exceed the prepared format/capacity envelope`. Fix: set the
  model's preprocessing input format to `out_format`, for example `NV16` for a
  4:2:2 USB camera. Read in the source, not verified on a device.

## Used in

- [multi-stream-object-detector](../../../examples/object-detection/multi-stream-object-detector/) (`SimaDecode` after `RtspEncodedInput`, `CapsRaw` NV12)
- [high-density-multi-stream-object-detector](../../../examples/object-detection/high-density-multi-stream-object-detector/) (`next_element`, `input_buffers`, `decoder_tuning`, `memory_opt`)
- [rfdetr-detection-segmentation](../../../examples/object-detection/rfdetr-detection-segmentation/) (`num_buffers` for MJPEG)
- [single-stream-object-detector](../../../examples/object-detection/single-stream-object-detector/) (`SimaDecode` MJPEG after `UsbCameraInput`)
- [multi-stream-multi-model](../../../examples/multi-model/multi-stream-multi-model/) (one `SimaDecode` per stream, each feeding its own model)
- [yolo26-tiny-drone-tracker](../../../examples/tracking/yolo26-tiny-drone-tracker/) (`VideoRate`, `CapsRaw`)

## Source

- C++ header: `include/nodes/sima/SimaDecode.h` (`SimaDecodeType`, `SimaDecodeOptions`, `SimaDecode`), `include/nodes/common/Caps.h` (`CapsRaw`, `CapsNV12SysMem`, `CapsI420`), `include/contracts/ContractTypes.h` (`CapsMemory`), `include/nodes/common/VideoConvert.h`, `include/nodes/common/VideoScale.h`, `include/nodes/common/VideoRate.h`, `include/builder/NextCpuPolicy.h` (APU)
- Python binding: `python/src/module.cpp`, `SimaDecodeType`, `SimaDecodeOptions`, `CapsMemory`, `nodes.sima_decode`, `nodes.caps_raw`, `nodes.format_filter`, `nodes.caps_nv12_sys_mem`, `nodes.video_convert`, `nodes.video_scale`, `nodes.video_rate`
