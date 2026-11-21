# Input

## What it is

`nodes::Input` is the node through which the application hands its own data to
a graph. The application produces a frame or a tensor, for example with OpenCV,
and passes it in with `Run::push()`. This is called push mode. A graph that
starts with a source node such as an RTSP or camera input reads data by itself
instead.

`InputOptions` describes what the application pushes: the payload type (image,
tensor, or encoded video), the format, the size, and the frame rate. Core
turns these fields into caps: the format, size and frame rate that the input
announces to the next node. `InputOptions` also sets how much data may wait at
the input and in which memory Core allocates the input buffers.
`InputMemoryPolicy` names those memory targets: plain memory of the board's
CPU, memory that EV74 can read, or DMS0. EV74, also called CVU, is the board's
vision processor that runs preprocessing. The MLA (Machine Learning
Accelerator) runs the model, and DMS0 is the memory it reads its input from.

`nodes::Input("name")` also gives the graph a named endpoint: an input that
other code finds by its name. A fragment is a graph made to be joined to other
graphs, for example the graph that `model.graph()` returns. `Graph::connect()`
joins another fragment to the named input, see the Links page of this reference.

## When to use it

Use `Input` when the application owns frame production: it reads images from
a directory, draws an overlay and sends the result to Insight, or decodes
frames on the board's CPU and feeds them to a model graph that runs on its own.

Do not use it when Core can read the source by itself. For an RTSP stream use
[RtspDecodedInput or RtspEncodedInput](rtsp.md). For a MIPI or USB camera use
[CameraInput or UsbCameraInput](cameras.md). Those groups read and decode the
stream without a copy through the application.

A graph that starts with `Input` is built with `Graph::build(seed)` and fed
with `Run::push()`. It is not started with `Run::run()`.

## Minimal example

Push BGR frames from OpenCV into a graph that sends them to Insight. The graph
is built once with a black frame, `seed`, and then each new frame, `tensor`, is
pushed. How to make a tensor from an OpenCV image is covered on the tensors
page.

C++

```cpp
neat::InputOptions input_options;
input_options.payload_type = neat::PayloadType::Image;
input_options.format = "BGR";
input_options.width = width;
input_options.height = height;
input_options.depth = 3;
input_options.fps_n = output_fps;
graph.add(neat::nodes::Input(input_options));
neat::Run run = graph.build(neat::TensorList{seed});
run.push(neat::TensorList{tensor});
```

Python

```python
input_options = pyneat.InputOptions()
input_options.payload_type = pyneat.PayloadType.Image
input_options.format = pyneat.Format.BGR
input_options.width = width
input_options.height = height
input_options.depth = 3
input_options.fps_n = output_fps
graph.add(pyneat.nodes.input(input_options))
run = graph.build([seed])
run.push([tensor])
```

Complete flow: [superpoint-feature-extractor](https://github.com/sima-neat/apps/tree/main/examples/feature-extraction/superpoint-feature-extractor)

To feed a model, take the options from the model instead of writing them by
hand: `model.input_appsrc_options(true)` returns the format, size, memory
policy and preprocessing metadata that the model expects. See
[face-detector](https://github.com/sima-neat/apps/tree/main/examples/face-detection/face-detector).

## Options

`nodes::Input` has these forms:

| Form | C++ | Python | What it does |
|---|---|---|---|
| Options only | `nodes::Input(opt)` | `pyneat.nodes.input(opt)` | Unnamed push input. Options default to `InputOptions{}` |
| Name only | `nodes::Input("name")` | `pyneat.nodes.input("name")` | Named endpoint with default options. Used by `Graph::connect()` |
| Name and options | `nodes::Input("name", opt)` | `pyneat.nodes.input("name", opt)` | Named endpoint with fixed caps, for example the model input of a graph that runs on its own |

`InputOptions`

| Option | C++ | Python | Default | What it does | Why and when to change it |
|---|---|---|---|---|---|
| Payload type | `payload_type` | `payload_type` | `PayloadType::Auto` | Kind of data pushed: image, tensor, or encoded video. `Auto` takes it from the seed sample passed to `build()`. See the `PayloadType` table | Set it whenever you set a format or size, so the caps do not depend on the seed |
| Format | `format` | `format` | Empty, taken from the seed | Pixel or tensor format, for example `BGR`, `NV12`, `FP32`, `H264` | Set it to the format you push. An encoded input needs `H264` or `H265` here |
| Width | `width` | `width` | `-1`, taken from the seed | Frame width in pixels. For a tensor, the second dimension | Set it when every frame has the same size. It also becomes the default `max_width` |
| Height | `height` | `height` | `-1`, taken from the seed | Frame height in pixels. For a tensor, the first dimension | Same as width |
| Depth | `depth` | `depth` | `-1`. `BGR` and `RGB` resolve to 3, `GRAY8` to 1 | Channels per pixel, or the third tensor dimension | `1` for `NV12`, `3` for `BGR` and `RGB`, the channel count for a tensor |
| Max width | `max_width` | `max_width` | `-1`, which becomes `width` when `width` is set | Largest width `push()` accepts. Also sizes the buffer pool | Frames change size at run time, for example a resolution switch. Set the largest size you push |
| Max height | `max_height` | `max_height` | `-1`, which becomes `height` when `height` is set | Largest height `push()` accepts | Same as max width |
| Max depth | `max_depth` | `max_depth` | `-1`, which becomes `depth` when `depth` is set | Largest depth `push()` accepts | Tensors whose channel count changes |
| Frame rate numerator | `fps_n` | `fps_n` | `0` (no frame rate in the caps) | Frame rate written into the caps, `fps_n / fps_d` | An encoder downstream needs a frame rate, for example `VideoSender` |
| Frame rate denominator | `fps_d` | `fps_d` | `1` | Denominator of the frame rate | Fractional rates such as `30000/1001` |
| Caps override | `caps_override` | `caps_override` | `""` | Full caps string that replaces every field above. It also locks the shape | Multi-tensor caps or caps that the fields cannot express. Otherwise leave it empty |
| Live source | `is_live` | `is_live` | `true` | Marks the input as live, so the pipeline paces buffers by the clock | Set `false` for offline batches that should run as fast as the graph can process them |
| Timestamp on push | `do_timestamp` | `do_timestamp` | `true` | Stamps each pushed buffer with the current time | Set `false` when you push samples that already carry their own timestamps |
| Block when full | `block` | `block` | `true` | `push()` waits when the input already holds `max_bytes` of data | Keep `true` so no frame is lost and push stays paired with pull. See the pitfalls before setting `false` |
| Stream type | `stream_type` | `stream_type` | `0` (stream) | Stream type of the input: `0` stream, `1` seekable, `2` random access | Rarely. Pushed frames are a stream |
| Max bytes | `max_bytes` | `max_bytes` | `0` | How much data may wait at the input before `push()` blocks. At build, Core replaces `0` with 4 frames' worth when the frame size is known | Bound the queue to the frames you allow in flight, for example `width * height * 3 / 2 * buffers` for NV12 |
| SiMa pool (deprecated) | `use_simaai_pool` | `use_simaai_pool` | `true` | Old switch for SiMa memory. `false` means `SystemMemory` | Never in new code. Use `memory_policy` |
| Pool min buffers | `pool_min_buffers` | `pool_min_buffers` | `1` | Minimum buffers in the input pool | Rarely |
| Pool max buffers | `pool_max_buffers` | `pool_max_buffers` | `2` | Maximum buffers in the input pool. A synchronous build is the one-frame-in, one-result-out build that `Graph::run(inputs)` makes; `build(seed)` is not one. In a synchronous build, when both pool options keep their defaults, Core raises this to the largest buffer count of the pipeline. A value you set is kept | More frames must be in flight at once, for example a model input shared by several streams |
| Memory policy | `memory_policy` | `memory_policy` | `InputMemoryPolicy::Auto` | Where Core allocates input buffers. See the `InputMemoryPolicy` table | Keep `Auto`. Set `SystemMemory` when the whole graph runs on the board's CPU. Set `Ev74` or `Dms0` only together with tensors created in that memory, otherwise `push()` fails with `CPU-backed Tensor pushed into a device-visible EV74/DMS route` |
| Buffer name | `buffer_name` | `buffer_name` | `""` | Name of the input buffer in SiMa metadata. Also the stream label in Core logs | A model route expects a specific buffer name |
| Preprocess metadata | `preprocess_meta` | `preprocess_meta` | Not set (`std::nullopt`, `None`) | Geometry of the model preprocessing (target size, letterbox, color), attached to each buffer so box decoding can map boxes back to the original frame | Do not fill it by hand. `Model::input_appsrc_options()` fills it |

The comments in `Input.h` are out of date for two options. The header says
`max_bytes = 0` means unlimited, but Core replaces `0` with 4 frames' worth at
build when the frame size is known. The header says `block` applies when the
buffer pool is exhausted, but Core applies it to the input queue, so `push()`
waits when the queue holds `max_bytes`. The table above describes what Core
does.

`PayloadType`

| Value | C++ | Python | What it does | When to use it |
|---|---|---|---|---|
| Auto | `PayloadType::Auto` | `pyneat.PayloadType.Auto` | Takes the type from the seed sample | You give no format or size and let the seed decide |
| Image | `PayloadType::Image` | `pyneat.PayloadType.Image` | Decoded pixels | OpenCV frames, NV12 frames from a decoder |
| Tensor | `PayloadType::Tensor` | `pyneat.PayloadType.Tensor` | Tensor data | Already preprocessed model input, for example `FP32` |
| Encoded | `PayloadType::Encoded` | `pyneat.PayloadType.Encoded` | Encoded video such as H.264, H.265 or JPEG | Forward an encoded stream to a decoder or to Insight |

`InputMemoryPolicy`

| Value | C++ | Python | What it does | When to use it |
|---|---|---|---|---|
| Auto | `InputMemoryPolicy::Auto` | `pyneat.InputMemoryPolicy.Auto` | Core looks at the first node after the input, skipping `Cast`. `Preproc`, `Quant`, `Tess`, `QuantTess` and `CastTess` get `Ev74`, and `push()` then requires device tensors. A model fragment gets `Dms0`, also with device tensors required. A node that prefers device memory, for example `VideoSender` fed NV12, gets `Ev74` but still accepts CPU tensors. Any other node gets system memory | Almost always |
| EV74 | `InputMemoryPolicy::Ev74` | `pyneat.InputMemoryPolicy.Ev74` | Memory visible to the EV74 processor. `push()` rejects CPU tensors, so create them with `TensorMemory::EV74` | The next node runs on EV74 and `Auto` cannot see it, for example a named endpoint that feeds Insight video |
| DMS0 | `InputMemoryPolicy::Dms0` | `pyneat.InputMemoryPolicy.Dms0` | DMS0 memory that the MLA reads. `push()` rejects CPU tensors, so create them with `TensorMemory::MLA` | Tensors go straight to the MLA |
| System memory | `InputMemoryPolicy::SystemMemory` | `pyneat.InputMemoryPolicy.SystemMemory` | Plain CPU memory | The graph runs only on the CPU, or the next node copies the data anyway |

## Language differences

- C++ `format` accepts a string such as `"BGR"` or a `FormatTag`. Python
  `format` accepts only `pyneat.Format`, for example `pyneat.Format.BGR`.
- Setting `use_simaai_pool` in Python prints a deprecation warning once.

## Pitfalls

- **A CPU tensor goes into a device input.** `push()` fails with
  `CPU-backed Tensor pushed into a device-visible EV74/DMS route` in these
  cases only:
  - `memory_policy` is `Auto` and the first node after the input, skipping
    `Cast`, is `Preproc`, `Quant`, `Tess`, `QuantTess` or `CastTess`;
  - `memory_policy` is `Auto` and the first node after the input is a model
    fragment;
  - `memory_policy` is set to `Ev74` or `Dms0`, for example on a named
    endpoint that feeds Insight.

  Other nodes, such as `VideoSender`, accept CPU tensors. Fix: create the
  tensor with `TensorMemory::EV74`, or with `TensorMemory::MLA` for a `Dms0`
  input.
- **A larger frame is rejected.** Setting `width` and `height` also sets
  `max_width` and `max_height` to the same values. A larger frame then fails
  with `input width ... exceeds configured capacity`. Fix: set `max_width` and
  `max_height` to the largest frame you push.
- **The seed decides the caps.** When you set no format or size, the first
  sample passed to `build()` decides them. A seed in another format or size
  than the real frames gives wrong caps or a rejected `push()`. Fix: build with
  a seed of the same format and size, or set the options.
- **`run()` on a push graph.** A graph that starts with `Input` waits for
  `push()`. Use `build(seed)` and `push()`, not `run()`.
- **`block = false` lets the input queue grow.** With `false`, `push()` does
  not wait when the input already holds `max_bytes`, so a producer faster than
  the graph keeps adding frames, and memory use and latency grow. Core also
  copies the payload of a non-blocking encoded input instead of passing it
  without a copy. The multi-stream examples set `block = true` explicitly.
  Not verified on a device.

## Used in

- [superpoint-feature-extractor](https://github.com/sima-neat/apps/tree/main/examples/feature-extraction/superpoint-feature-extractor) (BGR frames to Insight)
- [patchcore-anomaly-detector](https://github.com/sima-neat/apps/tree/main/examples/anomaly-detection/patchcore-anomaly-detector) (BGR or NV12 frames to Insight, named endpoints)
- [fastflow-anomaly-detector](https://github.com/sima-neat/apps/tree/main/examples/anomaly-detection/fastflow-anomaly-detector)
- [face-detector](https://github.com/sima-neat/apps/tree/main/examples/face-detection/face-detector) (`model.input_appsrc_options(true)`)
- [face-recognizer](https://github.com/sima-neat/apps/tree/main/examples/face-recognition/face-recognizer) (`model.input_appsrc_options(true)`)
- [multi-stream-multi-model](https://github.com/sima-neat/apps/tree/main/examples/multi-model/multi-stream-multi-model) (NV12 model input with `max_width`, encoded input)
- [high-density-multi-stream-object-detector](https://github.com/sima-neat/apps/tree/main/examples/object-detection/high-density-multi-stream-object-detector) (`max_bytes`, `pool_max_buffers`, `block`)
- [mipi-multi-model](https://github.com/sima-neat/apps/tree/main/examples/benchmarking/mipi-multi-model) (`memory_policy = Ev74`, Python only)
- [single-stream-thermal-face-detector](https://github.com/sima-neat/apps/tree/main/examples/face-detection/single-stream-thermal-face-detector) (named endpoints)

## Source

- C++ header: `include/nodes/io/Input.h` (`Input`, `InputOptions`, `InputMemoryPolicy`, `PreprocessMetaTemplate`), `include/pipeline/PayloadType.h` (`PayloadType`)
- Python binding: `python/src/module.cpp`, `InputOptions`, `InputMemoryPolicy`, `PayloadType`, `PreprocessMetaTemplate`, `nodes.input`
- Core commit: `b1f9556c`
