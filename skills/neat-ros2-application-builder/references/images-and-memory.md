# Images, decoder planes and memory

## Match the model before optimizing transport

Establish input color format, dimensions, row/plane strides, normalization,
resize policy and interpolation, quantization, tensor layout and output decoding.
Read the model package and compatible native example. Configure supported Neat
preprocessing explicitly when the reference does; a CPU bilinear resize is not
equivalent to the model's bicubic preprocessor. Preserve maximum-input handling,
letterbox padding and inverse geometry transforms.

Avoid performing a transformation twice. A tensor-mode graph may expect already
normalized data; an image-mode model may perform normalization itself. Confirm
which route is selected rather than inferring it from a model family name.
Some dense outputs require a documented transpose; some outputs are tiled or
encoded payloads. A successful `reshape` is not evidence of the correct layout.

## Identify each boundary separately

```text
encoded source -> decoder surface -> ROS-owned pixels -> DDS -> received pixels
              -> Neat input tensor -> accelerator -> output -> ROS/Insight
```

`OutputMemory.ZeroCopy` applies to a Neat output boundary. It does not prove that
packing, serialization, Python message construction or device transfer avoids
copies. Likewise, a NumPy view can avoid a host copy while a later `cvu()` or
model input transfer still copies. C++ intra-process and loaned-message transport
require compatible APIs, message types and middleware; verify support before
promising it. See the [ROS loaned-message guide](https://github.com/ros2/ros2_documentation/blob/jazzy/source/How-To-Guides/Configure-ZeroCopy-loaned-messages.rst)
for the distinction between loaning support and fallback allocation.

For each proposed optimization, record the owner, lifetime, mutability, byte size
and memory domain before and after the boundary. Remove one identified redundant
copy at a time. Keep exact input/output comparisons and ownership checks alongside
timing. Do not globally replace `copy=True` with `copy=False`.

## Pack actual decoder planes

NV12 contains Y and interleaved UV at half height; NV16 uses full-height UV.
Decoder surfaces can pad rows and place UV after an aligned height, so visible
`width * height` need not be the UV offset. Read roles, offsets and strides from
the tensor's plane metadata. Validate geometry and the visible row bounds against
the backing payload before constructing NumPy views or copying.

Use an application-documented planar wire format only between compatible nodes.
One possible convention uses even width/height, `step = width`, visible image
height, then packed Y and UV bytes. Its payload is `width * height * 3/2` for NV12
or `width * height * 2` for NV16. This convention is not the ordinary `step * height`
Image storage contract and is not supported by every ROS image viewer. State it
explicitly, or use an agreed custom planar message. Offer standard BGR/RGB when
needed; do not silently change visible height to include chroma rows.

When building a Neat tensor from planar ROS pixels, establish image format and
plane metadata before device transfer. Do not infer device plane offsets from a
plain two-dimensional ndarray. Follow the installed API's alignment/layout rules.

## Targeted copy reductions

- **BGR inference preparation:** use a read-only-compatible ndarray view of the
  ROS payload if the next operation already creates an owned tensor or converted
  image. Avoid calling an overlay helper that copies BGR just to read it.
- **Planar publication:** allocate the final ROS-owned payload and copy visible
  decoder rows into it. Avoid an intermediate packed ndarray followed by another
  full-frame copy into `Image.data`.
- **BGR publication:** when the supported OpenCV conversion accepts a destination,
  convert into a correctly shaped contiguous view of the final message buffer.
  Keep the decoder's required color conversion and plane packing unchanged.
- **Overlays:** drawing commonly modifies pixels. Keep an owned rendering buffer
  so visualization cannot corrupt the inference image or another subscriber.
- **Device transfer:** retaining a required CPU snapshot or accelerator copy is
  correct when the runtime cannot borrow that buffer safely. A partial reduction
  is useful even without end-to-end zero-copy.

For Python `sensor_msgs/Image`, supported `array('B')` storage can back a NumPy
view. Validate the generated message setter on the installed ROS version; private
message fields are not a portability shortcut. Account for allocation/zero-fill
cost when benchmarking direct-buffer packing. Keep the decoder Sample alive until
packing finishes. Retain the source message and tensor owners through async use;
never recycle a mutable message/array for the next published frame prematurely.

Do not reinterpret a read-only or strided native buffer as writable merely to
satisfy a DLPack importer. Use the documented compatible view or an owned copy.
Only call dense NumPy conversion on outputs that actually support it. Inspect
native payload/layout metadata when a tensor is tiled or carries encoded results.

## Focused validation

Use padded Y and UV strides, nonzero plane offsets, planar and BGR output, and
both supported planar formats. Assert visible bytes, encoding, geometry, row
step and the full source header. Reject malformed bounds and incompatible formats.
Retain one produced message while the decoder advances/reuses buffers; its bytes
must remain unchanged. Mutating one message must not affect the source or the next
message. Compare real decoded frames and resulting model outputs separately.

For source files, distinguish finite EOF from a stalled live source using the
installed runner's status/error contract. For RTSP, let actual stream metadata
determine rate where supported. A caps filter is not a frame-rate converter or
resizer. Apply decoder low-latency tuning only when its stream constraints are
satisfied; do not copy codec-specific tuning to arbitrary cameras.
