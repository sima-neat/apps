# API Decision Map

Choose the public API shape before writing code. Start with the artifact and
input owner, then select the smallest API that fits the application.

## Artifact and input boundary

- Classic applications consume a compiled model archive, not ONNX. Complete
  compilation before application design.
- Decide whether the caller supplies decoded inputs or the `Graph` owns a source
  and topology.
- Inspect the model package before adding preprocessing or postprocessing.

## Classic Model

Use `Model.run(...)` when:

- The caller supplies decoded images or tensors.
- Synchronous request/response behavior is acceptable.
- The model-owned route is the complete pipeline.

Use `Model.build(...)` when:

- The caller still owns input and request cadence.
- The app needs repeated `push` / `pull`.
- The model-owned `Runner` interface is enough.

## Graph

Use `Graph` when:

- The `Graph` owns a camera, file, RTSP stream, encoded source, or other
  producer.
- The application composes decoding, preprocessing, or output paths around the
  model route.
- The application needs multiple stages, named endpoints, branching, or fan-in.

With encoded RTSP input, decide whether to decode for inference, forward the
encoded stream to `VideoSender`, or branch to both. Use the same codec across
`RtspEncodedInput`, `SimaDecode`, and `VideoSenderOptions::Passthrough`. RTSP
inputs drop late packets by default (`drop_on_latency`); disable it only when
every frame matters more than freshness.

## Camera source

- MIPI/CSI camera: `nodes::CameraInput` / `pyneat.nodes.camera_input`. Capture
  is strict zero-copy; set `allow_cpu_fallback` only as a deliberate
  compatibility choice. Read Core's `mipi-camera-input.md` before changing
  capture buffer counts.
- USB (UVC) webcam: `nodes::UsbCameraInput` /
  `pyneat.nodes.usb_camera_input`, opened by a stable ID from
  `usb_camera_ids()` at an exact MJPEG mode the camera lists. Call
  `usb_camera_info(options)` once and pass the result to `UsbCameraInput`. Its
  `decoded_format` (NV12, or NV16 for 4:2:2 webcams) is the `SimaDecode`
  output format and the model's preprocess input format. Forward the JPEG
  frames to Insight with `Passthrough(RtspCodec::MJPEG)` rather than decoding
  and re-encoding. Core's `usb-camera-input.md` states which cameras and modes Neat supports.

Both camera docs live in `docs/develop-apps/advanced-concepts/application-design/` in Core
source.

## GenAI

Use `GenAIModel` when:

- The artifact is a deployed LLiMa model directory.
- The app calls an LLM, VLM, or ASR model in-process.
- The app needs `run(...)`, `stream(...)`, task detection, or model capability checks.

Use task-specific GenAI handles when the app knows the task:

- `VisionLanguageModel` for text-only LLMs and image-capable VLMs.
- `ASRModel` for speech-to-text.

Use `GenAIServer` when:

- The app boundary is HTTP.
- A browser, UI, companion service, or remote client should call one or more models.
- The server owns model registration and request routing.

Use GenAI graph fragments only when GenAI is one stage inside a larger Neat
`Graph`.
