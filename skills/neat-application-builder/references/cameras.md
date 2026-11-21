# Cameras

Core's `mipi-camera-input.md` and `usb-camera-input.md`, under
`docs/develop-apps/advanced-concepts/application-design/`, list the supported cameras and
modes. `include/nodes/io/CameraInput.h` and `UsbCameraInput.h` define the options, and
[Cameras](api/inputs/cameras.md) explains them.

- **MIPI camera:** capture is zero-copy. Set `allow_cpu_fallback` only as a deliberate
  compatibility choice. `CameraInputOptions::leaky_queue` defaults to `true` with a
  `queue_depth` of 2, so a graph slower than the camera drops frames without an error. Set it
  to `false` on a path that must process every frame.
- **Capture buffers:** on Platform 3.0, create the camera with `CameraInput` rather than
  `CameraInputWithCaptureBuffers`. Platform 3.0's `libcamerasrc` has no `buffer-count`
  property, so a nonzero capture buffer count fails at graph build.
- **MIPI camera into a model:** make the camera the graph's first node and add the model with
  `Model::RouteOptions`: `include_input = false`, and `upstream_name` and `buffer_name` set to
  the camera's `buffer_name`. The frames then reach the model without a copy or a CPU
  conversion.
- **USB webcam:** open it by a stable ID from `usb_camera_ids()` at an exact MJPEG mode the
  camera lists. Call `usb_camera_info(options)` once and pass the result to `UsbCameraInput`.
  Its `decoded_format` (NV12, or NV16 for 4:2:2 webcams) is the `SimaDecode` output format and
  the model's preprocessing input format. Forward the JPEG frames to Insight with
  `Passthrough(RtspCodec::MJPEG)`.
