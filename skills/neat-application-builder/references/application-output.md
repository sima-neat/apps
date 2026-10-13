# Application output

Choose output handling from the application's contract. Apps repository
contributions use Insight for real-time visualization. Offline and batch
applications may save results or display annotated images locally. For a
standalone application, follow its project's visualization requirements and
prefer Insight for remote browser viewing.

## Insight video and metadata

- Add the public `VideoSender` graph fragment for video. Use
  `VideoSenderOptions::H264RtpUdpFromRaw(...)` for raw frames or
  `VideoSenderOptions::Passthrough(codec)` for encoded H.264 or H.265.
- Pair it with `MetadataSender` when Insight needs JSON detections or other
  structured overlays.
- Inspect `include/nodes/groups/VideoSender.h` and
  `include/nodes/io/MetadataSender.h`. Read the matching Core docs under
  `docs/develop-apps/advanced-concepts/application-design/`, specifically
  `video_sender.md` and `metadata_sender.md`, when choosing ports, channels,
  codecs, or graph links.
- Use the Insight documentation or the installed Insight skill for viewer
  setup and service operations.

## Local display when required

For permitted local display, verify the selected GStreamer sink or OpenCV
backend on the target. Follow the installed platform documentation for display
session, DRM, connector, and driver requirements. Device setup and desktop
service management belong to platform tooling, outside application code.

For an OpenCV window, handle the close button as well as keyboard exit. Check
`getWindowProperty(name, WND_PROP_VISIBLE) < 1` before another `imshow` call
recreates a closed window. Verify close and keyboard behavior in the target's
actual display session.
