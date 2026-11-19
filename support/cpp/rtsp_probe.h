#pragma once

// Read an RTSP stream's frame size and frame rate before the graph is built. Mirrors
// sima_examples.rtsp_probe in support/python.

#include <string>

namespace sima_examples {

// Only rtsp_tcp changes the probe. The other fields stay for the examples not yet converted.
struct RtspProbeOptions {
  int payload_type = 96;
  int latency_ms = 200;
  bool rtsp_tcp = true;
  bool debug = false;
};

struct RtspStreamInfo {
  int width = 0;
  int height = 0;
  int fps = 0;
};

// Fills `out` from OpenCV's FFmpeg backend, over TCP when opt.rtsp_tcp is set. Returns whether
// the frame size is known.
bool probe_rtsp_stream_info(const std::string& url, const RtspProbeOptions& opt,
                            RtspStreamInfo& out);

} // namespace sima_examples
