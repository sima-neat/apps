#pragma once

// Read an RTSP stream's frame size and frame rate before the graph is built. Mirrors
// sima_examples.rtsp_probe in support/python.

#include <string>

namespace sima_examples {

struct RtspStreamInfo {
  int width = 0;
  int height = 0;
  int fps = 0;
};

// Reads the stream with OpenCV's FFmpeg backend, over TCP when `tcp` is set. Throws when the
// frame rate is unknown, including when the source cannot be opened.
RtspStreamInfo probe_rtsp_stream_info(const std::string& url, bool tcp);

// The form the examples not yet converted call. Only rtsp_tcp changes the probe. Returns
// whether the frame size is known and leaves an unknown frame rate at 0.
struct RtspProbeOptions {
  int payload_type = 96;
  int latency_ms = 200;
  bool rtsp_tcp = true;
  bool debug = false;
};

bool probe_rtsp_stream_info(const std::string& url, const RtspProbeOptions& opt,
                            RtspStreamInfo& out);

} // namespace sima_examples
