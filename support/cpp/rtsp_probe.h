#pragma once

// Read an RTSP stream's frame size and frame rate before the graph is built. Mirrors
// sima_examples.rtsp_probe in support/python.

#include <string>

namespace sima_examples {

struct RtspProbeOptions {
  int payload_type = 96;
  int latency_ms = 200;
  bool rtsp_tcp = true;
  bool debug = false;
  int decoder_num_buffers = 7;
};

struct RtspStreamInfo {
  int width = 0;
  int height = 0;
  int fps = 0;
};

// Fills `out` from ffprobe, then OpenCV for any value ffprobe left unset. Returns whether
// the frame size is known.
bool probe_rtsp_stream_info(const std::string& url, const RtspProbeOptions& opt,
                            RtspStreamInfo& out);

// The ffprobe command the probe runs, with the URL quoted for the shell.
std::string build_ffprobe_rtsp_stream_info_command(const std::string& url, bool rtsp_tcp);

} // namespace sima_examples
