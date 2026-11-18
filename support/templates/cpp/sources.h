#pragma once

// Probe a video source for its frame size and rate before the graph is built.

#include <neat.h>

#include <string>

namespace utils {

// A source's frame size and rate. 0 means unknown.
struct StreamInfo {
  int width = 0;
  int height = 0;
  int fps = 0;
};

// Copies each value dst does not know yet from src.
void fill_missing(StreamInfo& dst, const StreamInfo& src);

// The ffprobe command that prints the first video stream's size and rates. rtsp_tcp asks for
// RTSP over TCP; tls_verify false accepts a self-signed HTTPS certificate.
std::string build_ffprobe_command(const std::string& url, bool rtsp_tcp, bool tls_verify);

// The stream's size and rate from ffprobe; each value is 0 when ffprobe is missing, fails or
// does not report it.
StreamInfo probe_ffprobe(const std::string& url, bool rtsp_tcp, bool tls_verify);

// An RTSP stream's size and rate: ffprobe, with OpenCV filling in what ffprobe did not report.
StreamInfo probe_rtsp_stream_info(const std::string& url, bool rtsp_tcp);

// The width and height of the first frame a decoded-source graph produces. The source runs in
// a graph of its own, which is closed before this returns.
StreamInfo probe_decoded_size(simaai::neat::Graph source);

} // namespace utils
