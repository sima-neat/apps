#include "sima_examples/rtsp_probe.h"

#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace sima_examples {
namespace {

RtspStreamInfo read_stream_info(const std::string& url, bool tcp) {
  // OpenCV's FFmpeg backend reads its RTSP options from this variable when it opens the URL.
  setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", tcp ? "rtsp_transport;tcp" : "rtsp_transport;udp", 1);
  cv::VideoCapture capture(url, cv::CAP_FFMPEG);
  RtspStreamInfo info;
  if (capture.isOpened()) {
    info.width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
    info.height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    info.fps = static_cast<int>(std::lround(capture.get(cv::CAP_PROP_FPS)));
  }
  return info;
}

} // namespace

RtspStreamInfo probe_rtsp_stream_info(const std::string& url, bool tcp,
                                      const RtspStreamInfo& configured) {
  if (std::min({configured.width, configured.height, configured.fps}) > 0) {
    return configured;
  }
  const RtspStreamInfo probed = read_stream_info(url, tcp);
  RtspStreamInfo info;
  info.width = configured.width > 0 ? configured.width : probed.width;
  info.height = configured.height > 0 ? configured.height : probed.height;
  info.fps = configured.fps > 0 ? configured.fps : probed.fps;
  // An unreachable source fails here too, as in Python.
  if (info.width <= 0 || info.height <= 0) {
    throw std::runtime_error("failed to probe RTSP frame size: " + url +
                             "; set input.width and input.height");
  }
  if (info.fps <= 0) {
    throw std::runtime_error("failed to probe RTSP frame rate: " + url + "; set input.fps");
  }
  return info;
}

std::vector<RtspStreamInfo> probe_rtsp_streams(const std::vector<std::string>& urls, bool tcp,
                                               const RtspStreamInfo& configured) {
  std::map<std::string, RtspStreamInfo> by_url;
  std::string first_url;
  for (const auto& url : urls) {
    if (by_url.count(url) > 0) {
      continue;
    }
    const RtspStreamInfo stream = probe_rtsp_stream_info(url, tcp, configured);
    by_url[url] = stream;
    if (first_url.empty()) {
      first_url = url;
      continue;
    }
    // Core gap: a model shared by several streams accepts one input size, and a stream of
    // another size fails the build with "input spec mismatch".
    const RtspStreamInfo& first = by_url.at(first_url);
    if (stream.width != first.width || stream.height != first.height) {
      throw std::runtime_error(
          "every input.source stream needs the same frame size, because the streams share one "
          "model: " +
          first_url + " is " + std::to_string(first.width) + "x" + std::to_string(first.height) +
          ", " + url + " is " + std::to_string(stream.width) + "x" + std::to_string(stream.height));
    }
  }
  std::vector<RtspStreamInfo> streams;
  for (const auto& url : urls) {
    streams.push_back(by_url.at(url));
  }
  return streams;
}

bool probe_rtsp_stream_info(const std::string& url, const RtspProbeOptions& opt,
                            RtspStreamInfo& out) {
  out = read_stream_info(url, opt.rtsp_tcp);
  return out.width > 0 && out.height > 0;
}

} // namespace sima_examples
