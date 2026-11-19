#include "sima_examples/rtsp_probe.h"

#include <opencv2/videoio.hpp>

#include <cmath>
#include <cstdlib>
#include <stdexcept>

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

RtspStreamInfo probe_rtsp_stream_info(const std::string& url, bool tcp) {
  const RtspStreamInfo info = read_stream_info(url, tcp);
  // An unreachable source and one that reports no frame rate fail the same way, as in Python.
  if (info.fps <= 0) {
    throw std::runtime_error("failed to probe RTSP frame rate: " + url);
  }
  return info;
}

bool probe_rtsp_stream_info(const std::string& url, const RtspProbeOptions& opt,
                            RtspStreamInfo& out) {
  out = read_stream_info(url, opt.rtsp_tcp);
  return out.width > 0 && out.height > 0;
}

} // namespace sima_examples
