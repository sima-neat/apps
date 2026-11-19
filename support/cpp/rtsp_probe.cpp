#include "support/cpp/rtsp_probe.h"

#include <opencv2/videoio.hpp>

#include <cmath>
#include <cstdlib>

namespace sima_examples {

bool probe_rtsp_stream_info(const std::string& url, const RtspProbeOptions& opt,
                            RtspStreamInfo& out) {
  out = RtspStreamInfo{};
  // OpenCV's FFmpeg backend reads its RTSP options from this variable when it opens the URL.
  setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS",
         opt.rtsp_tcp ? "rtsp_transport;tcp" : "rtsp_transport;udp", 1);
  cv::VideoCapture capture(url, cv::CAP_FFMPEG);
  if (!capture.isOpened()) {
    return false;
  }
  out.width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
  out.height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
  out.fps = static_cast<int>(std::lround(capture.get(cv::CAP_PROP_FPS)));
  return out.width > 0 && out.height > 0;
}

} // namespace sima_examples
