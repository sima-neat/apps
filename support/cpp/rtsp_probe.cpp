#include "support/cpp/rtsp_probe.h"

#include <opencv2/videoio.hpp>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace sima_examples {

namespace {

int fps_from_rate(const std::string& value) {
  if (value.empty() || value == "0/0" || value == "0/1")
    return 0;
  try {
    const auto slash = value.find('/');
    double fps = 0.0;
    if (slash == std::string::npos) {
      fps = std::stod(value);
    } else {
      const double den = std::stod(value.substr(slash + 1));
      if (den <= 0.0)
        return 0;
      fps = std::stod(value.substr(0, slash)) / den;
    }
    return fps > 0.0 ? static_cast<int>(std::lround(fps)) : 0;
  } catch (...) {
    return 0;
  }
}

void fill_missing_stream_info(RtspStreamInfo& dst, const RtspStreamInfo& src) {
  if (dst.width <= 0)
    dst.width = src.width;
  if (dst.height <= 0)
    dst.height = src.height;
  if (dst.fps <= 0)
    dst.fps = src.fps;
}

RtspStreamInfo probe_ffprobe_rtsp_stream_info(const std::string& url, bool rtsp_tcp) {
  RtspStreamInfo info;
  const std::string command = build_ffprobe_rtsp_stream_info_command(url, rtsp_tcp);

  FILE* pipe = popen(command.c_str(), "r");
  if (!pipe) {
    return info;
  }

  int avg_fps = 0;
  int r_fps = 0;
  std::array<char, 256> buffer{};
  while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe)) {
    std::string line(buffer.data());
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
      line.pop_back();
    }
    const auto eq = line.find('=');
    if (eq == std::string::npos) {
      continue;
    }
    const std::string key = line.substr(0, eq);
    const std::string value = line.substr(eq + 1);
    if (key == "width") {
      info.width = std::atoi(value.c_str());
    } else if (key == "height") {
      info.height = std::atoi(value.c_str());
    } else if (key == "avg_frame_rate") {
      avg_fps = fps_from_rate(value);
    } else if (key == "r_frame_rate") {
      r_fps = fps_from_rate(value);
    }
  }
  pclose(pipe);
  info.fps = avg_fps > 0 ? avg_fps : r_fps;
  return info;
}

RtspStreamInfo probe_opencv_rtsp_stream_info(const std::string& url) {
  RtspStreamInfo info;
  cv::VideoCapture cap(url);
  if (!cap.isOpened()) {
    return info;
  }
  info.width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
  info.height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  info.fps = static_cast<int>(std::lround(cap.get(cv::CAP_PROP_FPS)));
  cap.release();
  return info;
}

} // namespace

std::string build_ffprobe_rtsp_stream_info_command(const std::string& url, bool rtsp_tcp) {
  std::string quoted_url = "'";
  for (const char c : url) {
    quoted_url += c == '\'' ? "'\\''" : std::string(1, c);
  }
  quoted_url += "'";

  return "ffprobe -v error " + std::string(rtsp_tcp ? "-rtsp_transport tcp " : "") +
         "-rw_timeout 5000000 -select_streams v:0 "
         "-show_entries stream=width,height,r_frame_rate,avg_frame_rate -of default=nw=1 " +
         quoted_url + " 2>/dev/null";
}

bool probe_rtsp_stream_info(const std::string& url, const RtspProbeOptions& opt,
                            RtspStreamInfo& out) {
  out = RtspStreamInfo{};
  fill_missing_stream_info(out, probe_ffprobe_rtsp_stream_info(url, opt.rtsp_tcp));
  fill_missing_stream_info(out, probe_opencv_rtsp_stream_info(url));

  return out.width > 0 && out.height > 0;
}

} // namespace sima_examples
