#include "sources.h"

#include "frames.h"

#include <opencv2/videoio.hpp>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <utility>

namespace utils {
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

std::string shell_quote(const std::string& value) {
  std::string out = "'";
  for (const char c : value) {
    out += c == '\'' ? "'\\''" : std::string(1, c);
  }
  out += "'";
  return out;
}

StreamInfo probe_opencv(const std::string& url) {
  StreamInfo info;
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

void fill_missing(StreamInfo& dst, const StreamInfo& src) {
  if (dst.width <= 0)
    dst.width = src.width;
  if (dst.height <= 0)
    dst.height = src.height;
  if (dst.fps <= 0)
    dst.fps = src.fps;
}

std::string build_ffprobe_command(const std::string& url, bool rtsp_tcp, bool tls_verify) {
  return "ffprobe -v error " + std::string(rtsp_tcp ? "-rtsp_transport tcp " : "") +
         "-rw_timeout 5000000 -select_streams v:0 "
         "-show_entries stream=width,height,r_frame_rate,avg_frame_rate -of default=nw=1 " +
         std::string(tls_verify ? "" : "-tls_verify 0 ") + shell_quote(url) + " 2>/dev/null";
}

StreamInfo probe_ffprobe(const std::string& url, bool rtsp_tcp, bool tls_verify) {
  StreamInfo info;
  const std::string command = build_ffprobe_command(url, rtsp_tcp, tls_verify);

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

StreamInfo probe_rtsp_stream_info(const std::string& url, bool rtsp_tcp) {
  StreamInfo info;
  fill_missing(info, probe_ffprobe(url, rtsp_tcp, true));
  fill_missing(info, probe_opencv(url));
  return info;
}

StreamInfo probe_decoded_size(simaai::neat::Graph source) {
  simaai::neat::Graph probe_graph("source_probe");
  probe_graph.add(std::move(source));
  probe_graph.add(simaai::neat::nodes::Output("frame", simaai::neat::OutputOptions::EveryFrame(1)));

  simaai::neat::RunOptions run_options;
  run_options.preset = simaai::neat::RunPreset::Realtime;
  run_options.queue_depth = 3;
  run_options.overflow_policy = simaai::neat::OverflowPolicy::KeepLatest;
  run_options.output_memory = simaai::neat::OutputMemory::ZeroCopy;
  simaai::neat::Run run = probe_graph.build(run_options);

  simaai::neat::Sample sample;
  simaai::neat::PullError pull_error;
  const auto status = run.pull("frame", -1, sample, &pull_error);
  run.close();
  if (status != simaai::neat::PullStatus::Ok) {
    throw std::runtime_error("failed to probe decoded source frame: " + pull_error.message);
  }

  StreamInfo info;
  const auto tensors = simaai::neat::tensors_from_sample(sample, false);
  if (!tensors.empty()) {
    (void)infer_dims(tensors.front(), info.width, info.height);
  }
  return info;
}

} // namespace utils
