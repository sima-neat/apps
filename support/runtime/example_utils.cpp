#include "example_utils.h"

#include "ffprobe_command.h"

#include <neat.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>

using json = nlohmann::json;

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

void require(bool cond, const std::string& msg) {
  if (!cond)
    throw std::runtime_error(msg);
}

double time_ms() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

bool probe_rtsp_stream_info(const std::string& url, const RtspProbeOptions& opt,
                            RtspStreamInfo& out) {
  out = RtspStreamInfo{};
  fill_missing_stream_info(out, probe_ffprobe_rtsp_stream_info(url, opt.rtsp_tcp));
  fill_missing_stream_info(out, probe_opencv_rtsp_stream_info(url));

  return out.width > 0 && out.height > 0;
}

cv::Mat load_rgb_resized(const std::string& image_path, int w, int h) {
  cv::Mat bgr = cv::imread(image_path, cv::IMREAD_COLOR);
  if (bgr.empty()) {
    throw std::runtime_error("Failed to read image: " + image_path);
  }

  if (w > 0 && h > 0 && (bgr.cols != w || bgr.rows != h)) {
    cv::resize(bgr, bgr, cv::Size(w, h), 0, 0, cv::INTER_AREA);
  }

  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  return rgb;
}

bool infer_dims(const simaai::neat::Tensor& t, int& w, int& h) {
  w = t.width();
  h = t.height();
  if ((w <= 0 || h <= 0) && t.shape.size() >= 2) {
    h = static_cast<int>(t.shape[0]);
    w = static_cast<int>(t.shape[1]);
  }
  return (w > 0 && h > 0);
}

bool nv12_to_bgr(const simaai::neat::Tensor& t, cv::Mat& out, std::string& err) {
  if (!t.is_nv12()) {
    err = "expected NV12 tensor";
    return false;
  }
  int w = 0;
  int h = 0;
  if (!infer_dims(t, w, h)) {
    err = "invalid tensor dimensions";
    return false;
  }
  std::vector<uint8_t> nv12 = t.copy_nv12_contiguous();
  if (nv12.empty()) {
    err = "NV12 copy failed";
    return false;
  }
  cv::Mat yuv(h + h / 2, w, CV_8UC1, nv12.data());
  cv::cvtColor(yuv, out, cv::COLOR_YUV2BGR_NV12);
  return true;
}

std::vector<float> tensor_to_floats(const simaai::neat::Tensor& t) {
  if (t.dtype != simaai::neat::TensorDType::Float32) {
    throw std::runtime_error("Expected Float32 tensor output");
  }
  std::vector<uint8_t> raw = t.copy_dense_bytes_tight();
  if (raw.empty()) {
    throw std::runtime_error("Tensor output is empty");
  }
  const size_t bytes = raw.size();
  if (bytes % sizeof(float) != 0) {
    throw std::runtime_error("Tensor plane size is not a multiple of float");
  }

  const size_t elems = bytes / sizeof(float);
  std::vector<float> out(elems);
  std::memcpy(out.data(), raw.data(), elems * sizeof(float));
  return out;
}

std::vector<float> scores_from_tensor(const simaai::neat::Tensor& t, const std::string& label) {
  auto scores_full = tensor_to_floats(t);
  if (scores_full.empty()) {
    throw std::runtime_error(label + ": empty tensor output");
  }
  if (scores_full.size() < 1000) {
    throw std::runtime_error(label + ": expected at least 1000 scores, got " +
                             std::to_string(scores_full.size()));
  }
  if (scores_full.size() > 1000) {
    scores_full.resize(1000);
  }
  return scores_full;
}

std::vector<ScoredIndex> topk_with_softmax(const std::vector<float>& v, int k) {
  if (v.empty() || k <= 0)
    return {};
  const int n = static_cast<int>(v.size());
  k = std::min(k, n);

  std::vector<int> idx(n);
  std::iota(idx.begin(), idx.end(), 0);
  std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                    [&v](int a, int b) { return v[a] > v[b]; });

  const float maxv = *std::max_element(v.begin(), v.end());
  double sum = 0.0;
  for (float x : v) {
    sum += std::exp(static_cast<double>(x - maxv));
  }

  std::vector<ScoredIndex> out;
  out.reserve(k);
  for (int i = 0; i < k; ++i) {
    const int id = idx[i];
    const double prob = std::exp(static_cast<double>(v[id] - maxv)) / sum;
    out.push_back(ScoredIndex{id, v[id], static_cast<float>(prob)});
  }
  return out;
}

void check_top1(const std::vector<float>& scores, int expected_id, float min_prob,
                const std::string& label) {
  const auto top = topk_with_softmax(scores, 5);
  std::cout << "[" << label << "] top1 index=" << top[0].index << " score=" << top[0].value
            << " prob=" << top[0].prob << "\n";
  std::cout << "[" << label << "] top5:";
  for (const auto& t : top) {
    std::cout << " " << t.index << ":" << t.prob;
  }
  std::cout << "\n";

  if (expected_id < 0)
    return;

  if (top[0].index != expected_id) {
    throw std::runtime_error(label + ": top-1 mismatch: expected " + std::to_string(expected_id) +
                             " got " + std::to_string(top[0].index));
  }
  if (min_prob > 0.0f && top[0].prob < min_prob) {
    throw std::runtime_error(label + ": top-1 probability too low: " + std::to_string(top[0].prob) +
                             " < " + std::to_string(min_prob));
  }
  std::cout << "[" << label << "] top-1 matches expected class " << expected_id << "\n";
}

std::string metadata_boxes_data_json(const std::string& array_key,
                                     const std::vector<MetadataBox>& boxes) {
  json data;
  data[array_key] = json::array();
  for (const auto& box : boxes) {
    data[array_key].push_back({
        {"id", box.id},
        {"label", box.label},
        {"confidence", box.confidence},
        {"bbox", {box.x, box.y, box.w, box.h}},
    });
  }
  return data.dump();
}

} // namespace sima_examples
