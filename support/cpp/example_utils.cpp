#include "support/cpp/example_utils.h"

#include <neat.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>

using json = nlohmann::json;

namespace sima_examples {

void require(bool cond, const std::string& msg) {
  if (!cond)
    throw std::runtime_error(msg);
}

double time_ms() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
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

bool nv16_to_yuy2(const simaai::neat::Tensor& t, cv::Mat& out, std::string& err) {
  if (t.image_format() != simaai::neat::ImageSpec::PixelFormat::NV16) {
    err = "expected NV16 tensor";
    return false;
  }
  const simaai::neat::Plane* y = nullptr;
  const simaai::neat::Plane* uv = nullptr;
  for (const auto& plane : t.planes) {
    if (plane.role == simaai::neat::PlaneRole::Y)
      y = &plane;
    if (plane.role == simaai::neat::PlaneRole::UV)
      uv = &plane;
  }
  int w = 0;
  int h = 0;
  if (!y || !uv || y->strides_bytes.empty() || uv->strides_bytes.empty() || !infer_dims(t, w, h)) {
    err = "NV16 tensor without Y/UV planes";
    return false;
  }
  // The payload is the whole mapped buffer, so plane offsets index into it.
  const std::vector<uint8_t> payload = t.copy_payload_bytes();
  const auto y_stride = static_cast<std::size_t>(y->strides_bytes[0]);
  const auto uv_stride = static_cast<std::size_t>(uv->strides_bytes[0]);
  if (static_cast<std::size_t>(uv->byte_offset) + uv_stride * static_cast<std::size_t>(h) >
      payload.size()) {
    err = "NV16 payload smaller than its planes";
    return false;
  }
  // Each row keeps its own chroma: Y and UV bytes interleave into packed 4:2:2 (Y0 U0 Y1 V0).
  out.create(h, w, CV_8UC2);
  for (int row = 0; row < h; ++row) {
    const uint8_t* y_row = payload.data() + y->byte_offset + row * y_stride;
    const uint8_t* uv_row = payload.data() + uv->byte_offset + row * uv_stride;
    uint8_t* dst = out.ptr(row);
    for (int x = 0; x < w; ++x) {
      dst[2 * x] = y_row[x];
      dst[2 * x + 1] = uv_row[x];
    }
  }
  return true;
}

bool nv16_to_bgr(const simaai::neat::Tensor& t, cv::Mat& out, std::string& err) {
  cv::Mat yuy2;
  if (!nv16_to_yuy2(t, yuy2, err))
    return false;
  cv::cvtColor(yuy2, out, cv::COLOR_YUV2BGR_YUY2);
  return true;
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
  // Tie-break by ascending index so equal scores resolve deterministically and
  // identically to the Python implementation's np.lexsort((arange(n), -scores)).
  std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&v](int a, int b) {
    if (v[a] != v[b])
      return v[a] > v[b];
    return a < b;
  });

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
