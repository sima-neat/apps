// Copied from support/templates/cpp/detections.cpp;
// edit the template and run scripts/sync_utils.py.
#include "detections.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <exception>
#include <stdexcept>

namespace utils {
namespace {

struct RawBox {
  int32_t x = 0;
  int32_t y = 0;
  int32_t w = 0;
  int32_t h = 0;
  float score = 0.0f;
  int32_t cls = 0;
};

std::string upper_ascii_copy(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    out.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
  }
  return out;
}

bool extract_tensor_bbox_payload(const simaai::neat::Sample& result, std::vector<uint8_t>& payload,
                                 std::string& err) {
  if (result.kind != simaai::neat::SampleKind::Tensor) {
    err = "capture_expected_tensor";
    return false;
  }
  if (!result.tensor.has_value()) {
    err = "capture_missing_tensor";
    return false;
  }

  const auto& tensor = result.tensor.value();
  std::string fmt = result.payload_tag;
  if (fmt.empty() && !result.format.empty())
    fmt = result.format;
  if (fmt.empty() && tensor.semantic.tess.has_value()) {
    fmt = tensor.semantic.tess->format;
  }

  const std::string fmt_upper = upper_ascii_copy(fmt);
  if (!fmt_upper.empty() && fmt_upper != "BBOX") {
    err = "capture_expected_bbox format=" + fmt_upper;
    return false;
  }

  try {
    payload = tensor.copy_payload_bytes();
  } catch (const std::exception& ex) {
    err = "capture_payload_failed err=";
    err += ex.what();
    return false;
  }

  if (payload.empty()) {
    err = "capture_empty_payload";
    return false;
  }
  return true;
}

} // namespace

bool extract_bbox_payload(const simaai::neat::Sample& sample, std::vector<uint8_t>& payload,
                          std::string& err) {
  if (sample.kind == simaai::neat::SampleKind::Bundle) {
    for (const auto& field : sample.fields) {
      if (extract_bbox_payload(field, payload, err)) {
        return true;
      }
    }
    err = "bundle missing BBOX field";
    return false;
  }
  if (sample.kind == simaai::neat::SampleKind::TensorSet && !sample.tensors.empty()) {
    simaai::neat::Sample tensor_sample = sample;
    tensor_sample.kind = simaai::neat::SampleKind::Tensor;
    tensor_sample.tensor = sample.tensors.front();
    tensor_sample.tensors.clear();
    return extract_tensor_bbox_payload(tensor_sample, payload, err);
  }
  return extract_tensor_bbox_payload(sample, payload, err);
}

std::vector<Box> parse_boxes_strict(const std::vector<uint8_t>& bytes, int img_w, int img_h,
                                    int expected_topk) {
  if (bytes.size() < 4) {
    throw std::runtime_error("bbox buffer too small");
  }
  uint32_t header = 0;
  std::memcpy(&header, bytes.data(), sizeof(header));

  const size_t max_boxes = (bytes.size() - sizeof(header)) / sizeof(RawBox);
  if (header > max_boxes) {
    throw std::runtime_error("bbox header exceeds payload count");
  }
  if (expected_topk > 0 && header > static_cast<uint32_t>(expected_topk)) {
    throw std::runtime_error("bbox header exceeds expected topk");
  }

  std::vector<Box> out;
  out.reserve(header);
  const uint8_t* base = bytes.data() + sizeof(header);
  for (size_t i = 0; i < header; ++i) {
    RawBox r{};
    std::memcpy(&r, base + i * sizeof(RawBox), sizeof(r));

    float x1 = static_cast<float>(r.x);
    float y1 = static_cast<float>(r.y);
    float x2 = static_cast<float>(r.x + r.w);
    float y2 = static_cast<float>(r.y + r.h);

    x1 = std::max(0.0f, std::min(x1, static_cast<float>(img_w)));
    y1 = std::max(0.0f, std::min(y1, static_cast<float>(img_h)));
    x2 = std::max(0.0f, std::min(x2, static_cast<float>(img_w)));
    y2 = std::max(0.0f, std::min(y2, static_cast<float>(img_h)));

    out.push_back(Box{x1, y1, x2, y2, r.score, r.cls});
  }
  return out;
}

void draw_boxes(cv::Mat& img, const std::vector<Box>& boxes, float min_score) {
  const cv::Scalar green(0, 255, 0);
  for (const auto& b : boxes) {
    if (b.score < min_score)
      continue;
    const int x1 = std::max(0, static_cast<int>(std::round(b.x1)));
    const int y1 = std::max(0, static_cast<int>(std::round(b.y1)));
    const int x2 = std::min(img.cols - 1, static_cast<int>(std::round(b.x2)));
    const int y2 = std::min(img.rows - 1, static_cast<int>(std::round(b.y2)));
    if (x2 <= x1 || y2 <= y1)
      continue;

    cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), green, 2);
    const std::string label =
        "id=" + std::to_string(b.class_id) + " score=" + std::to_string(b.score);
    cv::putText(img, label, cv::Point(x1, std::max(0, y1 - 4)), cv::FONT_HERSHEY_SIMPLEX, 0.4,
                green, 1);
  }
}

} // namespace utils
