#include "obj_detection_utils.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

namespace objdet {
namespace {

struct RawBox {
  int32_t x = 0;
  int32_t y = 0;
  int32_t w = 0;
  int32_t h = 0;
  float score = 0.0f;
  int32_t cls = 0;
};

} // namespace

std::vector<Box> parse_boxes_strict(const std::vector<uint8_t>& bytes, int img_w, int img_h,
                                    int expected_topk, bool debug) {
  std::vector<Box> out;
  parse_boxes_strict_into(bytes, img_w, img_h, expected_topk, debug, out);
  return out;
}

void parse_boxes_strict_into(const std::vector<uint8_t>& bytes, int img_w, int img_h,
                             int expected_topk, bool debug, std::vector<Box>& out) {
  parse_boxes_strict_into(std::span<const uint8_t>(bytes), img_w, img_h, expected_topk, debug, out);
}

void parse_boxes_strict_into(std::span<const uint8_t> bytes, int img_w, int img_h,
                             int expected_topk, bool debug, std::vector<Box>& out) {
  require(bytes.size() >= 4, "bbox buffer too small");
  uint32_t header = 0;
  std::memcpy(&header, bytes.data(), sizeof(header));

  const size_t payload = bytes.size() - sizeof(header);
  const size_t max_boxes = payload / sizeof(RawBox);
  const size_t trailing = payload % sizeof(RawBox);

  if (debug) {
    std::cerr << "[DBG] bbox header=" << header << " expected_topk=" << expected_topk
              << " payload=" << payload << " trailing=" << trailing << "\n";
  }

  require(header <= max_boxes, "bbox header exceeds payload count");
  if (expected_topk > 0) {
    require(header <= static_cast<uint32_t>(expected_topk), "bbox header exceeds expected topk");
  }

  const size_t count = header;
  out.clear();
  out.reserve(count);

  const uint8_t* base = bytes.data() + 4;
  for (size_t i = 0; i < count; ++i) {
    RawBox r{};
    std::memcpy(&r, base + i * 24, sizeof(r));

    float x1 = static_cast<float>(r.x);
    float y1 = static_cast<float>(r.y);
    float x2 = static_cast<float>(r.x + r.w);
    float y2 = static_cast<float>(r.y + r.h);

    x1 = std::max(0.0f, std::min(x1, static_cast<float>(img_w)));
    y1 = std::max(0.0f, std::min(y1, static_cast<float>(img_h)));
    x2 = std::max(0.0f, std::min(x2, static_cast<float>(img_w)));
    y2 = std::max(0.0f, std::min(y2, static_cast<float>(img_h)));

    out.push_back(Box{x1, y1, x2, y2, r.score, r.cls});
    if (debug && i < 4) {
      std::cerr << "[DBG] box[" << i << "]=" << r.x << "," << r.y << "," << r.w << "," << r.h
                << " score=" << r.score << " class=" << r.cls << "\n";
    }
  }
  return;
}

void draw_boxes(cv::Mat& img, const std::vector<Box>& boxes, float min_score,
                const cv::Scalar& color, const std::string& label_prefix) {
  for (const auto& b : boxes) {
    if (b.score < min_score)
      continue;
    const int x1 = std::max(0, static_cast<int>(std::round(b.x1)));
    const int y1 = std::max(0, static_cast<int>(std::round(b.y1)));
    const int x2 = std::min(img.cols - 1, static_cast<int>(std::round(b.x2)));
    const int y2 = std::min(img.rows - 1, static_cast<int>(std::round(b.y2)));
    if (x2 <= x1 || y2 <= y1)
      continue;

    cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);
    const std::string label =
        label_prefix + "id=" + std::to_string(b.class_id) + " score=" + std::to_string(b.score);
    cv::putText(img, label, cv::Point(x1, std::max(0, y1 - 4)), cv::FONT_HERSHEY_SIMPLEX, 0.4,
                color, 1);
  }
}

} // namespace objdet
