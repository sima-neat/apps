#include "obj_detection_utils.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>

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

std::string format_xyxy(float x1, float y1, float x2, float y2) {
  std::ostringstream oss;
  oss << static_cast<int>(std::round(x1)) << "," << static_cast<int>(std::round(y1)) << ","
      << static_cast<int>(std::round(x2)) << "," << static_cast<int>(std::round(y2));
  return oss.str();
}

std::string format_box(const Box& b) {
  std::ostringstream oss;
  oss << "class=" << b.class_id << " score=" << std::fixed << std::setprecision(2) << b.score
      << " box=" << format_xyxy(b.x1, b.y1, b.x2, b.y2);
  return oss.str();
}

std::string format_expected(const ExpectedBox& b) {
  std::ostringstream oss;
  oss << "class=" << b.class_id << " box=" << format_xyxy(b.x1, b.y1, b.x2, b.y2);
  return oss.str();
}

float box_iou_xyxy(float ax1, float ay1, float ax2, float ay2, float bx1, float by1, float bx2,
                   float by2) {
  const float inter_w = std::max(0.0f, std::min(ax2, bx2) - std::max(ax1, bx1));
  const float inter_h = std::max(0.0f, std::min(ay2, by2) - std::max(ay1, by1));
  const float inter = inter_w * inter_h;
  const float area_a = std::max(0.0f, ax2 - ax1) * std::max(0.0f, ay2 - ay1);
  const float area_b = std::max(0.0f, bx2 - bx1) * std::max(0.0f, by2 - by1);
  const float denom = area_a + area_b - inter;
  if (denom <= 0.0f)
    return 0.0f;
  return inter / denom;
}

float box_iou(const ExpectedBox& exp, const Box& pred) {
  return box_iou_xyxy(exp.x1, exp.y1, exp.x2, exp.y2, pred.x1, pred.y1, pred.x2, pred.y2);
}

std::vector<ExpectedBox> expected_people_boxes() {
  return {
      {747.0f, 42.0f, 1131.0f, 711.0f, 0},
      {149.0f, 201.0f, 1092.0f, 710.0f, 0},
      {437.0f, 434.0f, 532.0f, 717.0f, 27},
  };
}

MatchResult match_expected_boxes(const std::vector<Box>& boxes,
                                 const std::vector<ExpectedBox>& expected, float min_score,
                                 float min_iou) {
  MatchResult res;
  if (expected.empty()) {
    res.ok = true;
    return res;
  }

  std::vector<Box> candidates;
  candidates.reserve(boxes.size());
  for (const auto& b : boxes) {
    if (b.score >= min_score)
      candidates.push_back(b);
  }

  std::vector<bool> used(candidates.size(), false);
  std::ostringstream note;
  bool ok = true;
  for (size_t i = 0; i < expected.size(); ++i) {
    const auto& exp = expected[i];
    float best_iou = 0.0f;
    int best_idx = -1;
    for (size_t j = 0; j < candidates.size(); ++j) {
      if (used[j])
        continue;
      if (candidates[j].class_id != exp.class_id)
        continue;
      const float iou = box_iou(exp, candidates[j]);
      if (iou > best_iou) {
        best_iou = iou;
        best_idx = static_cast<int>(j);
      }
    }
    if (best_idx < 0 || best_iou < min_iou) {
      if (!note.str().empty())
        note << ";";
      note << "expected[" << i << "]=" << format_expected(exp) << " best_iou=" << std::fixed
           << std::setprecision(2) << best_iou;
      if (best_idx >= 0) {
        note << " best_pred=" << format_box(candidates[best_idx]);
      } else {
        note << " best_pred=none";
      }
      ok = false;
    } else {
      used[best_idx] = true;
    }
  }

  if (!ok) {
    note << ";pred=";
    if (candidates.empty()) {
      note << "none";
    } else {
      for (size_t i = 0; i < candidates.size(); ++i) {
        if (i > 0)
          note << "|";
        note << format_box(candidates[i]);
      }
    }
  }

  res.ok = ok;
  res.note = note.str();
  return res;
}

std::vector<Box> parse_boxes_strict(const std::vector<uint8_t>& bytes, int img_w, int img_h,
                                    int expected_topk, bool debug) {
  std::vector<Box> out;
  parse_boxes_strict_into(bytes, img_w, img_h, expected_topk, debug, out);
  return out;
}

void parse_boxes_strict_into(const std::vector<uint8_t>& bytes, int img_w, int img_h,
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

std::vector<Box> parse_boxes_lenient(const std::vector<uint8_t>& bytes, int img_w, int img_h,
                                     int expected_topk) {
  std::vector<Box> out;
  if (bytes.size() < sizeof(uint32_t))
    return out;

  uint32_t header = 0;
  std::memcpy(&header, bytes.data(), sizeof(header));
  const size_t payload = bytes.size() - sizeof(header);
  // Some runtimes can append trailing metadata bytes; accept and ignore remainder.
  if (payload < sizeof(RawBox))
    return out;

  const size_t max_boxes = payload / sizeof(RawBox);
  if (header > max_boxes)
    header = static_cast<uint32_t>(max_boxes);
  if (expected_topk > 0) {
    const size_t expected_size = sizeof(header) + static_cast<size_t>(expected_topk) * 24;
    if (bytes.size() < expected_size) {
      header = static_cast<uint32_t>(
          std::min<std::size_t>(header, static_cast<size_t>(expected_topk)));
    }
  }

  const uint8_t* base = bytes.data() + sizeof(header);
  for (size_t i = 0; i < header; ++i) {
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
    if (x2 <= x1 || y2 <= y1)
      continue;
    out.push_back(Box{x1, y1, x2, y2, r.score, static_cast<int>(r.cls)});
  }

  return out;
}

BoxSummary summarize_boxes(const std::vector<Box>& boxes, float min_score) {
  BoxSummary s;
  bool init = false;
  for (const auto& b : boxes) {
    if (b.score < min_score)
      continue;
    s.count += 1;
    if (!init) {
      s.min_score = b.score;
      s.max_score = b.score;
      init = true;
    } else {
      s.min_score = std::min(s.min_score, b.score);
      s.max_score = std::max(s.max_score, b.score);
    }
  }
  if (!init) {
    s.min_score = 0.0f;
    s.max_score = 0.0f;
  }
  return s;
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

void draw_expected_boxes(cv::Mat& img, const std::vector<ExpectedBox>& expected) {
  for (const auto& b : expected) {
    const int x1 = std::max(0, static_cast<int>(std::round(b.x1)));
    const int y1 = std::max(0, static_cast<int>(std::round(b.y1)));
    const int x2 = std::min(img.cols - 1, static_cast<int>(std::round(b.x2)));
    const int y2 = std::min(img.rows - 1, static_cast<int>(std::round(b.y2)));
    if (x2 <= x1 || y2 <= y1)
      continue;

    cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 255), 2);
    const std::string label = "exp id=" + std::to_string(b.class_id);
    cv::putText(img, label, cv::Point(x1, std::max(0, y1 - 4)), cv::FONT_HERSHEY_SIMPLEX, 0.4,
                cv::Scalar(0, 0, 255), 1);
  }
}

void save_overlay_boxes(const cv::Mat& img, const std::vector<Box>& boxes,
                        const std::vector<ExpectedBox>& expected, float min_score,
                        const std::filesystem::path& out_path) {
  cv::Mat overlay = img.clone();
  draw_expected_boxes(overlay, expected);
  draw_boxes(overlay, boxes, min_score, cv::Scalar(0, 255, 0), "pred ");
  require(cv::imwrite(out_path.string(), overlay),
          "Failed to write overlay image: " + out_path.string());
}

} // namespace objdet
