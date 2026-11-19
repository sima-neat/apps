// Copyright 2026 SiMa Technologies, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "postprocess.h"

#include <nlohmann/json.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <utility>

namespace rfdetr {
namespace {

// RF-DETR's reference postprocess ranks the 300 best (query, class) scores.
constexpr std::size_t kNumSelect = 300;
// The metadata sender rejects messages over 65,507 bytes; 32 KiB of segments keeps the whole
// message, envelope included, under that limit.
constexpr std::size_t kSegmentsByteBudget = 32'768;

// Indices of the `count` largest values, largest first; equal values keep index order.
std::vector<int> top_indices(std::span<const float> values, std::size_t count) {
  count = std::min(count, values.size());
  std::vector<int> indices(values.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::partial_sort(indices.begin(), indices.begin() + static_cast<std::ptrdiff_t>(count),
                    indices.end(), [&values](int left, int right) {
                      const float left_value = values[static_cast<std::size_t>(left)];
                      const float right_value = values[static_cast<std::size_t>(right)];
                      return left_value != right_value ? left_value > right_value : left < right;
                    });
  indices.resize(count);
  return indices;
}

float sigmoid(float value) {
  // Clamping keeps exp() finite in float32.
  return 1.0F / (1.0F + std::exp(-std::clamp(value, -80.0F, 80.0F)));
}

struct RankedObject {
  int query = 0;
  const std::string* label = nullptr;
  float score = 0.0F;
};

// Each object scoring at least `min_score`, best first.
std::vector<RankedObject> ranked_objects(std::span<const float> logits,
                                         const std::vector<std::string>& labels, float min_score) {
  std::vector<float> scores(logits.size());
  std::transform(logits.begin(), logits.end(), scores.begin(), sigmoid);
  std::vector<RankedObject> objects;
  for (const int index : top_indices(scores, kNumSelect)) {
    const float score = scores[static_cast<std::size_t>(index)];
    if (score < min_score) {
      break;
    }
    const auto& label = labels[static_cast<std::size_t>(index % kNumClasses)];
    if (index % kNumClasses != 0 && label != "unused") {
      objects.push_back({index / kNumClasses, &label, score});
    }
  }
  return objects;
}

// A normalized (cx, cy, w, h) box as whole pixels inside the frame.
cv::Rect frame_box(std::span<const float> boxes, int query, int width, int height) {
  const auto offset = static_cast<std::size_t>(query) * 4U;
  const float cx = boxes[offset];
  const float cy = boxes[offset + 1U];
  const float box_width = boxes[offset + 2U];
  const float box_height = boxes[offset + 3U];
  const auto pixel = [](float value) { return static_cast<int>(std::nearbyint(value)); };
  const int x0 = std::clamp(pixel((cx - box_width / 2.0F) * width), 0, width - 1);
  const int y0 = std::clamp(pixel((cy - box_height / 2.0F) * height), 0, height - 1);
  const int x1 = std::clamp(pixel((cx + box_width / 2.0F) * width), x0 + 1, width);
  const int y1 = std::clamp(pixel((cy + box_height / 2.0F) * height), y0 + 1, height);
  return {x0, y0, x1 - x0, y1 - y0};
}

// The outline of one query's mask inside its box, in frame pixels; empty when none.
std::vector<cv::Point> mask_polygon(std::span<const float> masks, int query, int queries,
                                    int mask_size, const cv::Rect& box,
                                    const PostprocessOptions& options) {
  const double scale_to_mask_x = static_cast<double>(mask_size) / options.width;
  const double scale_to_mask_y = static_cast<double>(mask_size) / options.height;
  const int x0 =
      std::clamp(static_cast<int>(std::floor(box.x * scale_to_mask_x)), 0, mask_size - 1);
  const int y0 =
      std::clamp(static_cast<int>(std::floor(box.y * scale_to_mask_y)), 0, mask_size - 1);
  const int x1 = std::clamp(static_cast<int>(std::ceil((box.x + box.width) * scale_to_mask_x)),
                            x0 + 1, mask_size);
  const int y1 = std::clamp(static_cast<int>(std::ceil((box.y + box.height) * scale_to_mask_y)),
                            y0 + 1, mask_size);
  cv::Mat mask(y1 - y0, x1 - x0, CV_32FC1);
  for (int y = y0; y < y1; ++y) {
    auto* row = mask.ptr<float>(y - y0);
    for (int x = x0; x < x1; ++x) {
      row[x - x0] = sigmoid(masks[static_cast<std::size_t>((y * mask_size + x) * queries + query)]);
    }
  }
  if (options.mask_grid_size != mask_size) {
    const cv::Size size((mask.cols * options.mask_grid_size + mask_size - 1) / mask_size,
                        (mask.rows * options.mask_grid_size + mask_size - 1) / mask_size);
    cv::resize(mask, mask, size, 0.0, 0.0, cv::INTER_LINEAR);
  }
  cv::Mat binary;
  cv::compare(mask, options.mask_threshold, binary, cv::CMP_GE);
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  if (contours.empty()) {
    return {};
  }
  const auto largest =
      std::max_element(contours.begin(), contours.end(), [](const auto& left, const auto& right) {
        return cv::contourArea(left) < cv::contourArea(right);
      });
  std::vector<cv::Point> polygon;
  cv::approxPolyDP(*largest, polygon, 0.004 * cv::arcLength(*largest, true), true);
  if (polygon.size() < 3U) {
    return {};
  }
  const double scale_x = static_cast<double>(box.width - 1) / std::max(mask.cols - 1, 1);
  const double scale_y = static_cast<double>(box.height - 1) / std::max(mask.rows - 1, 1);
  for (auto& point : polygon) {
    point.x = std::min(box.x + box.width - 1,
                       box.x + static_cast<int>(std::nearbyint(point.x * scale_x)));
    point.y = std::min(box.y + box.height - 1,
                       box.y + static_cast<int>(std::nearbyint(point.y * scale_y)));
  }
  return polygon;
}

} // namespace

std::vector<float> top_proposals(std::span<const float> scores, std::span<const float> proposals,
                                 int count) {
  std::vector<float> best;
  best.reserve(static_cast<std::size_t>(count) * 4U);
  for (const int index : top_indices(scores, static_cast<std::size_t>(count))) {
    const auto box = proposals.subspan(static_cast<std::size_t>(index) * 4U, 4U);
    best.insert(best.end(), box.begin(), box.end());
  }
  return best;
}

std::string detection_json(std::span<const float> boxes, std::span<const float> logits,
                           const std::vector<std::string>& labels,
                           const PostprocessOptions& options) {
  const auto width = static_cast<float>(options.width);
  const auto height = static_cast<float>(options.height);
  nlohmann::json objects = nlohmann::json::array();
  for (const auto& object : ranked_objects(logits, labels, options.min_score)) {
    if (objects.size() == static_cast<std::size_t>(options.max_detections)) {
      break;
    }
    const auto offset = static_cast<std::size_t>(object.query) * 4U;
    const float cx = boxes[offset];
    const float cy = boxes[offset + 1U];
    const float box_width = boxes[offset + 2U];
    const float box_height = boxes[offset + 3U];
    const float x = std::clamp((cx - box_width / 2.0F) * width, 0.0F, width);
    const float y = std::clamp((cy - box_height / 2.0F) * height, 0.0F, height);
    const float x2 = std::clamp((cx + box_width / 2.0F) * width, x, width);
    const float y2 = std::clamp((cy + box_height / 2.0F) * height, y, height);
    objects.push_back({
        {"id", "obj_" + std::to_string(objects.size() + 1U)},
        {"label", *object.label},
        {"confidence", object.score},
        {"bbox", {x, y, x2 - x, y2 - y}},
    });
  }
  return nlohmann::json{{"objects", std::move(objects)}}.dump();
}

std::string segmentation_json(std::span<const float> boxes, std::span<const float> logits,
                              std::span<const float> masks, int mask_size,
                              const std::vector<std::string>& labels,
                              const PostprocessOptions& options) {
  const int queries = static_cast<int>(boxes.size() / 4U);
  nlohmann::json segments = nlohmann::json::array();
  std::size_t size = sizeof(R"({"segments":[]})") - 1U;
  for (const auto& object : ranked_objects(logits, labels, options.min_score)) {
    if (segments.size() == static_cast<std::size_t>(options.max_detections)) {
      break;
    }
    const cv::Rect box = frame_box(boxes, object.query, options.width, options.height);
    const auto polygon = mask_polygon(masks, object.query, queries, mask_size, box, options);
    if (polygon.empty()) {
      continue;
    }
    nlohmann::json points = nlohmann::json::array();
    for (const auto& point : polygon) {
      points.push_back({point.x, point.y});
    }
    nlohmann::json segment = {
        {"id", "seg_" + std::to_string(segments.size() + 1U)},
        {"label", *object.label},
        {"confidence", object.score},
        {"bbox", {box.x, box.y, box.width, box.height}},
        {"mask_format", "polygon"},
        {"mask", std::move(points)},
    };
    const std::size_t segment_size = segment.dump().size() + 1U;
    if (size + segment_size > kSegmentsByteBudget) {
      break;
    }
    size += segment_size;
    segments.push_back(std::move(segment));
  }
  return nlohmann::json{{"segments", std::move(segments)}}.dump();
}

} // namespace rfdetr
