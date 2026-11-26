// Copyright 2026 SiMa Technologies, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "insight_metadata.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace insight_metadata {
namespace {

constexpr const char* kUnknownLabel = "unknown";

std::string label(const std::vector<std::string>& labels, int class_id) {
  return class_id >= 0 && class_id < static_cast<int>(labels.size())
             ? labels[static_cast<std::size_t>(class_id)]
             : kUnknownLabel;
}

} // namespace

double round_confidence(double value) {
  return std::floor(value * 1000.0 + 0.5) / 1000.0;
}

std::vector<Entry> objects(const std::vector<simaai::neat::Box>& boxes,
                           const std::vector<std::string>& labels, int frame_w, int frame_h) {
  std::vector<Entry> entries;
  int index = 1;
  for (const auto& box : boxes) {
    const int x = std::max(0, static_cast<int>(box.x1));
    const int y = std::max(0, static_cast<int>(box.y1));
    const int width = std::max(0, std::min(static_cast<int>(box.x2 - box.x1), frame_w - x));
    const int height = std::max(0, std::min(static_cast<int>(box.y2 - box.y1), frame_h - y));
    entries.push_back({{"id", "obj_" + std::to_string(index++)},
                       {"label", label(labels, box.class_id)},
                       {"confidence", round_confidence(box.score)},
                       {"bbox",
                        {static_cast<double>(x), static_cast<double>(y),
                         static_cast<double>(width), static_cast<double>(height)}}});
  }
  return entries;
}

cv::Rect frame_rect(const float* box, int frame_w, int frame_h) {
  const int x0 = std::clamp(static_cast<int>(std::floor(box[0])), 0, frame_w - 1);
  const int y0 = std::clamp(static_cast<int>(std::floor(box[1])), 0, frame_h - 1);
  const int x1 = std::clamp(static_cast<int>(std::ceil(box[2])), x0 + 1, frame_w);
  const int y1 = std::clamp(static_cast<int>(std::ceil(box[3])), y0 + 1, frame_h);
  return {x0, y0, x1 - x0, y1 - y0};
}

cv::Rect mask_rect(const cv::Rect& rect, int frame_w, int frame_h, int mask_w, int mask_h) {
  // The mask grid covers the model's letterboxed input, so a frame point maps to a cell through
  // the same letterbox.
  const double scale = std::min(static_cast<double>(mask_w) / frame_w,
                                static_cast<double>(mask_h) / frame_h);
  const double pad_x = (mask_w - frame_w * scale) * 0.5;
  const double pad_y = (mask_h - frame_h * scale) * 0.5;
  const auto cell_x = [&](double x) { return x * scale + pad_x; };
  const auto cell_y = [&](double y) { return y * scale + pad_y; };

  const int x0 = std::clamp(static_cast<int>(std::floor(cell_x(rect.x))), 0, mask_w - 1);
  const int y0 = std::clamp(static_cast<int>(std::floor(cell_y(rect.y))), 0, mask_h - 1);
  const int x1 =
      std::clamp(static_cast<int>(std::ceil(cell_x(rect.x + rect.width))), x0 + 1, mask_w);
  const int y1 =
      std::clamp(static_cast<int>(std::ceil(cell_y(rect.y + rect.height))), y0 + 1, mask_h);
  return {x0, y0, x1 - x0, y1 - y0};
}

std::vector<cv::Point> mask_polygon(const cv::Mat& mask, const cv::Rect& rect, int frame_w,
                                    int frame_h) {
  cv::Mat roi;
  cv::resize(mask(mask_rect(rect, frame_w, frame_h, mask.cols, mask.rows)), roi, rect.size(), 0,
             0, cv::INTER_LINEAR);
  cv::Mat binary;
  cv::threshold(roi, binary, kMaskThreshold * 255.0, 255, cv::THRESH_BINARY);
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  if (contours.empty()) {
    return {};
  }
  // The first of equal areas, as Python's max().
  const auto largest = std::max_element(
      contours.begin(), contours.end(),
      [](const auto& a, const auto& b) { return cv::contourArea(a) < cv::contourArea(b); });
  std::vector<cv::Point> polygon;
  cv::approxPolyDP(*largest, polygon, 0.004 * cv::arcLength(*largest, true), true);
  if (polygon.size() < 3) {
    return {};
  }
  for (auto& point : polygon) {
    point += rect.tl();
  }
  return polygon;
}

std::vector<Entry> segments(const std::vector<float>& boxes, const std::vector<std::uint8_t>& masks,
                            int mask_w, int mask_h, const std::vector<std::string>& labels,
                            int frame_w, int frame_h) {
  const std::size_t mask_bytes = static_cast<std::size_t>(mask_w) * mask_h;
  if (boxes.size() / 6 != masks.size() / mask_bytes) {
    throw std::runtime_error("segmentation decode returned " + std::to_string(boxes.size() / 6) +
                             " boxes but " + std::to_string(masks.size() / mask_bytes) +
                             " masks");
  }
  std::vector<Entry> entries;
  for (std::size_t i = 0; i < boxes.size() / 6; ++i) {
    const float* box = boxes.data() + i * 6;
    // A view of the decoded mask: OpenCV reads it without a copy.
    const cv::Mat mask(mask_h, mask_w, CV_8UC1,
                       const_cast<std::uint8_t*>(masks.data() + i * mask_bytes));
    const cv::Rect rect = frame_rect(box, frame_w, frame_h);
    const auto polygon = mask_polygon(mask, rect, frame_w, frame_h);
    if (polygon.empty()) {
      continue;
    }
    Entry points = Entry::array();
    for (const auto& point : polygon) {
      points.push_back({point.x, point.y});
    }
    entries.push_back({{"id", "seg_" + std::to_string(entries.size() + 1)},
                       {"label", label(labels, static_cast<int>(box[5]))},
                       {"confidence", round_confidence(box[4])},
                       {"bbox", {rect.x, rect.y, rect.width, rect.height}},
                       {"mask_format", "polygon"},
                       {"mask", std::move(points)}});
  }
  return entries;
}

std::vector<Entry> poses(const std::vector<float>& boxes, const std::vector<float>& keypoints) {
  const std::size_t keypoint_set = kCocoKeypointNames.size() * 3;
  if (boxes.size() / 6 != keypoints.size() / keypoint_set) {
    throw std::runtime_error("pose decode returned " + std::to_string(boxes.size() / 6) +
                             " boxes but " + std::to_string(keypoints.size() / keypoint_set) +
                             " keypoint sets");
  }
  std::vector<Entry> entries;
  for (std::size_t i = 0; i < boxes.size() / 6; ++i) {
    const float* box = boxes.data() + i * 6;
    const float* points = keypoints.data() + i * keypoint_set;
    Entry joints = Entry::array();
    for (std::size_t k = 0; k < kCocoKeypointNames.size(); ++k) {
      joints.push_back({{"name", kCocoKeypointNames[k]},
                        {"x", std::lround(points[k * 3])},
                        {"y", std::lround(points[k * 3 + 1])},
                        {"confidence", round_confidence(points[k * 3 + 2])}});
    }
    entries.push_back({{"id", "pose_" + std::to_string(i + 1)},
                       {"label", "person"},
                       {"confidence", round_confidence(box[4])},
                       {"bbox",
                        {std::lround(box[0]), std::lround(box[1]),
                         std::lround(std::max(0.0f, box[2] - box[0])),
                         std::lround(std::max(0.0f, box[3] - box[1]))}},
                       {"keypoints", std::move(joints)}});
  }
  return entries;
}

std::pair<std::string, int> data_json(const std::string& key, std::vector<Entry> entries) {
  // Stable, so entries with the same confidence keep their order, as in Python.
  std::stable_sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
    return a["confidence"].get<double>() > b["confidence"].get<double>();
  });
  Entry kept = Entry::array();
  std::size_t size = Entry{{key, Entry::array()}}.dump().size();
  for (auto& entry : entries) {
    const std::size_t entry_size = entry.dump().size() + 1;
    if (size + entry_size > kMetadataByteBudget) {
      break;
    }
    size += entry_size;
    kept.push_back(std::move(entry));
  }
  const int left_out = static_cast<int>(entries.size() - kept.size());
  return {Entry{{key, std::move(kept)}}.dump(), left_out};
}

} // namespace insight_metadata
