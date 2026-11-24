// Copyright 2026 SiMa Technologies, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "oriented_boxes.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstddef>

namespace obb {

namespace {

constexpr const char* kUnknownLabel = "unknown";

// Rounds halves to even, like NumPy in the Python implementation, so both send the same values.
double round_to(double value, double scale) {
  return std::nearbyint(value * scale) / scale;
}

double px(double value) {
  return round_to(value, 10.0);
}

} // namespace

std::string detections_json(const std::vector<OrientedBox>& boxes,
                            const std::vector<std::string>& labels) {
  nlohmann::json objects = nlohmann::json::array();
  for (std::size_t i = 0; i < boxes.size(); ++i) {
    const OrientedBox& box = boxes[i];
    const double cos = std::abs(std::cos(box.angle_radians));
    const double sin = std::abs(std::sin(box.angle_radians));
    const double extent_w = box.width * cos + box.height * sin;
    const double extent_h = box.width * sin + box.height * cos;
    const bool known = box.class_id >= 0 && box.class_id < static_cast<int>(labels.size());
    objects.push_back(
        {{"id", "obj_" + std::to_string(i + 1)},
         {"label", known ? labels[static_cast<std::size_t>(box.class_id)] : kUnknownLabel},
         {"confidence", round_to(box.score, 1000.0)},
         {"bbox",
          {px(box.cx - extent_w / 2), px(box.cy - extent_h / 2), px(extent_w), px(extent_h)}},
         {"oriented_bbox",
          {px(box.cx), px(box.cy), px(box.width), px(box.height),
           round_to(box.angle_radians, 10000.0)}}});
  }
  return nlohmann::json{{"objects", objects}}.dump();
}

} // namespace obb
