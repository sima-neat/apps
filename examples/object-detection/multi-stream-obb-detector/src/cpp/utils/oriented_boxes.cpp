// Copyright 2026 SiMa Technologies, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "oriented_boxes.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>

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

std::string detections_json(const simaai::neat::OrientedDecodeTensors& boxes,
                            const std::vector<std::string>& labels) {
  nlohmann::json objects = nlohmann::json::array();
  const auto count = static_cast<std::size_t>(boxes.scores.shape.at(0));
  // A frame without detections has no rows to read.
  if (count > 0) {
    // decode_obb_tensor returns dense CPU tensors.
    const float* geometry = boxes.geometry.data_ptr<float>();
    const float* scores = boxes.scores.data_ptr<float>();
    const std::int32_t* class_ids = boxes.class_ids.data_ptr<std::int32_t>();
    for (std::size_t i = 0; i < count; ++i) {
      const float* row = geometry + i * 5;
      const double cx = row[0];
      const double cy = row[1];
      const double width = row[2];
      const double height = row[3];
      const double angle = row[4];
      const double cos = std::abs(std::cos(angle));
      const double sin = std::abs(std::sin(angle));
      const double extent_w = width * cos + height * sin;
      const double extent_h = width * sin + height * cos;
      // decode_obb_tensor rejects a negative class ID.
      const auto class_id = static_cast<std::size_t>(class_ids[i]);
      objects.push_back(
          {{"id", "obj_" + std::to_string(i + 1)},
           {"label", class_id < labels.size() ? labels[class_id] : kUnknownLabel},
           {"confidence", round_to(scores[i], 1000.0)},
           {"bbox", {px(cx - extent_w / 2), px(cy - extent_h / 2), px(extent_w), px(extent_h)}},
           {"oriented_bbox", {px(cx), px(cy), px(width), px(height), round_to(angle, 10000.0)}}});
    }
  }
  return nlohmann::json{{"objects", objects}}.dump();
}

} // namespace obb
