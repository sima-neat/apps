// Copyright 2026 SiMa Technologies, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "feature_points.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstddef>
#include <cstring>
#include <utility>

namespace superpoint {

std::vector<float> floats(const simaai::neat::Tensor& tensor) {
  const auto bytes = tensor.copy_dense_bytes_tight();
  std::vector<float> values(bytes.size() / sizeof(float));
  std::memcpy(values.data(), bytes.data(), values.size() * sizeof(float));
  return values;
}

std::vector<Point> source_points(const std::vector<float>& keypoints, const FrameSize& model_size,
                                 const FrameSize& source_size) {
  const float scale_x =
      static_cast<float>(source_size.width) / static_cast<float>(model_size.width);
  const float scale_y =
      static_cast<float>(source_size.height) / static_cast<float>(model_size.height);
  std::vector<Point> points;
  for (std::size_t i = 0; i + 1 < keypoints.size(); i += 2) {
    const float x = keypoints[i] * scale_x;
    const float y = keypoints[i + 1] * scale_y;
    if (!std::isfinite(x) || !std::isfinite(y) || x < 0.0F || y < 0.0F) {
      continue;
    }
    const Point pixel{std::lround(static_cast<double>(x)), std::lround(static_cast<double>(y))};
    if (pixel.x < source_size.width && pixel.y < source_size.height) {
      points.push_back(pixel);
    }
  }
  return points;
}

std::string points_json(const std::vector<Point>& points) {
  // Whole-pixel {"x", "y"} keypoints keep 600 points near 10 KB, which sends in about 3 ms on a
  // DevKit; one named, scored pose per point took 37 KB and 9 ms.
  nlohmann::json keypoints = nlohmann::json::array();
  for (const auto& point : points) {
    keypoints.push_back({{"x", point.x}, {"y", point.y}});
  }
  nlohmann::json pose;
  pose["keypoints"] = std::move(keypoints);
  nlohmann::json data;
  data["poses"] = nlohmann::json::array({std::move(pose)});
  return data.dump();
}

} // namespace superpoint
