// Copyright 2026 SiMa Technologies, Inc.
// SPDX-License-Identifier: Apache-2.0

// SuperPoint feature points, from the decoder's output to Insight's metadata. Mirrors
// utils/feature_points.py.

#pragma once

#include "neat.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace superpoint {

struct FrameSize {
  int width = 0;
  int height = 0;
};

struct Point {
  long x = 0;
  long y = 0;
};

// The values of a Float32 tensor, such as the decoded keypoints.
inline std::vector<float> floats(const simaai::neat::Tensor& tensor) {
  std::vector<float> values(tensor.dense_bytes_tight() / sizeof(float));
  if (!values.empty() &&
      !tensor.copy_dense_bytes_tight_to(reinterpret_cast<std::uint8_t*>(values.data()),
                                        values.size() * sizeof(float))) {
    throw std::runtime_error("SuperPoint keypoints could not be read");
  }
  return values;
}

// The keypoints, interleaved x, y, scaled from model-input pixels (sima-neat/core#1328) to whole
// pixels on the source frame. Points on or past a frame edge are dropped.
inline std::vector<Point> source_points(const std::vector<float>& keypoints,
                                        const FrameSize& model_size, const FrameSize& source_size) {
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

// Insight's pose-estimation data: one pose whose unnamed keypoints Insight draws as dots.
inline std::string points_json(const std::vector<Point>& points) {
  nlohmann::json keypoints = nlohmann::json::array();
  for (const auto& point : points) {
    keypoints.push_back({{"x", point.x}, {"y", point.y}});
  }
  nlohmann::json pose = nlohmann::json::object();
  pose["keypoints"] = std::move(keypoints);
  nlohmann::json data = nlohmann::json::object();
  data["poses"] = nlohmann::json::array({std::move(pose)});
  return data.dump();
}

} // namespace superpoint
