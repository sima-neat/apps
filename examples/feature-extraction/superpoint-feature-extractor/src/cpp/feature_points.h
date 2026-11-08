#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

namespace superpoint_app {

struct FrameSize {
  int width = 0;
  int height = 0;
};

// Maps decoder keypoints (model-input pixels, interleaved x,y; sima-neat/core#1328) onto the
// source frame that Preproc stretched to the model input, and drops points on or beyond a source
// frame edge.
inline std::vector<float> source_points(const std::vector<float>& model_points,
                                        const FrameSize& model_size, const FrameSize& source_size) {
  const float scale_x =
      static_cast<float>(source_size.width) / static_cast<float>(model_size.width);
  const float scale_y =
      static_cast<float>(source_size.height) / static_cast<float>(model_size.height);
  std::vector<float> points;
  points.reserve(model_points.size());
  for (std::size_t i = 0; i + 1 < model_points.size(); i += 2) {
    const float x = model_points[i] * scale_x;
    const float y = model_points[i + 1] * scale_y;
    // The frame edges are exclusive coordinates. Discard isolated decoder outliers instead of
    // aborting the stream when a point lands on or beyond an edge.
    if (!std::isfinite(x) || !std::isfinite(y) || x < 0.0F || y < 0.0F ||
        x >= static_cast<float>(source_size.width) || y >= static_cast<float>(source_size.height)) {
      continue;
    }
    points.push_back(x);
    points.push_back(y);
  }
  return points;
}

} // namespace superpoint_app
