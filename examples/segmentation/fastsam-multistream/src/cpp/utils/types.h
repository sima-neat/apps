#pragma once

#include <opencv2/core.hpp>

#include <cstdint>
#include <vector>

namespace app {

struct Segment {
  float confidence = 0.0f;
  std::vector<cv::Point> polygon;  // frame pixels
};

struct StreamInfo {
  int width = 0;
  int height = 0;
  int fps = 0;
};

struct Frame {
  cv::Mat rgb;
  std::int64_t pts_ns = -1;
  std::int64_t frame_id = -1;
};

}  // namespace app
