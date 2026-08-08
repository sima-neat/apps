#pragma once

#include "examples/tracking/multi-stream-people-tracker/src/cpp/utils/tracker_api.cpp"

#include <cstdint>
#include <vector>

#include <opencv2/core.hpp>

namespace multi_stream_people_tracker {

struct TrackColor {
  std::uint8_t blue = 0;
  std::uint8_t green = 0;
  std::uint8_t red = 0;

  bool operator==(const TrackColor&) const = default;
};

TrackColor track_color(int track_id);
void draw_tracks_bgr(cv::Mat& frame, const std::vector<TrackedDetection>& tracks,
                     float min_score = 0.0f);

} // namespace multi_stream_people_tracker
