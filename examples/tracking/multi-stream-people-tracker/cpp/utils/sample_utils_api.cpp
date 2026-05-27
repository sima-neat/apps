#pragma once

#include "examples/tracking/multi-stream-people-tracker/cpp/utils/tracker_api.cpp"
#include "support/runtime/example_utils.h"

#include <opencv2/core/mat.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace simaai::neat {
struct Sample;
}

namespace multi_stream_people_tracker {

struct InsightTrackingPayload {
  std::vector<sima_examples::MetadataBox> tracks;
};

std::vector<Detection> filter_person_detections(const std::vector<Detection>& boxes,
                                                int person_class_id = 0);
InsightTrackingPayload
make_insight_tracking_detection(const std::vector<TrackedDetection>& tracked);
std::vector<std::uint8_t> extract_bbox_payload(const simaai::neat::Sample& sample);
std::vector<Detection> parse_bbox_payload(const std::vector<std::uint8_t>& payload, int img_w,
                                          int img_h);
cv::Mat tensor_rgb_from_sample(const simaai::neat::Sample& sample);

} // namespace multi_stream_people_tracker
