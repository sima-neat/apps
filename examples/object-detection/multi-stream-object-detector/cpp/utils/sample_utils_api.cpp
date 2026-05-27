#pragma once

#include "model_family_api.cpp"

#include "neat/session.h"
#include "support/runtime/example_utils.h"

#include <opencv2/core/mat.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace multi_stream_object_detector {

struct Detection {
  float x1 = 0.0f;
  float y1 = 0.0f;
  float x2 = 0.0f;
  float y2 = 0.0f;
  float score = 0.0f;
  int class_id = -1;
};

struct InsightDetectionPayload {
  std::vector<sima_examples::MetadataBox> objects;
};

std::vector<std::uint8_t> extract_bbox_payload(const simaai::neat::Sample& sample);
std::vector<Detection> parse_bbox_payload(const std::vector<std::uint8_t>& payload, int img_w,
                                          int img_h);
void require_detector_output_kind(ModelFamily family, const simaai::neat::Sample& sample);
std::vector<Detection> detections_from_detector_sample(ModelFamily family,
                                                       const simaai::neat::Sample& sample,
                                                       int img_w, int img_h);
std::string insight_frame_id(const simaai::neat::Sample& sample, int fallback_frame_index);
std::int64_t insight_timestamp_ms(double publish_time_s, double offset_ms);
InsightDetectionPayload
build_insight_detection_payload(const std::vector<Detection>& detections, int img_w, int img_h,
                                const std::vector<std::string>& class_labels);
cv::Mat tensor_rgb_from_sample(const simaai::neat::Sample& sample);

} // namespace multi_stream_object_detector
