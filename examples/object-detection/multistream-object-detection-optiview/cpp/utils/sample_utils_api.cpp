#pragma once

#include "model_family_api.cpp"

#include "neat/session.h"
#include "support/runtime/example_utils.h"

#include <opencv2/core/mat.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace multistream_object_detection_optiview {

struct Detection {
  float x1 = 0.0f;
  float y1 = 0.0f;
  float x2 = 0.0f;
  float y2 = 0.0f;
  float score = 0.0f;
  int class_id = -1;
};

struct OptiViewDetectionPayload {
  std::vector<sima_examples::OptiViewObject> objects;
  std::vector<std::string> labels;
};

std::vector<std::uint8_t> extract_bbox_payload(const simaai::neat::Sample& sample);
std::vector<Detection> parse_bbox_payload(const std::vector<std::uint8_t>& payload, int img_w,
                                          int img_h);
void require_detector_output_kind(ModelFamily family, const simaai::neat::Sample& sample);
std::vector<Detection> detections_from_detector_sample(ModelFamily family,
                                                       const simaai::neat::Sample& sample,
                                                       int img_w, int img_h);
std::string optiview_frame_id(const simaai::neat::Sample& sample, int fallback_frame_index);
std::int64_t optiview_timestamp_ms(double publish_time_s, double offset_ms);
OptiViewDetectionPayload
build_optiview_detection_payload(const std::vector<Detection>& detections, int img_w, int img_h,
                                 const std::vector<std::string>& class_labels);
cv::Mat tensor_rgb_from_sample(const simaai::neat::Sample& sample);

} // namespace multistream_object_detection_optiview
