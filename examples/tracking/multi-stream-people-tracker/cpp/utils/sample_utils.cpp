#include "examples/tracking/multi-stream-people-tracker/cpp/utils/sample_utils_api.cpp"

#include "neat/runtime.h"
#include "support/object_detection/obj_detection_utils.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace multi_stream_people_tracker {
namespace {

const simaai::neat::Tensor* first_tensor_impl(const simaai::neat::Sample& sample) {
  if (sample.kind == simaai::neat::SampleKind::Tensor && sample.tensor.has_value()) {
    return &sample.tensor.value();
  }
  if (sample.kind == simaai::neat::SampleKind::TensorSet && !sample.tensors.empty()) {
    return &sample.tensors.front();
  }
  for (const auto& field : sample.fields) {
    if (const auto* tensor = first_tensor_impl(field); tensor != nullptr) {
      return tensor;
    }
  }
  return nullptr;
}

bool extract_bbox_payload_impl(const simaai::neat::Sample& sample,
                               std::vector<std::uint8_t>& payload, std::string& error) {
  if (sample.kind == simaai::neat::SampleKind::Bundle) {
    for (const auto& field : sample.fields) {
      if (extract_bbox_payload_impl(field, payload, error)) {
        return true;
      }
    }
    error = "bundle missing BBOX field";
    return false;
  }
  if (sample.kind == simaai::neat::SampleKind::TensorSet && !sample.tensors.empty()) {
    simaai::neat::Sample tensor_sample = sample;
    tensor_sample.kind = simaai::neat::SampleKind::Tensor;
    tensor_sample.tensor = sample.tensors.front();
    tensor_sample.tensors.clear();
    return objdet::extract_bbox_payload(tensor_sample, payload, error);
  }
  return objdet::extract_bbox_payload(sample, payload, error);
}

} // namespace

std::vector<Detection> filter_person_detections(const std::vector<Detection>& boxes,
                                                int person_class_id) {
  std::vector<Detection> filtered;
  for (const auto& box : boxes) {
    if (box.class_id == person_class_id) {
      filtered.push_back(box);
    }
  }
  return filtered;
}

InsightTrackingPayload
make_insight_tracking_detection(const std::vector<TrackedDetection>& tracked) {
  InsightTrackingPayload payload;
  payload.tracks.reserve(tracked.size());
  for (const auto& det : tracked) {
    sima_examples::MetadataBox track;
    track.id = std::to_string(det.track_id);
    track.label = "person";
    track.confidence = det.score;
    track.x = det.x1;
    track.y = det.y1;
    track.w = std::max(0.0f, det.x2 - det.x1);
    track.h = std::max(0.0f, det.y2 - det.y1);
    payload.tracks.push_back(track);
  }
  return payload;
}

std::vector<std::uint8_t> extract_bbox_payload(const simaai::neat::Sample& sample) {
  std::vector<std::uint8_t> payload;
  std::string error;
  if (!extract_bbox_payload_impl(sample, payload, error)) {
    return {};
  }
  return payload;
}

std::vector<Detection> parse_bbox_payload(const std::vector<std::uint8_t>& payload, int img_w,
                                          int img_h) {
  const auto parsed = objdet::parse_boxes_lenient(payload, img_w, img_h, 0);
  std::vector<Detection> out;
  out.reserve(parsed.size());
  for (const auto& box : parsed) {
    out.push_back(Detection{
        box.x1,
        box.y1,
        box.x2,
        box.y2,
        box.score,
        box.class_id,
    });
  }
  return out;
}

cv::Mat tensor_rgb_from_sample(const simaai::neat::Sample& sample) {
  const auto* tensor = first_tensor_impl(sample);
  if (tensor == nullptr) {
    throw std::runtime_error("no tensor payload found in decoded RTSP sample");
  }
  return tensor->to_cv_mat_copy(simaai::neat::ImageSpec::PixelFormat::RGB);
}

} // namespace multi_stream_people_tracker
