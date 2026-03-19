#include "examples/tracking/multi-camera-people-detection-and-tracking-optiview/cpp/sample_utils_api.cpp"

#include "support/object_detection/obj_detection_utils.h"

#include "pipeline/TensorCore.h"

#include <stdexcept>
#include <vector>

namespace multi_camera_people_tracking {
namespace {

const simaai::neat::Tensor* first_tensor_impl(const simaai::neat::Sample& sample) {
  if (sample.kind == simaai::neat::SampleKind::Tensor && sample.tensor.has_value()) {
    return &sample.tensor.value();
  }
  for (const auto& field : sample.fields) {
    if (const auto* tensor = first_tensor_impl(field); tensor != nullptr) {
      return tensor;
    }
  }
  return nullptr;
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

OptiViewTrackingPayload
make_optiview_tracking_detection(const std::vector<TrackedDetection>& tracked) {
  OptiViewTrackingPayload payload;
  payload.objects.reserve(tracked.size());
  payload.labels.reserve(tracked.size());
  for (std::size_t index = 0; index < tracked.size(); ++index) {
    const auto& det = tracked[index];
    sima_examples::OptiViewObject object;
    object.x = static_cast<int>(det.x1);
    object.y = static_cast<int>(det.y1);
    object.w = static_cast<int>(std::max(0.0f, det.x2 - det.x1));
    object.h = static_cast<int>(std::max(0.0f, det.y2 - det.y1));
    object.score = det.score;
    object.class_id = static_cast<int>(index);
    payload.objects.push_back(object);
    payload.labels.push_back("Track ID: " + std::to_string(det.track_id));
  }
  return payload;
}

std::vector<std::uint8_t> extract_bbox_payload(const simaai::neat::Sample& sample) {
  std::vector<std::uint8_t> payload;
  std::string error;
  if (!sima_examples::extract_bbox_payload(sample, payload, error)) {
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

} // namespace multi_camera_people_tracking
