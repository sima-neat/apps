#include "sample_utils_api.cpp"

#include "neat/session.h"
#include "support/object_detection/obj_detection_utils.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace multistream_yolox_yolov8_optiview {
namespace {

std::string sample_payload_tag_upper(const simaai::neat::Sample& sample) {
  std::string tag = sample.payload_tag.empty() ? sample.format : sample.payload_tag;
  tag = objdet::upper_ascii_copy(tag);
  if (!tag.empty()) {
    return tag;
  }
  for (const auto& field : sample.fields) {
    tag = sample_payload_tag_upper(field);
    if (!tag.empty()) {
      return tag;
    }
  }
  return {};
}

std::string class_label(const std::vector<std::string>& class_labels, int class_id) {
  if (class_id >= 0 && class_id < static_cast<int>(class_labels.size())) {
    return class_labels[static_cast<std::size_t>(class_id)];
  }
  return std::to_string(class_id);
}

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

std::string to_string(DetectorOutputKind kind) {
  switch (kind) {
  case DetectorOutputKind::BBox:
    return "BBOX";
  case DetectorOutputKind::DetessDequant:
    return "DETESSDEQUANT";
  }
  return "UNKNOWN";
}

std::vector<std::uint8_t> extract_bbox_payload(const simaai::neat::Sample& sample) {
  std::vector<std::uint8_t> payload;
  std::string error;
  if (!objdet::extract_bbox_payload(sample, payload, error)) {
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

DetectorOutputKind require_detector_output_kind(ModelFamily family,
                                                const simaai::neat::Sample& sample) {
  std::string expected;
  DetectorOutputKind kind = DetectorOutputKind::BBox;
  switch (family) {
  case ModelFamily::YoloV8:
    expected = "BBOX";
    kind = DetectorOutputKind::BBox;
    break;
  case ModelFamily::YoloX:
    expected = "DETESSDEQUANT";
    kind = DetectorOutputKind::DetessDequant;
    break;
  case ModelFamily::Auto:
    throw std::invalid_argument("unsupported model family");
  }

  const std::string actual = sample_payload_tag_upper(sample);
  if (actual != expected) {
    throw std::runtime_error("unsupported detector output: family=" + to_string(family) +
                             " expected=" + expected + " actual=" +
                             (actual.empty() ? std::string("<empty>") : actual));
  }
  return kind;
}

std::vector<Detection> detections_from_detector_sample(ModelFamily family,
                                                       const simaai::neat::Sample& sample,
                                                       int img_w, int img_h) {
  const DetectorOutputKind kind = require_detector_output_kind(family, sample);
  const auto payload = extract_bbox_payload(sample);
  if (!payload.empty()) {
    return parse_bbox_payload(payload, img_w, img_h);
  }
  if (kind == DetectorOutputKind::DetessDequant) {
    throw std::runtime_error(
        "YOLOX detector output did not expose a BBOX payload; "
        "DetessDequant output parsing could not be finalized for this model pack");
  }
  return {};
}

std::string optiview_frame_id(const simaai::neat::Sample& sample, int fallback_frame_index) {
  if (sample.frame_id >= 0) {
    return std::to_string(sample.frame_id);
  }
  return std::to_string(fallback_frame_index);
}

std::int64_t optiview_timestamp_ms(double publish_time_s, double offset_ms) {
  return static_cast<std::int64_t>(std::llround(publish_time_s * 1000.0 + offset_ms));
}

OptiViewDetectionPayload build_optiview_detection_payload(const std::vector<Detection>& detections,
                                                          int img_w, int img_h,
                                                          const std::vector<std::string>& class_labels) {
  OptiViewDetectionPayload payload;
  payload.objects.reserve(detections.size());
  for (const auto& det : detections) {
    const int x1 = std::max(0, std::min(img_w, static_cast<int>(det.x1)));
    const int y1 = std::max(0, std::min(img_h, static_cast<int>(det.y1)));
    const int x2 = std::max(0, std::min(img_w, static_cast<int>(det.x2)));
    const int y2 = std::max(0, std::min(img_h, static_cast<int>(det.y2)));
    if (x2 <= x1 || y2 <= y1) {
      continue;
    }

    sima_examples::OptiViewObject object;
    object.x = x1;
    object.y = y1;
    object.w = std::max(0, x2 - x1);
    object.h = std::max(0, y2 - y1);
    object.score = det.score;
    object.class_id = det.class_id;
    payload.objects.push_back(object);
  }
  // OptiView resolves object labels by indexing this full label table with object.class_id.
  payload.labels = class_labels;
  return payload;
}

cv::Mat tensor_rgb_from_sample(const simaai::neat::Sample& sample) {
  const auto* tensor = first_tensor_impl(sample);
  if (tensor == nullptr) {
    throw std::runtime_error("no tensor payload found in decoded RTSP sample");
  }
  return tensor->to_cv_mat_copy(simaai::neat::ImageSpec::PixelFormat::RGB);
}

} // namespace multistream_yolox_yolov8_optiview
