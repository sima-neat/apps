#include "scrfd_decode.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace face_recog {

namespace {

// scrfd_2.5g_bnkps: three strides, two anchors per cell, one score per anchor. The outputs
// are the scores for strides 8, 16 and 32, then the boxes, then the landmarks.
constexpr int kStrides[3] = {8, 16, 32};
constexpr int kAnchorsPerCell = 2;
constexpr int kBoxValues = 4;
constexpr int kLandmarkValues = 10;
// Highest-scoring candidates kept for overlap removal.
constexpr std::size_t kMaxCandidates = 5000;

float overlap(const Detection& a, const Detection& b) {
  const float width = std::max(0.0f, std::min(a.x2, b.x2) - std::max(a.x1, b.x1) + 1.0f);
  const float height = std::max(0.0f, std::min(a.y2, b.y2) - std::max(a.y1, b.y1) + 1.0f);
  const float intersection = width * height;
  const float area_a = (a.x2 - a.x1 + 1.0f) * (a.y2 - a.y1 + 1.0f);
  const float area_b = (b.x2 - b.x1 + 1.0f) * (b.y2 - b.y1 + 1.0f);
  return intersection / (area_a + area_b - intersection);
}

} // namespace

Letterbox letterbox_for(int frame_w, int frame_h) {
  Letterbox letterbox;
  letterbox.scale = std::min(static_cast<float>(kScrfdInputW) / frame_w,
                             static_cast<float>(kScrfdInputH) / frame_h);
  letterbox.pad_left = (kScrfdInputW - static_cast<int>(std::round(frame_w * letterbox.scale))) / 2;
  letterbox.pad_top = (kScrfdInputH - static_cast<int>(std::round(frame_h * letterbox.scale))) / 2;
  return letterbox;
}

std::pair<simaai::neat::Tensor, Letterbox> preprocess_scrfd(const cv::Mat& bgr) {
  if (bgr.empty()) {
    throw std::runtime_error("cannot detect faces in an empty image");
  }
  const Letterbox letterbox = letterbox_for(bgr.cols, bgr.rows);
  const cv::Size scaled_size(static_cast<int>(std::round(bgr.cols * letterbox.scale)),
                             static_cast<int>(std::round(bgr.rows * letterbox.scale)));
  cv::Mat scaled;
  cv::resize(bgr, scaled, scaled_size, 0, 0, cv::INTER_LINEAR);
  cv::Mat canvas(kScrfdInputH, kScrfdInputW, CV_8UC3, cv::Scalar(0, 0, 0));
  scaled.copyTo(canvas(cv::Rect(letterbox.pad_left, letterbox.pad_top, scaled.cols, scaled.rows)));
  cv::cvtColor(canvas, canvas, cv::COLOR_BGR2RGB);
  cv::Mat input;
  canvas.convertTo(input, CV_32FC3, 1.0 / 255.0);
  return {tensor_from_hwc_f32(input), letterbox};
}

std::vector<Detection> decode_scrfd(const std::vector<simaai::neat::Tensor>& tensors,
                                    const ScrfdConfig& cfg, const Letterbox& letterbox) {
  if (tensors.size() != 9) {
    throw std::runtime_error("expected 9 SCRFD output tensors, got " +
                             std::to_string(tensors.size()));
  }
  // sigmoid(logit) >= t is logit >= log(t / (1 - t)), so most anchors skip the exponential.
  const float logit_threshold = std::log(cfg.min_score / (1.0f - cfg.min_score));
  // Model-input pixels back to source-frame pixels.
  const auto source_x = [&](float x) { return (x - letterbox.pad_left) / letterbox.scale; };
  const auto source_y = [&](float y) { return (y - letterbox.pad_top) / letterbox.scale; };

  std::vector<Detection> candidates;
  for (int s = 0; s < 3; ++s) {
    const int stride = kStrides[s];
    const int columns = kScrfdInputW / stride;
    const std::size_t anchors =
        static_cast<std::size_t>(kScrfdInputH / stride) * columns * kAnchorsPerCell;
    const auto scores = tensor_to_f32(tensors[s]);
    const auto boxes = tensor_to_f32(tensors[3 + s]);
    const auto landmarks = tensor_to_f32(tensors[6 + s]);
    if (scores.size() != anchors || boxes.size() != anchors * kBoxValues ||
        landmarks.size() != anchors * kLandmarkValues) {
      throw std::runtime_error("unexpected SCRFD output size at stride " + std::to_string(stride));
    }

    for (std::size_t i = 0; i < anchors; ++i) {
      if (scores[i] < logit_threshold) {
        continue;
      }
      // Both anchors of a cell share its top-left corner as their centre.
      const std::size_t cell = i / kAnchorsPerCell;
      const float centre_x = static_cast<float>((cell % columns) * stride);
      const float centre_y = static_cast<float>((cell / columns) * stride);

      Detection face;
      face.score = 1.0f / (1.0f + std::exp(-scores[i]));
      // Boxes and landmarks are distances from the anchor centre, in units of the stride.
      const float* box = &boxes[i * kBoxValues];
      face.x1 = source_x(centre_x - box[0] * stride);
      face.y1 = source_y(centre_y - box[1] * stride);
      face.x2 = source_x(centre_x + box[2] * stride);
      face.y2 = source_y(centre_y + box[3] * stride);
      const float* points = &landmarks[i * kLandmarkValues];
      for (int k = 0; k < 5; ++k) {
        face.landmarks[k * 2] = source_x(centre_x + points[k * 2] * stride);
        face.landmarks[k * 2 + 1] = source_y(centre_y + points[k * 2 + 1] * stride);
      }
      candidates.push_back(face);
    }
  }

  std::sort(candidates.begin(), candidates.end(),
            [](const Detection& a, const Detection& b) { return a.score > b.score; });
  if (candidates.size() > kMaxCandidates) {
    candidates.resize(kMaxCandidates);
  }

  std::vector<Detection> faces;
  for (const auto& candidate : candidates) {
    const bool overlaps = std::any_of(faces.begin(), faces.end(), [&](const Detection& face) {
      return overlap(face, candidate) > cfg.nms_iou;
    });
    if (overlaps) {
      continue;
    }
    faces.push_back(candidate);
    if (faces.size() == static_cast<std::size_t>(cfg.max_faces)) {
      break;
    }
  }
  return faces;
}

std::vector<simaai::neat::Tensor> collect_tensors(const simaai::neat::Sample& sample) {
  if (sample.kind == simaai::neat::SampleKind::Tensor && sample.tensor.has_value()) {
    return {*sample.tensor};
  }
  if (sample.kind == simaai::neat::SampleKind::TensorSet) {
    return sample.tensors;
  }
  std::vector<simaai::neat::Tensor> tensors;
  for (const auto& field : sample.fields) {
    const auto nested = collect_tensors(field);
    tensors.insert(tensors.end(), nested.begin(), nested.end());
  }
  return tensors;
}

std::vector<float> tensor_to_f32(const simaai::neat::Tensor& tensor) {
  const auto bytes = tensor.copy_dense_bytes_tight();
  if (tensor.dtype == simaai::neat::TensorDType::BFloat16) {
    // A bfloat16 value is the upper half of a float32.
    std::vector<float> values(bytes.size() / 2);
    for (std::size_t i = 0; i < values.size(); ++i) {
      std::uint16_t half = 0;
      std::memcpy(&half, bytes.data() + i * 2, sizeof(half));
      const std::uint32_t bits = static_cast<std::uint32_t>(half) << 16;
      std::memcpy(&values[i], &bits, sizeof(float));
    }
    return values;
  }
  if (tensor.dtype != simaai::neat::TensorDType::Float32) {
    throw std::runtime_error("model output is neither float32 nor bfloat16");
  }
  std::vector<float> values(bytes.size() / sizeof(float));
  std::memcpy(values.data(), bytes.data(), values.size() * sizeof(float));
  return values;
}

simaai::neat::Tensor tensor_from_hwc_f32(const cv::Mat& image) {
  if (image.empty() || image.type() != CV_32FC3) {
    throw std::runtime_error("model input must be a float32 image with three channels");
  }
  const cv::Mat packed = image.isContinuous() ? image : image.clone();
  const int height = packed.rows;
  const int width = packed.cols;
  const std::vector<float> values(
      packed.ptr<float>(), packed.ptr<float>() + static_cast<std::size_t>(height) * width * 3);
  auto tensor = simaai::neat::Tensor::from_vector(values, {height, width, 3},
                                                  simaai::neat::TensorMemory::EV74);
  tensor.layout = simaai::neat::TensorLayout::HWC;
  tensor.shape = {height, width, 3};
  tensor.strides_bytes = {static_cast<int64_t>(width * 3 * sizeof(float)),
                          static_cast<int64_t>(3 * sizeof(float)),
                          static_cast<int64_t>(sizeof(float))};
  return tensor;
}

} // namespace face_recog
