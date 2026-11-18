// Copied from support/templates/cpp/frames.cpp;
// edit the template and run scripts/sync_utils.py.
#include "frames.h"

#include <opencv2/imgproc.hpp>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace utils {
namespace {

// The frame as packed 4:2:2 (CV_8UC2, Y0 U0 Y1 V0), which cv::COLOR_YUV2BGR_YUY2 converts.
bool nv16_to_yuy2(const simaai::neat::Tensor& t, cv::Mat& out, std::string& err) {
  if (t.image_format() != simaai::neat::ImageSpec::PixelFormat::NV16) {
    err = "expected NV16 tensor";
    return false;
  }
  const simaai::neat::Plane* y = nullptr;
  const simaai::neat::Plane* uv = nullptr;
  for (const auto& plane : t.planes) {
    if (plane.role == simaai::neat::PlaneRole::Y)
      y = &plane;
    if (plane.role == simaai::neat::PlaneRole::UV)
      uv = &plane;
  }
  int w = 0;
  int h = 0;
  if (!y || !uv || y->strides_bytes.empty() || uv->strides_bytes.empty() || !infer_dims(t, w, h)) {
    err = "NV16 tensor without Y/UV planes";
    return false;
  }
  // The payload is the whole mapped buffer, so plane offsets index into it.
  const std::vector<uint8_t> payload = t.copy_payload_bytes();
  const auto y_stride = static_cast<std::size_t>(y->strides_bytes[0]);
  const auto uv_stride = static_cast<std::size_t>(uv->strides_bytes[0]);
  const auto rows = static_cast<std::size_t>(h);
  if (static_cast<std::size_t>(y->byte_offset) + y_stride * rows > payload.size() ||
      static_cast<std::size_t>(uv->byte_offset) + uv_stride * rows > payload.size()) {
    err = "NV16 payload smaller than its planes";
    return false;
  }
  // Each row keeps its own chroma: Y and UV bytes interleave into packed 4:2:2 (Y0 U0 Y1 V0).
  out.create(h, w, CV_8UC2);
  for (int row = 0; row < h; ++row) {
    const uint8_t* y_row = payload.data() + y->byte_offset + row * y_stride;
    const uint8_t* uv_row = payload.data() + uv->byte_offset + row * uv_stride;
    uint8_t* dst = out.ptr(row);
    for (int x = 0; x < w; ++x) {
      dst[2 * x] = y_row[x];
      dst[2 * x + 1] = uv_row[x];
    }
  }
  return true;
}

} // namespace

bool infer_dims(const simaai::neat::Tensor& t, int& w, int& h) {
  w = t.width();
  h = t.height();
  if ((w <= 0 || h <= 0) && t.shape.size() >= 2) {
    h = static_cast<int>(t.shape[0]);
    w = static_cast<int>(t.shape[1]);
  }
  return (w > 0 && h > 0);
}

bool nv12_to_bgr(const simaai::neat::Tensor& t, cv::Mat& out, std::string& err) {
  if (!t.is_nv12()) {
    err = "expected NV12 tensor";
    return false;
  }
  int w = 0;
  int h = 0;
  if (!infer_dims(t, w, h)) {
    err = "invalid tensor dimensions";
    return false;
  }
  std::vector<uint8_t> nv12 = t.copy_nv12_contiguous();
  if (nv12.empty()) {
    err = "NV12 copy failed";
    return false;
  }
  cv::Mat yuv(h + h / 2, w, CV_8UC1, nv12.data());
  cv::cvtColor(yuv, out, cv::COLOR_YUV2BGR_NV12);
  return true;
}

bool nv16_to_bgr(const simaai::neat::Tensor& t, cv::Mat& out, std::string& err) {
  cv::Mat yuy2;
  if (!nv16_to_yuy2(t, yuy2, err))
    return false;
  cv::cvtColor(yuy2, out, cv::COLOR_YUV2BGR_YUY2);
  return true;
}

cv::Mat tensor_bgr_from_decoded(const simaai::neat::Tensor& tensor) {
  cv::Mat bgr;
  std::string err;
  if (nv12_to_bgr(tensor, bgr, err)) {
    return bgr;
  }
  if (nv16_to_bgr(tensor, bgr, err)) {
    return bgr;
  }
  return tensor.to_cv_mat_copy(simaai::neat::ImageSpec::PixelFormat::BGR);
}

const simaai::neat::Sample* find_field(const simaai::neat::Sample& sample,
                                       const std::string& label) {
  if (sample.stream_label == label) {
    return &sample;
  }
  for (const auto& field : sample.fields) {
    if (const auto* found = find_field(field, label)) {
      return found;
    }
  }
  return nullptr;
}

const simaai::neat::Sample& joined_field(const simaai::neat::Sample& sample,
                                         const std::string& label, std::size_t bundle_index) {
  if (const auto* field = find_field(sample, label)) {
    return *field;
  }
  if (sample.kind == simaai::neat::SampleKind::Bundle && sample.fields.size() > bundle_index) {
    return sample.fields[bundle_index];
  }
  throw std::runtime_error("joined output missing " + label + " field");
}

} // namespace utils
