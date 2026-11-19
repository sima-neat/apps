#pragma once

#include <neat.h>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

#include <cstdint>
#include <cctype>
#include <exception>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace objdet {

struct Box {
  float x1 = 0.0f;
  float y1 = 0.0f;
  float x2 = 0.0f;
  float y2 = 0.0f;
  float score = 0.0f;
  int class_id = -1;
};

inline void require(bool cond, const std::string& msg) {
  if (!cond)
    throw std::runtime_error(msg);
}

std::vector<Box> parse_boxes_strict(const std::vector<uint8_t>& bytes, int img_w, int img_h,
                                    int expected_topk, bool debug);
void parse_boxes_strict_into(const std::vector<uint8_t>& bytes, int img_w, int img_h,
                             int expected_topk, bool debug, std::vector<Box>& out);
void parse_boxes_strict_into(std::span<const uint8_t> bytes, int img_w, int img_h,
                             int expected_topk, bool debug, std::vector<Box>& out);

void draw_boxes(cv::Mat& img, const std::vector<Box>& boxes, float min_score,
                const cv::Scalar& color, const std::string& label_prefix);

inline std::string upper_ascii_copy(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    out.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
  }
  return out;
}

inline std::string append_context(const std::string& msg, const std::string& context) {
  if (context.empty())
    return msg;
  return msg + " " + context;
}

inline bool extract_bbox_payload_impl(const simaai::neat::Sample& result,
                                      const std::string& context, std::vector<uint8_t>& payload,
                                      std::string& err) {
  if (result.kind == simaai::neat::SampleKind::Bundle) {
    for (const auto& field : result.fields) {
      if (extract_bbox_payload_impl(field, context, payload, err))
        return true;
    }
    err = append_context("bundle missing BBOX field", context);
    return false;
  }
  if (result.kind != simaai::neat::SampleKind::Tensor) {
    err = append_context("capture_expected_tensor", context);
    return false;
  }
  if (!result.tensor.has_value()) {
    err = append_context("capture_missing_tensor", context);
    return false;
  }

  const auto& tensor = result.tensor.value();
  std::string fmt = result.payload_tag;
  if (fmt.empty() && !result.format.empty())
    fmt = result.format;
  if (fmt.empty() && tensor.semantic.tess.has_value()) {
    fmt = tensor.semantic.tess->format;
  }

  const std::string fmt_upper = upper_ascii_copy(fmt);
  if (!fmt_upper.empty() && fmt_upper != "BBOX") {
    err = append_context("capture_expected_bbox", context);
    err += " format=" + fmt_upper;
    return false;
  }

  try {
    payload = tensor.copy_payload_bytes();
  } catch (const std::exception& ex) {
    err = append_context("capture_payload_failed", context);
    err += " err=";
    err += ex.what();
    return false;
  }

  if (payload.empty()) {
    err = append_context("capture_empty_payload", context);
    return false;
  }

  return true;
}

inline bool extract_bbox_payload(const simaai::neat::Sample& result, std::vector<uint8_t>& payload,
                                 std::string& err) {
  return extract_bbox_payload_impl(result, "", payload, err);
}

inline bool extract_bbox_payload(const simaai::neat::Sample& result, int iter,
                                 std::vector<uint8_t>& payload, std::string& err) {
  return extract_bbox_payload_impl(result, "iter=" + std::to_string(iter), payload, err);
}

} // namespace objdet
