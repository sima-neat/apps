// Copyright 2026 SiMa Technologies, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <nlohmann/json.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace instance_seg {

/// MetadataSender rejects a payload above 65507 bytes, and the rejection surfaces as an error the
/// application has to handle mid-stream. Half of that leaves room for the envelope and keeps the
/// datagram count low enough for Insight to reassemble within its 250 ms window.
inline constexpr std::size_t kMetadataByteBudget = 32768;

/// One instance in frame pixels: `bbox` is the detection rectangle, `polygon` its silhouette.
struct MetadataSegment {
  std::string id;
  std::string label;
  float confidence = 0.0f;
  cv::Rect bbox;
  std::vector<cv::Point> polygon;
};

struct EncodedSegments {
  std::string data_json;
  int dropped = 0;
};

/// YOLO26 emits masks at one quarter of the model input per dimension, so a 160x160 head
/// corresponds to a 640x640 input.
inline constexpr int kMaskStride = 4;

/// Mask-head region covering `frame_rect`. The head is a fixed grid over the letterboxed model
/// input, so a frame rectangle reaches it through the same scale and padding the preprocessor used.
inline cv::Rect mask_rect_for_frame_rect(const cv::Rect& frame_rect, const cv::Size& frame_size,
                                         const cv::Size& mask_size) {
  const int model_w = mask_size.width * kMaskStride;
  const int model_h = mask_size.height * kMaskStride;
  const double scale =
      std::min(static_cast<double>(model_w) / static_cast<double>(frame_size.width),
               static_cast<double>(model_h) / static_cast<double>(frame_size.height));
  const double pad_x =
      (static_cast<double>(model_w) - static_cast<double>(frame_size.width) * scale) * 0.5;
  const double pad_y =
      (static_cast<double>(model_h) - static_cast<double>(frame_size.height) * scale) * 0.5;
  const auto to_mask_x = [&](double frame_x) {
    return (frame_x * scale + pad_x) * static_cast<double>(mask_size.width) /
           static_cast<double>(model_w);
  };
  const auto to_mask_y = [&](double frame_y) {
    return (frame_y * scale + pad_y) * static_cast<double>(mask_size.height) /
           static_cast<double>(model_h);
  };

  const int x0 = std::clamp(static_cast<int>(std::floor(to_mask_x(frame_rect.x))), 0,
                            std::max(0, mask_size.width - 1));
  const int y0 = std::clamp(static_cast<int>(std::floor(to_mask_y(frame_rect.y))), 0,
                            std::max(0, mask_size.height - 1));
  const int x1 = std::clamp(static_cast<int>(std::ceil(to_mask_x(frame_rect.x + frame_rect.width))),
                            x0 + 1, mask_size.width);
  const int y1 =
      std::clamp(static_cast<int>(std::ceil(to_mask_y(frame_rect.y + frame_rect.height))), y0 + 1,
                 mask_size.height);
  return cv::Rect(x0, y0, x1 - x0, y1 - y0);
}

/// Mask-head region for `frame_rect`, resized to frame pixels.
inline cv::Mat project_letterbox_mask_roi(const cv::Mat& mask, const cv::Rect& frame_rect,
                                          const cv::Size& frame_size) {
  const cv::Rect mask_rect =
      mask_rect_for_frame_rect(frame_rect, frame_size, cv::Size(mask.cols, mask.rows));
  cv::Mat projected;
  cv::resize(mask(mask_rect), projected, frame_rect.size(), 0, 0, cv::INTER_LINEAR);
  return projected;
}

/// Frame-absolute silhouette of `mask` inside `frame_rect`, empty when the thresholded mask holds
/// nothing Insight can draw. `threshold` is a fraction of full scale, as `output.mask_threshold`
/// is. Upscaling before thresholding is what makes the outline match the rendered overlay.
inline std::vector<cv::Point> mask_polygon(const cv::Mat& mask, const cv::Rect& frame_rect,
                                           const cv::Size& frame_size, double threshold) {
  cv::Mat binary;
  cv::threshold(project_letterbox_mask_roi(mask, frame_rect, frame_size), binary, threshold * 255.0,
                255, cv::THRESH_BINARY);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  if (contours.empty()) {
    return {};
  }
  const auto& largest =
      *std::max_element(contours.begin(), contours.end(),
                        [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                          return cv::contourArea(a) < cv::contourArea(b);
                        });

  std::vector<cv::Point> polygon;
  cv::approxPolyDP(largest, polygon, 0.004 * cv::arcLength(largest, true), true);
  if (polygon.size() < 3) {
    return {};
  }
  // Contour points lie inside frame_rect, which is already clamped to the frame, so shifting them
  // into frame space cannot leave the image.
  for (auto& point : polygon) {
    point += frame_rect.tl();
  }
  return polygon;
}

/// `data` object of a `segmentation` metadata message. Segments that do not fit the byte budget are
/// dropped lowest-confidence first and counted.
inline EncodedSegments encode_segments(std::vector<MetadataSegment> segments) {
  // Stable, so segments tying on confidence are dropped in the same order the Python
  // implementation drops them.
  std::stable_sort(segments.begin(), segments.end(),
                   [](const MetadataSegment& a, const MetadataSegment& b) {
                     return a.confidence > b.confidence;
                   });

  nlohmann::json entries = nlohmann::json::array();
  std::size_t bytes = sizeof(R"({"segments":[]})") - 1;
  for (const auto& segment : segments) {
    nlohmann::json points = nlohmann::json::array();
    for (const auto& point : segment.polygon) {
      points.push_back({point.x, point.y});
    }
    nlohmann::json entry = {
        {"id", segment.id},
        {"label", segment.label},
        {"confidence", segment.confidence},
        {"bbox", {segment.bbox.x, segment.bbox.y, segment.bbox.width, segment.bbox.height}},
        {"mask_format", "polygon"},
        {"mask", std::move(points)},
    };
    const std::size_t entry_bytes = entry.dump().size() + 1;
    if (bytes + entry_bytes > kMetadataByteBudget) {
      break;
    }
    bytes += entry_bytes;
    entries.push_back(std::move(entry));
  }

  const int dropped = static_cast<int>(segments.size() - entries.size());
  return {nlohmann::json{{"segments", std::move(entries)}}.dump(), dropped};
}

} // namespace instance_seg
