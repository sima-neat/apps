// Copyright 2026 SiMa Technologies, Inc.
// SPDX-License-Identifier: Apache-2.0

// The annotated images. Mirrors utils/annotate.py in the Python implementation. A Box is any type
// with x1, y1, x2, y2, score, and class_id, such as Neat's.

#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <filesystem>
#include <string>
#include <vector>

namespace annotate {

// Box colors in BGR, one per class, repeating every eight classes.
inline const std::array<cv::Scalar, 8> kBoxColors = {
    cv::Scalar(0, 255, 0),   cv::Scalar(255, 0, 0),   cv::Scalar(0, 0, 255),
    cv::Scalar(255, 255, 0), cv::Scalar(255, 0, 255), cv::Scalar(0, 255, 255),
    cv::Scalar(128, 255, 0), cv::Scalar(255, 128, 0),
};

// The annotated PNG's file name.
inline std::string output_name(const std::filesystem::path& image_path) {
  return image_path.stem().string() + "_detr.png";
}

// A copy of image with each box, its label, and its score drawn on it.
template <typename Box>
cv::Mat draw_detections(const cv::Mat& image, const std::vector<Box>& boxes,
                        const std::vector<std::string>& labels) {
  cv::Mat annotated = image.clone();
  for (const Box& box : boxes) {
    const cv::Scalar& color = kBoxColors[static_cast<size_t>(box.class_id) % kBoxColors.size()];
    // The decoder clamps boxes to the image, so the coordinates are not negative.
    const cv::Point top_left(static_cast<int>(box.x1 + 0.5f), static_cast<int>(box.y1 + 0.5f));
    const cv::Point bottom_right(static_cast<int>(box.x2 + 0.5f), static_cast<int>(box.y2 + 0.5f));
    cv::rectangle(annotated, top_left, bottom_right, color, 2);
    const std::string text = labels.at(static_cast<size_t>(box.class_id)) + " " +
                             cv::format("%.2f", static_cast<double>(box.score));
    cv::putText(annotated, text, cv::Point(top_left.x, std::max(0, top_left.y - 4)),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv::LINE_AA);
  }
  return annotated;
}

} // namespace annotate
