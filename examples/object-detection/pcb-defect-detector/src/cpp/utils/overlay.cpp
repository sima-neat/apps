// Copyright 2026 SiMa Technologies, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "overlay.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>

namespace pcb {

int pixel(float value, int limit) {
  return std::clamp(static_cast<int>(std::round(value)), 0, limit - 1);
}

void draw_boxes(cv::Mat& image, const std::vector<simaai::neat::Box>& boxes,
                const std::vector<std::string>& labels) {
  for (const auto& box : boxes) {
    const int x1 = pixel(box.x1, image.cols);
    const int y1 = pixel(box.y1, image.rows);
    const int x2 = pixel(box.x2, image.cols);
    const int y2 = pixel(box.y2, image.rows);
    if (x2 <= x1 || y2 <= y1) {
      continue;
    }
    const cv::Scalar color = kClassColors[static_cast<size_t>(box.class_id) % kClassColors.size()];
    cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);
    const std::string text =
        labels.at(static_cast<size_t>(box.class_id)) + " " + cv::format("%.2f", box.score);
    int baseline = 0;
    const cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    const int label_top = std::max(0, y1 - text_size.height - 4);
    cv::rectangle(image, cv::Point(x1, label_top), cv::Point(x1 + text_size.width + 2, y1), color,
                  cv::FILLED);
    cv::putText(image, text, cv::Point(x1 + 1, std::max(10, y1 - 3)), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
  }
}

} // namespace pcb
