// Copyright 2026 SiMa Technologies, Inc.
// SPDX-License-Identifier: Apache-2.0

// The annotated images and the JSON detections report. Mirrors utils/annotate.py in the Python
// implementation. A Box is any type with x1, y1, x2, y2, score, and class_id, such as Neat's.

#pragma once

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace annotate {

// Box colors in BGR, one per class, repeating every eight classes.
inline const std::array<cv::Scalar, 8> kBoxColors = {
    cv::Scalar(0, 255, 0),   cv::Scalar(255, 0, 0),   cv::Scalar(0, 0, 255),
    cv::Scalar(255, 255, 0), cv::Scalar(255, 0, 255), cv::Scalar(0, 255, 255),
    cv::Scalar(128, 255, 0), cv::Scalar(255, 128, 0),
};

// The annotated PNG's file name. It keeps the source extension, so frame.jpg and frame.png do
// not write the same file.
inline std::string output_name(const std::filesystem::path& image_path) {
  return image_path.stem().string() + "_" + image_path.extension().string().substr(1) + ".png";
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

// One image's entry in the report: its name, size, and every box with its label.
template <typename Box>
nlohmann::json detections_record(const std::filesystem::path& image_path, const cv::Mat& image,
                                 const std::vector<Box>& boxes,
                                 const std::vector<std::string>& labels) {
  nlohmann::json detections = nlohmann::json::array();
  for (const Box& box : boxes) {
    detections.push_back({{"class_id", box.class_id},
                          {"label", labels.at(static_cast<size_t>(box.class_id))},
                          {"score", box.score},
                          {"box", {box.x1, box.y1, box.x2, box.y2}}});
  }
  return {{"image", image_path.filename().string()},
          {"width", image.cols},
          {"height", image.rows},
          {"detections", detections}};
}

inline void write_report(const std::filesystem::path& path, const nlohmann::json& records) {
  if (path.has_parent_path()) {
    std::filesystem::create_directories(path.parent_path());
  }
  std::ofstream out(path);
  out << nlohmann::json{{"images", records}}.dump(2) << "\n";
  out.flush();
  if (!out) {
    throw std::runtime_error("cannot write " + path.string());
  }
}

} // namespace annotate
