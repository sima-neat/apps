// Copyright 2026 SiMa Technologies, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "images.h"

#include <opencv2/imgcodecs.hpp>

namespace fs = std::filesystem;

namespace pcb {

std::string image_problem(const cv::Mat& image, ImageSize max_size) {
  if (image.empty()) {
    return "not a readable image";
  }
  if (image.cols > max_size.first || image.rows > max_size.second) {
    return std::to_string(image.cols) + "x" + std::to_string(image.rows) +
           " is larger than app.input_max_width x app.input_max_height (" +
           std::to_string(max_size.first) + "x" + std::to_string(max_size.second) + ")";
  }
  return "";
}

bool save_image(const fs::path& path, const cv::Mat& image) {
  try {
    return cv::imwrite(path.string(), image);
  } catch (const cv::Exception&) {
    return false;
  }
}

} // namespace pcb
