// Copyright 2026 SiMa Technologies, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "images.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace fs = std::filesystem;

namespace pcb {

std::vector<fs::path> find_images(const fs::path& folder) {
  std::vector<fs::path> images;
  for (const auto& entry : fs::directory_iterator(folder)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    std::string extension = entry.path().extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (extension == ".jpg" || extension == ".jpeg" || extension == ".png" || extension == ".bmp") {
      images.push_back(entry.path());
    }
  }
  if (images.empty()) {
    throw std::runtime_error("input.source: no .jpg, .jpeg, .png, or .bmp images in " +
                             folder.string());
  }
  std::sort(images.begin(), images.end());
  return images;
}

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

} // namespace pcb
