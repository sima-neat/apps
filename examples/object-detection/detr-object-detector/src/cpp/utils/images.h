// Copyright 2026 SiMa Technologies, Inc.
// SPDX-License-Identifier: Apache-2.0

// The input images: the files in the folder, and the largest size among them. Mirrors
// utils/images.py in the Python implementation.

#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace images {

inline const std::set<std::string> kImageSuffixes = {".jpg", ".jpeg", ".png", ".bmp"};

// The images directly in folder, in file-name order.
inline std::vector<std::filesystem::path> image_paths(const std::filesystem::path& folder) {
  if (!std::filesystem::is_directory(folder)) {
    throw std::runtime_error("input.source is not a folder: " + folder.string());
  }
  std::vector<std::filesystem::path> paths;
  for (const auto& entry : std::filesystem::directory_iterator(folder)) {
    std::string suffix = entry.path().extension().string();
    std::transform(suffix.begin(), suffix.end(), suffix.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (entry.is_regular_file() && kImageSuffixes.count(suffix) > 0) {
      paths.push_back(entry.path());
    }
  }
  if (paths.empty()) {
    throw std::runtime_error("input.source has no .jpg, .jpeg, .png, or .bmp files: " +
                             folder.string());
  }
  std::sort(paths.begin(), paths.end());
  return paths;
}

inline cv::Mat read_image(const std::filesystem::path& path) {
  cv::Mat image = cv::imread(path.string(), cv::IMREAD_COLOR);
  if (image.empty()) {
    throw std::runtime_error("cannot read image: " + path.string());
  }
  return image;
}

// The largest width and the largest height among the images, which may be different images.
inline cv::Size largest_image_size(const std::vector<std::filesystem::path>& paths) {
  cv::Size largest(0, 0);
  for (const auto& path : paths) {
    const cv::Mat image = read_image(path);
    largest.width = std::max(largest.width, image.cols);
    largest.height = std::max(largest.height, image.rows);
  }
  return largest;
}

} // namespace images
