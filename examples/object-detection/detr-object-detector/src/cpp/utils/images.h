// Copyright 2026 SiMa Technologies, Inc.
// SPDX-License-Identifier: Apache-2.0

// The largest size among the input images. Mirrors utils/images.py in the Python implementation.

#pragma once

#include "sima_examples/image_io.h"

#include <opencv2/core.hpp>

#include <algorithm>
#include <filesystem>
#include <vector>

namespace images {

// The largest width and the largest height among the images, which may be different images.
inline cv::Size largest_image_size(const std::vector<std::filesystem::path>& paths) {
  cv::Size largest(0, 0);
  for (const auto& path : paths) {
    const cv::Mat image = sima_examples::read_image(path);
    largest.width = std::max(largest.width, image.cols);
    largest.height = std::max(largest.height, image.rows);
  }
  return largest;
}

} // namespace images
