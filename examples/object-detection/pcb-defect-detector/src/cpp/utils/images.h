// Copyright 2026 SiMa Technologies, Inc.
// SPDX-License-Identifier: Apache-2.0

// Finds the input images, checks that the model can take each one, and saves the results.
// Mirrors utils/images.py in the Python implementation.

#pragma once

#include <opencv2/core.hpp>

#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace pcb {

// The largest image the model accepts: width, height.
using ImageSize = std::pair<int, int>;

// The .jpg, .jpeg, .png, and .bmp files in folder, in name order. Throws when there are none.
std::vector<std::filesystem::path> find_images(const std::filesystem::path& folder);

// Why the model cannot take image, or "" when it can.
std::string image_problem(const cv::Mat& image, ImageSize max_size);

// Writes image to path; false when it could not be written, so one image does not stop the batch.
// OpenCV reports some failures by return value and throws for others, such as an encoder error.
bool save_image(const std::filesystem::path& path, const cv::Mat& image);

} // namespace pcb
