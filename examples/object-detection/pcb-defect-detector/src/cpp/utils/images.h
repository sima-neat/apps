// Copyright 2026 SiMa Technologies, Inc.
// SPDX-License-Identifier: Apache-2.0

// Finds the input images and checks that the model can take each one. Mirrors utils/images.py in
// the Python implementation.

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

} // namespace pcb
