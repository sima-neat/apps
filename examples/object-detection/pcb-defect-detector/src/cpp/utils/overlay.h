// Copyright 2026 SiMa Technologies, Inc.
// SPDX-License-Identifier: Apache-2.0

// Draws defect boxes and their labels on a BGR image. Mirrors utils/overlay.py in the Python
// implementation, so both save the same annotated images.

#pragma once

#include "neat.h"

#include <opencv2/core.hpp>

#include <array>
#include <string>
#include <vector>

namespace pcb {

// BGR, one per class in pcb_label.txt order.
inline const std::array<cv::Scalar, 6> kClassColors = {
    cv::Scalar(56, 56, 255), cv::Scalar(29, 178, 255), cv::Scalar(10, 249, 72),
    cv::Scalar(255, 194, 0), cv::Scalar(255, 0, 200),  cv::Scalar(49, 210, 207)};

// A box coordinate as a pixel index from 0 to limit - 1, rounded half away from zero.
int pixel(float value, int limit);

// Draws each box in its class's color, with its label and score, on image in place. A box with no
// width or height inside the image is skipped.
void draw_boxes(cv::Mat& image, const std::vector<simaai::neat::Box>& boxes,
                const std::vector<std::string>& labels);

} // namespace pcb
