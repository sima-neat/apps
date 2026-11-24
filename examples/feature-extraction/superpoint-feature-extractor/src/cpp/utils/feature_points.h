// Copyright 2026 SiMa Technologies, Inc.
// SPDX-License-Identifier: Apache-2.0

// SuperPoint feature points: from the decoder's model-input pixels to the metadata Insight draws.
// Mirrors utils/feature_points.py in the Python implementation, so both send the same metadata.

#pragma once

#include "neat.h"

#include <string>
#include <vector>

namespace superpoint {

struct FrameSize {
  int width = 0;
  int height = 0;
};

struct Point {
  long x = 0;
  long y = 0;
};

// The values of a Float32 tensor, such as the decoded keypoints; a frame without feature points has
// none. Python reads them with Tensor.to_numpy.
std::vector<float> floats(const simaai::neat::Tensor& tensor);

// The keypoints, interleaved x, y, on the source frame, in whole pixels. The decoder reports
// model-input pixels (sima-neat/core#1328). After a Stretch resize (resize.mode = Stretch with
// resize.enable = On, as main sets it), each axis maps back by its own ratio. A letterbox would
// need its padding removed first. Each point rounds half up with std::lround, and a point that
// lands on or beyond a frame edge is dropped: the edges are exclusive coordinates.
std::vector<Point> source_points(const std::vector<float>& keypoints, const FrameSize& model_size,
                                 const FrameSize& source_size);

// The `data` of Insight's pose-estimation overlay: one pose whose keypoints are the points.
// Insight draws each keypoint as a dot. It draws a skeleton only between COCO joint names and a
// box only around keypoints with a confidence above 0.3, so unnamed points without a confidence
// get neither.
std::string points_json(const std::vector<Point>& points);

} // namespace superpoint
