// Copyright 2026 SiMa Technologies, Inc.
// SPDX-License-Identifier: Apache-2.0

// The Insight object-detection data for one frame's oriented boxes. Mirrors
// utils/oriented_boxes.py in the Python implementation, so both send the same JSON.

#pragma once

#include <string>
#include <vector>

namespace obb {

// One detection in source-frame pixels. A positive angle turns the box clockwise.
struct OrientedBox {
  double cx = 0.0;
  double cy = 0.0;
  double width = 0.0;
  double height = 0.0;
  double angle_radians = 0.0;
  double score = 0.0;
  int class_id = 0;
};

// The `data` of one frame's `object-detection` message, as {"objects": [...]}. `oriented_bbox`
// is what Insight draws. `bbox` is the box's upright extent, which places the label. Pixels are
// rounded to 0.1, angles to 4 decimals, and scores to 3, so that a frame with hundreds of boxes
// stays under the 65507-byte limit of one metadata message.
std::string detections_json(const std::vector<OrientedBox>& boxes,
                            const std::vector<std::string>& labels);

} // namespace obb
