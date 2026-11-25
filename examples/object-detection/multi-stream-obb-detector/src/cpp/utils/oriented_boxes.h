// Copyright 2026 SiMa Technologies, Inc.
// SPDX-License-Identifier: Apache-2.0

// The Insight object-detection data for one frame's oriented boxes. Mirrors
// utils/oriented_boxes.py in the Python implementation, so both send the same JSON.

#pragma once

#include "neat.h"

#include <string>
#include <vector>

namespace obb {

// The `data` of one frame's `object-detection` message, as {"objects": [...]}. `boxes` is what
// decode_obb_tensor returns: geometry [N, 5] as cx, cy, width, height, and the angle in radians,
// in source-frame pixels, and scores and class IDs [N]. A positive angle turns the box clockwise.
// `oriented_bbox` is what Insight draws. `bbox` is the box's upright extent, which places the
// label. Pixels are rounded to 0.1, angles to 4 decimals, and scores to 3, so that a frame with
// hundreds of boxes stays under the 65507-byte limit of one metadata message.
std::string detections_json(const simaai::neat::OrientedDecodeTensors& boxes,
                            const std::vector<std::string>& labels);

} // namespace obb
