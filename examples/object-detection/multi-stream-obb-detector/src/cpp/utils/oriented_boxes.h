// Copyright 2026 SiMa Technologies, Inc.
// SPDX-License-Identifier: Apache-2.0

// The Insight object-detection data for one frame's oriented boxes. Mirrors
// utils/oriented_boxes.py in the Python implementation, so both send the same JSON.

#pragma once

#include "neat.h"

#include <string>
#include <vector>

namespace obb {

// One frame's `object-detection` data for Insight. Values are rounded so that a frame with
// hundreds of boxes stays under the 65507-byte limit of one metadata message.
std::string detections_json(const simaai::neat::OrientedDecodeTensors& boxes,
                            const std::vector<std::string>& labels);

} // namespace obb
