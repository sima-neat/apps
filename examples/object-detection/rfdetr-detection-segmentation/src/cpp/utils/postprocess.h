// Copyright 2026 SiMa Technologies, Inc.
// SPDX-License-Identifier: Apache-2.0

// RF-DETR postprocessing: the proposals the transformer reads and the metadata Insight draws.
// Mirrors utils/postprocess.py in the Python implementation, so both send the same metadata.

#pragma once

#include <span>
#include <string>
#include <vector>

namespace rfdetr {

// COCO class IDs 0 to 90, one label each. ID 0 and the "unused" labels name no object.
constexpr int kNumClasses = 91;

struct PostprocessOptions {
  int width = 0; // Frame size the boxes and polygons are scaled to, in pixels.
  int height = 0;
  float min_score = 0.0F;
  int max_detections = 0;
  float mask_threshold = 0.0F; // Segmentation only.
  int mask_grid_size = 0;      // Segmentation only.
};

// The `count` proposal boxes with the highest scores, best first, as count x 4 values.
std::vector<float> top_proposals(std::span<const float> scores, std::span<const float> proposals,
                                 int count);

// The "objects" metadata: each box in frame pixels as [x, y, width, height].
std::string detection_json(std::span<const float> boxes, std::span<const float> logits,
                           const std::vector<std::string>& labels,
                           const PostprocessOptions& options);

// The "segments" metadata: each object's box and its outline as a polygon, in frame pixels.
// `masks` holds mask_size x mask_size x queries values.
std::string segmentation_json(std::span<const float> boxes, std::span<const float> logits,
                              std::span<const float> masks, int mask_size,
                              const std::vector<std::string>& labels,
                              const PostprocessOptions& options);

} // namespace rfdetr
