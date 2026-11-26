// Copyright 2026 SiMa Technologies, Inc.
// SPDX-License-Identifier: Apache-2.0

// The defect regions of an anomaly map, as the segments Insight draws. Mirrors
// sima_examples.anomaly_regions in support/python, so both languages send the same metadata.

#pragma once

#include <opencv2/core.hpp>

#include <array>
#include <string>
#include <vector>

namespace sima_examples {

struct RegionOptions {
  int width = 0; // Frame size the regions are scaled to, in pixels.
  int height = 0;
  double threshold = 0.0; // Map value at which a pixel counts as anomalous.
  int min_region_px = 0;  // Smallest region, in map pixels, that counts as a defect.
  int max_regions = 0;    // Regions kept, highest score first.
};

struct Region {
  std::array<int, 4> bbox{};               // x, y, width, height, in frame pixels.
  double score = 0.0;                      // Highest map value inside the region.
  std::vector<std::array<int, 2>> polygon; // Outline as [x, y] points, in frame pixels.
};

// The connected regions at or above the threshold that cover at least min_region_px map pixels,
// highest score first, scaled from the map to the frame. `anomaly_map` is CV_32FC1.
std::vector<Region> defect_regions(const cv::Mat& anomaly_map, const RegionOptions& options);

// The "segmentation" metadata: each region's box and outline, labelled as an anomaly.
std::string segments_json(const std::vector<Region>& regions);

} // namespace sima_examples
