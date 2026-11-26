// Copyright 2026 SiMa Technologies, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "sima_examples/anomaly_regions.h"

#include <nlohmann/json.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstddef>
#include <utility>

namespace sima_examples {
namespace {

constexpr const char* kLabel = "anomaly";

} // namespace

std::vector<Region> defect_regions(const cv::Mat& anomaly_map, const RegionOptions& options) {
  const int map_width = anomaly_map.cols;
  const int map_height = anomaly_map.rows;
  const cv::Mat above = anomaly_map >= options.threshold;
  cv::Mat labels;
  cv::Mat stats;
  cv::Mat centroids;
  const int count = cv::connectedComponentsWithStats(above, labels, stats, centroids, 8);
  std::vector<Region> regions;
  for (int index = 1; index < count; ++index) {
    const int* stat = stats.ptr<int>(index);
    if (stat[cv::CC_STAT_AREA] < options.min_region_px) {
      continue;
    }
    const cv::Mat member = labels == index;
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(member, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    const int x = stat[cv::CC_STAT_LEFT];
    const int y = stat[cv::CC_STAT_TOP];
    // Integer scaling, the same in both languages: map pixel m covers frame pixels from
    // m * width / map_width.
    const int x0 = x * options.width / map_width;
    const int y0 = y * options.height / map_height;
    const int x1 = (x + stat[cv::CC_STAT_WIDTH]) * options.width / map_width;
    const int y1 = (y + stat[cv::CC_STAT_HEIGHT]) * options.height / map_height;
    Region region;
    region.bbox = {x0, y0, x1 - x0, y1 - y0};
    cv::minMaxLoc(anomaly_map, nullptr, &region.score, nullptr, nullptr, member);
    std::vector<cv::Point> outline;
    cv::approxPolyDP(contours.front(), outline, 1.0, true);
    if (outline.size() >= 3) {
      // Contour points are map pixels, so they scale to points inside the frame.
      for (const cv::Point& point : outline) {
        region.polygon.push_back(
            {point.x * options.width / map_width, point.y * options.height / map_height});
      }
    } else {
      // A polygon needs three points and an area; a dot or a line is sent as its box, kept at
      // least one pixel wide and high and inside the frame.
      const int left = std::min(x0, options.width - 2);
      const int top = std::min(y0, options.height - 2);
      const int right = std::min(std::max(x1, left + 1), options.width - 1);
      const int bottom = std::min(std::max(y1, top + 1), options.height - 1);
      region.polygon = {{left, top}, {left, bottom}, {right, bottom}, {right, top}};
    }
    regions.push_back(std::move(region));
  }
  std::stable_sort(regions.begin(), regions.end(),
                   [](const Region& a, const Region& b) { return a.score > b.score; });
  if (regions.size() > static_cast<std::size_t>(options.max_regions)) {
    regions.resize(static_cast<std::size_t>(options.max_regions));
  }
  return regions;
}

std::string segments_json(const std::vector<Region>& regions) {
  nlohmann::json segments = nlohmann::json::array();
  for (std::size_t index = 0; index < regions.size(); ++index) {
    const Region& region = regions[index];
    nlohmann::json polygon = nlohmann::json::array();
    for (const auto& point : region.polygon) {
      polygon.push_back({point[0], point[1]});
    }
    segments.push_back({
        {"id", "seg_" + std::to_string(index + 1)},
        {"label", kLabel},
        {"confidence", region.score},
        {"bbox", region.bbox},
        {"mask_format", "polygon"},
        {"mask", std::move(polygon)},
    });
  }
  return nlohmann::json{{"segments", std::move(segments)}}.dump();
}

} // namespace sima_examples
