#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

namespace ssd_mobilenet {

// Optional application policy for hiding detector outputs that represent a same-class crowd
// region rather than one object. Core SSD decode/NMS deliberately remains model-faithful.
struct AggregateSuppressionOptions {
  bool enabled = true;
  float min_parent_area_fraction = 0.20f;
  float min_child_containment = 0.90f;
  float max_child_area_ratio = 0.25f;
  int min_children = 2;
};

template <typename Box>
std::vector<Box> suppress_aggregate_boxes(const std::vector<Box>& boxes, int image_width,
                                          int image_height,
                                          const AggregateSuppressionOptions& options) {
  if (!options.enabled || boxes.size() < static_cast<std::size_t>(options.min_children + 1) ||
      image_width <= 0 || image_height <= 0) {
    return boxes;
  }

  const float image_area = static_cast<float>(image_width) * static_cast<float>(image_height);
  std::vector<float> areas;
  areas.reserve(boxes.size());
  for (const Box& box : boxes) {
    areas.push_back(std::max(0.0f, box.x2 - box.x1) * std::max(0.0f, box.y2 - box.y1));
  }

  std::vector<bool> suppressed(boxes.size(), false);
  for (std::size_t parent_index = 0; parent_index < boxes.size(); ++parent_index) {
    const Box& parent = boxes[parent_index];
    const float parent_area = areas[parent_index];
    if (parent_area < image_area * options.min_parent_area_fraction) {
      continue;
    }

    int children = 0;
    const float max_child_area = parent_area * options.max_child_area_ratio;
    for (std::size_t child_index = 0; child_index < boxes.size(); ++child_index) {
      if (child_index == parent_index || boxes[child_index].class_id != parent.class_id) {
        continue;
      }
      const float child_area = areas[child_index];
      if (child_area <= 0.0f || child_area > max_child_area) {
        continue;
      }

      const Box& child = boxes[child_index];
      const float intersection_width =
          std::max(0.0f, std::min(parent.x2, child.x2) - std::max(parent.x1, child.x1));
      const float intersection_height =
          std::max(0.0f, std::min(parent.y2, child.y2) - std::max(parent.y1, child.y1));
      if ((intersection_width * intersection_height) / child_area >=
              options.min_child_containment &&
          ++children >= options.min_children) {
        suppressed[parent_index] = true;
        break;
      }
    }
  }

  std::vector<Box> filtered;
  filtered.reserve(boxes.size());
  for (std::size_t index = 0; index < boxes.size(); ++index) {
    if (!suppressed[index]) {
      filtered.push_back(boxes[index]);
    }
  }
  return filtered;
}

} // namespace ssd_mobilenet
