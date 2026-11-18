// Copied from support/templates/cpp/insight.cpp;
// edit the template and run scripts/sync_utils.py.
#include "insight.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <iostream>

namespace utils {

std::vector<MetadataBox> build_metadata_boxes(const std::vector<Box>& boxes,
                                              const std::vector<std::string>& labels, int frame_w,
                                              int frame_h) {
  std::vector<MetadataBox> metadata_boxes;
  metadata_boxes.reserve(boxes.size());
  int object_index = 1;
  for (const auto& box : boxes) {
    int x1 = std::max(0, static_cast<int>(box.x1));
    int y1 = std::max(0, static_cast<int>(box.y1));
    int w = std::max(0, static_cast<int>(box.x2 - box.x1));
    int h = std::max(0, static_cast<int>(box.y2 - box.y1));
    if (x1 + w > frame_w)
      w = frame_w - x1;
    if (y1 + h > frame_h)
      h = frame_h - y1;

    MetadataBox obj;
    obj.id = "obj_" + std::to_string(object_index++);
    obj.label = (box.class_id >= 0 && box.class_id < static_cast<int>(labels.size()))
                    ? labels[box.class_id]
                    : "unknown";
    obj.confidence = box.score;
    obj.x = static_cast<float>(x1);
    obj.y = static_cast<float>(y1);
    obj.w = static_cast<float>(std::max(0, w));
    obj.h = static_cast<float>(std::max(0, h));
    metadata_boxes.push_back(obj);
  }
  return metadata_boxes;
}

std::string metadata_boxes_data_json(const std::string& array_key,
                                     const std::vector<MetadataBox>& boxes) {
  nlohmann::json data;
  data[array_key] = nlohmann::json::array();
  for (const auto& box : boxes) {
    data[array_key].push_back({
        {"id", box.id},
        {"label", box.label},
        {"confidence", box.confidence},
        {"bbox", {box.x, box.y, box.w, box.h}},
    });
  }
  return data.dump();
}

void send_metadata(simaai::neat::MetadataSender& sender, const std::string& metadata_type,
                   const std::string& data_json, const simaai::neat::Sample& sample) {
  const int64_t ts_ms = sample.pts_ns >= 0 ? sample.pts_ns / 1'000'000 : -1;
  const std::string frame_id = sample.frame_id >= 0 ? std::to_string(sample.frame_id) : "";
  std::string err;
  if (!sender.send_metadata(metadata_type, data_json, ts_ms, frame_id, &err)) {
    std::cerr << "[warn] insight metadata send failed: " << err << "\n";
  }
}

} // namespace utils
