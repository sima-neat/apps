// Copied from support/templates/cpp/insight.h;
// edit the template and run scripts/sync_utils.py.
#pragma once

// Detection metadata for Insight, in the format it draws over the video.

#include "detections.h"

#include <neat.h>
#include <nodes/io/MetadataSender.h>

#include <string>
#include <vector>

namespace utils {

struct MetadataBox {
  std::string id;
  std::string label;
  float confidence = 0.0f;
  float x = 0.0f;
  float y = 0.0f;
  float w = 0.0f;
  float h = 0.0f;
};

// Insight objects for parsed boxes: an id, a class label, the score and a clipped x, y, w, h.
std::vector<MetadataBox> build_metadata_boxes(const std::vector<Box>& boxes,
                                              const std::vector<std::string>& labels, int frame_w,
                                              int frame_h);

// {"<array_key>": [{"id", "label", "confidence", "bbox": [x, y, w, h]}, ...]} as JSON.
std::string metadata_boxes_data_json(const std::string& array_key,
                                     const std::vector<MetadataBox>& boxes);

// Sends data_json stamped with the sample's timestamp and frame ID, which Insight matches to the
// RTP timestamp of the video frame it came from. A failed send prints a warning.
void send_metadata(simaai::neat::MetadataSender& sender, const std::string& metadata_type,
                   const std::string& data_json, const simaai::neat::Sample& sample);

} // namespace utils
