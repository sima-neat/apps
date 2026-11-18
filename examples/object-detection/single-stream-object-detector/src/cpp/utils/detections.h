// Copied from support/templates/cpp/detections.h;
// edit the template and run scripts/sync_utils.py.
#pragma once

// Detections from a box-decoding model: the BBOX payload, its boxes, and drawing them.

#include <neat.h>

#include <opencv2/core/mat.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace utils {

struct Box {
  float x1 = 0.0f;
  float y1 = 0.0f;
  float x2 = 0.0f;
  float y2 = 0.0f;
  float score = 0.0f;
  int class_id = -1;
};

// The BBOX payload of a tensor sample, a tensor set, or the first bundle field that has one.
// False with err set when there is none.
bool extract_bbox_payload(const simaai::neat::Sample& sample, std::vector<uint8_t>& payload,
                          std::string& err);

// Boxes from a BBOX payload, clipped to the image. The payload is a little-endian uint32 count
// followed by that many 24-byte records: int32 x, y, w, h, float32 score, int32 class ID. A
// count the payload cannot hold, or one above expected_topk when that is positive, throws.
std::vector<Box> parse_boxes_strict(const std::vector<uint8_t>& bytes, int img_w, int img_h,
                                    int expected_topk);

// Draws each box at or above min_score in green, labelled with its class ID and score.
void draw_boxes(cv::Mat& img, const std::vector<Box>& boxes, float min_score);

} // namespace utils
