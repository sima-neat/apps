// Copyright 2026 SiMa Technologies, Inc.
// SPDX-License-Identifier: Apache-2.0

// The Insight metadata entries of one frame: objects, segments, or poses in frame pixels.
// Mirrors utils/insight_metadata.py in the Python implementation, so both send the same JSON.

#pragma once

#include "neat.h"

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace insight_metadata {

// Keeps its keys in insertion order, as Python's json does, so both languages send the same bytes.
using Entry = nlohmann::ordered_json;

// MetadataSender rejects a message above 65507 bytes. Entries beyond this budget are left out,
// lowest confidence first, so a crowded frame still sends its best entries.
constexpr std::size_t kMetadataByteBudget = 32768;
// A mask cell belongs to the object when its value is at least half of full scale (255).
constexpr double kMaskThreshold = 0.5;
// The order in which the BoxDecode pose payload emits the 17 COCO keypoints. Insight joins the
// skeleton by these names.
constexpr std::array<const char*, 17> kCocoKeypointNames = {
    "nose",           "left_eye",   "right_eye",   "left_ear",   "right_ear",   "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip",
    "right_hip",      "left_knee",  "right_knee",  "left_ankle", "right_ankle"};

double round_confidence(double value);

// The `objects` entries of `boxes`, in frame pixels.
std::vector<Entry> objects(const std::vector<simaai::neat::Box>& boxes,
                           const std::vector<std::string>& labels, int frame_w, int frame_h);

// The whole pixels the box (x1, y1, x2, y2) covers, at least one, inside the frame.
cv::Rect frame_rect(const float* box, int frame_w, int frame_h);

// The mask cells under the frame rectangle `rect`, through the model's letterbox.
cv::Rect mask_rect(const cv::Rect& rect, int frame_w, int frame_h, int mask_w, int mask_h);

// The outline, in frame pixels, of the largest region of `mask` inside `rect`, or empty when it
// has fewer than three corners.
std::vector<cv::Point> mask_polygon(const cv::Mat& mask, const cv::Rect& rect, int frame_w,
                                    int frame_h);

// The `segments` entries of `boxes` [N, 6] (x1, y1, x2, y2, score, class_id) in frame pixels and
// their `masks` [N, mask_h, mask_w], each with its outline as a polygon.
std::vector<Entry> segments(const std::vector<float>& boxes, const std::vector<std::uint8_t>& masks,
                            int mask_w, int mask_h, const std::vector<std::string>& labels,
                            int frame_w, int frame_h);

// The `poses` entries of `boxes` [N, 6] in frame pixels and their `keypoints` [N, 17, 3]
// (x, y, visibility).
std::vector<Entry> poses(const std::vector<float>& boxes, const std::vector<float>& keypoints);

// The `data` of one message, {key: [...]}, with the entries that fit kMetadataByteBudget,
// highest confidence first, and how many were left out.
std::pair<std::string, int> data_json(const std::string& key, std::vector<Entry> entries);

} // namespace insight_metadata
