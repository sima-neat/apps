// Copyright 2026 SiMa Technologies, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace sima_examples::detection_egress {

struct FrameMetadata {
  int stream_index = 0;
  std::string_view stream_id;
  std::int64_t frame_id = -1;
  std::int64_t pts_ns = -1;
  std::int64_t dts_ns = -1;
  std::int64_t duration_ns = -1;
  std::int64_t input_seq = -1;
  std::int64_t orig_input_seq = -1;
  std::optional<std::uint32_t> rtp_timestamp;
};

inline const std::string kUnknownLabel = "unknown";

// Serialize the Insight object-detection envelope in one JSON DOM pass. BoxRange is intentionally
// structural so host applications can use it without linking the NEAT target runtime.
template <typename BoxRange>
std::string serialize(const BoxRange& boxes, const std::vector<std::string>& labels, int frame_w,
                      int frame_h, const FrameMetadata& frame) {
  nlohmann::json payload;
  payload["type"] = "object-detection";
  payload["timestamp"] = frame.pts_ns >= 0 ? frame.pts_ns / 1'000'000 : -1;
  payload["frame_id"] = frame.frame_id >= 0 ? std::to_string(frame.frame_id) : "";
  payload["stream_id"] = std::string(frame.stream_id);
  payload["stream_index"] = frame.stream_index;
  payload["pts_ns"] = frame.pts_ns;
  payload["dts_ns"] = frame.dts_ns;
  payload["duration_ns"] = frame.duration_ns;
  payload["input_seq"] = frame.input_seq;
  payload["orig_input_seq"] = frame.orig_input_seq;
  if (frame.rtp_timestamp.has_value()) {
    payload["rtp_timestamp"] = *frame.rtp_timestamp;
  }

  auto& objects = payload["data"]["objects"];
  objects = nlohmann::json::array();
  int object_index = 1;
  for (const auto& box : boxes) {
    const int x1 = std::max(0, static_cast<int>(box.x1));
    const int y1 = std::max(0, static_cast<int>(box.y1));
    int width = std::max(0, static_cast<int>(box.x2 - box.x1));
    int height = std::max(0, static_cast<int>(box.y2 - box.y1));
    if (x1 + width > frame_w) {
      width = frame_w - x1;
    }
    if (y1 + height > frame_h) {
      height = frame_h - y1;
    }

    const std::string& label = box.class_id >= 0 && box.class_id < static_cast<int>(labels.size())
                                   ? labels[static_cast<std::size_t>(box.class_id)]
                                   : kUnknownLabel;
    objects.push_back({
        {"id", "obj_" + std::to_string(object_index++)},
        {"label", label},
        {"confidence", box.score},
        {"bbox",
         {static_cast<float>(x1), static_cast<float>(y1), static_cast<float>(std::max(0, width)),
          static_cast<float>(std::max(0, height))}},
    });
  }

  return payload.dump();
}

} // namespace sima_examples::detection_egress
