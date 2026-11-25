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

// The object-detection and tracking metadata messages sent to Insight. Mirrors
// sima_examples.detection_egress in support/python.

#pragma once

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
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
  // The RTP timestamp an application read from the source packets, sent when set. Insight
  // pairs metadata with video by `timestamp`, the frame's presentation time in milliseconds.
  std::optional<std::uint32_t> rtp_timestamp;
};

inline const std::string kUnknownLabel = "unknown";

// The identity of a pulled sample, such as a simaai::neat::Sample, on stream `stream_index`.
// Mirrors detection_egress.frame_metadata in support/python. The result views the sample's
// stream_id, so the sample must outlive it.
template <typename SampleT> FrameMetadata frame_metadata(const SampleT& sample, int stream_index) {
  FrameMetadata metadata;
  metadata.stream_index = stream_index;
  metadata.stream_id = sample.stream_id;
  metadata.frame_id = sample.frame_id;
  metadata.pts_ns = sample.pts_ns;
  metadata.dts_ns = sample.dts_ns;
  metadata.duration_ns = sample.duration_ns;
  metadata.input_seq = sample.input_seq;
  metadata.orig_input_seq = sample.orig_input_seq;
  return metadata;
}

namespace detail {

inline double clip(double value, int limit) {
  return std::min(std::max(value, 0.0), static_cast<double>(limit));
}

// One box as Insight reads it: [x, y, width, height] in frame pixels. Both corners are clipped to
// the frame before the size is measured, so a box past an edge gets its visible width, and a box
// outside the frame gets zero size.
template <typename BoxT>
nlohmann::json entry(std::string entry_id, const BoxT& box, const std::vector<std::string>& labels,
                     int frame_w, int frame_h) {
  const int x1 = static_cast<int>(clip(box.x1, frame_w));
  const int y1 = static_cast<int>(clip(box.y1, frame_h));
  const int width = std::max(0, static_cast<int>(clip(box.x2, frame_w)) - x1);
  const int height = std::max(0, static_cast<int>(clip(box.y2, frame_h)) - y1);
  const std::string& label = box.class_id >= 0 && box.class_id < static_cast<int>(labels.size())
                                 ? labels[static_cast<std::size_t>(box.class_id)]
                                 : kUnknownLabel;
  return {
      {"id", std::move(entry_id)},
      {"label", label},
      {"confidence", box.score},
      {"bbox",
       {static_cast<float>(x1), static_cast<float>(y1), static_cast<float>(width),
        static_cast<float>(height)}},
  };
}

inline std::string message(const char* metadata_type, nlohmann::json data,
                           const FrameMetadata& frame) {
  nlohmann::json payload;
  payload["type"] = metadata_type;
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
  payload["data"] = std::move(data);
  return payload.dump();
}

} // namespace detail

// The object-detection message: each box with its label, numbered obj_1, obj_2, and so on.
// BoxRange is structural, so host applications can use it without linking the Neat runtime.
template <typename BoxRange>
std::string serialize(const BoxRange& boxes, const std::vector<std::string>& labels, int frame_w,
                      int frame_h, const FrameMetadata& frame) {
  nlohmann::json objects = nlohmann::json::array();
  int object_index = 1;
  for (const auto& box : boxes) {
    objects.push_back(
        detail::entry("obj_" + std::to_string(object_index++), box, labels, frame_w, frame_h));
  }
  return detail::message("object-detection", {{"objects", std::move(objects)}}, frame);
}

// The `data.tracks` entries of a tracking message, for an application that builds its own
// message: each box with its label and its track ID (`track_id`) as `id`.
template <typename TrackRange>
nlohmann::json track_entries(const TrackRange& tracks, const std::vector<std::string>& labels,
                             int frame_w, int frame_h) {
  nlohmann::json entries = nlohmann::json::array();
  for (const auto& track : tracks) {
    entries.push_back(
        detail::entry(std::to_string(track.track_id), track, labels, frame_w, frame_h));
  }
  return entries;
}

// The tracking message: each box with its label and its track ID (`track_id`), which Insight
// colors and draws with a trail.
template <typename TrackRange>
std::string serialize_tracks(const TrackRange& tracks, const std::vector<std::string>& labels,
                             int frame_w, int frame_h, const FrameMetadata& frame) {
  return detail::message("tracking", {{"tracks", track_entries(tracks, labels, frame_w, frame_h)}},
                         frame);
}

} // namespace sima_examples::detection_egress
