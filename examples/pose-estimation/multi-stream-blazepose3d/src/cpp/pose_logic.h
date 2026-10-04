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
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace blazepose_app {

constexpr std::size_t kBodyLandmarkCount = 33;
constexpr std::size_t kRawLandmarkCount = 39;
constexpr std::size_t kRawLandmarkWidth = 5;
constexpr std::size_t kRawWorldLandmarkWidth = 3;

// Stores `value` as the newest work for one stream. Returns true when it replaced
// older work, which is dropped so a slow stage never accumulates stale frames.
template <typename T> bool keep_latest(std::optional<T>& slot, T value) {
  const bool replaced = slot.has_value();
  slot = std::move(value);
  return replaced;
}

// Claims `frame_id` for publication when it is newer than every frame already
// claimed on the stream. Completed older inference must not be sent after a
// newer empty frame has cleared the viewer and its temporal-filter state.
inline bool claim_newer_frame(std::uint64_t frame_id, std::uint64_t& last_claimed_frame_id) {
  if (frame_id <= last_claimed_frame_id) {
    return false;
  }
  last_claimed_frame_id = frame_id;
  return true;
}

// Accepted model input with no output for this long means the shared Run is stuck.
constexpr std::chrono::seconds kInferenceStallTimeout{5};

// Times how long a model has had accepted input but no output; true past the timeout.
inline bool inference_stalled(std::optional<std::chrono::steady_clock::time_point>& waiting_since,
                              bool input_pending, std::chrono::steady_clock::time_point now) {
  if (!input_pending) {
    waiting_since.reset();
    return false;
  }
  if (!waiting_since) {
    waiting_since = now;
  }
  return now - *waiting_since > kInferenceStallTimeout;
}

constexpr std::array<const char*, kBodyLandmarkCount> kLandmarkNames = {
    "nose",        "left_eye_inner",  "left_eye",        "left_eye_outer", "right_eye_inner",
    "right_eye",   "right_eye_outer", "left_ear",        "right_ear",      "mouth_left",
    "mouth_right", "left_shoulder",   "right_shoulder",  "left_elbow",     "right_elbow",
    "left_wrist",  "right_wrist",     "left_pinky",      "right_pinky",    "left_index",
    "right_index", "left_thumb",      "right_thumb",     "left_hip",       "right_hip",
    "left_knee",   "right_knee",      "left_ankle",      "right_ankle",    "left_heel",
    "right_heel",  "left_foot_index", "right_foot_index"};

// Sends one frame's 2D/3D metadata pair through `send(type)`. Both sends are
// always attempted; the pair counts as published only when both succeed.
template <typename Send> bool send_metadata_pair(Send&& send) {
  const bool overlay_sent = send("pose-estimation");
  const bool world_sent = send("auxiliary-visualization");
  return overlay_sent && world_sent;
}

struct Box {
  float x1 = 0.0F;
  float y1 = 0.0F;
  float x2 = 0.0F;
  float y2 = 0.0F;
  float score = 0.0F;
  int class_id = -1;
};

struct Roi {
  int x = 0;
  int y = 0;
  int width = 0;
  int height = 0;
};

struct Affine {
  double m00 = 1.0;
  double m01 = 0.0;
  double m02 = 0.0;
  double m10 = 0.0;
  double m11 = 1.0;
  double m12 = 0.0;
};

struct Keypoint {
  float x = 0.0F;
  float y = 0.0F;
  float confidence = 0.0F;
};

struct WorldKeypoint {
  float x = 0.0F;
  float y = 0.0F;
  float z = 0.0F;
  float confidence = 0.0F;
};

struct Pose {
  int roi_index = 0;
  float presence = 0.0F;
  Box box;
  std::array<Keypoint, kBodyLandmarkCount> keypoints{};
  std::array<WorldKeypoint, kBodyLandmarkCount> world_keypoints{};
};

inline float box_iou(const Box& left, const Box& right) {
  const float intersection_width =
      std::max(0.0F, std::min(left.x2, right.x2) - std::max(left.x1, right.x1));
  const float intersection_height =
      std::max(0.0F, std::min(left.y2, right.y2) - std::max(left.y1, right.y1));
  const float intersection = intersection_width * intersection_height;
  const float left_area = std::max(0.0F, left.x2 - left.x1) * std::max(0.0F, left.y2 - left.y1);
  const float right_area =
      std::max(0.0F, right.x2 - right.x1) * std::max(0.0F, right.y2 - right.y1);
  const float union_area = left_area + right_area - intersection;
  return union_area > 0.0F ? intersection / union_area : 0.0F;
}

inline float blend(float previous, float current, float alpha) {
  return previous + alpha * (current - previous);
}

// Per-stream exponential smoothing of 2D and world landmarks. Each pose is
// matched to the previous frame's pose with the highest box IoU; matched
// landmarks move kAlpha of the way toward the new estimate, unmatched poses
// pass through unchanged.
class PoseSmoother {
public:
  static constexpr float kAlpha = 0.5F;
  static constexpr float kMinimumMatchIou = 0.15F;

  std::vector<Pose> filter(std::vector<Pose> poses) {
    std::vector<bool> used(previous_.size(), false);
    for (Pose& current : poses) {
      int match = -1;
      float best_iou = kMinimumMatchIou;
      for (std::size_t index = 0; index < previous_.size(); ++index) {
        const float overlap = used[index] ? 0.0F : box_iou(current.box, previous_[index].box);
        if (overlap >= best_iou) {
          best_iou = overlap;
          match = static_cast<int>(index);
        }
      }
      if (match < 0) {
        continue;
      }
      used[static_cast<std::size_t>(match)] = true;
      const Pose& previous = previous_[static_cast<std::size_t>(match)];
      for (std::size_t landmark = 0; landmark < kBodyLandmarkCount; ++landmark) {
        Keypoint& point = current.keypoints[landmark];
        point.x = blend(previous.keypoints[landmark].x, point.x, kAlpha);
        point.y = blend(previous.keypoints[landmark].y, point.y, kAlpha);
        WorldKeypoint& world = current.world_keypoints[landmark];
        world.x = blend(previous.world_keypoints[landmark].x, world.x, kAlpha);
        world.y = blend(previous.world_keypoints[landmark].y, world.y, kAlpha);
        world.z = blend(previous.world_keypoints[landmark].z, world.z, kAlpha);
      }
    }
    previous_ = poses;
    return poses;
  }

private:
  std::vector<Pose> previous_;
};

inline int round_half_away_from_zero(double value) {
  return value >= 0.0 ? static_cast<int>(std::floor(value + 0.5))
                      : static_cast<int>(std::ceil(value - 0.5));
}

inline Roi square_roi(const Box& box, double scale) {
  const double width = std::max(0.0, static_cast<double>(box.x2 - box.x1));
  const double height = std::max(0.0, static_cast<double>(box.y2 - box.y1));
  const int side = std::max(1, round_half_away_from_zero(std::max(width, height) * scale));
  const double center_x = (static_cast<double>(box.x1) + box.x2) * 0.5;
  const double center_y = (static_cast<double>(box.y1) + box.y2) * 0.5;
  return {round_half_away_from_zero(center_x - static_cast<double>(side) * 0.5),
          round_half_away_from_zero(center_y - static_cast<double>(side) * 0.5), side, side};
}

inline float sigmoid(float value) {
  if (value >= 0.0F) {
    const float z = std::exp(-value);
    return 1.0F / (1.0F + z);
  }
  const float z = std::exp(value);
  return z / (1.0F + z);
}

inline Pose decode_pose(const std::vector<float>& raw_landmarks,
                        const std::vector<float>& raw_world_landmarks, const Affine& affine,
                        const Box& box, float presence, int roi_index) {
  if (raw_landmarks.size() != kRawLandmarkCount * kRawLandmarkWidth) {
    throw std::runtime_error("BlazePose screen-landmark output must contain 195 floats");
  }
  if (raw_world_landmarks.size() != kRawLandmarkCount * kRawWorldLandmarkWidth) {
    throw std::runtime_error("BlazePose world-landmark output must contain 117 floats");
  }

  Pose pose;
  pose.roi_index = roi_index;
  pose.presence = presence;
  pose.box = box;
  for (std::size_t index = 0; index < kBodyLandmarkCount; ++index) {
    const float* raw = raw_landmarks.data() + index * kRawLandmarkWidth;
    const double source_x = affine.m00 * raw[0] + affine.m01 * raw[1] + affine.m02;
    const double source_y = affine.m10 * raw[0] + affine.m11 * raw[1] + affine.m12;
    const float confidence = std::min(sigmoid(raw[3]), sigmoid(raw[4]));
    pose.keypoints[index] = {static_cast<float>(source_x), static_cast<float>(source_y),
                             confidence};
    const float* world = raw_world_landmarks.data() + index * kRawWorldLandmarkWidth;
    pose.world_keypoints[index] = {world[0], world[1], world[2], confidence};
  }
  return pose;
}

// Decodes a pose only from finite model outputs. A NaN or infinite landmark would
// reach integer rounding and JSON serialization, so that ROI's pose is discarded
// and the frame publishes without it. One pass over the floats, no allocation.
inline std::optional<Pose> decode_finite_pose(const std::vector<float>& raw_landmarks,
                                              const std::vector<float>& raw_world_landmarks,
                                              const Affine& affine, const Box& box, float presence,
                                              int roi_index) {
  const auto finite = [](float value) { return std::isfinite(value); };
  if (!std::isfinite(presence) ||
      !std::all_of(raw_landmarks.begin(), raw_landmarks.end(), finite) ||
      !std::all_of(raw_world_landmarks.begin(), raw_world_landmarks.end(), finite)) {
    return std::nullopt;
  }
  return decode_pose(raw_landmarks, raw_world_landmarks, affine, box, presence, roi_index);
}

inline nlohmann::json poses_data_json(std::vector<Pose> poses, const std::string& stream_id) {
  std::sort(poses.begin(), poses.end(),
            [](const Pose& left, const Pose& right) { return left.roi_index < right.roi_index; });
  nlohmann::json data;
  data["stream_id"] = stream_id;
  data["poses"] = nlohmann::json::array();
  for (const Pose& pose : poses) {
    nlohmann::json keypoints = nlohmann::json::array();
    nlohmann::json world_keypoints = nlohmann::json::array();
    for (std::size_t index = 0; index < pose.keypoints.size(); ++index) {
      const Keypoint& point = pose.keypoints[index];
      keypoints.push_back({{"name", kLandmarkNames[index]},
                           {"x", round_half_away_from_zero(point.x)},
                           {"y", round_half_away_from_zero(point.y)},
                           {"confidence", std::round(point.confidence * 1000.0F) / 1000.0F}});
      const WorldKeypoint& world = pose.world_keypoints[index];
      world_keypoints.push_back({{"name", kLandmarkNames[index]},
                                 {"x", std::round(world.x * 1'000'000.0F) / 1'000'000.0F},
                                 {"y", std::round(world.y * 1'000'000.0F) / 1'000'000.0F},
                                 {"z", std::round(world.z * 1'000'000.0F) / 1'000'000.0F},
                                 {"confidence", std::round(world.confidence * 1000.0F) / 1000.0F}});
    }
    data["poses"].push_back(
        {{"id", "pose_" + std::to_string(pose.roi_index + 1)},
         {"label", "person"},
         {"presence", std::round(pose.presence * 1000.0F) / 1000.0F},
         {"confidence", std::round(pose.box.score * 1000.0F) / 1000.0F},
         {"bbox",
          {round_half_away_from_zero(pose.box.x1), round_half_away_from_zero(pose.box.y1),
           round_half_away_from_zero(std::max(0.0F, pose.box.x2 - pose.box.x1)),
           round_half_away_from_zero(std::max(0.0F, pose.box.y2 - pose.box.y1))}},
         {"keypoints", std::move(keypoints)},
         {"world_keypoints", std::move(world_keypoints)}});
  }
  return data;
}

inline nlohmann::json
auxiliary_visualization_data_json(std::string id, std::string renderer, nlohmann::json payload,
                                  std::optional<std::string> title = std::nullopt) {
  nlohmann::json data = {{"schema_version", 1},
                         {"id", std::move(id)},
                         {"renderer", std::move(renderer)},
                         {"payload", std::move(payload)}};
  if (title.has_value()) {
    data["title"] = std::move(*title);
  }
  return data;
}

// Builds the 3D view from the 2D message's world keypoints, so both carry the
// same rounded values without decoding them twice. Serialize `overlay` first:
// its id, presence and world keypoints are moved into the result.
inline nlohmann::json world_pose_auxiliary_from_overlay(nlohmann::json overlay) {
  nlohmann::json world_poses = nlohmann::json::array();
  for (nlohmann::json& pose : overlay["poses"]) {
    world_poses.push_back({{"id", std::move(pose["id"])},
                           {"presence", std::move(pose["presence"])},
                           {"keypoints", std::move(pose["world_keypoints"])}});
  }
  nlohmann::json data = auxiliary_visualization_data_json(
      "world-pose", "blazepose-3d", {{"poses", std::move(world_poses)}}, "3D Pose");
  data["stream_id"] = std::move(overlay["stream_id"]);
  return data;
}

} // namespace blazepose_app
