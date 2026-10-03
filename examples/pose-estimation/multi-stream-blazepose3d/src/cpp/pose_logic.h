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

constexpr std::array<const char*, kBodyLandmarkCount> kLandmarkNames = {
    "nose",        "left_eye_inner",  "left_eye",        "left_eye_outer", "right_eye_inner",
    "right_eye",   "right_eye_outer", "left_ear",        "right_ear",      "mouth_left",
    "mouth_right", "left_shoulder",   "right_shoulder",  "left_elbow",     "right_elbow",
    "left_wrist",  "right_wrist",     "left_pinky",      "right_pinky",    "left_index",
    "right_index", "left_thumb",      "right_thumb",     "left_hip",       "right_hip",
    "left_knee",   "right_knee",      "left_ankle",      "right_ankle",    "left_heel",
    "right_heel",  "left_foot_index", "right_foot_index"};

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

class PoseSmoother {
public:
  std::vector<Pose> filter(std::vector<Pose> poses, int64_t pts_ns) {
    if (pts_ns >= 0 && last_pts_ns_ >= 0 &&
        (pts_ns <= last_pts_ns_ || pts_ns - last_pts_ns_ > kResetAfterNs)) {
      previous_.clear();
      missing_frames_ = 0;
    }
    if (poses.empty()) {
      if (!previous_.empty() && ++missing_frames_ <= kMaxCoastFrames) {
        auto coasted = previous_;
        const float decay = std::pow(kCoastDecay, missing_frames_);
        for (Pose& pose : coasted) {
          pose.presence *= decay;
          pose.box.score *= decay;
          for (std::size_t index = 0; index < pose.keypoints.size(); ++index) {
            pose.keypoints[index].confidence *= decay;
            pose.world_keypoints[index].confidence = pose.keypoints[index].confidence;
          }
        }
        return coasted;
      }
      return poses;
    }
    missing_frames_ = 0;
    std::vector<bool> used(previous_.size(), false);
    for (Pose& current : poses) {
      float best_iou = kMinimumMatchIou;
      int match = -1;
      for (std::size_t previous_index = 0; previous_index < previous_.size(); ++previous_index) {
        if (used[previous_index]) {
          continue;
        }
        const float overlap = box_iou(current.box, previous_[previous_index].box);
        if (overlap >= best_iou) {
          best_iou = overlap;
          match = static_cast<int>(previous_index);
        }
      }
      if (match < 0) {
        continue;
      }
      used[static_cast<std::size_t>(match)] = true;
      const Pose& previous = previous_[static_cast<std::size_t>(match)];
      const float width = std::max(0.0F, current.box.x2 - current.box.x1);
      const float height = std::max(0.0F, current.box.y2 - current.box.y1);
      const float scale = std::max(1.0F, std::hypot(width, height));
      const float center_motion =
          std::hypot((current.box.x1 + current.box.x2 - previous.box.x1 - previous.box.x2) * 0.5F,
                     (current.box.y1 + current.box.y2 - previous.box.y1 - previous.box.y2) * 0.5F) /
          scale;
      const float box_alpha = motion_alpha(center_motion);
      current.box.x1 = blend(previous.box.x1, current.box.x1, box_alpha);
      current.box.y1 = blend(previous.box.y1, current.box.y1, box_alpha);
      current.box.x2 = blend(previous.box.x2, current.box.x2, box_alpha);
      current.box.y2 = blend(previous.box.y2, current.box.y2, box_alpha);
      current.presence = blend(previous.presence, current.presence, kConfidenceAlpha);
      current.box.score = blend(previous.box.score, current.box.score, kConfidenceAlpha);

      for (std::size_t landmark = 0; landmark < current.keypoints.size(); ++landmark) {
        Keypoint& point = current.keypoints[landmark];
        const Keypoint& previous_point = previous.keypoints[landmark];
        const float motion =
            std::hypot(point.x - previous_point.x, point.y - previous_point.y) / scale;
        const float alpha = motion_alpha(motion);
        point.x = blend(previous_point.x, point.x, alpha);
        point.y = blend(previous_point.y, point.y, alpha);
        point.confidence = blend(previous_point.confidence, point.confidence, kConfidenceAlpha);

        WorldKeypoint& world = current.world_keypoints[landmark];
        const WorldKeypoint& previous_world = previous.world_keypoints[landmark];
        world.x = blend(previous_world.x, world.x, alpha);
        world.y = blend(previous_world.y, world.y, alpha);
        world.z = blend(previous_world.z, world.z, alpha);
        world.confidence = point.confidence;
      }
    }

    previous_ = poses;
    if (pts_ns >= 0) {
      last_pts_ns_ = pts_ns;
    }
    return poses;
  }

private:
  static float motion_alpha(float motion) {
    return motion >= kFastMotionThreshold ? kFastMotionAlpha : kPositionAlpha;
  }

  static constexpr float kPositionAlpha = 0.45F;
  static constexpr float kConfidenceAlpha = 0.20F;
  static constexpr float kFastMotionAlpha = 0.90F;
  static constexpr float kFastMotionThreshold = 0.08F;
  static constexpr float kMinimumMatchIou = 0.15F;
  static constexpr int64_t kResetAfterNs = 250'000'000;
  static constexpr int kMaxCoastFrames = 2;
  static constexpr float kCoastDecay = 0.85F;
  std::vector<Pose> previous_;
  int64_t last_pts_ns_ = -1;
  int missing_frames_ = 0;
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
                        const Box& box, float presence) {
  if (raw_landmarks.size() != kRawLandmarkCount * kRawLandmarkWidth) {
    throw std::runtime_error("BlazePose screen-landmark output must contain 195 floats");
  }
  if (raw_world_landmarks.size() != kRawLandmarkCount * kRawWorldLandmarkWidth) {
    throw std::runtime_error("BlazePose world-landmark output must contain 117 floats");
  }

  Pose pose;
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

inline nlohmann::json poses_data_json(std::vector<Pose> poses, const std::string& stream_id) {
  nlohmann::json data;
  data["stream_id"] = stream_id;
  data["poses"] = nlohmann::json::array();
  for (std::size_t pose_index = 0; pose_index < poses.size(); ++pose_index) {
    const Pose& pose = poses[pose_index];
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
        {{"id", "pose_" + std::to_string(pose_index + 1)},
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

inline nlohmann::json world_pose_auxiliary_data_json(nlohmann::json overlay) {
  auto world_poses = nlohmann::json::array();
  for (auto& pose : overlay["poses"]) {
    world_poses.push_back({{"id", std::move(pose["id"])},
                           {"presence", std::move(pose["presence"])},
                           {"keypoints", std::move(pose["world_keypoints"])}});
  }
  return {{"schema_version", 1},
          {"id", "world-pose"},
          {"renderer", "blazepose-3d"},
          {"title", "3D Pose"},
          {"stream_id", std::move(overlay["stream_id"])},
          {"payload", {{"poses", std::move(world_poses)}}}};
}

} // namespace blazepose_app
