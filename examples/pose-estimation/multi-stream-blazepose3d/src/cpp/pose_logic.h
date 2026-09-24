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

struct CropPlan {
  Roi image;
  Roi roi;
};

struct BatchCropPlan {
  Roi image;
  std::vector<std::size_t> indices;
  std::vector<Roi> rois;
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
  Box box;
  std::array<Keypoint, kBodyLandmarkCount> keypoints{};
  std::array<WorldKeypoint, kBodyLandmarkCount> world_keypoints{};
};

struct PoseSmoothingOptions {
  float position_alpha = 0.45F;
  float confidence_alpha = 0.20F;
  float fast_motion_alpha = 0.90F;
  float fast_motion_threshold = 0.08F;
  float minimum_match_iou = 0.15F;
  int reset_after_ms = 250;
  int max_coast_frames = 2;
  float coast_confidence_decay = 0.85F;
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
  explicit PoseSmoother(PoseSmoothingOptions options = {}) : options_(options) {}

  std::vector<Pose> filter(std::vector<Pose> poses, int64_t pts_ns) {
    if (poses.empty()) {
      if (is_reset_gap(pts_ns)) {
        reset();
        return poses;
      }
      if (!previous_.empty() && ++missing_frames_ <= options_.max_coast_frames) {
        auto coasted = previous_;
        const float decay = std::pow(options_.coast_confidence_decay, missing_frames_);
        for (Pose& pose : coasted) {
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
    if (is_reset_gap(pts_ns) || (pts_ns >= 0 && last_pts_ns_ >= 0 && pts_ns <= last_pts_ns_)) {
      reset();
    }
    missing_frames_ = 0;

    std::vector<int> matches(poses.size(), -1);
    std::vector<bool> used(previous_.size(), false);
    for (std::size_t current_index = 0; current_index < poses.size(); ++current_index) {
      float best_iou = options_.minimum_match_iou;
      for (std::size_t previous_index = 0; previous_index < previous_.size(); ++previous_index) {
        if (used[previous_index]) {
          continue;
        }
        const float overlap = box_iou(poses[current_index].box, previous_[previous_index].box);
        if (overlap >= best_iou) {
          best_iou = overlap;
          matches[current_index] = static_cast<int>(previous_index);
        }
      }
      if (matches[current_index] >= 0) {
        used[static_cast<std::size_t>(matches[current_index])] = true;
      }
    }

    const float elapsed_frames = elapsed_frame_count(pts_ns);
    for (std::size_t current_index = 0; current_index < poses.size(); ++current_index) {
      const int previous_index = matches[current_index];
      if (previous_index < 0) {
        continue;
      }
      Pose& current = poses[current_index];
      const Pose& previous = previous_[static_cast<std::size_t>(previous_index)];
      const float width = std::max(0.0F, current.box.x2 - current.box.x1);
      const float height = std::max(0.0F, current.box.y2 - current.box.y1);
      const float scale = std::max(1.0F, std::hypot(width, height));
      const float center_motion =
          std::hypot((current.box.x1 + current.box.x2 - previous.box.x1 - previous.box.x2) * 0.5F,
                     (current.box.y1 + current.box.y2 - previous.box.y1 - previous.box.y2) * 0.5F) /
          scale;
      const float box_alpha = motion_alpha(center_motion, elapsed_frames);
      current.box.x1 = blend(previous.box.x1, current.box.x1, box_alpha);
      current.box.y1 = blend(previous.box.y1, current.box.y1, box_alpha);
      current.box.x2 = blend(previous.box.x2, current.box.x2, box_alpha);
      current.box.y2 = blend(previous.box.y2, current.box.y2, box_alpha);
      current.box.score = blend(previous.box.score, current.box.score,
                                adjusted_alpha(options_.confidence_alpha, elapsed_frames));

      for (std::size_t landmark = 0; landmark < current.keypoints.size(); ++landmark) {
        Keypoint& point = current.keypoints[landmark];
        const Keypoint& previous_point = previous.keypoints[landmark];
        const float motion =
            std::hypot(point.x - previous_point.x, point.y - previous_point.y) / scale;
        const float alpha = motion_alpha(motion, elapsed_frames);
        point.x = blend(previous_point.x, point.x, alpha);
        point.y = blend(previous_point.y, point.y, alpha);
        point.confidence = blend(previous_point.confidence, point.confidence,
                                 adjusted_alpha(options_.confidence_alpha, elapsed_frames));

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

  void reset() {
    previous_.clear();
    last_pts_ns_ = -1;
    missing_frames_ = 0;
  }

private:
  bool is_reset_gap(int64_t pts_ns) const {
    return pts_ns >= 0 && last_pts_ns_ >= 0 &&
           pts_ns - last_pts_ns_ > static_cast<int64_t>(options_.reset_after_ms) * 1'000'000;
  }

  float elapsed_frame_count(int64_t pts_ns) const {
    if (pts_ns < 0 || last_pts_ns_ < 0 || pts_ns <= last_pts_ns_) {
      return 1.0F;
    }
    constexpr double kNominalFrameNs = 40'000'000.0;
    return static_cast<float>(std::clamp((pts_ns - last_pts_ns_) / kNominalFrameNs, 1.0, 6.0));
  }

  static float adjusted_alpha(float alpha, float elapsed_frames) {
    return 1.0F - std::pow(1.0F - alpha, elapsed_frames);
  }

  float motion_alpha(float normalized_motion, float elapsed_frames) const {
    const float amount = std::clamp(normalized_motion / options_.fast_motion_threshold, 0.0F, 1.0F);
    const float alpha = blend(options_.position_alpha, options_.fast_motion_alpha, amount);
    return adjusted_alpha(alpha, elapsed_frames);
  }

  PoseSmoothingOptions options_;
  std::vector<Pose> previous_;
  int64_t last_pts_ns_ = -1;
  int missing_frames_ = 0;
};

inline int64_t select_frame_id(int64_t frame_id, int64_t orig_input_seq, int64_t input_seq,
                               std::uint64_t pull_sequence) {
  if (frame_id >= 0) {
    return frame_id;
  }
  if (orig_input_seq >= 0) {
    return orig_input_seq;
  }
  if (input_seq >= 0) {
    return input_seq;
  }
  return static_cast<int64_t>(pull_sequence);
}

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

inline std::optional<CropPlan> crop_plan(const Roi& roi, int frame_width, int frame_height) {
  if (frame_width <= 0 || frame_height <= 0 || roi.width <= 0 || roi.height <= 0) {
    return std::nullopt;
  }
  const int64_t left = std::max<int64_t>(0, roi.x);
  const int64_t top = std::max<int64_t>(0, roi.y);
  const int64_t right = std::min<int64_t>(frame_width, static_cast<int64_t>(roi.x) + roi.width);
  const int64_t bottom = std::min<int64_t>(frame_height, static_cast<int64_t>(roi.y) + roi.height);
  if (right <= left || bottom <= top) {
    return std::nullopt;
  }

  const Roi image{static_cast<int>(left), static_cast<int>(top), static_cast<int>(right - left),
                  static_cast<int>(bottom - top)};
  return CropPlan{image, {roi.x - image.x, roi.y - image.y, roi.width, roi.height}};
}

inline std::optional<BatchCropPlan> batch_crop_plan(const std::vector<Roi>& rois, int frame_width,
                                                    int frame_height) {
  BatchCropPlan result;
  bool has_image = false;
  for (std::size_t index = 0; index < rois.size(); ++index) {
    const auto crop = crop_plan(rois[index], frame_width, frame_height);
    if (!crop.has_value()) {
      continue;
    }
    if (!has_image) {
      result.image = crop->image;
      has_image = true;
    } else {
      const int right =
          std::max(result.image.x + result.image.width, crop->image.x + crop->image.width);
      const int bottom =
          std::max(result.image.y + result.image.height, crop->image.y + crop->image.height);
      result.image.x = std::min(result.image.x, crop->image.x);
      result.image.y = std::min(result.image.y, crop->image.y);
      result.image.width = right - result.image.x;
      result.image.height = bottom - result.image.y;
    }
    result.indices.push_back(index);
  }
  if (!has_image) {
    return std::nullopt;
  }
  result.rois.reserve(result.indices.size());
  for (const std::size_t index : result.indices) {
    const Roi& roi = rois[index];
    result.rois.push_back({roi.x - result.image.x, roi.y - result.image.y, roi.width, roi.height});
  }
  return result;
}

inline Affine offset_affine(Affine affine, int x, int y) {
  affine.m02 += x;
  affine.m12 += y;
  return affine;
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
                        const Box& box, int roi_index) {
  if (raw_landmarks.size() != kRawLandmarkCount * kRawLandmarkWidth) {
    throw std::runtime_error("BlazePose screen-landmark output must contain 195 floats");
  }
  if (raw_world_landmarks.size() != kRawLandmarkCount * kRawWorldLandmarkWidth) {
    throw std::runtime_error("BlazePose world-landmark output must contain 117 floats");
  }

  Pose pose;
  pose.roi_index = roi_index;
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
  std::sort(poses.begin(), poses.end(),
            [](const Pose& left, const Pose& right) { return left.roi_index < right.roi_index; });
  nlohmann::json data;
  data["stream_id"] = stream_id;
  data["poses"] = nlohmann::json::array();
  for (const Pose& pose : poses) {
    nlohmann::json keypoints = nlohmann::json::array();
    for (std::size_t index = 0; index < pose.keypoints.size(); ++index) {
      const Keypoint& point = pose.keypoints[index];
      keypoints.push_back({{"name", kLandmarkNames[index]},
                           {"x", round_half_away_from_zero(point.x)},
                           {"y", round_half_away_from_zero(point.y)},
                           {"confidence", std::round(point.confidence * 1000.0F) / 1000.0F}});
    }
    data["poses"].push_back(
        {{"id", "pose_" + std::to_string(pose.roi_index + 1)},
         {"label", "person"},
         {"confidence", std::round(pose.box.score * 1000.0F) / 1000.0F},
         {"bbox",
          {round_half_away_from_zero(pose.box.x1), round_half_away_from_zero(pose.box.y1),
           round_half_away_from_zero(std::max(0.0F, pose.box.x2 - pose.box.x1)),
           round_half_away_from_zero(std::max(0.0F, pose.box.y2 - pose.box.y1))}},
         {"keypoints", std::move(keypoints)}});
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

inline nlohmann::json world_pose_auxiliary_data_json(std::vector<Pose> poses,
                                                     const std::string& stream_id) {
  std::sort(poses.begin(), poses.end(),
            [](const Pose& left, const Pose& right) { return left.roi_index < right.roi_index; });
  nlohmann::json world_poses = nlohmann::json::array();
  for (const Pose& pose : poses) {
    nlohmann::json keypoints = nlohmann::json::array();
    for (std::size_t index = 0; index < pose.world_keypoints.size(); ++index) {
      const WorldKeypoint& point = pose.world_keypoints[index];
      keypoints.push_back({{"name", kLandmarkNames[index]},
                           {"x", point.x},
                           {"y", point.y},
                           {"z", point.z},
                           {"confidence", std::round(point.confidence * 1000.0F) / 1000.0F}});
    }
    world_poses.push_back({{"id", "pose_" + std::to_string(pose.roi_index + 1)},
                           {"keypoints", std::move(keypoints)}});
  }

  nlohmann::json data = auxiliary_visualization_data_json(
      "world-pose", "blazepose-3d", {{"poses", std::move(world_poses)}}, "3D Pose");
  data["stream_id"] = stream_id;
  return data;
}

} // namespace blazepose_app
