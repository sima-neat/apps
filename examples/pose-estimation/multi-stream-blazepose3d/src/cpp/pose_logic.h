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
#include <vector>

namespace blazepose_app {

constexpr std::size_t kBodyLandmarkCount = 33;
constexpr std::size_t kRawLandmarkCount = 39;
constexpr std::size_t kRawLandmarkWidth = 5;

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

struct Pose {
  int roi_index = 0;
  Box box;
  std::array<Keypoint, kBodyLandmarkCount> keypoints{};
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

inline Pose decode_pose(const std::vector<float>& raw_landmarks, const Affine& affine,
                        const Box& box, int roi_index) {
  if (raw_landmarks.size() != kRawLandmarkCount * kRawLandmarkWidth) {
    throw std::runtime_error("BlazePose screen-landmark output must contain 195 floats");
  }

  Pose pose;
  pose.roi_index = roi_index;
  pose.box = box;
  for (std::size_t index = 0; index < kBodyLandmarkCount; ++index) {
    const float* raw = raw_landmarks.data() + index * kRawLandmarkWidth;
    const double source_x = affine.m00 * raw[0] + affine.m01 * raw[1] + affine.m02;
    const double source_y = affine.m10 * raw[0] + affine.m11 * raw[1] + affine.m12;
    pose.keypoints[index] = {static_cast<float>(source_x), static_cast<float>(source_y),
                             std::min(sigmoid(raw[3]), sigmoid(raw[4]))};
  }
  return pose;
}

inline nlohmann::json poses_data_json(std::vector<Pose> poses) {
  std::sort(poses.begin(), poses.end(),
            [](const Pose& left, const Pose& right) { return left.roi_index < right.roi_index; });
  nlohmann::json data;
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

} // namespace blazepose_app
