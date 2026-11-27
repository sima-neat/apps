// Copyright 2026 SiMa Technologies, Inc.
// SPDX-License-Identifier: Apache-2.0

// Faces with five landmarks from yolov5s-face's six raw heads, and their Insight pose message.
// Neat's BBOX format has no landmark slots, so the decode runs here. Mirrors utils/yolov5_face.py.

#pragma once

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace yolov5_face {

// The model was compiled for an 800x800 canvas; its levels are 100, 50, and 25 cells wide.
constexpr int kInferSize = 800;
constexpr int kNumAnchors = 3;
constexpr int kNumLandmarks = 5;
constexpr int kBoxChannels = 18;
constexpr int kLandmarkChannels = 30;
// Fixed by the model: the stride and the three anchor sizes of each level, largest grid first.
constexpr std::array<float, 3> kStrides = {8.0f, 16.0f, 32.0f};
constexpr float kAnchors[3][kNumAnchors][2] = {
    {{4.f, 5.f}, {8.f, 10.f}, {13.f, 16.f}},
    {{23.f, 29.f}, {43.f, 55.f}, {73.f, 105.f}},
    {{146.f, 217.f}, {231.f, 300.f}, {335.f, 433.f}},
};
// Landmark names in model output order. Insight joins keypoints whose names are COCO body
// joints, such as nose or left_eye; these names are not, so no lines are drawn across the face.
inline const std::array<std::string, kNumLandmarks> kLandmarkNames = {"eye_l", "eye_r", "nose_tip",
                                                                      "mouth_l", "mouth_r"};

// The scale and padding that fit a frame into the model canvas.
struct Letterbox {
  float scale = 1.0f;
  int pad_left = 0;
  int pad_top = 0;
};

struct Point {
  float x = 0.0f;
  float y = 0.0f;
};

struct Face {
  float x1 = 0.0f;
  float y1 = 0.0f;
  float x2 = 0.0f;
  float y2 = 0.0f;
  float score = 0.0f;
  std::array<Point, kNumLandmarks> landmarks{};
};

// One split head as the model emits it: a [1, ...] shape and its float values in that order.
struct SplitHead {
  std::vector<int64_t> shape;
  std::vector<float> values;
};

// One split head as (H, W, C).
struct ChannelsLast {
  int rows = 0;
  int cols = 0;
  int channels = 0;
  std::vector<float> values;
};

// The centered letterbox the model's preprocessing applies to a frame of this size.
inline Letterbox letterbox(int frame_w, int frame_h) {
  Letterbox box;
  box.scale = std::min(static_cast<float>(kInferSize) / static_cast<float>(frame_w),
                       static_cast<float>(kInferSize) / static_cast<float>(frame_h));
  box.pad_left = (kInferSize - static_cast<int>(std::round(frame_w * box.scale))) / 2;
  box.pad_top = (kInferSize - static_cast<int>(std::round(frame_h * box.scale))) / 2;
  return box;
}

// A split head as (H, W, C). The channel axis is found by its size, 18 or 30, which no grid
// size equals: [1, H, W, C], Neat's raw [1, H, C, W], or ONNX's [1, C, H, W].
inline ChannelsLast channels_last(const SplitHead& head) {
  if (head.shape.size() != 4 || head.shape[0] != 1) {
    throw std::runtime_error("expected a [1, ...] split head");
  }
  const auto is_channels = [](int64_t size) {
    return size == kBoxChannels || size == kLandmarkChannels;
  };
  int channel_axis = 0;
  for (int axis = 3; axis >= 1 && channel_axis == 0; --axis) {
    if (is_channels(head.shape[axis])) {
      channel_axis = axis;
    }
  }
  if (channel_axis == 0) {
    throw std::runtime_error("split head shape [1," + std::to_string(head.shape[1]) + "," +
                             std::to_string(head.shape[2]) + "," + std::to_string(head.shape[3]) +
                             "] has no axis of 18 or 30 channels");
  }
  ChannelsLast out;
  out.channels = static_cast<int>(head.shape[channel_axis]);
  out.rows = static_cast<int>(channel_axis == 1 ? head.shape[2] : head.shape[1]);
  out.cols = static_cast<int>(channel_axis == 3 ? head.shape[2] : head.shape[3]);
  const std::size_t count = static_cast<std::size_t>(out.rows) * out.cols * out.channels;
  if (head.values.size() != count) {
    throw std::runtime_error("split head has " + std::to_string(head.values.size()) +
                             " values, its shape needs " + std::to_string(count));
  }
  out.values.resize(count);
  for (int y = 0; y < out.rows; ++y) {
    for (int x = 0; x < out.cols; ++x) {
      for (int c = 0; c < out.channels; ++c) {
        const std::size_t source =
            channel_axis == 3   ? (static_cast<std::size_t>(y) * out.cols + x) * out.channels + c
            : channel_axis == 2 ? (static_cast<std::size_t>(y) * out.channels + c) * out.cols + x
                                : (static_cast<std::size_t>(c) * out.rows + y) * out.cols + x;
        out.values[(static_cast<std::size_t>(y) * out.cols + x) * out.channels + c] =
            head.values[source];
      }
    }
  }
  return out;
}

inline float sigmoid(float value) {
  return 1.0f / (1.0f + std::exp(-value));
}

// The faces of one pyramid level whose score is above `min_score`, in canvas pixels.
inline void decode_level(const ChannelsLast& box_head, const ChannelsLast& landmark_head, int level,
                         float min_score, std::vector<Face>& faces) {
  const float stride = kStrides[level];
  for (int y = 0; y < box_head.rows; ++y) {
    for (int x = 0; x < box_head.cols; ++x) {
      const std::size_t cell = static_cast<std::size_t>(y) * box_head.cols + x;
      for (int anchor = 0; anchor < kNumAnchors; ++anchor) {
        const float* box = &box_head.values[cell * kBoxChannels + anchor * 6];
        const float* points =
            &landmark_head.values[cell * kLandmarkChannels + anchor * 2 * kNumLandmarks];
        const float score = sigmoid(box[4]) * sigmoid(box[5]);
        if (score <= min_score) {
          continue;
        }
        const float anchor_w = kAnchors[level][anchor][0];
        const float anchor_h = kAnchors[level][anchor][1];
        const float center_x = (sigmoid(box[0]) * 2.0f - 0.5f + static_cast<float>(x)) * stride;
        const float center_y = (sigmoid(box[1]) * 2.0f - 0.5f + static_cast<float>(y)) * stride;
        const float half_w = std::pow(sigmoid(box[2]) * 2.0f, 2.0f) * anchor_w / 2.0f;
        const float half_h = std::pow(sigmoid(box[3]) * 2.0f, 2.0f) * anchor_h / 2.0f;
        Face face{center_x - half_w, center_y - half_h, center_x + half_w, center_y + half_h,
                  score};
        for (int k = 0; k < kNumLandmarks; ++k) {
          face.landmarks[k] = {points[2 * k] * anchor_w + static_cast<float>(x) * stride,
                               points[2 * k + 1] * anchor_h + static_cast<float>(y) * stride};
        }
        faces.push_back(face);
      }
    }
  }
}

inline float iou(const Face& a, const Face& b) {
  const float overlap_w = std::max(0.0f, std::min(a.x2, b.x2) - std::max(a.x1, b.x1));
  const float overlap_h = std::max(0.0f, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
  const float overlap = overlap_w * overlap_h;
  const float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
  const float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
  const float union_area = area_a + area_b - overlap;
  return union_area > 0.0f ? overlap / union_area : 0.0f;
}

// Greedy non-maximum suppression, highest score first, stopping at `max_detections`.
inline std::vector<Face> keep_after_nms(std::vector<Face> faces, float nms_iou,
                                        int max_detections) {
  std::stable_sort(faces.begin(), faces.end(),
                   [](const Face& a, const Face& b) { return a.score > b.score; });
  std::vector<Face> kept;
  for (const Face& face : faces) {
    if (static_cast<int>(kept.size()) >= max_detections) {
      break;
    }
    if (std::none_of(kept.begin(), kept.end(),
                     [&](const Face& other) { return iou(other, face) > nms_iou; })) {
      kept.push_back(face);
    }
  }
  return kept;
}

// The faces in the model's six split heads, in model canvas pixels, after NMS.
inline std::vector<Face> decode_faces(const std::vector<SplitHead>& heads, float min_score,
                                      float nms_iou, int max_detections) {
  if (heads.size() != 6) {
    throw std::runtime_error("expected 6 split heads, got " + std::to_string(heads.size()));
  }
  std::map<std::pair<int, int>, std::map<int, ChannelsLast>, std::greater<>> by_grid;
  for (const SplitHead& head : heads) {
    ChannelsLast hwc = channels_last(head);
    const int channels = hwc.channels;
    by_grid[{hwc.rows, hwc.cols}][channels] = std::move(hwc);
  }
  bool paired = by_grid.size() == 3;
  for (const auto& [grid, level_heads] : by_grid) {
    paired = paired && level_heads.size() == 2;
  }
  if (!paired) {
    throw std::runtime_error("expected a box and a landmark head on 3 grids, got " +
                             std::to_string(by_grid.size()) + " grids");
  }
  std::vector<Face> candidates;
  int level = 0;
  for (const auto& [grid, level_heads] : by_grid) {
    decode_level(level_heads.at(kBoxChannels), level_heads.at(kLandmarkChannels), level++,
                 min_score, candidates);
  }
  return keep_after_nms(std::move(candidates), nms_iou, max_detections);
}

// Canvas pixels to frame pixels. Boxes are clipped to the frame, landmarks are not.
inline std::vector<Face> to_frame(std::vector<Face> faces, const Letterbox& box, int frame_w,
                                  int frame_h) {
  const auto frame_x = [&](float x) { return (x - box.pad_left) / box.scale; };
  const auto frame_y = [&](float y) { return (y - box.pad_top) / box.scale; };
  const auto width = static_cast<float>(frame_w);
  const auto height = static_cast<float>(frame_h);
  for (Face& face : faces) {
    face.x1 = std::clamp(frame_x(face.x1), 0.0f, width);
    face.y1 = std::clamp(frame_y(face.y1), 0.0f, height);
    face.x2 = std::clamp(frame_x(face.x2), 0.0f, width);
    face.y2 = std::clamp(frame_y(face.y2), 0.0f, height);
    for (Point& point : face.landmarks) {
      point = {frame_x(point.x), frame_y(point.y)};
    }
  }
  return faces;
}

// The `data` of an Insight pose-estimation message: one pose of five keypoints per face, each
// keypoint carrying the face's score.
inline std::string poses_json(const std::vector<Face>& faces, const std::string& label) {
  nlohmann::json poses = nlohmann::json::array();
  for (std::size_t index = 0; index < faces.size(); ++index) {
    nlohmann::json keypoints = nlohmann::json::array();
    for (int k = 0; k < kNumLandmarks; ++k) {
      keypoints.push_back({{"name", kLandmarkNames[k]},
                           {"x", faces[index].landmarks[k].x},
                           {"y", faces[index].landmarks[k].y},
                           {"confidence", faces[index].score}});
    }
    poses.push_back({{"id", "face_" + std::to_string(index + 1)},
                     {"label", label},
                     {"keypoints", std::move(keypoints)}});
  }
  return nlohmann::json{{"poses", std::move(poses)}}.dump();
}

} // namespace yolov5_face
