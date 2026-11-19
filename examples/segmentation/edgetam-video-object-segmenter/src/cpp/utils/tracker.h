// Copyright 2026 SiMa Technologies, Inc.
// SPDX-License-Identifier: Apache-2.0

// The CPU steps of EdgeTAM tracking: the prompt encoding, the mask choice and the memory bank.
#pragma once

#include <opencv2/core.hpp>

#include <cstdint>
#include <deque>
#include <optional>
#include <vector>

namespace edgetam {

// The model's interfaces, channels last as the packages take and return them.
constexpr int kImageSize = 1024;
constexpr int kMasks = 4;
constexpr int kMaskSize = 256;
constexpr int kPointerDim = 256;
constexpr int kMemoryDim = 64;
constexpr int kMemoryTokens = 512;
constexpr int kMemorySlots = 7;
constexpr int kObjectPointers = 16;
constexpr int kPromptPoints = 3;
// Padding, background, foreground, box top-left and box bottom-right.
constexpr int kPointTypes = 5;
constexpr int kPaddingPoint = 0;
constexpr int kForegroundPoint = 2;
// The mask logits of a frame where the model's object score says the object is absent.
constexpr float kNoObjectLogit = -1024.0F;

using Shape = std::vector<int64_t>;
const Shape kEmbeddingShape = {1, 64, 64, 256};
const std::vector<Shape> kHighResShapes = {{1, 256, 256, 32}, {1, 128, 128, 64}};
const Shape kPointCoordsShape = {1, 1, kPromptPoints, 2};
const Shape kPointTypesShape = {1, 1, kPromptPoints, kPointTypes};
const Shape kMemoryMaskShape = {1, kImageSize, kImageSize, 1};
const Shape kMemoriesShape = {1, kMemorySlots* kMemoryTokens, kMemoryDim};
const Shape kUsedMemoriesShape = {1, kMemorySlots};
const Shape kPointersShape = {1, kObjectPointers, kPointerDim};
const Shape kUsedPointersShape = {1, kObjectPointers};

// Where the object is in one frame, in fractions of the frame rounded to four decimals.
struct ObjectLocation {
  bool visible = false;
  // The share of the frame the mask covers.
  double area = 0.0;
  // The mask's centre of mass, empty when the object is not visible.
  std::optional<double> center_x;
  std::optional<double> center_y;
};

// The mask decoder's point_coords and point_types.
struct PromptArrays {
  std::vector<float> coords;
  std::vector<float> types;
};

// The object's [256, 256] mask logits and its object pointer.
struct ChosenMask {
  cv::Mat mask;
  std::vector<float> pointer;
};

// The memory attention's memories, memory_mask, object_pointers and object_pointer_mask.
struct MemoryArrays {
  std::vector<float> memories;
  std::vector<float> used_memories;
  std::vector<float> pointers;
  std::vector<float> used_pointers;
};

// One foreground point at (x, y) frame fractions on the prompted frame, and only padding on a
// tracked frame, where point is empty. A coordinate is the centre of its pixel of the 1024x1024
// model input, normalized to [-1, 1].
PromptArrays prompt_arrays(const std::optional<cv::Point2d>& point);

// Chooses from the mask decoder's outputs. The decoder returns four candidate masks. With one
// point or none the model's authors keep the best of masks 1 to 3 by predicted IoU. A frame whose
// object score is not positive has no object.
ChosenMask choose_mask(const std::vector<float>& masks, const std::vector<float>& iou_scores,
                       const std::vector<float>& object_score, const std::vector<float>& pointers);

// The memory encoder's [1, 1024, 1024, 1] mask: the logits upsampled bilinearly, thresholded on
// the prompted frame and as probabilities on a tracked one.
std::vector<float> memory_mask(const cv::Mat& mask, bool prompted);

// The area and centre of the mask's positive logits.
ObjectLocation locate_object(const cv::Mat& mask);

// The frames the tracker remembers: the prompted frame, then the latest frames, newest first.
class MemoryBank {
public:
  // Remembers one frame: the memory encoder's memory and the chosen mask's object pointer.
  void add(std::vector<float> memory, std::vector<float> pointer);
  // An unused slot stays zero with a mask of 0.
  [[nodiscard]] MemoryArrays arrays() const;

private:
  std::optional<std::vector<float>> prompted_memory_;
  std::optional<std::vector<float>> prompted_pointer_;
  std::deque<std::vector<float>> memories_;
  std::deque<std::vector<float>> pointers_;
};

} // namespace edgetam
