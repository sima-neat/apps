// Copyright 2026 SiMa Technologies, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "tracker.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace edgetam {

namespace {

double round_to_4_decimals(double value) {
  return std::round(value * 1e4) / 1e4;
}

// `output` names the model output and `model` its config key, for the message.
void require_size(const std::vector<float>& values, std::size_t size, const std::string& output,
                  const std::string& model) {
  if (values.size() != size) {
    throw std::runtime_error(output + " has " + std::to_string(values.size()) + " values, not " +
                             std::to_string(size) + "; point model." + model +
                             " at EdgeTAM's package");
  }
}

// Copies each kept row into its slot of `rows` and marks the slot used.
void fill_slots(const std::vector<const std::vector<float>*>& kept, std::vector<float>& rows,
                std::vector<float>& used) {
  for (std::size_t slot = 0; slot < kept.size(); ++slot) {
    std::copy(kept[slot]->begin(), kept[slot]->end(), rows.begin() + slot * kept[slot]->size());
    used[slot] = 1.0F;
  }
}

} // namespace

PromptArrays prompt_arrays(const std::optional<cv::Point2d>& point) {
  PromptArrays prompt{std::vector<float>(kPromptPoints * 2, 0.0F),
                      std::vector<float>(kPromptPoints * kPointTypes, 0.0F)};
  for (int slot = 0; slot < kPromptPoints; ++slot) {
    prompt.types[slot * kPointTypes + kPaddingPoint] = 1.0F;
  }
  if (point) {
    prompt.coords[0] = static_cast<float>((point->x * kImageSize + 0.5) / kImageSize * 2 - 1);
    prompt.coords[1] = static_cast<float>((point->y * kImageSize + 0.5) / kImageSize * 2 - 1);
    prompt.types[kPaddingPoint] = 0.0F;
    prompt.types[kForegroundPoint] = 1.0F;
  }
  return prompt;
}

ChosenMask choose_mask(const std::vector<float>& masks, const std::vector<float>& iou_scores,
                       const std::vector<float>& object_score, const std::vector<float>& pointers) {
  require_size(masks, static_cast<std::size_t>(kMaskSize) * kMaskSize * kMasks, "mask_logits",
               "mask_decoder");
  require_size(iou_scores, kMasks, "iou_scores", "mask_decoder");
  require_size(object_score, 1, "object_score_logits", "mask_decoder");
  require_size(pointers, static_cast<std::size_t>(kMasks) * kPointerDim, "object_pointers",
               "mask_decoder");

  const int best = static_cast<int>(std::max_element(iou_scores.begin() + 1, iou_scores.end()) -
                                    iou_scores.begin());
  ChosenMask chosen;
  // The Mat reads the caller's values, and extractChannel copies the chosen mask out of them.
  const cv::Mat candidates(kMaskSize, kMaskSize, CV_32FC(kMasks), const_cast<float*>(masks.data()));
  cv::extractChannel(candidates, chosen.mask, best);
  if (object_score[0] <= 0.0F) {
    chosen.mask.setTo(kNoObjectLogit);
  }
  chosen.pointer.assign(pointers.begin() + best * kPointerDim,
                        pointers.begin() + (best + 1) * kPointerDim);
  return chosen;
}

std::vector<float> memory_mask(const cv::Mat& mask, bool prompted) {
  cv::Mat values;
  cv::resize(mask, values, cv::Size(kImageSize, kImageSize), 0, 0, cv::INTER_LINEAR);
  if (prompted) {
    cv::threshold(values, values, 0.0, 1.0, cv::THRESH_BINARY);
  } else {
    // 1 / (1 + exp(-logit)). OpenCV's exp overflows to infinity for the no-object logit, which
    // gives the probability 0 it should.
    cv::Mat exponent;
    cv::exp(-values, exponent);
    cv::divide(1.0, exponent + 1.0, values);
  }
  return {values.begin<float>(), values.end<float>()};
}

ObjectLocation locate_object(const cv::Mat& mask) {
  const cv::Mat covered = mask > 0;
  const int pixels = cv::countNonZero(covered);
  if (pixels == 0) {
    return {};
  }
  const cv::Moments moments = cv::moments(covered, true);
  return {true, round_to_4_decimals(static_cast<double>(pixels) / mask.total()),
          round_to_4_decimals((moments.m10 / moments.m00 + 0.5) / mask.cols),
          round_to_4_decimals((moments.m01 / moments.m00 + 0.5) / mask.rows)};
}

void MemoryBank::add(std::vector<float> memory, std::vector<float> pointer) {
  require_size(memory, static_cast<std::size_t>(kMemoryTokens) * kMemoryDim, "memory",
               "memory_encoder");
  if (!prompted_memory_) {
    prompted_memory_ = std::move(memory);
    prompted_pointer_ = std::move(pointer);
    return;
  }
  memories_.push_front(std::move(memory));
  pointers_.push_front(std::move(pointer));
  if (memories_.size() > kMemorySlots - 1) {
    memories_.pop_back();
  }
  if (pointers_.size() > kObjectPointers - 1) {
    pointers_.pop_back();
  }
}

MemoryArrays MemoryBank::arrays() const {
  if (!prompted_memory_) {
    throw std::logic_error("the memory bank is empty; add the prompted frame first");
  }
  MemoryArrays arrays{
      std::vector<float>(static_cast<std::size_t>(kMemorySlots) * kMemoryTokens * kMemoryDim),
      std::vector<float>(kMemorySlots), std::vector<float>(kObjectPointers * kPointerDim),
      std::vector<float>(kObjectPointers)};
  std::vector<const std::vector<float>*> memories = {&*prompted_memory_};
  for (const std::vector<float>& memory : memories_) {
    memories.push_back(&memory);
  }
  std::vector<const std::vector<float>*> pointers = {&*prompted_pointer_};
  for (const std::vector<float>& pointer : pointers_) {
    pointers.push_back(&pointer);
  }
  fill_slots(memories, arrays.memories, arrays.used_memories);
  fill_slots(pointers, arrays.pointers, arrays.used_pointers);
  return arrays;
}

} // namespace edgetam
