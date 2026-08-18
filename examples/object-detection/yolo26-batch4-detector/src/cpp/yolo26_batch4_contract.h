// Copyright 2026 SiMa Technologies, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace yolo26_batch4 {

inline constexpr int kBatchSize = 4;
inline constexpr int kBboxChannels = 4;
inline constexpr int kLevels = 3;
inline constexpr int kExpectedOutputs = 2 * kLevels;

struct GridShape {
  int height = 0;
  int width = 0;
};

struct ModelContract {
  int net = 0;
  int class_count = 0;
  std::array<GridShape, kLevels> grids{};
};

inline ModelContract
validate_model_contract(const std::vector<std::vector<std::int64_t>>& input_shapes,
                        const std::vector<std::vector<std::int64_t>>& output_shapes,
                        std::size_t label_count) {
  if (input_shapes.size() != 1U) {
    throw std::runtime_error("expected exactly one model input");
  }
  const auto& input = input_shapes.front();
  if (input.size() != 4U ||
      std::any_of(input.begin(), input.end(), [](std::int64_t dim) { return dim <= 0; })) {
    throw std::runtime_error("model input must be a concrete positive [4,N,N,3] tensor");
  }
  if (input[0] != kBatchSize || input[1] != input[2] || input[3] != 3) {
    throw std::runtime_error("model input must have shape [4,N,N,3]");
  }
  if (output_shapes.size() != static_cast<std::size_t>(kExpectedOutputs)) {
    throw std::runtime_error("expected exactly six YOLO26 output tensors");
  }

  std::vector<GridShape> bbox_grids;
  std::vector<GridShape> class_grids;
  int class_count = 0;
  for (const auto& output : output_shapes) {
    if (output.size() != 4U ||
        std::any_of(output.begin(), output.end(), [](std::int64_t dim) { return dim <= 0; }) ||
        output[0] != kBatchSize || output[1] != output[2]) {
      throw std::runtime_error("every model output must be a concrete positive [4,G,G,C] tensor");
    }
    const GridShape grid{static_cast<int>(output[1]), static_cast<int>(output[2])};
    const int channels = static_cast<int>(output[3]);
    if (channels == kBboxChannels) {
      bbox_grids.push_back(grid);
      continue;
    }
    if (class_count == 0) {
      class_count = channels;
    } else if (channels != class_count) {
      throw std::runtime_error("YOLO26 class heads must have one common class count");
    }
    class_grids.push_back(grid);
  }
  if (bbox_grids.size() != kLevels || class_grids.size() != kLevels) {
    throw std::runtime_error(
        "expected three bbox heads and three class heads; four-class models are ambiguous");
  }
  if (label_count != static_cast<std::size_t>(class_count)) {
    throw std::runtime_error("label count must match the model class-head channel count");
  }

  const auto descending = [](const GridShape& lhs, const GridShape& rhs) {
    if (lhs.height != rhs.height) {
      return lhs.height > rhs.height;
    }
    return lhs.width > rhs.width;
  };
  std::sort(bbox_grids.begin(), bbox_grids.end(), descending);
  std::sort(class_grids.begin(), class_grids.end(), descending);

  ModelContract contract;
  contract.net = static_cast<int>(input[1]);
  contract.class_count = class_count;
  for (std::size_t i = 0U; i < kLevels; ++i) {
    if (bbox_grids[i].height != class_grids[i].height ||
        bbox_grids[i].width != class_grids[i].width ||
        (i > 0U && bbox_grids[i].height == bbox_grids[i - 1U].height &&
         bbox_grids[i].width == bbox_grids[i - 1U].width)) {
      throw std::runtime_error("bbox and class heads must form three distinct matching grids");
    }
    contract.grids[i] = bbox_grids[i];
  }
  return contract;
}

inline std::size_t head_cell_offset(int row, int col, int width, int channels) {
  return (static_cast<std::size_t>(row) * static_cast<std::size_t>(width) +
          static_cast<std::size_t>(col)) *
         static_cast<std::size_t>(channels);
}

} // namespace yolo26_batch4
