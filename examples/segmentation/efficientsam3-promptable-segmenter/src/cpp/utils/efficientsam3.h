// Copyright 2026 SiMa Technologies, Inc.
// SPDX-License-Identifier: Apache-2.0

// EfficientSAM3's host work: the model input made from a decoded frame, the instances read from
// the model's outputs, and the metadata Insight draws. Mirrors utils/efficientsam3.py in the
// Python implementation, so both send the same metadata.

#pragma once

#include <opencv2/core.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace efficientsam3 {

// The model reads a 1008x1008 RGB image.
constexpr int kModelSize = 1008;
// MetadataSender rejects a message above 65507 bytes; half of it leaves room for the envelope.
constexpr std::size_t kMetadataByteBudget = 32'768;
// The model sometimes returns one object twice, e.g. a whole person and their upper body. An
// instance whose box overlaps a higher-scoring one by more than this intersection-over-union is
// dropped as a duplicate.
constexpr double kDuplicateIou = 0.6;

struct SegmentOptions {
  int width = 0; // Frame size the boxes and outlines are scaled to, in pixels.
  int height = 0;
  std::string label; // The prompt, which labels every instance.
  double min_score = 0.0;
  int max_detections = 0;
  double mask_threshold = 0.0;
};

using Box = std::array<int, 4>; // x0, y0, x1, y1 in frame pixels

struct Segment {
  std::string id;
  std::string label;
  double confidence = 0.0;
  Box box{};
  std::vector<cv::Point> mask;
};

// Write the model input for an NV12 frame (height * 3 / 2 rows) into `out`: RGB at kModelSize x
// kModelSize, scaled to -1..1.
void preprocess(const cv::Mat& frame, cv::Mat& out);

// The outline of the largest region of one instance's mask inside its box, in frame pixels; empty
// when there is none.
std::vector<cv::Point> mask_outline(const cv::Mat& mask_logits, const Box& box, int width,
                                    int height, float logit_threshold);

// Intersection over union of two boxes.
double box_iou(const Box& a, const Box& b);

// The instances of one result, best first. `detections` holds queries x 6 values (x1, y1, x2, y2,
// score, -) in model pixels, and `masks` mask_h x mask_w logits with one channel per query.
std::vector<Segment> segments_of(const cv::Mat& detections, const cv::Mat& masks,
                                 const SegmentOptions& options);

// The "segments" metadata: the best-scoring segments that fit the metadata byte budget.
std::string segments_json(const std::vector<Segment>& segments);

// Pairs every video frame with the result of the segmented frame nearest to it in time. Insight
// draws metadata only on the frame with its timestamp, and the model segments only some of the
// frames, so every frame is sent the nearest finished result.
class OverlayClock {
public:
  struct Message {
    std::string data;
    int64_t timestamp_ms;
    std::string frame_id;
  };

  void add_frame(int64_t pts_ms, const std::string& frame_id);
  // The messages of the frames up to this result.
  std::vector<Message> add_result(int64_t pts_ms, const std::string& data);

private:
  std::deque<std::pair<int64_t, std::string>> waiting_;
  std::optional<std::pair<int64_t, std::string>> previous_;
};

} // namespace efficientsam3
