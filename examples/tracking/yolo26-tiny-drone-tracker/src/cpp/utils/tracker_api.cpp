#pragma once

#include <memory>
#include <vector>

namespace yolo26_tiny_drone_tracker {

struct Detection {
  float x1 = 0.0f;
  float y1 = 0.0f;
  float x2 = 0.0f;
  float y2 = 0.0f;
  float score = 0.0f;
  int class_id = -1;
};

struct TrackedDetection {
  int track_id = 0;
  float x1 = 0.0f;
  float y1 = 0.0f;
  float x2 = 0.0f;
  float y2 = 0.0f;
  float score = 0.0f;
  int class_id = -1;
};

struct TrackerConfig {
  float high_score_threshold = 0.30f;
  float new_track_threshold = 0.30f;
  float match_iou_threshold = 0.30f;
  float max_center_distance = 2.5f;
  float velocity_momentum = 0.80f;
  int max_missing_frames = 15;
  int min_confirmed_hits = 1;
  int max_active_tracks = 256;
  bool center_distance_enabled = false;
};

// High-confidence detections establish tracks. Lower-confidence detections
// may recover a confirmed track, but can never create or confirm one.
class ObjectTracker {
public:
  explicit ObjectTracker(TrackerConfig config = {});
  ~ObjectTracker();

  ObjectTracker(ObjectTracker&&) noexcept;
  ObjectTracker& operator=(ObjectTracker&&) noexcept;
  ObjectTracker(const ObjectTracker&) = delete;
  ObjectTracker& operator=(const ObjectTracker&) = delete;

  int active_track_count() const;
  std::vector<TrackedDetection> update(const std::vector<Detection>& detections, int frame_index);

private:
  TrackerConfig config_;
  int next_track_id_ = 1;

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace yolo26_tiny_drone_tracker
