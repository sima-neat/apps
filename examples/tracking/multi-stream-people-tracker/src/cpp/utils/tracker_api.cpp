#pragma once

#include <memory>
#include <vector>

namespace multi_stream_people_tracker {

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
  float match_iou_threshold = 0.10f;
  float max_center_distance = 2.5f;
  float velocity_momentum = 0.80f;
  int max_missing_frames = 15;
  int min_confirmed_hits = 1;
  bool center_distance_enabled = true;
};

// A compact two-stage tracker for detections whose boxes may be only a few
// pixels wide. High-confidence detections establish tracks. Lower-confidence
// detections may recover an established track, but can never create one.
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

} // namespace multi_stream_people_tracker
