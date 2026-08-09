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
  bool predicted = false;
};

// Partial-affine transform from the previous decoded frame into the current
// frame. Invalid transforms let the tracker fall back to detection consensus.
struct CameraTransform {
  float a = 1.0f;
  float b = 0.0f;
  float tx = 0.0f;
  float c = 0.0f;
  float d = 1.0f;
  float ty = 0.0f;
  bool valid = false;
};

struct TrackerConfig {
  float high_score_threshold = 0.30f;
  float new_track_threshold = 0.30f;
  float match_iou_threshold = 0.10f;
  float max_center_distance = 2.5f;
  float velocity_momentum = 0.80f;
  float box_smoothing_alpha = 1.0f;
  int max_missing_frames = 15;
  int min_confirmed_hits = 1;
  int max_prediction_frames = 0;
  bool center_distance_enabled = true;
  bool camera_motion_compensation = false;
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
  std::vector<TrackedDetection> update(const std::vector<Detection>& detections, int frame_index,
                                       const CameraTransform& camera_transform = {});
  void update_into(const std::vector<Detection>& detections, int frame_index,
                   std::vector<TrackedDetection>& tracked,
                   const CameraTransform& camera_transform = {});

private:
  TrackerConfig config_;
  int next_track_id_ = 1;

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace multi_stream_people_tracker
