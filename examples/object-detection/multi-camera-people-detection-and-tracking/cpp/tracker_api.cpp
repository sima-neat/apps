#pragma once

#include <memory>
#include <vector>

namespace multi_camera_people_tracking {

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

class PeopleTracker {
public:
  explicit PeopleTracker(float iou_threshold = 0.3f, int max_missing_frames = 2);
  ~PeopleTracker();

  PeopleTracker(PeopleTracker&&) noexcept;
  PeopleTracker& operator=(PeopleTracker&&) noexcept;
  PeopleTracker(const PeopleTracker&) = delete;
  PeopleTracker& operator=(const PeopleTracker&) = delete;

  int active_track_count() const;
  std::vector<TrackedDetection> update(const std::vector<Detection>& detections, int frame_index);

private:
  float iou_threshold_ = 0.3f;
  int max_missing_frames_ = 2;
  int next_track_id_ = 1;

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace multi_camera_people_tracking
