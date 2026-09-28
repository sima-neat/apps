#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace multi_stream_tracker {

inline constexpr std::size_t kMaxClasses = 5;

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
  std::string label;
};

/// Tracker settings for one detection class. Defaults follow ByteTrack.
struct ClassTrackerConfig {
  int class_id = -1;
  std::string label;
  double high_score_threshold = 0.50;
  double low_score_threshold = 0.10;
  double new_track_threshold = 0.60;
  double match_iou_threshold = 0.20;
  double low_match_iou_threshold = 0.50;
  int max_missing_frames = 30;
  int min_confirmed_hits = 2;
  double position_noise = 0.05;
  double velocity_noise = 0.00625;

  /// Throws std::runtime_error with the offending key.
  void validate() const;
};

/// COCO class labels in YOLO26 detector class-id order.
const std::vector<std::string>& coco_labels();

/// Maps a COCO class name or numeric id to (class_id, label). Throws on unknown classes.
std::pair<int, std::string> resolve_class(const std::string& value);

/// One `key: value` mapping from a `tracking.classes` entry, in file order.
using ClassEntry = std::vector<std::pair<std::string, std::string>>;

/// Parses and validates 1 to kMaxClasses entries. Throws on invalid or duplicate classes.
std::vector<ClassTrackerConfig> parse_class_configs(const std::vector<ClassEntry>& entries);

/// Minimum-cost assignment (Hungarian); pairs whose cost exceeds `max_cost` are dropped.
std::vector<std::pair<int, int>> linear_assignment(const std::vector<std::vector<double>>& cost,
                                                   double max_cost);

/// ByteTrack-style tracker for one class. Track IDs come from a counter that
/// can be shared with other class trackers on the same stream.
class ByteTracker {
public:
  ByteTracker(ClassTrackerConfig config, std::shared_ptr<int> next_track_id);
  ~ByteTracker();

  ByteTracker(ByteTracker&&) noexcept;
  ByteTracker& operator=(ByteTracker&&) noexcept;
  ByteTracker(const ByteTracker&) = delete;
  ByteTracker& operator=(const ByteTracker&) = delete;

  const ClassTrackerConfig& config() const { return config_; }
  int active_track_count() const;
  std::vector<TrackedDetection> update(const std::vector<Detection>& detections, int frame_index);

private:
  ClassTrackerConfig config_;
  std::shared_ptr<int> next_track_id_;

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// One ByteTracker per configured class with stream-unique track IDs.
class MultiClassTracker {
public:
  MultiClassTracker() = default;
  explicit MultiClassTracker(const std::vector<ClassTrackerConfig>& configs);

  int active_track_count() const;
  std::vector<TrackedDetection> update(const std::vector<Detection>& detections, int frame_index);

private:
  std::vector<ByteTracker> trackers_;
};

} // namespace multi_stream_tracker
