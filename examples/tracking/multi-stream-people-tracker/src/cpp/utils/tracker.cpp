#include "examples/tracking/multi-stream-people-tracker/src/cpp/utils/tracker_api.cpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace multi_stream_people_tracker {
namespace {

struct TrackState {
  int track_id = 0;
  Detection detection;
  float velocity_x = 0.0f;
  float velocity_y = 0.0f;
  float velocity_w = 0.0f;
  float velocity_h = 0.0f;
  int last_frame_index = 0;
  int missing_frames = 0;
  int hits = 1;
};

float width(const Detection& detection) {
  return std::max(0.0f, detection.x2 - detection.x1);
}

float height(const Detection& detection) {
  return std::max(0.0f, detection.y2 - detection.y1);
}

float center_x(const Detection& detection) {
  return 0.5f * (detection.x1 + detection.x2);
}

float center_y(const Detection& detection) {
  return 0.5f * (detection.y1 + detection.y2);
}

float iou_xyxy(const Detection& a, const Detection& b) {
  const float xx1 = std::max(a.x1, b.x1);
  const float yy1 = std::max(a.y1, b.y1);
  const float xx2 = std::min(a.x2, b.x2);
  const float yy2 = std::min(a.y2, b.y2);
  const float intersection = std::max(0.0f, xx2 - xx1) * std::max(0.0f, yy2 - yy1);
  const float union_area = width(a) * height(a) + width(b) * height(b) - intersection;
  return union_area > 0.0f ? intersection / union_area : 0.0f;
}

float normalized_center_distance(const Detection& a, const Detection& b) {
  const float dx = center_x(a) - center_x(b);
  const float dy = center_y(a) - center_y(b);
  const float diagonal_a = std::hypot(width(a), height(a));
  const float diagonal_b = std::hypot(width(b), height(b));
  const float scale = std::max(1.0f, 0.5f * (diagonal_a + diagonal_b));
  return std::hypot(dx, dy) / scale;
}

Detection predict(const TrackState& track, int frame_index) {
  const int elapsed = std::max(0, frame_index - track.last_frame_index);
  const float predicted_x = center_x(track.detection) + track.velocity_x * elapsed;
  const float predicted_y = center_y(track.detection) + track.velocity_y * elapsed;
  const float predicted_w = std::max(1.0f, width(track.detection) + track.velocity_w * elapsed);
  const float predicted_h = std::max(1.0f, height(track.detection) + track.velocity_h * elapsed);
  return Detection{
      predicted_x - predicted_w * 0.5f,
      predicted_y - predicted_h * 0.5f,
      predicted_x + predicted_w * 0.5f,
      predicted_y + predicted_h * 0.5f,
      track.detection.score,
      track.detection.class_id,
  };
}

void validate_config(const TrackerConfig& config) {
  if (config.high_score_threshold < 0.0f || config.high_score_threshold > 1.0f) {
    throw std::invalid_argument("high_score_threshold must be in [0, 1]");
  }
  if (config.new_track_threshold < config.high_score_threshold ||
      config.new_track_threshold > 1.0f) {
    throw std::invalid_argument("new_track_threshold must be in [high_score_threshold, 1]");
  }
  if (config.match_iou_threshold < 0.0f || config.match_iou_threshold > 1.0f) {
    throw std::invalid_argument("match_iou_threshold must be in [0, 1]");
  }
  if (!std::isfinite(config.max_center_distance) || config.max_center_distance < 0.0f) {
    throw std::invalid_argument("max_center_distance must be >= 0");
  }
  if (config.velocity_momentum < 0.0f || config.velocity_momentum >= 1.0f) {
    throw std::invalid_argument("velocity_momentum must be in [0, 1)");
  }
  if (config.max_missing_frames < 0) {
    throw std::invalid_argument("max_missing_frames must be >= 0");
  }
  if (config.min_confirmed_hits < 1) {
    throw std::invalid_argument("min_confirmed_hits must be >= 1");
  }
}

} // namespace

struct ObjectTracker::Impl {
  std::unordered_map<int, TrackState> tracks;
  int last_frame_index = -1;
};

ObjectTracker::ObjectTracker(TrackerConfig config)
    : config_(config), impl_(std::make_unique<Impl>()) {
  validate_config(config_);
}

ObjectTracker::~ObjectTracker() = default;
ObjectTracker::ObjectTracker(ObjectTracker&&) noexcept = default;
ObjectTracker& ObjectTracker::operator=(ObjectTracker&&) noexcept = default;

int ObjectTracker::active_track_count() const {
  return static_cast<int>(impl_->tracks.size());
}

std::vector<TrackedDetection> ObjectTracker::update(const std::vector<Detection>& detections,
                                                    int frame_index) {
  if (frame_index < 0) {
    throw std::invalid_argument("frame_index must be >= 0");
  }
  if (frame_index < impl_->last_frame_index) {
    throw std::invalid_argument("frame_index must be monotonic");
  }
  impl_->last_frame_index = frame_index;

  std::vector<int> stale_tracks;
  for (const auto& [track_id, track] : impl_->tracks) {
    const int intervening_frames = std::max(0, frame_index - track.last_frame_index - 1);
    if (intervening_frames > config_.max_missing_frames) {
      stale_tracks.push_back(track_id);
    }
  }
  for (const int track_id : stale_tracks) {
    impl_->tracks.erase(track_id);
  }

  std::vector<int> high_detections;
  std::vector<int> low_detections;
  for (std::size_t index = 0; index < detections.size(); ++index) {
    if (detections[index].score >= config_.high_score_threshold) {
      high_detections.push_back(static_cast<int>(index));
    } else {
      low_detections.push_back(static_cast<int>(index));
    }
  }

  std::unordered_set<int> matched_tracks;
  std::unordered_set<int> matched_detections;
  std::unordered_map<int, int> assignments;

  const auto associate = [&](const std::vector<int>& detection_indices) {
    std::vector<std::tuple<float, int, int>> candidates;
    for (const auto& [track_id, track] : impl_->tracks) {
      if (matched_tracks.count(track_id) != 0) {
        continue;
      }
      const Detection predicted = predict(track, frame_index);
      for (const int detection_index : detection_indices) {
        if (matched_detections.count(detection_index) != 0) {
          continue;
        }
        const Detection& detection = detections[static_cast<std::size_t>(detection_index)];
        if (detection.class_id != track.detection.class_id) {
          continue;
        }
        const float iou = iou_xyxy(predicted, detection);
        const float center_distance = normalized_center_distance(predicted, detection);
        if (iou < config_.match_iou_threshold && center_distance > config_.max_center_distance) {
          continue;
        }
        const float affinity = iou + 1.0f / (1.0f + center_distance);
        candidates.emplace_back(affinity, track_id, detection_index);
      }
    }
    std::sort(candidates.begin(), candidates.end(), std::greater<>());
    for (const auto& [affinity, track_id, detection_index] : candidates) {
      static_cast<void>(affinity);
      if (matched_tracks.count(track_id) != 0 || matched_detections.count(detection_index) != 0) {
        continue;
      }
      matched_tracks.insert(track_id);
      matched_detections.insert(detection_index);
      assignments[detection_index] = track_id;
    }
  };

  // ByteTrack-style score staging: recover existing tracks with low scores,
  // but only high-score detections participate in track creation.
  associate(high_detections);
  associate(low_detections);

  for (const auto& [detection_index, track_id] : assignments) {
    TrackState& track = impl_->tracks.at(track_id);
    const Detection& detection = detections[static_cast<std::size_t>(detection_index)];
    const int elapsed = std::max(1, frame_index - track.last_frame_index);
    const float measured_vx = (center_x(detection) - center_x(track.detection)) / elapsed;
    const float measured_vy = (center_y(detection) - center_y(track.detection)) / elapsed;
    const float measured_vw = (width(detection) - width(track.detection)) / elapsed;
    const float measured_vh = (height(detection) - height(track.detection)) / elapsed;
    const float previous = config_.velocity_momentum;
    const float measured = 1.0f - previous;
    track.velocity_x = previous * track.velocity_x + measured * measured_vx;
    track.velocity_y = previous * track.velocity_y + measured * measured_vy;
    track.velocity_w = previous * track.velocity_w + measured * measured_vw;
    track.velocity_h = previous * track.velocity_h + measured * measured_vh;
    track.detection = detection;
    track.last_frame_index = frame_index;
    track.missing_frames = 0;
    ++track.hits;
  }

  for (const int detection_index : high_detections) {
    if (matched_detections.count(detection_index) != 0) {
      continue;
    }
    const Detection& detection = detections[static_cast<std::size_t>(detection_index)];
    if (detection.score < config_.new_track_threshold) {
      continue;
    }
    const int track_id = next_track_id_++;
    impl_->tracks.emplace(
        track_id, TrackState{track_id, detection, 0.0f, 0.0f, 0.0f, 0.0f, frame_index, 0, 1});
    matched_tracks.insert(track_id);
    matched_detections.insert(detection_index);
    assignments[detection_index] = track_id;
  }

  std::vector<int> expired_tracks;
  for (auto& [track_id, track] : impl_->tracks) {
    if (matched_tracks.count(track_id) != 0) {
      continue;
    }
    track.missing_frames = frame_index - track.last_frame_index;
    if (track.missing_frames > config_.max_missing_frames) {
      expired_tracks.push_back(track_id);
    }
  }
  for (const int track_id : expired_tracks) {
    impl_->tracks.erase(track_id);
  }

  std::vector<TrackedDetection> tracked;
  tracked.reserve(assignments.size());
  for (std::size_t detection_index = 0; detection_index < detections.size(); ++detection_index) {
    const auto assignment = assignments.find(static_cast<int>(detection_index));
    if (assignment == assignments.end()) {
      continue;
    }
    const TrackState& track = impl_->tracks.at(assignment->second);
    if (track.hits < config_.min_confirmed_hits) {
      continue;
    }
    const Detection& detection = detections[detection_index];
    tracked.push_back(TrackedDetection{track.track_id, detection.x1, detection.y1, detection.x2,
                                       detection.y2, detection.score, detection.class_id});
  }
  return tracked;
}

} // namespace multi_stream_people_tracker
