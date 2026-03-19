#include "examples/object-detection/multi-camera-people-detection-and-tracking/cpp/tracker_api.cpp"

#include <algorithm>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace multi_camera_people_tracking {
namespace {

float iou_xyxy(const Detection& a, const Detection& b) {
  const float xx1 = std::max(a.x1, b.x1);
  const float yy1 = std::max(a.y1, b.y1);
  const float xx2 = std::min(a.x2, b.x2);
  const float yy2 = std::min(a.y2, b.y2);
  const float width = std::max(0.0f, xx2 - xx1);
  const float height = std::max(0.0f, yy2 - yy1);
  const float inter = width * height;
  const float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
  const float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
  const float denom = area_a + area_b - inter;
  return denom > 0.0f ? (inter / denom) : 0.0f;
}

struct TrackState {
  int track_id = 0;
  Detection detection;
  int last_frame_index = 0;
  int missing_frames = 0;
};

} // namespace

struct PeopleTracker::Impl {
  std::unordered_map<int, TrackState> tracks;
};

PeopleTracker::PeopleTracker(float iou_threshold, int max_missing_frames)
    : iou_threshold_(iou_threshold), max_missing_frames_(max_missing_frames),
      impl_(std::make_unique<Impl>()) {}

PeopleTracker::~PeopleTracker() = default;

PeopleTracker::PeopleTracker(PeopleTracker&&) noexcept = default;

PeopleTracker& PeopleTracker::operator=(PeopleTracker&&) noexcept = default;

int PeopleTracker::active_track_count() const {
  return static_cast<int>(impl_->tracks.size());
}

std::vector<TrackedDetection> PeopleTracker::update(const std::vector<Detection>& detections,
                                                    int frame_index) {
  std::vector<std::tuple<float, int, int>> candidates;
  for (const auto& [track_id, track] : impl_->tracks) {
    for (std::size_t det_index = 0; det_index < detections.size(); ++det_index) {
      if (detections[det_index].class_id != track.detection.class_id) {
        continue;
      }
      const float iou = iou_xyxy(track.detection, detections[det_index]);
      if (iou >= iou_threshold_) {
        candidates.emplace_back(iou, track_id, static_cast<int>(det_index));
      }
    }
  }
  std::sort(candidates.begin(), candidates.end(), std::greater<>());

  std::unordered_set<int> matched_tracks;
  std::unordered_set<int> matched_dets;
  std::unordered_map<int, int> assignments;

  for (const auto& [iou, track_id, det_index] : candidates) {
    static_cast<void>(iou);
    if (matched_tracks.count(track_id) > 0 || matched_dets.count(det_index) > 0) {
      continue;
    }
    matched_tracks.insert(track_id);
    matched_dets.insert(det_index);
    assignments[det_index] = track_id;
  }

  for (const auto& [det_index, track_id] : assignments) {
    TrackState state;
    state.track_id = track_id;
    state.detection = detections[static_cast<std::size_t>(det_index)];
    state.last_frame_index = frame_index;
    state.missing_frames = 0;
    impl_->tracks[track_id] = state;
  }

  for (std::size_t det_index = 0; det_index < detections.size(); ++det_index) {
    if (matched_dets.count(static_cast<int>(det_index)) > 0) {
      continue;
    }
    const int track_id = next_track_id_++;
    impl_->tracks[track_id] = TrackState{
        track_id,
        detections[det_index],
        frame_index,
        0,
    };
    assignments[static_cast<int>(det_index)] = track_id;
  }

  std::vector<int> expired;
  for (auto& [track_id, track] : impl_->tracks) {
    if (matched_tracks.count(track_id) > 0) {
      continue;
    }
    bool assigned_now = false;
    for (const auto& [det_index, assigned_track_id] : assignments) {
      static_cast<void>(det_index);
      if (assigned_track_id == track_id) {
        assigned_now = true;
        break;
      }
    }
    if (assigned_now) {
      continue;
    }
    track.missing_frames += 1;
    if (track.missing_frames > max_missing_frames_) {
      expired.push_back(track_id);
    }
  }
  for (const int track_id : expired) {
    impl_->tracks.erase(track_id);
  }

  std::vector<TrackedDetection> tracked;
  tracked.reserve(detections.size());
  for (std::size_t det_index = 0; det_index < detections.size(); ++det_index) {
    const int track_id = assignments[static_cast<int>(det_index)];
    const Detection& det = detections[det_index];
    tracked.push_back(TrackedDetection{
        track_id,
        det.x1,
        det.y1,
        det.x2,
        det.y2,
        det.score,
        det.class_id,
    });
  }
  return tracked;
}

} // namespace multi_camera_people_tracking
