// Copyright 2026 SiMa Technologies, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "tracker.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <tuple>

namespace drone_tracker {
namespace {

using BBox = std::array<double, 4>;

double width(const BBox& box) {
  return std::max(0.0, box[2] - box[0]);
}

double height(const BBox& box) {
  return std::max(0.0, box[3] - box[1]);
}

double center_x(const BBox& box) {
  return 0.5 * (box[0] + box[2]);
}

double center_y(const BBox& box) {
  return 0.5 * (box[1] + box[3]);
}

double iou(const BBox& a, const BBox& b) {
  const double overlap_w = std::max(0.0, std::min(a[2], b[2]) - std::max(a[0], b[0]));
  const double overlap_h = std::max(0.0, std::min(a[3], b[3]) - std::max(a[1], b[1]));
  const double intersection = overlap_w * overlap_h;
  const double union_area = width(a) * height(a) + width(b) * height(b) - intersection;
  return union_area > 0.0 ? intersection / union_area : 0.0;
}

// Center distance in mean box diagonals.
double center_distance(const BBox& a, const BBox& b) {
  const double scale =
      std::max(1.0, 0.5 * (std::hypot(width(a), height(a)) + std::hypot(width(b), height(b))));
  return std::hypot(center_x(a) - center_x(b), center_y(a) - center_y(b)) / scale;
}

} // namespace

ObjectTracker::ObjectTracker(TrackerSettings settings) : settings_(settings) {}

int ObjectTracker::tracks_started() const {
  return next_track_id_ - 1;
}

std::vector<TrackedDetection>
ObjectTracker::update(const std::vector<simaai::neat::Box>& detections, int frame_index) {
  std::erase_if(tracks_, [&](const auto& item) {
    return frame_index - item.second.last_frame_index - 1 > settings_.max_missing_frames;
  });
  std::vector<BBox> boxes;
  std::vector<int> high;
  std::vector<int> low;
  for (std::size_t index = 0; index < detections.size(); ++index) {
    const auto& detection = detections[index];
    boxes.push_back({detection.x1, detection.y1, detection.x2, detection.y2});
    (detection.score >= settings_.high_score_threshold ? high : low)
        .push_back(static_cast<int>(index));
  }

  std::map<int, int> assignments; // Detection index -> track ID.
  associate(boxes, high, frame_index, assignments, /*confirmed_only=*/false);
  associate(boxes, low, frame_index, assignments, /*confirmed_only=*/true);

  // Move each matched track to its box and blend the measured motion into its velocity.
  std::set<int> matched_tracks;
  for (const auto& [detection_index, track_id] : assignments) {
    Track& track = tracks_.at(track_id);
    const BBox& box = boxes[static_cast<std::size_t>(detection_index)];
    const int elapsed = std::max(1, frame_index - track.last_frame_index);
    const std::array<double, 4> measured{
        (center_x(box) - center_x(track.bbox)) / elapsed,
        (center_y(box) - center_y(track.bbox)) / elapsed,
        (width(box) - width(track.bbox)) / elapsed,
        (height(box) - height(track.bbox)) / elapsed,
    };
    for (std::size_t i = 0; i < measured.size(); ++i) {
      track.velocity[i] =
          kVelocityMomentum * track.velocity[i] + (1.0 - kVelocityMomentum) * measured[i];
    }
    track.bbox = box;
    track.last_frame_index = frame_index;
    ++track.hits;
    matched_tracks.insert(track_id);
  }
  std::erase_if(tracks_, [&](const auto& item) {
    return matched_tracks.count(item.first) == 0 &&
           frame_index - item.second.last_frame_index > settings_.max_missing_frames;
  });

  for (const int detection_index : high) {
    if (assignments.count(detection_index) != 0 ||
        detections[static_cast<std::size_t>(detection_index)].score <
            settings_.new_track_threshold) {
      continue;
    }
    if (static_cast<int>(tracks_.size()) >= kMaxActiveTracks) {
      break;
    }
    const int track_id = next_track_id_++;
    tracks_[track_id] = Track{boxes[static_cast<std::size_t>(detection_index)], frame_index};
    assignments[detection_index] = track_id;
  }

  std::vector<TrackedDetection> tracked;
  for (const auto& [detection_index, track_id] : assignments) {
    if (tracks_.at(track_id).hits < settings_.min_confirmed_hits) {
      continue;
    }
    const auto& detection = detections[static_cast<std::size_t>(detection_index)];
    tracked.push_back({track_id, detection.x1, detection.y1, detection.x2, detection.y2,
                       detection.score, detection.class_id});
  }
  return tracked;
}

// Assigns free detections to free tracks, best affinity first.
void ObjectTracker::associate(const std::vector<BBox>& boxes,
                              const std::vector<int>& detection_indices, int frame_index,
                              std::map<int, int>& assignments, bool confirmed_only) const {
  std::set<int> matched_tracks;
  for (const auto& [detection_index, track_id] : assignments) {
    matched_tracks.insert(track_id);
  }
  // (affinity, track ID, detection index) of every pair that passes a gate.
  std::vector<std::tuple<double, int, int>> pairs;
  for (const auto& [track_id, track] : tracks_) {
    if (matched_tracks.count(track_id) != 0 ||
        (confirmed_only && track.hits < settings_.min_confirmed_hits)) {
      continue;
    }
    // The box moved by its velocity to frame_index, at least 1 pixel on each side.
    const int elapsed = frame_index - track.last_frame_index;
    const double predicted_x = center_x(track.bbox) + track.velocity[0] * elapsed;
    const double predicted_y = center_y(track.bbox) + track.velocity[1] * elapsed;
    const double predicted_w = std::max(1.0, width(track.bbox) + track.velocity[2] * elapsed);
    const double predicted_h = std::max(1.0, height(track.bbox) + track.velocity[3] * elapsed);
    const BBox predicted{predicted_x - predicted_w * 0.5, predicted_y - predicted_h * 0.5,
                         predicted_x + predicted_w * 0.5, predicted_y + predicted_h * 0.5};
    for (const int detection_index : detection_indices) {
      if (assignments.count(detection_index) != 0) {
        continue;
      }
      const BBox& box = boxes[static_cast<std::size_t>(detection_index)];
      const double overlap = iou(predicted, box);
      const double distance = center_distance(predicted, box);
      if (overlap >= kMatchIou || distance <= kMaxCenterDistance) {
        pairs.emplace_back(overlap + 1.0 / (1.0 + distance), track_id, detection_index);
      }
    }
  }
  // Highest affinity first; ties go to the lower track ID, then the lower detection index.
  std::sort(pairs.begin(), pairs.end(), [](const auto& lhs, const auto& rhs) {
    if (std::get<0>(lhs) != std::get<0>(rhs)) {
      return std::get<0>(lhs) > std::get<0>(rhs);
    }
    return std::make_pair(std::get<1>(lhs), std::get<2>(lhs)) <
           std::make_pair(std::get<1>(rhs), std::get<2>(rhs));
  });
  for (const auto& [affinity, track_id, detection_index] : pairs) {
    if (matched_tracks.count(track_id) != 0 || assignments.count(detection_index) != 0) {
      continue;
    }
    matched_tracks.insert(track_id);
    assignments[detection_index] = track_id;
  }
}

} // namespace drone_tracker
