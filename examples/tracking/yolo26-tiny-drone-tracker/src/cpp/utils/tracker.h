// Copyright 2026 SiMa Technologies, Inc.
// SPDX-License-Identifier: Apache-2.0

// Two-stage, motion-aware tracking of small detections, one tracker per stream. Mirrors
// utils/tracker.py in the Python implementation, so both languages give the same track IDs to the
// same boxes. High-score detections start and confirm tracks. A low-score detection may continue
// a confirmed track, but it cannot start or confirm one.

#pragma once

#include "neat.h"

#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <vector>

namespace drone_tracker {

// A match needs the detection's center within this distance of the track's predicted center, in
// units of the mean box diagonal of the two boxes. Tiny boxes that move a few pixels rarely
// overlap, so this gate does all of the matching: two boxes that overlap are always closer than
// one mean diagonal. IoU only ranks the pairs that pass it.
constexpr double kMaxCenterDistance = 3.0;
// Weight of the previous velocity when a match updates it; the rest is the measured motion.
constexpr double kVelocityMomentum = 0.75;
// New tracks stop at this many active tracks per stream.
constexpr int kMaxActiveTracks = 256;

struct TrackerSettings {
  // Doubles, as Python reads them, so a score at a threshold falls on the same side in both.
  double high_score_threshold = 0.0; // Score that lets a detection continue any track.
  double new_track_threshold = 0.0;  // Score an unmatched detection needs to start a track.
  int max_missing_frames = 0;        // Frames a track survives without a match.
  int min_confirmed_hits = 0;        // Detections, counting the first, before it is reported.
};

// A detection of this frame and the ID of the track it belongs to.
struct TrackedDetection {
  int track_id = 0;
  float x1 = 0.0f;
  float y1 = 0.0f;
  float x2 = 0.0f;
  float y2 = 0.0f;
  float score = 0.0f;
  int class_id = -1;
};

// Gives each detection of one stream the ID of the track it continues, frame by frame.
class ObjectTracker {
public:
  ObjectTracker(TrackerSettings settings, int fps);

  int tracks_started() const;

  // Matches the detections of the frame at `pts_ns` to the tracks, and returns those of confirmed
  // tracks in detection order.
  std::vector<TrackedDetection> update(const std::vector<simaai::neat::Box>& detections,
                                       std::int64_t pts_ns);

private:
  using BBox = std::array<double, 4>;

  struct Track {
    BBox bbox{};
    int last_frame_index = 0;
    std::array<double, 4> velocity{};
    int hits = 1;
  };

  // The stream's frame number at `pts_ns`, counted from its first frame at `fps`. A realtime link
  // drops frames, so the frames this tracker sees are not consecutive.
  int frame_index_at(std::int64_t pts_ns);

  void associate(const std::vector<BBox>& boxes, const std::vector<int>& detection_indices,
                 int frame_index, std::map<int, int>& assignments, bool confirmed_only) const;

  TrackerSettings settings_;
  int fps_ = 0;
  int next_track_id_ = 1;
  std::optional<std::int64_t> first_pts_ns_;
  int frame_index_ = 0;
  // Ordered by track ID, as Python's insertion-ordered dict is, so ties resolve alike.
  std::map<int, Track> tracks_;
};

} // namespace drone_tracker
