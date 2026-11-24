// Copyright 2026 SiMa Technologies, Inc.
// SPDX-License-Identifier: Apache-2.0

// Two-stage, motion-aware tracking of small detections, one tracker per stream. Mirrors
// utils/tracker.py in the Python implementation, so both languages give the same track IDs to the
// same boxes. High-score detections start and confirm tracks. A low-score detection may continue
// a confirmed track, but it cannot start or confirm one.

#pragma once

#include "neat.h"

#include <array>
#include <map>
#include <vector>

namespace drone_tracker {

// A match needs this IoU between a track's predicted box and the detection, or a center distance
// within kMaxCenterDistance. Tiny boxes that move a few pixels rarely overlap, so the IoU gate is
// low and the distance gate does most of the matching.
constexpr double kMatchIou = 0.05;
// Center distance in units of the mean box diagonal of the two boxes.
constexpr double kMaxCenterDistance = 3.0;
// Weight of the previous velocity when a match updates it; the rest is the measured motion.
constexpr double kVelocityMomentum = 0.75;
// New tracks stop at this many active tracks per stream.
constexpr int kMaxActiveTracks = 256;

struct TrackerSettings {
  float high_score_threshold = 0.0f; // Score that lets a detection start or confirm a track.
  float new_track_threshold = 0.0f;  // Score an unmatched detection needs to start a track.
  int max_missing_frames = 0;        // Frames a track survives without a match.
  int min_confirmed_hits = 0;        // Matches before a track is reported.
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

// Gives each detection of a stream the ID of the track it continues, frame by frame.
class ObjectTracker {
public:
  explicit ObjectTracker(TrackerSettings settings);

  int tracks_started() const;

  // Matches this frame's detections to the tracks, and returns those of confirmed tracks in
  // detection order. `frame_index` counts the stream's frames and only grows.
  std::vector<TrackedDetection> update(const std::vector<simaai::neat::Box>& detections,
                                       int frame_index);

private:
  using BBox = std::array<double, 4>;

  struct Track {
    BBox bbox{};
    int last_frame_index = 0;
    std::array<double, 4> velocity{};
    int hits = 1;
  };

  void associate(const std::vector<BBox>& boxes, const std::vector<int>& detection_indices,
                 int frame_index, std::map<int, int>& assignments, bool confirmed_only) const;

  TrackerSettings settings_;
  int next_track_id_ = 1;
  // Ordered by track ID, as Python's insertion-ordered dict is, so ties resolve alike.
  std::map<int, Track> tracks_;
};

} // namespace drone_tracker
