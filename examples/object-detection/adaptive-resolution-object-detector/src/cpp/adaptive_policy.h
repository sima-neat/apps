// Copyright 2026 SiMa Technologies, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Pure, dependency-free resolution-tier policy and shared compute budget.
//
// This header has NO Neat/OpenCV dependencies on purpose: the decision logic is
// the heart of the example and must be unit-testable on the host with nothing
// but a C++ compiler. main.cpp adapts real detections into FrameStats and feeds
// the results back into the pipeline. src/python/adaptive_policy.py mirrors this
// logic line-for-line so both languages behave identically.

#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace adaptive {

// One detection reduced to just what the policy needs.
struct DetBox {
  float w = 0.0f;
  float h = 0.0f;
  float score = 0.0f;
};

// Per-frame scene summary the policy reasons over.
struct FrameStats {
  int object_count = 0;
  float min_object_px = std::numeric_limits<float>::max(); // smallest min(w,h); max => "no small objects"
  float min_confidence = 1.0f;                             // lowest score; 1.0 => "nothing uncertain"
};

// Policy knobs. The first four map directly to the issue's `adaptive:` block;
// the rest are step-down/stability tuning with sensible defaults.
struct PolicyConfig {
  std::vector<int> resolutions{320, 640, 960}; // candidate tiers, ascending
  float confidence_low = 0.40f;                // below this on any object => step up
  float min_object_px = 24.0f;                 // an object smaller than this => step up
  int hysteresis_frames = 15;                  // frames a new tier must persist before it commits
  int density_high = 20;                       // this many objects => crowded => step up
  int density_low = 5;                         // at most this many objects => eligible to step down
  float down_size_factor = 2.0f;               // objects must be >= factor*min_object_px to step down
  float confidence_margin = 0.10f;             // extra confidence headroom required to step down
};

// Per-stream mutable state carried across frames.
struct PolicyState {
  int tier_index = 0;     // committed tier (index into resolutions)
  int pending_index = 0;  // tier currently being "voted" for
  int pending_count = 0;  // consecutive frames pending_index has been desired
};

// Reduce a set of detections to a FrameStats. Boxes below min_score are ignored.
inline FrameStats frame_stats(const std::vector<DetBox>& boxes, float min_score) {
  FrameStats s;
  for (const auto& b : boxes) {
    if (b.score < min_score)
      continue;
    ++s.object_count;
    const float dim = std::min(b.w, b.h);
    if (dim < s.min_object_px)
      s.min_object_px = dim;
    if (b.score < s.min_confidence)
      s.min_confidence = b.score;
  }
  return s;
}

// Cost of a tier, proportional to pixel area, normalised so the smallest tier costs 1.
inline double tier_cost(int size, int base_size) {
  if (base_size <= 0)
    return 1.0;
  const double ratio = static_cast<double>(size) / static_cast<double>(base_size);
  return ratio * ratio;
}

// Highest tier index a stream may use given the shared budget split across
// `active_count` streams. Always allows at least the lowest tier.
inline int budget_allowed_index(int active_count, const std::vector<int>& resolutions,
                                double budget_units) {
  if (resolutions.empty())
    return 0;
  const int n = std::max(1, active_count);
  const double share = budget_units / static_cast<double>(n);
  const int base = resolutions.front();
  int allowed = 0;
  for (std::size_t i = 0; i < resolutions.size(); ++i) {
    if (tier_cost(resolutions[i], base) <= share + 1e-9)
      allowed = static_cast<int>(i);
    else
      break;
  }
  return allowed;
}

// The tier the scene "wants" this frame, moving at most one step from `current`.
inline int desired_tier(int current, const FrameStats& stats, const PolicyConfig& cfg) {
  const int ntiers = static_cast<int>(cfg.resolutions.size());
  if (ntiers == 0)
    return 0;
  const bool want_up = (stats.min_object_px < cfg.min_object_px) ||
                       (stats.object_count > 0 && stats.min_confidence < cfg.confidence_low) ||
                       (stats.object_count >= cfg.density_high);
  if (want_up)
    return std::min(current + 1, ntiers - 1);

  const bool want_down =
      (stats.object_count <= cfg.density_low) &&
      (stats.min_object_px >= cfg.min_object_px * cfg.down_size_factor) &&
      (stats.object_count == 0 || stats.min_confidence >= cfg.confidence_low + cfg.confidence_margin);
  if (want_down)
    return std::max(current - 1, 0);
  return current;
}

// Advance the hysteresis state machine one frame and return the committed tier.
// A change only commits after the desired tier has persisted hysteresis_frames.
inline int select_tier(PolicyState& state, const FrameStats& stats, const PolicyConfig& cfg) {
  const int desired = desired_tier(state.tier_index, stats, cfg);
  if (desired == state.tier_index) {
    state.pending_index = state.tier_index;
    state.pending_count = 0;
    return state.tier_index;
  }
  if (desired == state.pending_index) {
    ++state.pending_count;
  } else {
    state.pending_index = desired;
    state.pending_count = 1;
  }
  if (state.pending_count >= cfg.hysteresis_frames) {
    state.tier_index = desired;
    state.pending_index = desired;
    state.pending_count = 0;
  }
  return state.tier_index;
}

// Content desire (hysteresis-gated) clamped by the shared budget. This is the
// tier the stream should actually run at this frame.
inline int effective_tier(PolicyState& state, const FrameStats& stats, const PolicyConfig& cfg,
                          int active_count, double budget_units) {
  const int content = select_tier(state, stats, cfg);
  const int allowed = budget_allowed_index(active_count, cfg.resolutions, budget_units);
  return std::min(content, allowed);
}

} // namespace adaptive
