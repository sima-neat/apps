// Copyright 2026 SiMa Technologies, Inc.
// SPDX-License-Identifier: Apache-2.0

// Stops a stream whose model stopped returning results while its camera still sends frames.
// Mirrors utils/pending_results.py in the Python implementation, so both fail the same way.

#pragma once

#include <optional>
#include <string>

namespace pending_results {

// The pending-result deadline of one stream. From the stream's first result on, a frame that
// arrives after the last result has a result pending. When no result comes within `deadline_s` of
// the first such frame, check() throws with the stream and its model. Before the first result the
// decoder may still wait for a keyframe, so nothing is pending. A camera that pauses sends no
// frames, so nothing is pending and the stream waits without a limit.
class PendingResults {
public:
  PendingResults(int stream_index, std::string model_path, double deadline_s);

  void frame_arrived(double now_s);
  void result_arrived();
  void check(double now_s) const;

private:
  int stream_index_;
  std::string model_path_;
  double deadline_s_;
  bool started_ = false;
  std::optional<double> pending_since_;
};

} // namespace pending_results
