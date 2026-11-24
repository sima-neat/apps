// Copyright 2026 SiMa Technologies, Inc.
// SPDX-License-Identifier: Apache-2.0

// Stops a stream whose model stopped returning results while its decoder still sends frames.
// Mirrors utils/pending_results.py in the Python implementation, so both fail the same way.

#pragma once

#include <optional>
#include <string>

namespace pending_results {

// The pending-result deadline of one stream. A decoded frame that arrives after the stream's last
// result has a result pending. When no result comes within the deadline of the first such frame,
// observe() throws with the stream and its model. The first result gets `first_result_s`, because
// Core builds the rest of the pipeline when the first frame reaches it. Every later result gets
// `deadline_s`. A camera that pauses sends no frames, so nothing is pending and the stream waits
// without a limit.
class PendingResults {
public:
  PendingResults(int stream_index, std::string model_path, double deadline_s,
                 double first_result_s);

  // Records one pull round of the stream, then throws when its pending result is late. A frame
  // that arrives with a result in the same round starts no deadline: the result may be its own.
  void observe(bool frame_arrived, bool result_arrived, double now_s);

private:
  int stream_index_;
  std::string model_path_;
  double deadline_s_;
  double first_result_s_;
  bool has_result_ = false;
  std::optional<double> pending_since_;
};

} // namespace pending_results
