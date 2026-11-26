// Copyright 2026 SiMa Technologies, Inc.
// SPDX-License-Identifier: Apache-2.0

// Stops a stream whose model stopped returning results while its decoder still sends frames.
// Mirrors utils/pending_results.py in the Python implementation, so both fail the same way.

#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace pending_results {

// A frame newer than the last result starts a deadline: `first_result_s` for the first result,
// while Core builds the pipeline, then `deadline_s`. A paused camera starts none.
class PendingResults {
public:
  PendingResults(int stream_index, std::string model_path, double deadline_s,
                 double first_result_s);

  // Records one pull round, given the frame ID of the frame and of the result it pulled (none for
  // no pull), then throws when the pending result is late. An ID of -1 is unknown.
  void observe(std::optional<std::int64_t> frame_id, std::optional<std::int64_t> result_frame_id,
               double now_s);

private:
  int stream_index_;
  std::string model_path_;
  double deadline_s_;
  double first_result_s_;
  bool has_result_ = false;
  std::int64_t last_result_frame_id_ = -1;
  std::optional<double> pending_since_;
};

} // namespace pending_results
