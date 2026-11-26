// Copyright 2026 SiMa Technologies, Inc.
// SPDX-License-Identifier: Apache-2.0

// Stops a stream whose model stopped returning results while its decoder still sends frames.
// Mirrors utils/pending_results.py in the Python implementation, so both fail the same way.

#pragma once

#include <optional>
#include <string>

namespace pending_results {

// A frame that arrives after the last result starts a deadline: `first_result_s` for the first
// result, while Core builds the pipeline, then `deadline_s`. A paused camera starts none.
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
