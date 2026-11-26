// Copyright 2026 SiMa Technologies, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "pending_results.h"

#include <sstream>
#include <stdexcept>
#include <utility>

namespace pending_results {

PendingResults::PendingResults(int stream_index, std::string model_path, double deadline_s,
                               double first_result_s)
    : stream_index_(stream_index), model_path_(std::move(model_path)), deadline_s_(deadline_s),
      first_result_s_(first_result_s) {}

void PendingResults::observe(std::optional<std::int64_t> frame_id,
                             std::optional<std::int64_t> result_frame_id, double now_s) {
  if (result_frame_id) {
    has_result_ = true;
    last_result_frame_id_ = *result_frame_id;
    pending_since_.reset();
  }
  if (frame_id && !pending_since_) {
    // With an unknown ID, a frame pulled with a result may be that result's own.
    const bool newer = *frame_id >= 0 && last_result_frame_id_ >= 0
                           ? *frame_id > last_result_frame_id_
                           : !result_frame_id;
    if (newer) {
      pending_since_ = now_s;
    }
  }
  const double limit_s = has_result_ ? deadline_s_ : first_result_s_;
  if (pending_since_ && now_s - *pending_since_ > limit_s) {
    std::ostringstream message;
    // Shortest form, as Python's {:g}: 10 prints "10", 0.5 prints "0.5".
    message << "stream " << stream_index_ << ": model " << model_path_ << " returned no result for "
            << limit_s << " s while its decoder sent frames";
    throw std::runtime_error(message.str());
  }
}

} // namespace pending_results
