// Copyright 2026 SiMa Technologies, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "pending_results.h"

#include <sstream>
#include <stdexcept>
#include <utility>

namespace pending_results {

PendingResults::PendingResults(int stream_index, std::string model_path, double deadline_s)
    : stream_index_(stream_index), model_path_(std::move(model_path)), deadline_s_(deadline_s) {}

void PendingResults::frame_arrived(double now_s) {
  if (started_ && !pending_since_) {
    pending_since_ = now_s;
  }
}

void PendingResults::result_arrived() {
  started_ = true;
  pending_since_.reset();
}

void PendingResults::check(double now_s) const {
  if (pending_since_ && now_s - *pending_since_ > deadline_s_) {
    std::ostringstream message;
    // Shortest form, as Python's {:g}: 10 prints "10", 0.5 prints "0.5".
    message << "stream " << stream_index_ << ": model " << model_path_ << " returned no result for "
            << deadline_s_ << " s while its camera sent frames";
    throw std::runtime_error(message.str());
  }
}

} // namespace pending_results
