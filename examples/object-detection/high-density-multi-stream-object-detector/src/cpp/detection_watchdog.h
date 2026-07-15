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

#pragma once

#include <chrono>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <vector>

namespace high_density {

// Verifies liveness per stream. Aggregate progress is deliberately
// insufficient: one healthy stream must not hide a source that never starts or
// later stops producing detections.
class DetectionWatchdog {
public:
  using Clock = std::chrono::steady_clock;
  using TimePoint = Clock::time_point;

  DetectionWatchdog(std::size_t stream_count, std::chrono::milliseconds startup_timeout,
                    std::chrono::milliseconds cadence_timeout, TimePoint start = Clock::now())
      : last_seen_(stream_count), startup_deadline_(start + startup_timeout),
        cadence_timeout_(cadence_timeout) {
    if (stream_count == 0) {
      throw std::invalid_argument("detection watchdog requires at least one stream");
    }
    if (startup_timeout <= std::chrono::milliseconds::zero() ||
        cadence_timeout <= std::chrono::milliseconds::zero()) {
      throw std::invalid_argument("detection watchdog timeouts must be positive");
    }
  }

  void observe(std::size_t stream_index, TimePoint now = Clock::now()) {
    if (stream_index >= last_seen_.size()) {
      throw std::out_of_range("detection watchdog stream index is out of range");
    }
    last_seen_[stream_index] = now;
  }

  [[nodiscard]] bool startup_complete() const {
    for (const auto& seen : last_seen_) {
      if (!seen.has_value()) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] std::vector<std::size_t> expired_streams(TimePoint now = Clock::now()) const {
    std::vector<std::size_t> expired;
    const bool started = startup_complete();
    if (!started && now < startup_deadline_) {
      return expired;
    }

    for (std::size_t index = 0; index < last_seen_.size(); ++index) {
      const auto& seen = last_seen_[index];
      if (!seen.has_value()) {
        expired.push_back(index);
      } else if (started && now - *seen >= cadence_timeout_) {
        expired.push_back(index);
      }
    }
    return expired;
  }

private:
  std::vector<std::optional<TimePoint>> last_seen_;
  TimePoint startup_deadline_;
  std::chrono::milliseconds cadence_timeout_;
};

} // namespace high_density
