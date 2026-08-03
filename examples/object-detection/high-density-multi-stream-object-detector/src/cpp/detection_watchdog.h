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

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace high_density {

enum class DetectionFailureKind {
  None,
  Startup,
  StreamStarvation,
  GlobalStall,
};

struct DetectionFailure {
  DetectionFailureKind kind = DetectionFailureKind::None;
  std::vector<std::size_t> streams;

  [[nodiscard]] explicit operator bool() const {
    return kind != DetectionFailureKind::None;
  }
};

// Verifies detector liveness without assuming round-robin scheduling. A shared
// accelerator may legitimately return streams in unequal bursts, so aggregate
// completions are not a valid per-stream starvation clock. Instead, startup,
// individual stream progress, and detector-wide progress have independent,
// configurable deadlines.
class DetectionWatchdog {
public:
  using Clock = std::chrono::steady_clock;
  using TimePoint = Clock::time_point;

  DetectionWatchdog(std::size_t stream_count, std::size_t priming_observations,
                    std::chrono::milliseconds startup_timeout,
                    std::chrono::milliseconds stream_timeout,
                    std::chrono::milliseconds no_progress_timeout, TimePoint start = Clock::now())
      : priming_counts_(stream_count, 0), last_seen_(stream_count, start),
        starvation_latched_(stream_count, false), priming_observations_(priming_observations),
        startup_deadline_(start + startup_timeout), stream_timeout_(stream_timeout),
        no_progress_timeout_(no_progress_timeout), last_any_seen_(start) {
    if (stream_count == 0) {
      throw std::invalid_argument("detection watchdog requires at least one stream");
    }
    if (priming_observations == 0) {
      throw std::invalid_argument("detection watchdog requires at least one priming observation");
    }
    if (startup_timeout <= std::chrono::milliseconds::zero() ||
        stream_timeout <= std::chrono::milliseconds::zero() ||
        no_progress_timeout <= std::chrono::milliseconds::zero()) {
      throw std::invalid_argument("detection watchdog timeouts must be positive");
    }
  }

  void observe(std::size_t stream_index, TimePoint now = Clock::now()) {
    if (stream_index >= last_seen_.size()) {
      throw std::out_of_range("detection watchdog stream index is out of range");
    }

    if (!running_ && !startup_expired_latched_ && now >= startup_deadline_) {
      startup_expired_latched_ = true;
      // Preserve the streams that had not completed priming before the
      // deadline. A late result in the same drain batch must not erase an
      // already-expired startup interval.
      for (std::size_t index = 0; index < priming_counts_.size(); ++index) {
        if (priming_counts_[index] < priming_observations_) {
          startup_failure_streams_.push_back(index);
        }
      }
    }

    // Preserve an already-expired detector-wide interval before recovered
    // progress advances the timestamp. Otherwise a late result arriving
    // between periodic checks could erase the failure.
    if (now - last_any_seen_ >= no_progress_timeout_) {
      global_stall_latched_ = true;
    }
    last_any_seen_ = now;
    if (running_) {
      // Check the returning stream before advancing its timestamp. This keeps
      // the completion path constant-time while ensuring that recovery later
      // in a drain batch cannot erase an already-crossed deadline. Other
      // expired streams are found by check().
      if (now - last_seen_[stream_index] >= stream_timeout_) {
        starvation_latched_[stream_index] = true;
      }
      last_seen_[stream_index] = now;
      return;
    }

    if (startup_expired_latched_) {
      return;
    }

    if (priming_counts_[stream_index] < priming_observations_) {
      ++priming_counts_[stream_index];
      if (priming_counts_[stream_index] == priming_observations_) {
        ++primed_streams_;
      }
    }
    if (primed_streams_ == priming_counts_.size()) {
      running_ = true;
      // Staggered startup is excluded from steady-state liveness accounting.
      // Every stream receives the same deadline baseline when priming ends.
      std::fill(last_seen_.begin(), last_seen_.end(), now);
    }
  }

  [[nodiscard]] bool startup_complete() const {
    return running_;
  }

  [[nodiscard]] DetectionFailure check(TimePoint now = Clock::now()) const {
    if (global_stall_latched_ || now - last_any_seen_ >= no_progress_timeout_) {
      return {DetectionFailureKind::GlobalStall, {}};
    }

    if (startup_expired_latched_) {
      return {DetectionFailureKind::Startup, startup_failure_streams_};
    }

    if (!running_) {
      if (now < startup_deadline_) {
        return {};
      }
      DetectionFailure failure{DetectionFailureKind::Startup, {}};
      for (std::size_t index = 0; index < priming_counts_.size(); ++index) {
        if (priming_counts_[index] < priming_observations_) {
          failure.streams.push_back(index);
        }
      }
      return failure;
    }

    DetectionFailure failure{DetectionFailureKind::StreamStarvation, {}};
    for (std::size_t index = 0; index < starvation_latched_.size(); ++index) {
      if (starvation_latched_[index] || now - last_seen_[index] >= stream_timeout_) {
        failure.streams.push_back(index);
      }
    }
    return failure.streams.empty() ? DetectionFailure{} : failure;
  }

private:
  std::vector<std::size_t> priming_counts_;
  std::vector<TimePoint> last_seen_;
  std::vector<bool> starvation_latched_;
  std::vector<std::size_t> startup_failure_streams_;
  std::size_t priming_observations_;
  std::size_t primed_streams_ = 0;
  TimePoint startup_deadline_;
  std::chrono::milliseconds stream_timeout_;
  std::chrono::milliseconds no_progress_timeout_;
  TimePoint last_any_seen_;
  bool running_ = false;
  bool startup_expired_latched_ = false;
  bool global_stall_latched_ = false;
};

} // namespace high_density
