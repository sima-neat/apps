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
#include <cstdint>
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

// Verifies detector liveness without assuming a wall-clock cadence for each
// stream. A shared accelerator may legitimately return streams in bursts, so
// steady-state fairness is measured in completed detector work. Wall time is
// retained only for bounded startup and the distinct case where the entire
// detector stops making progress.
class DetectionWatchdog {
public:
  using Clock = std::chrono::steady_clock;
  using TimePoint = Clock::time_point;

  DetectionWatchdog(std::size_t stream_count, std::size_t priming_observations,
                    std::chrono::milliseconds startup_timeout,
                    std::chrono::milliseconds no_progress_timeout,
                    std::uint64_t max_missed_completions, TimePoint start = Clock::now())
      : priming_counts_(stream_count, 0), last_seen_sequence_(stream_count, 0),
        starvation_latched_(stream_count, false), priming_observations_(priming_observations),
        startup_deadline_(start + startup_timeout), no_progress_timeout_(no_progress_timeout),
        max_missed_completions_(max_missed_completions), last_any_seen_(start) {
    if (stream_count == 0) {
      throw std::invalid_argument("detection watchdog requires at least one stream");
    }
    if (priming_observations == 0) {
      throw std::invalid_argument("detection watchdog requires at least one priming observation");
    }
    if (startup_timeout <= std::chrono::milliseconds::zero() ||
        no_progress_timeout <= std::chrono::milliseconds::zero()) {
      throw std::invalid_argument("detection watchdog timeouts must be positive");
    }
    if (max_missed_completions == 0) {
      throw std::invalid_argument("detection watchdog progress budget must be positive");
    }
  }

  void observe(std::size_t stream_index, TimePoint now = Clock::now()) {
    if (stream_index >= last_seen_sequence_.size()) {
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
    ++total_observations_;
    last_any_seen_ = now;
    if (running_) {
      last_seen_sequence_[stream_index] = total_observations_;
      // Latch the first violated progress boundary at the completion that
      // crosses it. A starved stream may return later in the same drain batch,
      // but that recovery must not erase the already-observed violation.
      for (std::size_t index = 0; index < last_seen_sequence_.size(); ++index) {
        if (total_observations_ - last_seen_sequence_[index] > max_missed_completions_) {
          starvation_latched_[index] = true;
        }
      }
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
      // Priming is intentionally excluded from starvation accounting. All
      // streams receive the same steady-state baseline even when startup was
      // staggered or returned in large per-stream bursts.
      for (auto& sequence : last_seen_sequence_) {
        sequence = total_observations_;
      }
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
      if (starvation_latched_[index]) {
        failure.streams.push_back(index);
      }
    }
    return failure.streams.empty() ? DetectionFailure{} : failure;
  }

private:
  std::vector<std::size_t> priming_counts_;
  std::vector<std::uint64_t> last_seen_sequence_;
  std::vector<bool> starvation_latched_;
  std::vector<std::size_t> startup_failure_streams_;
  std::size_t priming_observations_;
  std::size_t primed_streams_ = 0;
  std::uint64_t total_observations_ = 0;
  TimePoint startup_deadline_;
  std::chrono::milliseconds no_progress_timeout_;
  std::uint64_t max_missed_completions_;
  TimePoint last_any_seen_;
  bool running_ = false;
  bool startup_expired_latched_ = false;
  bool global_stall_latched_ = false;
};

} // namespace high_density
