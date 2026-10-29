#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace high_density::testing {

class ReceivedMetadataTracker {
public:
  using Clock = std::chrono::steady_clock;
  using TimePoint = Clock::time_point;

  ReceivedMetadataTracker(std::size_t stream_count, std::uint64_t warmup_per_stream,
                          std::uint64_t target_frames,
                          std::chrono::milliseconds initial_progress_timeout,
                          std::chrono::milliseconds stream_progress_timeout,
                          TimePoint start = Clock::now())
      : warmup_per_stream_(warmup_per_stream), target_frames_(target_frames),
        initial_progress_timeout_(initial_progress_timeout),
        stream_progress_timeout_(stream_progress_timeout), seen_(stream_count),
        warmup_frames_(stream_count), measured_frames_(stream_count),
        useful_detection_(stream_count), last_progress_(stream_count, start),
        measurement_started_(warmup_per_stream == 0), measurement_start_(start),
        measurement_end_(start) {
    if (stream_count == 0 || target_frames == 0) {
      throw std::invalid_argument("metadata tracker requires streams and a frame target");
    }
    if (initial_progress_timeout <= std::chrono::milliseconds::zero() ||
        stream_progress_timeout <= std::chrono::milliseconds::zero()) {
      throw std::invalid_argument("metadata tracker timeouts must be positive");
    }
  }

  bool observe(std::size_t stream_index, const std::string& frame_id, bool has_objects,
               TimePoint now = Clock::now()) {
    if (stream_index >= seen_.size()) {
      throw std::out_of_range("metadata stream index is out of range");
    }
    if (frame_id.empty() || complete_ || !seen_[stream_index].insert(frame_id).second) {
      return false;
    }

    last_progress_[stream_index] = now;
    useful_detection_[stream_index] = useful_detection_[stream_index] || has_objects;
    if (!measurement_started_) {
      ++warmup_frames_[stream_index];
      return false;
    }

    ++measured_frames_[stream_index];
    ++total_measured_;
    if (total_measured_ == target_frames_) {
      complete_ = true;
      measurement_end_ = now;
    }
    return complete_;
  }

  [[nodiscard]] bool warmup_complete() const {
    return std::all_of(warmup_frames_.begin(), warmup_frames_.end(),
                       [this](std::uint64_t count) { return count >= warmup_per_stream_; });
  }

  void start_measurement(TimePoint now = Clock::now()) {
    if (measurement_started_) {
      return;
    }
    if (!warmup_complete()) {
      throw std::logic_error("metadata measurement cannot start before warmup completes");
    }
    measurement_started_ = true;
    measurement_start_ = now;
    std::fill(last_progress_.begin(), last_progress_.end(), now);
  }

  [[nodiscard]] std::vector<std::size_t> stalled_streams(TimePoint now = Clock::now()) const {
    const auto timeout =
        measurement_started_ ? stream_progress_timeout_ : initial_progress_timeout_;
    std::vector<std::size_t> stalled;
    for (std::size_t index = 0; index < last_progress_.size(); ++index) {
      if (now - last_progress_[index] >= timeout) {
        stalled.push_back(index);
      }
    }
    return stalled;
  }

  [[nodiscard]] bool measurement_started() const {
    return measurement_started_;
  }
  [[nodiscard]] bool complete() const {
    return complete_;
  }
  [[nodiscard]] std::uint64_t total_measured() const {
    return total_measured_;
  }
  [[nodiscard]] const std::vector<std::uint64_t>& warmup_frames() const {
    return warmup_frames_;
  }
  [[nodiscard]] const std::vector<std::uint64_t>& measured_frames() const {
    return measured_frames_;
  }
  [[nodiscard]] const std::vector<bool>& useful_detection() const {
    return useful_detection_;
  }
  [[nodiscard]] double elapsed_seconds() const {
    if (!complete_) {
      return 0.0;
    }
    return std::chrono::duration<double>(measurement_end_ - measurement_start_).count();
  }

private:
  std::uint64_t warmup_per_stream_;
  std::uint64_t target_frames_;
  std::chrono::milliseconds initial_progress_timeout_;
  std::chrono::milliseconds stream_progress_timeout_;
  std::vector<std::unordered_set<std::string>> seen_;
  std::vector<std::uint64_t> warmup_frames_;
  std::vector<std::uint64_t> measured_frames_;
  std::vector<bool> useful_detection_;
  std::vector<TimePoint> last_progress_;
  bool measurement_started_ = false;
  bool complete_ = false;
  std::uint64_t total_measured_ = 0;
  TimePoint measurement_start_;
  TimePoint measurement_end_;
};

} // namespace high_density::testing
