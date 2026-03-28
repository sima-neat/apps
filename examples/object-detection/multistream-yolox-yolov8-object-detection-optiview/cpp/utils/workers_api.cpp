#pragma once

#include "config_api.cpp"
#include "model_family_api.cpp"
#include "pipeline_api.cpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

namespace multistream_yolox_yolov8_optiview {

struct DetectorRuntimeKey {
  ModelFamily family = ModelFamily::Auto;
  int width = 0;
  int height = 0;
};

struct StreamProbeSpec {
  ModelFamily family = ModelFamily::Auto;
  RtspProbe probe;
};

inline bool startup_trace_enabled_from_env() {
  const char* raw = std::getenv("SIMA_OPTIVIEW_STARTUP_TRACE");
  if (raw == nullptr) {
    return false;
  }
  std::string lowered{raw};
  std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  return lowered == "1" || lowered == "true" || lowered == "yes" || lowered == "on";
}

inline int source_startup_pull_timeout_ms() {
  return 50000;
}

inline int source_pull_timeout_ms() {
  return 10000;
}

inline double source_startup_stagger_s() {
  return 0.5;
}

std::string format_video_build_error(int stream_index, VideoMode video_mode,
                                     const std::string& detail);

std::vector<DetectorRuntimeKey> collect_detector_runtime_keys(
    const std::vector<StreamProbeSpec>& streams);

class ReadyStreamQueue {
public:
  void push(int stream_index) {
    std::lock_guard<std::mutex> lock(mu_);
    if (closed_) {
      return;
    }
    queue_.push_back(stream_index);
    cv_.notify_one();
  }

  bool pop_wait(int& out, int timeout_ms) {
    std::unique_lock<std::mutex> lock(mu_);
    const auto ready = [&] { return closed_ || !queue_.empty(); };
    if (timeout_ms < 0) {
      cv_.wait(lock, ready);
    } else if (!cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), ready)) {
      return false;
    }
    if (queue_.empty()) {
      return false;
    }
    out = queue_.front();
    queue_.pop_front();
    return true;
  }

  void close() {
    std::lock_guard<std::mutex> lock(mu_);
    closed_ = true;
    cv_.notify_all();
  }

private:
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<int> queue_;
  bool closed_ = false;
};

template <typename T> class LatestFrameMailbox {
public:
  LatestFrameMailbox(int stream_index, std::size_t capacity)
      : stream_index_(stream_index), capacity_(std::max<std::size_t>(1, capacity)) {}

  int push(T item, ReadyStreamQueue& ready_queue) {
    std::lock_guard<std::mutex> lock(mu_);
    if (closed_) {
      return 0;
    }
    int dropped = 0;
    while (queue_.size() >= capacity_) {
      queue_.pop_front();
      ++dropped;
    }
    queue_.push_back(std::move(item));
    if (!in_flight_ && !ready_notified_) {
      ready_queue.push(stream_index_);
      ready_notified_ = true;
    }
    return dropped;
  }

  bool take_for_processing(T& out) {
    std::lock_guard<std::mutex> lock(mu_);
    if (queue_.empty()) {
      ready_notified_ = false;
      return false;
    }
    out = std::move(queue_.front());
    queue_.pop_front();
    in_flight_ = true;
    ready_notified_ = false;
    return true;
  }

  void complete(ReadyStreamQueue& ready_queue) {
    std::lock_guard<std::mutex> lock(mu_);
    in_flight_ = false;
    if (!queue_.empty() && !ready_notified_) {
      ready_queue.push(stream_index_);
      ready_notified_ = true;
    }
  }

  void close() {
    std::lock_guard<std::mutex> lock(mu_);
    closed_ = true;
  }

  bool drained() const {
    std::lock_guard<std::mutex> lock(mu_);
    return closed_ && queue_.empty() && !in_flight_;
  }

private:
  int stream_index_ = 0;
  std::size_t capacity_ = 1;
  mutable std::mutex mu_;
  std::deque<T> queue_;
  bool closed_ = false;
  bool ready_notified_ = false;
  bool in_flight_ = false;
};

int run_app(const AppConfig& cfg, ModelFamily family);

} // namespace multistream_yolox_yolov8_optiview
