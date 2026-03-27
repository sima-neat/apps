#pragma once

#include "config_api.cpp"
#include "model_family_api.cpp"
#include "pipeline_api.cpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <unordered_map>
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

constexpr int decode_pull_timeout_ms() { return 5; }
constexpr std::size_t encoded_for_decode_queue_capacity() { return 20; }
constexpr int decoder_warmup_packets_inflight() { return 64; }
constexpr int decoder_steady_packets_inflight() { return 8; }
constexpr bool video_output_flows_from_source(bool video_enabled, VideoMode mode) {
  return video_enabled && mode == VideoMode::Clean;
}
constexpr bool video_output_flows_from_detector(bool video_enabled, VideoMode mode) {
  return video_enabled && mode == VideoMode::Annotated;
}

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

template <typename T> class BoundedQueue {
public:
  explicit BoundedQueue(std::size_t capacity) : capacity_(capacity) {}

  bool push_drop_oldest(T item, bool keep_latest = false) {
    std::lock_guard<std::mutex> lock(mu_);
    if (closed_ || capacity_ == 0) {
      return false;
    }
    if (queue_.size() >= capacity_) {
      if (keep_latest) {
        return true;
      }
      queue_.pop_front();
    }
    queue_.push_back(std::move(item));
    cv_.notify_one();
    return true;
  }

  bool pop_wait(T& out, int timeout_ms) {
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
    out = std::move(queue_.front());
    queue_.pop_front();
    return true;
  }

  void close() {
    std::lock_guard<std::mutex> lock(mu_);
    closed_ = true;
    cv_.notify_all();
  }

  bool closed() const {
    std::lock_guard<std::mutex> lock(mu_);
    return closed_;
  }

  bool drained() const {
    std::lock_guard<std::mutex> lock(mu_);
    return closed_ && queue_.empty();
  }

private:
  std::size_t capacity_ = 0;
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::deque<T> queue_;
  bool closed_ = false;
};

template <typename T> class PendingFrameStore {
public:
  explicit PendingFrameStore(std::size_t capacity)
      : capacity_(std::max<std::size_t>(1, capacity)) {}

  void put(std::int64_t frame_id, T item) {
    if (frame_id < 0) {
      return;
    }

    std::lock_guard<std::mutex> lock(mu_);
    const auto key = frame_id;
    auto existing = pending_.find(key);
    if (existing != pending_.end()) {
      existing->second = std::move(item);
      return;
    }

    if (pending_.size() >= capacity_) {
      const auto oldest = order_.front();
      order_.pop_front();
      pending_.erase(oldest);
    }

    order_.push_back(key);
    pending_.emplace(key, std::move(item));
  }

  std::optional<T> take(std::int64_t frame_id) {
    if (frame_id < 0) {
      return std::nullopt;
    }

    std::lock_guard<std::mutex> lock(mu_);
    const auto it = pending_.find(frame_id);
    if (it == pending_.end()) {
      return std::nullopt;
    }

    std::optional<T> item{std::move(it->second)};
    pending_.erase(it);
    const auto order_it = std::find(order_.begin(), order_.end(), frame_id);
    if (order_it != order_.end()) {
      order_.erase(order_it);
    }
    return item;
  }

  std::optional<T> take_oldest() {
    std::lock_guard<std::mutex> lock(mu_);
    if (order_.empty()) {
      return std::nullopt;
    }

    const auto frame_id = order_.front();
    order_.pop_front();
    const auto it = pending_.find(frame_id);
    if (it == pending_.end()) {
      return std::nullopt;
    }

    std::optional<T> item{std::move(it->second)};
    pending_.erase(it);
    return item;
  }

private:
  std::size_t capacity_ = 1;
  mutable std::mutex mu_;
  std::unordered_map<std::int64_t, T> pending_;
  std::deque<std::int64_t> order_;
};

template <typename T>
std::optional<T> take_pending_frame_match_or_oldest(PendingFrameStore<T>& store,
                                                    std::int64_t preferred_frame_id) {
  if (preferred_frame_id >= 0) {
    auto exact = store.take(preferred_frame_id);
    if (exact.has_value()) {
      return exact;
    }
  }
  return store.take_oldest();
}

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
    if (!queue_.empty() && !ready_notified_ && !closed_) {
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
