// Copyright 2026 SiMa Technologies, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <utility>

namespace app16 {

/**
 * Size the CPU-owned encoded-AU queue for both the intentional presentation
 * delay and a bounded RTSP catch-up burst.
 *
 * GStreamer's RTSP ingress can legally release roughly one second of queued
 * media at once after a short scheduling or network pause. Counting only the
 * nominal delay plus one source-second leaves no admission margin: at 10 fps,
 * a 400 ms delay produced the former 4 + 10 == 14 frame limit and the 15th AU
 * stopped the whole application. Encoded AUs live in ordinary CPU memory;
 * the separate byte limit remains the hard memory bound.
 */
inline std::size_t encoded_delivery_frame_capacity(int fps, int delay_ms) {
  if (fps <= 0 || delay_ms < 0) {
    return 0;
  }
  constexpr std::size_t kMinimumCatchUpFrames = 64;
  constexpr std::size_t kCatchUpSeconds = 2;
  const auto frame_rate = static_cast<std::size_t>(fps);
  const auto delay_frames = (static_cast<std::size_t>(delay_ms) * frame_rate + 999U) / 1000U;
  const auto catch_up_frames = std::max(kMinimumCatchUpFrames, kCatchUpSeconds * frame_rate);
  return delay_frames + catch_up_frames;
}

/**
 * CPU-owned delay and metadata queues for one Insight channel.
 *
 * The caller provides synchronization. Keeping this type independent of Neat
 * makes the ordering and overflow contract directly unit-testable.
 */
template <typename Frame> class EncodedDeliveryQueue {
public:
  using Clock = std::chrono::steady_clock;
  using TimePoint = Clock::time_point;

  enum class MetadataEnqueueStatus {
    Accepted,
    AcceptedDroppedOldest,
    UnresolvedEpoch,
  };

  explicit EncodedDeliveryQueue(std::size_t max_frames = 512, std::size_t max_metadata = 128,
                                std::size_t max_frame_bytes = 16U * 1024U * 1024U,
                                std::int64_t pts_reset_threshold_ns = 1000000000LL,
                                std::size_t max_frame_history = 0)
      : max_frames_(max_frames), max_metadata_(max_metadata), max_frame_bytes_(max_frame_bytes),
        pts_reset_threshold_ns_(pts_reset_threshold_ns),
        max_frame_history_(max_frame_history > 0 ? max_frame_history
                                                 : std::max<std::size_t>(256, max_frames * 4)) {}

  /** Reject a new encoded AU when full; never discard part of an H.264 stream. */
  bool enqueue_frame(Frame frame, TimePoint due, std::size_t payload_bytes = 0,
                     std::int64_t pts_ns = -1) {
    if (max_frames_ == 0 || frames_.size() >= max_frames_ ||
        (max_frame_bytes_ > 0 &&
         payload_bytes > max_frame_bytes_ - std::min(frame_bytes_, max_frame_bytes_))) {
      return false;
    }
    observe_pts(pts_ns, last_frame_pts_ns_, frame_epoch_, frame_epoch_resets_);
    frames_.push_back(DelayedFrame{std::move(frame), due, payload_bytes, pts_ns, frame_epoch_});
    if (pts_ns >= 0 && max_frame_history_ > 0) {
      frame_history_.push_back(FrameEpochStamp{pts_ns, frame_epoch_});
      while (frame_history_.size() > max_frame_history_) {
        frame_history_.pop_front();
      }
    }
    frame_bytes_ += payload_bytes;
    frame_count_high_water_ = std::max(frame_count_high_water_, frames_.size());
    frame_bytes_high_water_ = std::max(frame_bytes_high_water_, frame_bytes_);
    return true;
  }

  /** Pop the oldest AU only after its synchronization delay has elapsed. */
  bool pop_due_frame(TimePoint now, Frame& frame, std::uint64_t* epoch = nullptr) {
    if (frames_.empty() || frames_.front().due > now) {
      return false;
    }
    frame = std::move(frames_.front().frame);
    if (epoch) {
      *epoch = frames_.front().epoch;
    }
    frame_bytes_ -= std::min(frame_bytes_, frames_.front().payload_bytes);
    frames_.pop_front();
    return true;
  }

  /**
   * Copy the oldest due AU without consuming it.
   *
   * A nonblocking sender can use this to try one channel and move on when
   * that channel is backpressured. The caller removes the AU with
   * pop_due_frame() only after the sender accepted it, preserving H.264 order
   * without introducing head-of-line blocking across channels.
   */
  bool peek_due_frame(TimePoint now, Frame& frame, std::uint64_t* epoch = nullptr) const {
    if (frames_.empty() || frames_.front().due > now) {
      return false;
    }
    frame = frames_.front().frame;
    if (epoch) {
      *epoch = frames_.front().epoch;
    }
    return true;
  }

  /**
   * Queue detection metadata in PTS order. If inference ever outruns video,
   * discard the oldest metadata object and report that event to the caller.
   */
  MetadataEnqueueStatus enqueue_metadata_status(std::int64_t pts_ns, std::string payload) {
    bool dropped_oldest = false;
    if (max_metadata_ == 0) {
      return MetadataEnqueueStatus::AcceptedDroppedOldest;
    }

    const auto resolved_epoch = resolve_metadata_epoch(pts_ns);
    if (!resolved_epoch.has_value()) {
      ++metadata_epoch_unresolved_;
      return MetadataEnqueueStatus::UnresolvedEpoch;
    }
    const std::uint64_t epoch = *resolved_epoch;
    if (epoch > metadata_epoch_) {
      metadata_epoch_resets_ += epoch - metadata_epoch_;
      metadata_epoch_ = epoch;
    }

    const auto insertion = std::upper_bound(
        metadata_.begin(), metadata_.end(), PendingMetadata{pts_ns, {}, epoch},
        [](const PendingMetadata& lhs, const PendingMetadata& rhs) {
          return lhs.epoch < rhs.epoch || (lhs.epoch == rhs.epoch && lhs.pts_ns < rhs.pts_ns);
        });
    metadata_.insert(insertion, PendingMetadata{pts_ns, std::move(payload), epoch});
    if (metadata_.size() > max_metadata_) {
      metadata_.pop_front();
      dropped_oldest = true;
    }
    return dropped_oldest ? MetadataEnqueueStatus::AcceptedDroppedOldest
                          : MetadataEnqueueStatus::Accepted;
  }

  /**
   * Compatibility helper: true means metadata was not admitted cleanly.
   * Call enqueue_metadata_status() when the exact reason matters.
   */
  bool enqueue_metadata(std::int64_t pts_ns, std::string payload) {
    return enqueue_metadata_status(pts_ns, std::move(payload)) != MetadataEnqueueStatus::Accepted;
  }

  /**
   * Return the newest detection not newer than the video AU being sent.
   * Older eligible detections are intentionally coalesced so Insight remains
   * current rather than drawing stale intermediate results.
   */
  std::string take_metadata_through(std::uint64_t video_epoch, std::int64_t video_pts_ns,
                                    std::size_t* stale_epoch_count = nullptr) {
    std::string payload;
    std::size_t stale = 0;
    while (!metadata_.empty() && metadata_.front().epoch < video_epoch) {
      metadata_.pop_front();
      ++stale;
    }
    while (!metadata_.empty() && metadata_.front().epoch == video_epoch &&
           metadata_.front().pts_ns <= video_pts_ns) {
      payload = std::move(metadata_.front().payload);
      metadata_.pop_front();
    }
    if (stale_epoch_count) {
      *stale_epoch_count = stale;
    }
    return payload;
  }

  /** Backward-compatible single-epoch helper used by small unit fixtures. */
  std::string take_metadata_through(std::int64_t video_pts_ns) {
    return take_metadata_through(frame_epoch_, video_pts_ns);
  }

  [[nodiscard]] std::size_t frame_count() const {
    return frames_.size();
  }
  [[nodiscard]] std::size_t frame_bytes() const {
    return frame_bytes_;
  }
  [[nodiscard]] std::size_t frame_count_high_water() const {
    return frame_count_high_water_;
  }
  [[nodiscard]] std::size_t frame_bytes_high_water() const {
    return frame_bytes_high_water_;
  }
  [[nodiscard]] std::size_t max_frames() const {
    return max_frames_;
  }
  [[nodiscard]] std::size_t max_frame_bytes() const {
    return max_frame_bytes_;
  }
  [[nodiscard]] std::size_t metadata_count() const {
    return metadata_.size();
  }
  [[nodiscard]] std::uint64_t frame_epoch() const {
    return frame_epoch_;
  }
  [[nodiscard]] std::uint64_t metadata_epoch() const {
    return metadata_epoch_;
  }
  [[nodiscard]] std::uint64_t frame_epoch_resets() const {
    return frame_epoch_resets_;
  }
  [[nodiscard]] std::uint64_t metadata_epoch_resets() const {
    return metadata_epoch_resets_;
  }
  [[nodiscard]] std::uint64_t metadata_epoch_unresolved() const {
    return metadata_epoch_unresolved_;
  }
  [[nodiscard]] std::size_t frame_history_count() const {
    return frame_history_.size();
  }
  /**
   * Milliseconds until the oldest encoded AU becomes due.
   *
   * A negative value means the dispatcher is behind rather than the delay
   * window legitimately retaining the queue.  This is intentionally exposed
   * for low-rate failure diagnostics; callers must still provide the same
   * synchronization used for the other queue operations.
   */
  [[nodiscard]] std::int64_t oldest_frame_due_in_ms(TimePoint now) const {
    if (frames_.empty()) {
      return 0;
    }
    return std::chrono::duration_cast<std::chrono::milliseconds>(frames_.front().due - now).count();
  }
  [[nodiscard]] std::size_t pending_count() const {
    return frame_count() + metadata_count();
  }

private:
  struct DelayedFrame {
    Frame frame;
    TimePoint due;
    std::size_t payload_bytes = 0;
    std::int64_t pts_ns = -1;
    std::uint64_t epoch = 0;
  };

  struct PendingMetadata {
    std::int64_t pts_ns = -1;
    std::string payload;
    std::uint64_t epoch = 0;
  };

  struct FrameEpochStamp {
    std::int64_t pts_ns = -1;
    std::uint64_t epoch = 0;
  };

  std::optional<std::uint64_t> resolve_metadata_epoch(std::int64_t pts_ns) const {
    if (pts_ns < 0) {
      return std::nullopt;
    }

    std::optional<std::uint64_t> epoch;
    for (auto it = frame_history_.rbegin(); it != frame_history_.rend(); ++it) {
      if (it->pts_ns != pts_ns) {
        continue;
      }
      if (!epoch.has_value()) {
        epoch = it->epoch;
      } else if (*epoch != it->epoch) {
        // The same PTS is still retained from more than one source epoch.
        // Without a sequence token, guessing could draw a valid detection on
        // the wrong loop/reconnect. Drop it instead of poisoning later epochs.
        return std::nullopt;
      }
    }
    if (epoch.has_value()) {
      return epoch;
    }

    // Before the first discontinuity there is no epoch ambiguity. This also
    // keeps small unit fixtures useful when they exercise metadata ordering
    // without constructing encoded frames. After a reset, exact retained AU
    // history is required; a late result outside that bounded history is
    // intentionally rejected rather than inferred from completion order.
    if (frame_epoch_ == 0) {
      return 0;
    }
    return std::nullopt;
  }

  void observe_pts(std::int64_t pts_ns, std::int64_t& last_pts_ns, std::uint64_t& epoch,
                   std::uint64_t& resets) {
    if (pts_ns < 0) {
      return;
    }
    if (last_pts_ns >= 0 && pts_ns < last_pts_ns &&
        last_pts_ns - pts_ns > pts_reset_threshold_ns_) {
      ++epoch;
      ++resets;
      last_pts_ns = pts_ns;
      return;
    }
    last_pts_ns = std::max(last_pts_ns, pts_ns);
  }

  std::size_t max_frames_;
  std::size_t max_metadata_;
  std::size_t max_frame_bytes_;
  std::int64_t pts_reset_threshold_ns_;
  std::size_t max_frame_history_;
  std::size_t frame_bytes_ = 0;
  std::size_t frame_count_high_water_ = 0;
  std::size_t frame_bytes_high_water_ = 0;
  std::int64_t last_frame_pts_ns_ = -1;
  std::uint64_t frame_epoch_ = 0;
  std::uint64_t metadata_epoch_ = 0;
  std::uint64_t frame_epoch_resets_ = 0;
  std::uint64_t metadata_epoch_resets_ = 0;
  std::uint64_t metadata_epoch_unresolved_ = 0;
  std::deque<DelayedFrame> frames_;
  std::deque<PendingMetadata> metadata_;
  std::deque<FrameEpochStamp> frame_history_;
};

} // namespace app16
