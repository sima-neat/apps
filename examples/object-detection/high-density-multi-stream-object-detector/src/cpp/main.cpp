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

#include "neat.h"
#include "neat/models.h"
#include "neat/node_groups.h"
#include "neat/nodes.h"
#include "support/object_detection/obj_detection_utils.h"
#include "support/runtime/config_utils.h"
#include "support/runtime/example_utils.h"

#include <nodes/groups/VideoSender.h>
#include <nodes/io/MetadataSender.h>
#include <pipeline/LatestByStreamFrameTap.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <csignal>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

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
 * keeps the ordering and overflow contract local to the application.
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

namespace fs = std::filesystem;

namespace {

constexpr int kStreamLimit = 80;
constexpr int kDefaultInitialDetectionTimeoutMs = 30000;
constexpr int kDetectorResultTimeoutMs = 5000;
constexpr int kDefaultQueueDepth = 4;
constexpr int kDefaultInternalQueueDepth = 1;
constexpr int kDefaultMaxInflightPerStream = 4;
// Complete MLA work asynchronously so a busy model stage cannot hold the
// fused fan-in callback and starve a subset of decoder streams.
constexpr bool kInferenceAsync = true;
constexpr int kDefaultDecoderBuffers = 16;
constexpr int kDefaultDecoderInputBuffers = 2;
constexpr int kAllInsightStreams = -1;
// This is both the per-channel delayed-AU byte budget and the largest AU the
// VideoSender input will admit. Keeping one limit prevents a frame from being
// accepted by the delay queue and rejected later by the sender appsrc.
constexpr std::size_t kEncodedQueueMaxBytesPerStream = 16U * 1024U * 1024U;

volatile std::sig_atomic_t g_stop_requested = 0;
std::atomic<bool> g_runtime_stop_requested{false};

void request_stop(int) {
  g_stop_requested = 1;
}

enum class FanInPolicy {
  Latest,
  EveryFrame,
};

struct AppConfig {
  std::string model_path;
  std::string decode_type = "yolo26";
  fs::path labels_path;
  std::vector<std::string> rtsp_urls;
  int workers = 1;
  int queue_depth = kDefaultQueueDepth;
  int internal_queue_depth = kDefaultInternalQueueDepth;
  int max_inflight_per_stream = kDefaultMaxInflightPerStream;
  FanInPolicy fan_in_policy = FanInPolicy::Latest;
  int decoder_buffers = kDefaultDecoderBuffers;
  int decoder_input_buffers = kDefaultDecoderInputBuffers;
  std::string decoder_tuning = "auto";
  int input_width = 0;
  int input_height = 0;
  int input_fps = 0;
  int latency_ms = 100;
  bool tcp = true;
  bool skip_rtsp_probe = false;
  bool rtsp_drop_on_latency = false;
  double min_score = 0.55;
  double nms_iou = 0.60;
  int max_detections = 50;
  bool bypass_model = false;
  bool bypass_preprocess = false;
  bool copy_input = false;
  bool profile = false;
  int warmup_frames = 30;
  int initial_detection_timeout_ms = kDefaultInitialDetectionTimeoutMs;
  std::string insight_host = "127.0.0.1";
  int video_port_base = 9000;
  int metadata_port_base = 9100;
  int insight_visible_streams = kAllInsightStreams;
  bool video_enabled = true;
  int insight_video_sync_delay_ms = 400;
};

void apply_h264_caps(simaai::neat::nodes::groups::RtspDecodedInputOptions& opt, int width,
                     int height, int fps, int& width_out, int& height_out, int& fps_out) {
  if (width > 0 && height > 0) {
    opt.fallback_h264_width = width;
    opt.fallback_h264_height = height;
    opt.output_caps.width = width;
    opt.output_caps.height = height;
    width_out = width;
    height_out = height;
  }
  if (fps > 0) {
    opt.fallback_h264_fps = fps;
    opt.output_caps.fps = fps;
    fps_out = fps;
  }
}

simaai::neat::BoxDecodeType parse_box_decode_type(const std::string& token) {
  if (token == "yolo26" || token == "yolov26")
    return simaai::neat::BoxDecodeType::YoloV26;
  if (token == "yolov8")
    return simaai::neat::BoxDecodeType::YoloV8;
  throw std::runtime_error("model.decode_type must be one of: yolo26, yolov8");
}

std::string normalize_decoder_tuning(std::string token) {
  std::replace(token.begin(), token.end(), '_', '-');
  std::transform(token.begin(), token.end(), token.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (token.empty() || token == "auto")
    return "auto";
  if (token == "default")
    return "default";
  if (token == "low-memory" || token == "memory")
    return "low-memory";
  if (token == "throughput-low-latency" || token == "throughput" ||
      token == "low-latency-throughput")
    return "throughput-low-latency";
  throw std::runtime_error(
      "input.decoder_tuning must be one of: auto, default, low-memory, throughput-low-latency");
}

FanInPolicy parse_fan_in_policy(const std::string& token) {
  if (token == "latest")
    return FanInPolicy::Latest;
  if (token == "every_frame")
    return FanInPolicy::EveryFrame;
  throw std::runtime_error("inference.fan_in_policy must be one of: latest, every_frame");
}

const char* fan_in_policy_name(FanInPolicy policy) {
  switch (policy) {
  case FanInPolicy::Latest:
    return "latest";
  case FanInPolicy::EveryFrame:
    return "every_frame";
  }
  return "latest";
}

simaai::neat::GraphLinkPolicy graph_link_policy(FanInPolicy policy) {
  switch (policy) {
  case FanInPolicy::Latest:
    return simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
  case FanInPolicy::EveryFrame:
    return simaai::neat::GraphLinkPolicy::RealtimeEveryFrameByStream;
  }
  return simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
}

struct CliOptions {
  fs::path config_path;
  bool validate_config_only = false;
};

struct StreamProfile {
  bool enabled = false;
  int stream_index = 0;
  int interval = 100;
  int frames = 0;
  int boxes = 0;
  double start_ms = 0.0;
  double parse_ms = 0.0;
  double metadata_send_ms = 0.0;

  void add(double parse, double metadata_send, int box_count) {
    if (!enabled)
      return;
    if (frames == 0)
      start_ms = sima_examples::time_ms();
    ++frames;
    boxes += box_count;
    parse_ms += parse;
    metadata_send_ms += metadata_send;
    if (frames >= interval)
      flush();
  }

  void flush() {
    if (!enabled || frames == 0)
      return;
    const double elapsed = sima_examples::time_ms() - start_ms;
    const double metadata_fps =
        elapsed > 0.0 ? static_cast<double>(frames) * 1000.0 / elapsed : 0.0;
    const auto avg = [this](double value) { return value / static_cast<double>(frames); };
    std::cout << "[profile stream=" << stream_index << "] frames=" << frames
              << " metadata_fps=" << metadata_fps << " avg_parse_ms=" << avg(parse_ms)
              << " avg_metadata_send_ms=" << avg(metadata_send_ms)
              << " avg_boxes=" << static_cast<double>(boxes) / static_cast<double>(frames) << "\n";
    frames = 0;
    boxes = 0;
    start_ms = 0.0;
    parse_ms = 0.0;
    metadata_send_ms = 0.0;
  }
};
struct AggregateProfile {
  bool enabled = false;
  int stream_count = 0;
  std::uint64_t interval_frames = 400;
  std::uint64_t frames = 0;
  double start_ms = 0.0;
  std::mutex mu;

  void add() {
    if (!enabled)
      return;
    std::lock_guard<std::mutex> lock(mu);
    if (frames == 0)
      start_ms = sima_examples::time_ms();
    ++frames;
    if (frames >= interval_frames)
      flush_locked();
  }

  void flush() {
    if (!enabled)
      return;
    std::lock_guard<std::mutex> lock(mu);
    flush_locked();
  }

private:
  void flush_locked() {
    if (frames == 0)
      return;
    const double elapsed_s = std::max(0.001, (sima_examples::time_ms() - start_ms) / 1000.0);
    const double aggregate_fps = static_cast<double>(frames) / elapsed_s;
    std::cout << "[profile aggregate] frames=" << frames << " elapsed_s=" << elapsed_s
              << " aggregate_fps=" << aggregate_fps << " avg_per_stream_fps="
              << (stream_count > 0 ? aggregate_fps / static_cast<double>(stream_count) : 0.0)
              << "\n";
    frames = 0;
    start_ms = 0.0;
  }
};

struct SourceRuntime {
  int index = 0;
  std::string url;
  std::unique_ptr<simaai::neat::MetadataSender> metadata_sender;
  std::vector<std::string> labels;
  simaai::neat::nodes::groups::RtspDecodedInputOptions source_options;
  StreamProfile profile;
  int frame_w = 0;
  int frame_h = 0;
  int source_fps = 0;
  int video_port = 0;
  int processed = 0;
  simaai::neat::Run video_run;
  std::shared_ptr<std::mutex> video_mu = std::make_shared<std::mutex>();
  app16::EncodedDeliveryQueue<simaai::neat::Sample> delivery;
  std::uint64_t video_cached = 0;
  std::uint64_t video_match_ok = 0;
  std::uint64_t video_match_miss = 0;
  std::uint64_t video_push_ok = 0;
  std::uint64_t video_queue_reject = 0;
  std::uint64_t video_try_busy = 0;
  std::uint64_t video_try_busy_streak = 0;
  std::uint64_t video_try_busy_streak_max = 0;
  std::uint64_t metadata_send_ok = 0;
  std::uint64_t metadata_send_fail = 0;
  std::int64_t last_video_pts_ns = -1;
  std::int64_t last_detection_pts_ns = -1;

  struct VideoSnapshot {
    std::uint64_t cached = 0;
    std::uint64_t match_ok = 0;
    std::uint64_t match_miss = 0;
    std::uint64_t push_ok = 0;
    std::uint64_t queue_reject = 0;
    std::uint64_t try_busy = 0;
    std::uint64_t try_busy_streak = 0;
    std::uint64_t try_busy_streak_max = 0;
    std::uint64_t metadata_ok = 0;
    std::uint64_t metadata_fail = 0;
    std::size_t pending = 0;
    std::size_t pending_frames = 0;
    std::size_t pending_metadata = 0;
    std::size_t frame_count_high_water = 0;
    std::size_t frame_bytes_high_water = 0;
    std::int64_t video_pts_ns = -1;
    std::int64_t detection_pts_ns = -1;
    std::uint64_t frame_epoch = 0;
    std::uint64_t metadata_epoch = 0;
    std::uint64_t frame_epoch_resets = 0;
    std::uint64_t metadata_epoch_resets = 0;
    std::uint64_t metadata_epoch_unresolved = 0;
  };

  VideoSnapshot video_snapshot() const {
    std::lock_guard<std::mutex> lock(*video_mu);
    return VideoSnapshot{video_cached,
                         video_match_ok,
                         video_match_miss,
                         video_push_ok,
                         video_queue_reject,
                         video_try_busy,
                         video_try_busy_streak,
                         video_try_busy_streak_max,
                         metadata_send_ok,
                         metadata_send_fail,
                         delivery.pending_count(),
                         delivery.frame_count(),
                         delivery.metadata_count(),
                         delivery.frame_count_high_water(),
                         delivery.frame_bytes_high_water(),
                         last_video_pts_ns,
                         last_detection_pts_ns,
                         delivery.frame_epoch(),
                         delivery.metadata_epoch(),
                         delivery.frame_epoch_resets(),
                         delivery.metadata_epoch_resets(),
                         delivery.metadata_epoch_unresolved()};
  }
};
struct AppRuntime {
  explicit AppRuntime(int internal_queue_depth)
      : graph([internal_queue_depth] {
          simaai::neat::GraphOptions options;
          options.advanced_execution.internal_queue_depth = internal_queue_depth;
          options.advanced_execution.inference_async = kInferenceAsync;
          return options;
        }()) {}

  std::unique_ptr<simaai::neat::Model> model;
  simaai::neat::Graph graph;
  simaai::neat::Run run;
  std::vector<SourceRuntime> sources;
  std::atomic<bool> stop_encoded_video_dispatch{false};
  std::thread encoded_video_dispatch_thread;
  std::mutex encoded_video_error_mu;
  std::exception_ptr encoded_video_error;
};
CliOptions parse_args(int argc, char** argv) {
  CliOptions options;
  const fs::path adjacent_config = fs::path(argv[0]).parent_path() / "config.yaml";
  options.config_path = fs::exists(adjacent_config)
                            ? adjacent_config
                            : sima_examples::default_config_path(SIMANEAT_APPS_EXAMPLE_SOURCE_DIR);
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--config") {
      if (i + 1 >= argc) {
        throw std::runtime_error("--config requires a path");
      }
      options.config_path = argv[++i];
    } else if (arg == "--validate-config-only") {
      options.validate_config_only = true;
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: " << argv[0] << " [--config <path>] [--validate-config-only]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  return options;
}

fs::path resolve_config_relative_path(const fs::path& config_path, const std::string& value) {
  const fs::path path(value);
  if (path.empty() || path.is_absolute()) {
    return path;
  }
  return (config_path.parent_path() / path).lexically_normal();
}

std::string strip_inline_comment(const std::string& line) {
  bool in_single = false;
  bool in_double = false;
  std::string out;
  out.reserve(line.size());
  for (char c : line) {
    if (c == '\'' && !in_double) {
      in_single = !in_single;
    } else if (c == '"' && !in_single) {
      in_double = !in_double;
    } else if (c == '#' && !in_single && !in_double) {
      break;
    }
    out.push_back(c);
  }
  return out;
}

std::string unquote(std::string value) {
  value = sima_examples::trim_copy(value);
  if (value.size() >= 2 && ((value.front() == '"' && value.back() == '"') ||
                            (value.front() == '\'' && value.back() == '\''))) {
    return value.substr(1, value.size() - 2);
  }
  return value;
}

int leading_indent(const std::string& line) {
  int indent = 0;
  while (indent < static_cast<int>(line.size()) &&
         (line[static_cast<std::size_t>(indent)] == ' ' ||
          line[static_cast<std::size_t>(indent)] == '\t')) {
    ++indent;
  }
  return indent;
}

std::vector<std::string> parse_streams(const fs::path& config_path) {
  std::ifstream input(config_path);
  if (!input.is_open()) {
    throw std::runtime_error("failed to open config file: " + config_path.string());
  }

  std::vector<std::string> streams;
  bool in_streams = false;
  int streams_indent = -1;
  std::string raw_line;
  while (std::getline(input, raw_line)) {
    const std::string line_without_comment = strip_inline_comment(raw_line);
    if (sima_examples::trim_copy(line_without_comment).empty()) {
      continue;
    }

    const int indent = leading_indent(line_without_comment);
    const std::string line = sima_examples::trim_copy(line_without_comment);

    if (in_streams && indent <= streams_indent && line.rfind("- ", 0) != 0) {
      in_streams = false;
    }
    if (!in_streams && line == "streams:") {
      in_streams = true;
      streams_indent = indent;
      continue;
    }
    if (!in_streams) {
      continue;
    }

    if (line.rfind("- ", 0) == 0) {
      const std::string url = unquote(line.substr(2));
      if (url.empty()) {
        throw std::runtime_error("streams entries must be non-empty strings");
      }
      streams.push_back(url);
    }
  }
  if (streams.empty()) {
    throw std::runtime_error("streams must be a non-empty list");
  }
  return streams;
}

void validate_config(const AppConfig& cfg) {
  if (!cfg.bypass_model && !cfg.bypass_preprocess) {
    sima_examples::require(!cfg.model_path.empty(), "model.path must be set");
    (void)parse_box_decode_type(cfg.decode_type);
    sima_examples::require(!cfg.labels_path.empty(), "model.labels must be set");
  } else if (cfg.bypass_model) {
    sima_examples::require(!cfg.model_path.empty(),
                           "model.path must be set for inference.bypass_model preproc-only mode");
  }
  sima_examples::require(!cfg.rtsp_urls.empty(), "streams must be set");
  sima_examples::require(cfg.rtsp_urls.size() <= kStreamLimit,
                         "this example supports up to 80 streams");
  sima_examples::require(
      cfg.workers == 1,
      "graph-native multi-stream detector uses one shared worker; set inference.workers to 1");
  sima_examples::require(cfg.queue_depth > 0, "inference.queue_depth must be > 0");
  sima_examples::require(cfg.queue_depth <= 32, "inference.queue_depth must be <= 32");
  sima_examples::require(cfg.internal_queue_depth >= 0,
                         "inference.internal_queue_depth must be >= 0");
  sima_examples::require(cfg.internal_queue_depth <= 32,
                         "inference.internal_queue_depth must be <= 32");
  sima_examples::require(cfg.max_inflight_per_stream > 0,
                         "inference.max_inflight_per_stream must be > 0");
  sima_examples::require(cfg.max_inflight_per_stream <= 32,
                         "inference.max_inflight_per_stream must be <= 32");
  sima_examples::require(cfg.decoder_buffers > 0, "input.decoder_buffers must be > 0");
  sima_examples::require(cfg.decoder_buffers <= 64, "input.decoder_buffers must be <= 64");
  sima_examples::require(cfg.decoder_input_buffers > 0, "input.decoder_input_buffers must be > 0");
  (void)normalize_decoder_tuning(cfg.decoder_tuning);
  sima_examples::require(!cfg.insight_host.empty(), "output.insight.host must be set");
  sima_examples::require(cfg.latency_ms >= 0, "input.latency_ms must be >= 0");
  sima_examples::require(cfg.input_width >= 0, "input.width must be >= 0");
  sima_examples::require(cfg.input_height >= 0, "input.height must be >= 0");
  sima_examples::require(cfg.input_fps >= 0, "input.fps must be >= 0");
  sima_examples::require((cfg.input_width == 0) == (cfg.input_height == 0),
                         "input.width and input.height must be set together");
  if (cfg.skip_rtsp_probe) {
    sima_examples::require(
        cfg.input_width > 0 && cfg.input_height > 0 && cfg.input_fps > 0,
        "input.skip_rtsp_probe requires input.width, input.height, and input.fps");
  }
  sima_examples::require(cfg.min_score >= 0.0 && cfg.min_score <= 1.0,
                         "inference.min_score must be between 0 and 1");
  sima_examples::require(cfg.nms_iou >= 0.0 && cfg.nms_iou <= 1.0,
                         "inference.nms_iou must be between 0 and 1");
  sima_examples::require(cfg.max_detections > 0, "inference.max_detections must be > 0");
  if (cfg.bypass_model) {
    sima_examples::require(cfg.input_width > 0 && cfg.input_height > 0,
                           "inference.bypass_model requires input.width and input.height");
  }
  if (cfg.bypass_preprocess) {
    sima_examples::require(cfg.input_width > 0 && cfg.input_height > 0,
                           "diagnostics.bypass_preprocess requires input.width and input.height");
  }
  sima_examples::require(cfg.warmup_frames >= 0, "runtime.warmup_frames must be >= 0");
  sima_examples::require(cfg.initial_detection_timeout_ms > 0,
                         "runtime.initial_detection_timeout_ms must be > 0");
  sima_examples::require(cfg.video_port_base > 0, "output.insight.video_port_base must be > 0");
  sima_examples::require(cfg.video_port_base <= 65535,
                         "output.insight.video_port_base must be <= 65535");
  sima_examples::require(cfg.metadata_port_base > 0,
                         "output.insight.metadata_port_base must be > 0");
  sima_examples::require(cfg.metadata_port_base <= 65535,
                         "output.insight.metadata_port_base must be <= 65535");
  sima_examples::require(cfg.insight_video_sync_delay_ms >= 0 &&
                             cfg.insight_video_sync_delay_ms <= 2000,
                         "output.insight.sync_delay_ms must be between 0 and 2000");
  sima_examples::require(cfg.insight_visible_streams >= kAllInsightStreams,
                         "output.insight.max_visible_streams must be >= -1 (-1 means all)");
  if (cfg.insight_visible_streams >= 0) {
    sima_examples::require(static_cast<std::size_t>(cfg.insight_visible_streams) <=
                               cfg.rtsp_urls.size(),
                           "output.insight.max_visible_streams cannot exceed stream count");
  }
  const auto visible_streams = static_cast<std::int64_t>(
      cfg.insight_visible_streams < 0 ? cfg.rtsp_urls.size()
                                      : static_cast<std::size_t>(cfg.insight_visible_streams));
  if (visible_streams > 0) {
    const auto video_last = static_cast<std::int64_t>(cfg.video_port_base) + visible_streams - 1;
    const auto metadata_last =
        static_cast<std::int64_t>(cfg.metadata_port_base) + visible_streams - 1;
    sima_examples::require(video_last <= 65535, "output.insight video port range exceeds 65535");
    sima_examples::require(metadata_last <= 65535,
                           "output.insight metadata port range exceeds 65535");
    const bool disjoint =
        video_last < cfg.metadata_port_base || metadata_last < cfg.video_port_base;
    sima_examples::require(disjoint, "output.insight video and metadata port ranges overlap");
  }
}

AppConfig load_app_config(const fs::path& config_path) {
  const auto raw = sima_examples::ScalarConfig::load(config_path);
  sima_examples::require(!raw.string_value("inference.fps").has_value(),
                         "inference.fps is not supported; set stream FPS at the RTSP source");
  sima_examples::require(
      !raw.string_value("inference.target_fps").has_value(),
      "inference.target_fps is not supported; set stream FPS at the RTSP source");
  sima_examples::require(
      !raw.string_value("output.hidden_streams.video_sink").has_value() &&
          !raw.string_value("output.hidden_streams.metadata_enabled").has_value(),
      "output.hidden_streams was removed; set output.insight.max_visible_streams explicitly");
  sima_examples::require(
      !raw.string_value("output.debug_dir").has_value() &&
          !raw.string_value("output.save_every").has_value(),
      "output.debug_dir/output.save_every were removed from the realtime application");
  AppConfig cfg;
  cfg.model_path =
      resolve_config_relative_path(config_path, raw.string_or("model.path", "")).string();
  cfg.decode_type = raw.string_or("model.decode_type", "yolo26");
  cfg.labels_path =
      resolve_config_relative_path(config_path, raw.string_or("model.labels", "coco_label.txt"));
  cfg.rtsp_urls = parse_streams(config_path);
  cfg.tcp = raw.bool_or("input.tcp", true);
  cfg.latency_ms = raw.int_or("input.latency_ms", 100);
  cfg.decoder_buffers = raw.int_or("input.decoder_buffers", kDefaultDecoderBuffers);
  cfg.decoder_input_buffers =
      raw.int_or("input.decoder_input_buffers", kDefaultDecoderInputBuffers);
  cfg.decoder_tuning = normalize_decoder_tuning(raw.string_or("input.decoder_tuning", "auto"));
  cfg.input_width = raw.int_or("input.width", 0);
  cfg.input_height = raw.int_or("input.height", 0);
  cfg.input_fps = raw.int_or("input.fps", 0);
  cfg.skip_rtsp_probe = raw.bool_or("input.skip_rtsp_probe", false);
  cfg.rtsp_drop_on_latency = raw.bool_or("input.drop_on_latency", false);
  cfg.workers = raw.int_or("inference.workers", 1);
  cfg.queue_depth = raw.int_or("inference.queue_depth", kDefaultQueueDepth);
  if (!raw.string_value("inference.queue_depth").has_value()) {
    cfg.queue_depth = raw.int_or("inference.inflight",
                                 raw.int_or("inference.detector_queue_depth", kDefaultQueueDepth));
  }
  cfg.internal_queue_depth =
      raw.int_or("inference.internal_queue_depth", kDefaultInternalQueueDepth);
  cfg.max_inflight_per_stream =
      raw.int_or("inference.max_inflight_per_stream", kDefaultMaxInflightPerStream);
  cfg.fan_in_policy = parse_fan_in_policy(raw.string_or("inference.fan_in_policy", "latest"));
  cfg.min_score = raw.double_or("inference.min_score", 0.55);
  cfg.nms_iou = raw.double_or("inference.nms_iou", 0.60);
  cfg.max_detections = raw.int_or("inference.max_detections", 50);
  cfg.bypass_model =
      raw.bool_or("inference.bypass_model", raw.bool_or("diagnostics.bypass_model", false));
  cfg.bypass_preprocess = raw.bool_or("diagnostics.bypass_preprocess", false);
  cfg.copy_input = raw.bool_or("diagnostics.copy_input", false);
  cfg.profile = raw.bool_or("runtime.profile", false);
  cfg.warmup_frames = raw.int_or("runtime.warmup_frames", 30);
  cfg.initial_detection_timeout_ms =
      raw.int_or("runtime.initial_detection_timeout_ms", kDefaultInitialDetectionTimeoutMs);
  cfg.insight_host = raw.string_or("output.insight.host", "");
  cfg.video_port_base = raw.int_or("output.insight.video_port_base", 9000);
  cfg.metadata_port_base = raw.int_or("output.insight.metadata_port_base", 9100);
  cfg.insight_visible_streams =
      raw.int_or("output.insight.max_visible_streams", kAllInsightStreams);
  cfg.video_enabled = raw.bool_or("output.video_enabled", true);
  cfg.insight_video_sync_delay_ms = raw.int_or("output.insight.sync_delay_ms", 400);
  validate_config(cfg);
  return cfg;
}

std::vector<std::string> load_labels(const fs::path& labels_path) {
  std::ifstream in(labels_path);
  if (!in.good()) {
    throw std::runtime_error("labels file does not exist: " + labels_path.string());
  }
  std::vector<std::string> labels;
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty()) {
      labels.push_back(line);
    }
  }
  if (labels.empty()) {
    throw std::runtime_error("labels file is empty: " + labels_path.string());
  }
  return labels;
}

bool extract_bbox_payload(const simaai::neat::Sample& sample, std::vector<std::uint8_t>& payload,
                          std::string& err) {
  if (sample.kind == simaai::neat::SampleKind::Bundle) {
    for (const auto& field : sample.fields) {
      if (extract_bbox_payload(field, payload, err)) {
        return true;
      }
    }
    err = "bundle missing BBOX field";
    return false;
  }
  if (sample.kind == simaai::neat::SampleKind::TensorSet && !sample.tensors.empty()) {
    simaai::neat::Sample tensor_sample = sample;
    tensor_sample.kind = simaai::neat::SampleKind::Tensor;
    tensor_sample.tensor = sample.tensors.front();
    tensor_sample.tensors.clear();
    return objdet::extract_bbox_payload(tensor_sample, payload, err);
  }
  return objdet::extract_bbox_payload(sample, payload, err);
}

std::vector<sima_examples::MetadataBox> build_metadata_boxes(const std::vector<objdet::Box>& boxes,
                                                             const std::vector<std::string>& labels,
                                                             int frame_w, int frame_h) {
  std::vector<sima_examples::MetadataBox> metadata_boxes;
  metadata_boxes.reserve(boxes.size());
  int object_index = 1;
  for (const auto& box : boxes) {
    int x1 = std::max(0, static_cast<int>(box.x1));
    int y1 = std::max(0, static_cast<int>(box.y1));
    int w = std::max(0, static_cast<int>(box.x2 - box.x1));
    int h = std::max(0, static_cast<int>(box.y2 - box.y1));
    if (x1 + w > frame_w)
      w = frame_w - x1;
    if (y1 + h > frame_h)
      h = frame_h - y1;

    sima_examples::MetadataBox obj;
    obj.id = "obj_" + std::to_string(object_index++);
    obj.label = (box.class_id >= 0 && box.class_id < static_cast<int>(labels.size()))
                    ? labels[box.class_id]
                    : "unknown";
    obj.confidence = box.score;
    obj.x = static_cast<float>(x1);
    obj.y = static_cast<float>(y1);
    obj.w = static_cast<float>(std::max(0, w));
    obj.h = static_cast<float>(std::max(0, h));
    metadata_boxes.push_back(obj);
  }
  return metadata_boxes;
}

bool env_bool(const char* key, bool fallback = false) {
  const char* value = std::getenv(key);
  if (!value || !*value)
    return fallback;
  std::string v(value);
  std::transform(v.begin(), v.end(), v.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return v == "1" || v == "true" || v == "yes" || v == "on";
}

int env_int(const char* key, int fallback) {
  const char* value = std::getenv(key);
  if (!value || !*value)
    return fallback;
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (!end || *end != '\0')
    return fallback;
  return static_cast<int>(parsed);
}

bool app_verbose() {
  return env_bool("APP16_VERBOSE", false);
}

bool app_print_backend() {
  return app_verbose() || env_bool("APP16_PRINT_BACKEND", false);
}

int app_liveness_ms() {
  return std::max(0, env_int("APP16_LIVENESS_MS", 0));
}

void print_pull_liveness(const std::vector<SourceRuntime>& sources, const char* reason,
                         std::uint64_t total_pulls) {
  if (sources.empty()) {
    std::cerr << "[app16][liveness] reason=" << (reason ? reason : "snapshot")
              << " streams=0 total_pulls=" << total_pulls << "\n";
    return;
  }

  int min_processed = sources.front().processed;
  int max_processed = sources.front().processed;
  int zero_streams = 0;
  std::uint64_t min_video = std::numeric_limits<std::uint64_t>::max();
  std::uint64_t max_video = 0;
  std::uint64_t video_queue_rejects = 0;
  std::uint64_t video_try_busy = 0;
  std::uint64_t metadata_send_ok = 0;
  std::uint64_t metadata_send_fail = 0;
  std::int64_t min_sync_lag_ms = std::numeric_limits<std::int64_t>::max();
  std::int64_t max_sync_lag_ms = std::numeric_limits<std::int64_t>::min();
  int sync_lag_streams = 0;
  for (const auto& source : sources) {
    min_processed = std::min(min_processed, source.processed);
    max_processed = std::max(max_processed, source.processed);
    if (source.processed == 0) {
      ++zero_streams;
    }
    const auto video = source.video_snapshot();
    min_video = std::min(min_video, video.push_ok);
    max_video = std::max(max_video, video.push_ok);
    video_queue_rejects += video.queue_reject;
    video_try_busy += video.try_busy;
    metadata_send_ok += video.metadata_ok;
    metadata_send_fail += video.metadata_fail;
    if (video.video_pts_ns >= 0 && video.detection_pts_ns >= 0 &&
        video.frame_epoch == video.metadata_epoch) {
      const std::int64_t lag_ms = (video.video_pts_ns - video.detection_pts_ns) / 1000000;
      min_sync_lag_ms = std::min(min_sync_lag_ms, lag_ms);
      max_sync_lag_ms = std::max(max_sync_lag_ms, lag_ms);
      ++sync_lag_streams;
    }
  }

  std::vector<const SourceRuntime*> low;
  low.reserve(sources.size());
  const int low_cutoff = std::max(min_processed + 2, max_processed / 4);
  for (const auto& source : sources) {
    if (source.processed <= low_cutoff) {
      low.push_back(&source);
    }
  }
  std::sort(low.begin(), low.end(), [](const SourceRuntime* a, const SourceRuntime* b) {
    if (a->processed != b->processed) {
      return a->processed < b->processed;
    }
    return a->index < b->index;
  });

  std::cerr << "[app16][liveness] reason=" << (reason ? reason : "snapshot")
            << " streams=" << sources.size() << " total_pulls=" << total_pulls
            << " min_processed=" << min_processed << " max_processed=" << max_processed
            << " zero_streams=" << zero_streams << " video_min=" << min_video
            << " video_max=" << max_video << " video_queue_reject=" << video_queue_rejects
            << " video_try_busy=" << video_try_busy << " metadata_send_ok=" << metadata_send_ok
            << " metadata_send_fail=" << metadata_send_fail;
  if (sync_lag_streams > 0) {
    std::cerr << " av_pts_lag_ms_min=" << min_sync_lag_ms
              << " av_pts_lag_ms_max=" << max_sync_lag_ms;
  }
  std::cerr << " low_cutoff=" << low_cutoff << " low=";
  const std::size_t show = std::min<std::size_t>(low.size(), 12);
  for (std::size_t i = 0; i < show; ++i) {
    if (i != 0U) {
      std::cerr << ",";
    }
    std::cerr << "stream" << low[i]->index << ":" << low[i]->processed;
  }
  if (low.size() > show) {
    std::cerr << ",...";
  }
  std::cerr << "\n";
}

std::string stream_id_for(int stream_index) {
  return "stream" + std::to_string(stream_index);
}

int effective_insight_visible_streams(const AppConfig& cfg) {
  if (cfg.insight_visible_streams < 0) {
    return static_cast<int>(cfg.rtsp_urls.size());
  }
  return cfg.insight_visible_streams;
}

bool is_insight_visible_stream(const AppConfig& cfg, int stream_index) {
  return stream_index >= 0 && stream_index < effective_insight_visible_streams(cfg);
}

bool should_send_metadata(const AppConfig& cfg, int stream_index) {
  return is_insight_visible_stream(cfg, stream_index);
}

int stream_index_from_detection(const simaai::neat::Sample& sample, int stream_count) {
  const std::string prefix = "stream";
  if (sample.stream_id.rfind(prefix, 0) != 0) {
    if (stream_count == 1)
      return 0;
    throw std::runtime_error("detection sample missing stream id: " + sample.stream_id);
  }
  const std::string suffix = sample.stream_id.substr(prefix.size());
  if (suffix.empty() ||
      !std::all_of(suffix.begin(), suffix.end(), [](unsigned char c) { return std::isdigit(c); })) {
    throw std::runtime_error("invalid detection stream id: " + sample.stream_id);
  }
  const int index = std::stoi(suffix);
  if (index < 0 || index >= stream_count) {
    throw std::runtime_error("detection stream id out of range: " + sample.stream_id);
  }
  return index;
}

int target_frames_per_stream() {
  return std::max(0, env_int("APP16_FRAMES_PER_STREAM", 0));
}

bool target_reached(const std::vector<SourceRuntime>& sources) {
  const int target = target_frames_per_stream();
  if (target <= 0 || sources.empty()) {
    return false;
  }
  return std::all_of(sources.begin(), sources.end(),
                     [target](const SourceRuntime& source) { return source.processed >= target; });
}

simaai::neat::RunOptions realtime_options(
    int queue_depth = 3,
    simaai::neat::OverflowPolicy overflow_policy = simaai::neat::OverflowPolicy::KeepLatest,
    bool copy_input = false) {
  simaai::neat::RunOptions run_options;
  run_options.preset = simaai::neat::RunPreset::Realtime;
  run_options.queue_depth = queue_depth;
  run_options.overflow_policy = overflow_policy;
  run_options.output_memory = simaai::neat::OutputMemory::ZeroCopy;
  run_options.advanced.copy_input = copy_input;
  return run_options;
}

simaai::neat::nodes::groups::RtspDecodedInputOptions
make_source_options(const AppConfig& cfg, const std::string& url, int& fps_out, int& width_out,
                    int& height_out) {
  sima_examples::RtspStreamInfo probe;
  if (!cfg.skip_rtsp_probe) {
    sima_examples::RtspProbeOptions probe_options;
    probe_options.payload_type = 96;
    probe_options.latency_ms = cfg.latency_ms;
    probe_options.rtsp_tcp = cfg.tcp;
    probe_options.debug = cfg.profile;
    (void)sima_examples::probe_rtsp_stream_info(url, probe_options, probe);
  }

  simaai::neat::nodes::groups::RtspDecodedInputOptions opt;
  opt.url = url;
  opt.latency_ms = cfg.latency_ms;
  opt.tcp = cfg.tcp;
  opt.drop_on_latency = cfg.rtsp_drop_on_latency;
  opt.payload_type = 96;
  opt.insert_queue = true;
  opt.out_format = "NV12";
  opt.decoder_name = "decoder";
  opt.decoder_raw_output = true;
  opt.decoder_next_element = "CVU";
  opt.decoder_input_buffers = cfg.decoder_input_buffers;
  opt.decoder_tuning = cfg.decoder_tuning;
  opt.decoder_memory_opt =
      cfg.decoder_tuning == "low-memory" || cfg.decoder_tuning == "throughput-low-latency";
  opt.auto_caps_from_stream = !cfg.skip_rtsp_probe;
  opt.num_buffers = cfg.decoder_buffers;
  apply_h264_caps(opt, probe.width, probe.height, probe.fps, width_out, height_out, fps_out);
  apply_h264_caps(opt, cfg.input_width, cfg.input_height, cfg.input_fps, width_out, height_out,
                  fps_out);
  opt.output_caps.enable = true;
  opt.output_caps.format = simaai::neat::FormatTag::NV12;
  opt.output_caps.memory = simaai::neat::CapsMemory::Any;
  return opt;
}

simaai::neat::Graph
make_rtsp_h264_input(const simaai::neat::nodes::groups::RtspDecodedInputOptions& opt) {
  simaai::neat::nodes::groups::RtspEncodedInputOptions encoded;
  encoded.url = opt.url;
  encoded.codec = simaai::neat::nodes::groups::RtspCodec::H264;
  encoded.latency_ms = opt.latency_ms;
  encoded.tcp = opt.tcp;
  encoded.drop_on_latency = opt.drop_on_latency;
  encoded.buffer_mode = opt.buffer_mode;
  encoded.insert_queue = opt.insert_queue;
  encoded.sync_mode = opt.sync_mode;
  encoded.h264_payload_type = opt.payload_type;
  encoded.h264_parse_config_interval = opt.h264_parse_config_interval;
  encoded.h264_fps = opt.h264_fps;
  encoded.h264_width = opt.h264_width;
  encoded.h264_height = opt.h264_height;
  encoded.auto_caps_from_stream = opt.auto_caps_from_stream;
  encoded.fallback_h264_fps = opt.fallback_h264_fps;
  encoded.fallback_h264_width = opt.fallback_h264_width;
  encoded.fallback_h264_height = opt.fallback_h264_height;
  return simaai::neat::nodes::groups::RtspEncodedInput(encoded);
}

simaai::neat::InputOptions make_encoded_h264_input_options(bool block) {
  simaai::neat::InputOptions input_options;
  input_options.payload_type = simaai::neat::PayloadType::Encoded;
  input_options.format = simaai::neat::FormatTag::H264;
  input_options.block = block;
  input_options.max_bytes = kEncodedQueueMaxBytesPerStream;
  input_options.use_simaai_pool = false;
  input_options.memory_policy = simaai::neat::InputMemoryPolicy::SystemMemory;
  return input_options;
}

simaai::neat::InputOptions make_raw_nv12_input_options(const AppConfig& cfg, bool block) {
  simaai::neat::InputOptions input_options;
  input_options.payload_type = simaai::neat::PayloadType::Image;
  input_options.format = simaai::neat::FormatTag::NV12;
  input_options.width = cfg.input_width;
  input_options.height = cfg.input_height;
  input_options.max_width = cfg.input_width;
  input_options.max_height = cfg.input_height;
  input_options.fps_n = cfg.input_fps > 0 ? cfg.input_fps : 0;
  input_options.fps_d = 1;
  input_options.block = block;
  input_options.pool_min_buffers = 1;
  input_options.pool_max_buffers = std::max(1, cfg.queue_depth);
  input_options.max_bytes = static_cast<std::uint64_t>(cfg.input_width) *
                            static_cast<std::uint64_t>(cfg.input_height) * 3U / 2U *
                            static_cast<std::uint64_t>(input_options.pool_max_buffers);
  input_options.memory_policy = simaai::neat::InputMemoryPolicy::Auto;
  return input_options;
}

simaai::neat::Graph
make_rtsp_decoded_input(const simaai::neat::nodes::groups::RtspDecodedInputOptions& opt,
                        int decoder_buffers) {
  const int dec_w = (opt.h264_width > 0) ? opt.h264_width : opt.fallback_h264_width;
  const int dec_h = (opt.h264_height > 0) ? opt.h264_height : opt.fallback_h264_height;
  const int dec_fps = (opt.h264_fps > 0) ? opt.h264_fps : opt.fallback_h264_fps;

  simaai::neat::Graph graph = make_rtsp_h264_input(opt);
  simaai::neat::SimaDecodeOptions decode;
  decode.type = simaai::neat::SimaDecodeType::H264;
  decode.sima_allocator_type = opt.sima_allocator_type;
  decode.out_format = simaai::neat::FormatTag::NV12;
  decode.decoder_name = opt.decoder_name;
  decode.raw_output = opt.decoder_raw_output;
  decode.next_element = opt.decoder_next_element;
  decode.dec_width = dec_w;
  decode.dec_height = dec_h;
  decode.dec_fps = dec_fps;
  decode.num_buffers = decoder_buffers;
  decode.input_buffers = opt.decoder_input_buffers;
  decode.decoder_tuning = opt.decoder_tuning;
  decode.memory_opt = opt.decoder_memory_opt;
  graph.add(simaai::neat::nodes::SimaDecode(std::move(decode)));
  if (opt.output_caps.enable) {
    graph.add(simaai::neat::nodes::CapsRaw("NV12", opt.output_caps.width, opt.output_caps.height,
                                           opt.output_caps.fps, opt.output_caps.memory));
  }
  return graph;
}

std::unique_ptr<simaai::neat::Model> make_model(const AppConfig& cfg) {
  simaai::neat::Model::Options model_opt;
  if (!app_verbose()) {
    model_opt.verbose = simaai::neat::VerboseOptions::quiet();
  }
  model_opt.preprocess.kind = simaai::neat::InputKind::Image;
  model_opt.preprocess.enable = simaai::neat::AutoFlag::On;
  model_opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::NV12;
  model_opt.preprocess.preset = simaai::neat::NormalizePreset::COCO_YOLO;
  model_opt.decode_type = parse_box_decode_type(cfg.decode_type);
  model_opt.score_threshold = cfg.min_score;
  model_opt.nms_iou_threshold = cfg.nms_iou;
  model_opt.top_k = cfg.max_detections;
  return std::make_unique<simaai::neat::Model>(cfg.model_path, model_opt);
}

simaai::neat::nodes::groups::VideoSenderOptions make_video_options(const AppConfig& cfg,
                                                                   const SourceRuntime& source) {
  auto video_options = simaai::neat::nodes::groups::VideoSenderOptions::H264RtpUdpFromEncoded();
  video_options.host = cfg.insight_host;
  video_options.channel = source.index;
  video_options.video_port_base = cfg.video_port_base;
  video_options.async = true;
  return video_options;
}

SourceRuntime make_source_runtime(const AppConfig& cfg, int stream_index,
                                  const std::vector<std::string>& labels) {
  SourceRuntime source;
  source.index = stream_index;
  source.url = cfg.rtsp_urls[static_cast<std::size_t>(stream_index)];
  source.source_options =
      make_source_options(cfg, source.url, source.source_fps, source.frame_w, source.frame_h);
  sima_examples::require(source.frame_w > 0 && source.frame_h > 0,
                         "failed to probe RTSP frame dimensions");
  sima_examples::require(source.source_fps > 0, "failed to probe RTSP frame rate");
  source.labels = labels;
  source.profile.enabled = cfg.profile;
  source.profile.stream_index = stream_index;
  const std::size_t encoded_queue_frames =
      app16::encoded_delivery_frame_capacity(source.source_fps, cfg.insight_video_sync_delay_ms);
  source.delivery = app16::EncodedDeliveryQueue<simaai::neat::Sample>(
      encoded_queue_frames, /*max_metadata=*/128, kEncodedQueueMaxBytesPerStream);

  if (should_send_metadata(cfg, stream_index)) {
    simaai::neat::MetadataSenderOptions metadata_options;
    metadata_options.host = cfg.insight_host;
    metadata_options.channel = stream_index;
    metadata_options.metadata_port_base = cfg.metadata_port_base;
    simaai::neat::MetadataSenderSendOptions send_options;
    send_options.nonblocking = true;
    std::string metadata_err;
    source.metadata_sender = std::make_unique<simaai::neat::MetadataSender>(
        metadata_options, send_options, &metadata_err);
    sima_examples::require(source.metadata_sender->ok(), metadata_err);
  }
  return source;
}

simaai::neat::Graph build_detector_graph(const AppConfig& cfg,
                                         std::unique_ptr<simaai::neat::Model>& model) {
  if (cfg.bypass_preprocess) {
    simaai::neat::Graph input_graph;
    input_graph.add(
        simaai::neat::nodes::Input("detector_frame", make_raw_nv12_input_options(cfg, true)));

    simaai::neat::Graph output_graph;
    output_graph.add(simaai::neat::nodes::Output(
        "detections", simaai::neat::OutputOptions::EveryFrame(cfg.queue_depth)));

    simaai::neat::Graph graph;
    graph.connect(input_graph, output_graph);
    return graph;
  }

  if (cfg.bypass_model) {
    model = make_model(cfg);

    auto detector_input_options = model->input_appsrc_options(false);
    const int detector_input_buffers = std::max(1, detector_input_options.pool_max_buffers);
    if (cfg.input_width > 0 && cfg.input_height > 0) {
      const auto width = static_cast<std::uint64_t>(cfg.input_width);
      const auto height = static_cast<std::uint64_t>(cfg.input_height);
      detector_input_options.max_bytes =
          width * height * 3U / 2U * static_cast<std::uint64_t>(detector_input_buffers);
      detector_input_options.pool_max_buffers = detector_input_buffers;
    }
    detector_input_options.block = true;

    simaai::neat::Graph input_graph;
    input_graph.add(simaai::neat::nodes::Input("detector_frame", detector_input_options));

    simaai::neat::Graph preproc_graph = model->preprocess();

    simaai::neat::Graph output_graph;
    output_graph.add(simaai::neat::nodes::Output(
        "detections", simaai::neat::OutputOptions::EveryFrame(cfg.queue_depth)));

    simaai::neat::Graph graph;
    graph.connect(input_graph, preproc_graph);
    graph.connect(preproc_graph, output_graph);
    return graph;
  }

  model = make_model(cfg);

  simaai::neat::Graph input_graph;
  auto detector_input_options = model->input_appsrc_options(false);
  const int detector_input_buffers = std::max(1, detector_input_options.pool_max_buffers);
  if (cfg.input_width > 0 && cfg.input_height > 0) {
    const auto width = static_cast<std::uint64_t>(cfg.input_width);
    const auto height = static_cast<std::uint64_t>(cfg.input_height);
    detector_input_options.max_bytes =
        width * height * 3U / 2U * static_cast<std::uint64_t>(detector_input_buffers);
    detector_input_options.pool_max_buffers = detector_input_buffers;
  }
  detector_input_options.block = true;
  input_graph.add(simaai::neat::nodes::Input("detector_frame", detector_input_options));

  simaai::neat::Graph model_graph = model->graph();

  simaai::neat::Graph detections_graph;
  detections_graph.add(simaai::neat::nodes::Output(
      "detections", simaai::neat::OutputOptions::EveryFrame(cfg.queue_depth)));

  simaai::neat::Graph graph;
  graph.connect(input_graph, model_graph);
  graph.connect(model_graph, detections_graph);
  return graph;
}

void complete_passthrough_sample(SourceRuntime& source, const AppConfig& cfg,
                                 AggregateProfile& aggregate_profile) {
  ++source.processed;
  const bool warming_up = source.processed <= cfg.warmup_frames;
  if (!warming_up) {
    source.profile.add(/*parse_ms=*/0.0, /*metadata_send_ms=*/0.0, /*boxes=*/0);
    aggregate_profile.add();
  }
}

void connect_source_graph(AppRuntime& app, const AppConfig& cfg, SourceRuntime& source,
                          const simaai::neat::Graph& detector_graph) {
  auto branch = simaai::neat::graphs::Branch("source", {"detector_frame"});
  simaai::neat::RealtimeGraphLinkOptions detector_link;
  detector_link.policy = graph_link_policy(cfg.fan_in_policy);
  detector_link.queue_depth = cfg.queue_depth;
  detector_link.stream_id = stream_id_for(source.index);
  detector_link.max_inflight_per_stream = cfg.max_inflight_per_stream;

  auto rtsp = make_rtsp_decoded_input(source.source_options, cfg.decoder_buffers);
  app.graph.connect(rtsp, branch);
  app.graph.connect_realtime(branch, detector_graph, detector_link);

  if (cfg.video_enabled && is_insight_visible_stream(cfg, source.index)) {
    source.video_port = make_video_options(cfg, source).video_port();
  }

  std::cout << "[stream " << source.index << "] rtsp=" << source.url << " stream=" << source.frame_w
            << "x" << source.frame_h << "@" << source.source_fps << " insight=" << cfg.insight_host
            << " video=";
  if (!cfg.video_enabled) {
    std::cout << "disabled";
  } else if (is_insight_visible_stream(cfg, source.index)) {
    std::cout << source.video_port;
  } else {
    std::cout << "not-published";
  }
  std::cout << " metadata=";
  if (source.metadata_sender) {
    std::cout << source.metadata_sender->metadata_port();
  } else {
    std::cout << "disabled";
  }
  std::cout << "\n";
}

void validate_worker_pool_geometry(const std::vector<SourceRuntime>& sources) {
  if (sources.empty()) {
    return;
  }
  const int frame_w = sources.front().frame_w;
  const int frame_h = sources.front().frame_h;
  for (const auto& source : sources) {
    if (source.frame_w != frame_w || source.frame_h != frame_h) {
      throw std::runtime_error("detector graph requires all streams to share frame size");
    }
  }
}

std::uint32_t rtp_timestamp_from_pts_ns(int64_t pts_ns) {
  if (pts_ns < 0) {
    return 0;
  }
  const auto ticks = (static_cast<__int128>(pts_ns) * 90000) / 1000000000;
  return static_cast<std::uint32_t>(static_cast<std::uint64_t>(ticks));
}

void start_encoded_video_senders(AppRuntime& app, const AppConfig& cfg) {
  if (!cfg.video_enabled) {
    return;
  }
  for (auto& source : app.sources) {
    if (!is_insight_visible_stream(cfg, source.index)) {
      continue;
    }
    auto video_options = make_video_options(cfg, source);
    source.video_port = video_options.video_port();

    simaai::neat::Graph video_graph("encoded_insight_video_sender");
    video_graph.connect(
        simaai::neat::nodes::Input("video_h264", make_encoded_h264_input_options(false)),
        simaai::neat::nodes::groups::VideoSender(video_options));
    source.video_run = video_graph.build(realtime_options(
        /*queue_depth=*/1, simaai::neat::OverflowPolicy::Block, /*copy_input=*/false));
    std::cout << "[encoded_video] stream=" << source.index << " video_port=" << source.video_port
              << " cpu_au_queue_frames=" << source.delivery.max_frames()
              << " cpu_au_queue_bytes=" << source.delivery.max_frame_bytes() << "\n";
  }
}

std::size_t encoded_sample_bytes(const simaai::neat::Sample& sample) {
  const simaai::neat::Tensor* tensor = nullptr;
  if (sample.kind == simaai::neat::SampleKind::Tensor && sample.tensor.has_value()) {
    tensor = &*sample.tensor;
  } else if (sample.kind == simaai::neat::SampleKind::TensorSet && !sample.tensors.empty()) {
    tensor = &sample.tensors.front();
  }
  return tensor && tensor->storage ? tensor->storage->size_bytes : 0U;
}

void request_encoded_video_stop(AppRuntime& app, std::exception_ptr error = {}) noexcept {
  if (error) {
    try {
      std::lock_guard<std::mutex> lock(app.encoded_video_error_mu);
      if (!app.encoded_video_error) {
        app.encoded_video_error = std::move(error);
      }
    } catch (...) {
      // The stop flag is the non-allocating fallback. The pull loop will still
      // terminate with its generic encoded-delivery error if diagnostics could
      // not be retained during an allocation failure.
    }
  }
  g_runtime_stop_requested.store(true, std::memory_order_relaxed);
}

void install_encoded_video_frame_tap(AppRuntime& app, const AppConfig& cfg) {
  const std::size_t visible_video_runs = static_cast<std::size_t>(
      std::count_if(app.sources.begin(), app.sources.end(), [](const SourceRuntime& source) {
        return static_cast<bool>(source.video_run);
      }));
  if (!cfg.video_enabled || visible_video_runs == 0) {
    simaai::neat::clear_latest_by_stream_encoded_frame_callback();
    return;
  }
  if (visible_video_runs < app.sources.size()) {
    std::cerr << "[warn] encoded AU tap is graph-global: Core copies encoded CPU AUs for "
              << app.sources.size() << " source branches before App16 discards the "
              << (app.sources.size() - visible_video_runs)
              << " non-published channels; use all-visible output for the validated profile\n";
  }
  const int sync_delay_ms = cfg.insight_video_sync_delay_ms;
  simaai::neat::set_latest_by_stream_encoded_frame_callback([&app, sync_delay_ms](
                                                                simaai::neat::Sample frame) {
    try {
      if (g_runtime_stop_requested.load(std::memory_order_relaxed)) {
        return;
      }
      const int stream_index =
          stream_index_from_detection(frame, static_cast<int>(app.sources.size()));
      auto& source = app.sources[static_cast<std::size_t>(stream_index)];
      if (!source.video_run) {
        return;
      }
      const std::size_t frame_bytes = encoded_sample_bytes(frame);
      const std::int64_t frame_pts_ns = frame.pts_ns;
      const auto now = std::chrono::steady_clock::now();
      bool rejected = false;
      bool count_full = false;
      bool bytes_full = false;
      std::size_t queued_frames = 0;
      std::size_t queued_metadata = 0;
      std::size_t queued_bytes = 0;
      std::size_t max_frames = 0;
      std::size_t max_bytes = 0;
      std::int64_t oldest_due_in_ms = 0;
      {
        // The Core callback owns the AU bytes already. This short section
        // only moves that Sample into the per-stream deque and snapshots
        // overflow diagnostics; it never sends UDP, logs, or waits on a
        // VideoSender while holding the callback-path mutex.
        std::lock_guard<std::mutex> lock(*source.video_mu);
        ++source.video_cached;
        rejected = !source.delivery.enqueue_frame(std::move(frame),
                                                  now + std::chrono::milliseconds(sync_delay_ms),
                                                  frame_bytes, frame_pts_ns);
        if (rejected) {
          ++source.video_queue_reject;
          queued_frames = source.delivery.frame_count();
          queued_metadata = source.delivery.metadata_count();
          queued_bytes = source.delivery.frame_bytes();
          max_frames = source.delivery.max_frames();
          max_bytes = source.delivery.max_frame_bytes();
          count_full = queued_frames >= max_frames;
          bytes_full = max_bytes > 0 && frame_bytes > max_bytes - std::min(queued_bytes, max_bytes);
          oldest_due_in_ms = source.delivery.oldest_frame_due_in_ms(now);
        }
      }
      if (rejected) {
        request_encoded_video_stop(app);
        const char* limit = count_full ? (bytes_full ? "count+bytes" : "count") : "bytes";
        std::cerr << "[encoded_video] stream=" << source.index
                  << " delayed queue limit reached (limit=" << limit
                  << ", queued_frames=" << queued_frames << ", queued_metadata=" << queued_metadata
                  << ", queued_bytes=" << queued_bytes << ", attempted_au_bytes=" << frame_bytes
                  << ", max_frames=" << max_frames << ", max_bytes=" << max_bytes
                  << ", oldest_due_in_ms=" << oldest_due_in_ms
                  << "); stopping instead of corrupting H.264\n";
      }
    } catch (...) {
      // Core deliberately catches subscriber exceptions so an observability
      // tap cannot tear down a GStreamer streaming thread. Convert every
      // App16 callback failure into an explicit app stop here; otherwise
      // one silently lost H.264 AU could corrupt only the Insight branch.
      request_encoded_video_stop(app, std::current_exception());
    }
  });
}

void log_encoded_dispatch_stall(const char* operation, int stream_index,
                                std::chrono::steady_clock::duration elapsed) {
  constexpr auto kLogThreshold = std::chrono::milliseconds(20);
  if (elapsed < kLogThreshold) {
    return;
  }
  const auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  std::cerr << "[encoded_video][stall] operation=" << operation << " stream=" << stream_index
            << " elapsed_us=" << elapsed_us << "\n";
}

bool should_log_failure_count(std::uint64_t count) {
  // Log the first failure and then powers of two. Persistent UDP pressure is
  // visible without letting a failing receiver turn the realtime dispatcher
  // into a stderr benchmark.
  return count != 0 && (count & (count - 1U)) == 0;
}

void send_metadata_nonblocking(SourceRuntime& source, const std::string& payload) {
  if (!source.metadata_sender) {
    return;
  }
  std::string err;
  const bool sent = source.metadata_sender->send_raw_json(payload, &err);
  std::uint64_t failures = 0;
  {
    std::lock_guard<std::mutex> lock(*source.video_mu);
    if (sent) {
      ++source.metadata_send_ok;
    } else {
      failures = ++source.metadata_send_fail;
    }
  }
  if (!sent && should_log_failure_count(failures)) {
    std::cerr << "[warn] stream " << source.index
              << " nonblocking metadata send failed (count=" << failures << "): " << err << "\n";
  }
}

void encoded_video_dispatch_loop(AppRuntime& app) {
  try {
    auto previous_round_end = std::chrono::steady_clock::now();
    while (!app.stop_encoded_video_dispatch.load(std::memory_order_relaxed) &&
           !g_runtime_stop_requested.load(std::memory_order_relaxed)) {
      bool did_work = false;
      const auto now = std::chrono::steady_clock::now();
      log_encoded_dispatch_stall("round_gap", -1, now - previous_round_end);
      for (auto& source : app.sources) {
        simaai::neat::Sample frame;
        std::uint64_t frame_epoch = 0;
        bool have_frame = false;
        const auto peek_start = std::chrono::steady_clock::now();
        {
          std::lock_guard<std::mutex> lock(*source.video_mu);
          have_frame = source.delivery.peek_due_frame(now, frame, &frame_epoch);
        }
        log_encoded_dispatch_stall("peek", source.index,
                                   std::chrono::steady_clock::now() - peek_start);
        if (!have_frame) {
          continue;
        }

        // Publish matching metadata before admitting the encoded AU. Insight
        // can retain metadata until the corresponding RTP frame arrives, but
        // it cannot recover an exact match after the browser has already
        // presented that frame. Sending video first caused occasional
        // one-to-three-frame stale overlays even though both payloads carried
        // the same source PTS.
        std::string metadata_payload;
        {
          std::lock_guard<std::mutex> lock(*source.video_mu);
          std::size_t stale_epoch_metadata = 0;
          metadata_payload = source.delivery.take_metadata_through(frame_epoch, frame.pts_ns,
                                                                   &stale_epoch_metadata);
          source.video_match_miss += stale_epoch_metadata;
          if (!metadata_payload.empty()) {
            ++source.video_match_ok;
          }
        }
        if (!metadata_payload.empty() && source.metadata_sender) {
          const auto metadata_send_start = std::chrono::steady_clock::now();
          send_metadata_nonblocking(source, metadata_payload);
          log_encoded_dispatch_stall("metadata_send_before_video", source.index,
                                     std::chrono::steady_clock::now() - metadata_send_start);
        }

        // Try each channel at most once per round. A blocked sender keeps its
        // AU at the head of its own queue while the other channels continue.
        // Busy-waiting here creates global head-of-line blocking at 24/48
        // channels and eventually overflows every otherwise healthy queue.
        const auto try_push_start = std::chrono::steady_clock::now();
        const bool sender_accepted = source.video_run.try_push("video_h264", frame);
        log_encoded_dispatch_stall("try_push", source.index,
                                   std::chrono::steady_clock::now() - try_push_start);
        if (!sender_accepted) {
          const std::string sender_error = source.video_run.last_error();
          const bool sender_running = source.video_run.running();
          {
            std::lock_guard<std::mutex> lock(*source.video_mu);
            ++source.video_try_busy;
            ++source.video_try_busy_streak;
            source.video_try_busy_streak_max =
                std::max(source.video_try_busy_streak_max, source.video_try_busy_streak);
          }
          // A full queue is a normal nonblocking result and the AU remains at
          // the head for retry. A stopped/error Run can never recover, so fail
          // with the actual per-channel cause instead of waiting for the delay
          // queue to overflow with a misleading H.264 integrity error.
          if (!sender_running || !sender_error.empty()) {
            throw std::runtime_error("encoded VideoSender stopped for stream " +
                                     std::to_string(source.index) +
                                     " at pts_ns=" + std::to_string(frame.pts_ns) +
                                     (sender_error.empty() ? std::string{} : ": " + sender_error));
          }
          continue;
        }
        did_work = true;
        const std::int64_t frame_pts_ns = frame.pts_ns;
        const auto consume_start = std::chrono::steady_clock::now();
        {
          std::lock_guard<std::mutex> lock(*source.video_mu);
          simaai::neat::Sample consumed;
          std::uint64_t consumed_epoch = 0;
          if (!source.delivery.pop_due_frame(now, consumed, &consumed_epoch) ||
              consumed_epoch != frame_epoch) {
            throw std::runtime_error("encoded delivery queue changed after sender accepted AU");
          }
          ++source.video_push_ok;
          source.video_try_busy_streak = 0;
          source.last_video_pts_ns = frame_pts_ns;
        }
        log_encoded_dispatch_stall("consume", source.index,
                                   std::chrono::steady_clock::now() - consume_start);
      }
      if (!did_work) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      previous_round_end = std::chrono::steady_clock::now();
    }
  } catch (...) {
    {
      std::lock_guard<std::mutex> lock(app.encoded_video_error_mu);
      app.encoded_video_error = std::current_exception();
    }
    g_runtime_stop_requested.store(true, std::memory_order_relaxed);
  }
}

void start_encoded_video_dispatch(AppRuntime& app) {
  const bool have_video =
      std::any_of(app.sources.begin(), app.sources.end(),
                  [](const SourceRuntime& source) { return static_cast<bool>(source.video_run); });
  if (!have_video) {
    return;
  }
  app.stop_encoded_video_dispatch.store(false, std::memory_order_relaxed);
  app.encoded_video_dispatch_thread = std::thread(encoded_video_dispatch_loop, std::ref(app));
}

void stop_encoded_video_dispatch(AppRuntime& app) {
  app.stop_encoded_video_dispatch.store(true, std::memory_order_relaxed);
  if (app.encoded_video_dispatch_thread.joinable()) {
    app.encoded_video_dispatch_thread.join();
  }
}

void rethrow_encoded_video_error(AppRuntime& app) {
  std::exception_ptr error;
  {
    std::lock_guard<std::mutex> lock(app.encoded_video_error_mu);
    error = app.encoded_video_error;
  }
  if (error) {
    std::rethrow_exception(error);
  }
}

void stop_encoded_video_senders(AppRuntime& app) {
  for (auto& source : app.sources) {
    if (source.video_run) {
      source.video_run.close();
    }
  }
}

void send_metadata(SourceRuntime& source, const simaai::neat::Sample& frame,
                   const std::vector<objdet::Box>& boxes) {
  if (!source.metadata_sender) {
    return;
  }
  const auto metadata_boxes =
      build_metadata_boxes(boxes, source.labels, source.frame_w, source.frame_h);
  const std::string data_json = sima_examples::metadata_boxes_data_json("objects", metadata_boxes);
  const int64_t timestamp_ms = frame.pts_ns >= 0 ? frame.pts_ns / 1'000'000 : -1;
  const std::string frame_id = frame.frame_id >= 0 ? std::to_string(frame.frame_id) : "";

  nlohmann::json payload;
  try {
    payload["type"] = "object-detection";
    payload["data"] = nlohmann::json::parse(data_json);
    payload["timestamp"] = timestamp_ms;
    payload["frame_id"] = frame_id;
    payload["stream_id"] = frame.stream_id.empty() ? stream_id_for(source.index) : frame.stream_id;
    payload["stream_index"] = source.index;
    payload["pts_ns"] = frame.pts_ns;
    payload["dts_ns"] = frame.dts_ns;
    payload["duration_ns"] = frame.duration_ns;
    payload["input_seq"] = frame.input_seq;
    payload["orig_input_seq"] = frame.orig_input_seq;
    if (frame.pts_ns >= 0) {
      payload["rtp_timestamp"] = rtp_timestamp_from_pts_ns(frame.pts_ns);
    }
  } catch (const std::exception& ex) {
    std::cerr << "[warn] stream " << source.index << " metadata JSON build failed: " << ex.what()
              << "\n";
    return;
  }

  std::string payload_text = payload.dump();
  if (source.video_run && frame.pts_ns < 0) {
    std::lock_guard<std::mutex> lock(*source.video_mu);
    ++source.video_match_miss;
    return;
  }
  if (source.video_run) {
    std::lock_guard<std::mutex> lock(*source.video_mu);
    if (source.delivery.enqueue_metadata(frame.pts_ns, std::move(payload_text))) {
      ++source.video_match_miss;
    }
    return;
  }

  send_metadata_nonblocking(source, payload_text);
}

void complete_detection(SourceRuntime& source, const AppConfig& cfg,
                        AggregateProfile& aggregate_profile,
                        const simaai::neat::Sample& detections) {
  const double parse_start = sima_examples::time_ms();
  std::vector<std::uint8_t> payload;
  std::string err;
  if (!extract_bbox_payload(detections, payload, err)) {
    throw std::runtime_error("stream " + std::to_string(source.index) +
                             " bbox extract failed: " + err);
  }
  const auto boxes = objdet::parse_boxes_strict(payload, source.frame_w, source.frame_h,
                                                cfg.max_detections, false);
  const double parse_end = sima_examples::time_ms();

  ++source.processed;
  if (detections.pts_ns >= 0) {
    std::lock_guard<std::mutex> lock(*source.video_mu);
    source.last_detection_pts_ns = detections.pts_ns;
  }
  const bool warming_up = source.processed <= cfg.warmup_frames;

  if (!warming_up) {
    const double metadata_start = sima_examples::time_ms();
    send_metadata(source, detections, boxes);
    const double metadata_end = sima_examples::time_ms();
    source.profile.add(parse_end - parse_start, metadata_end - metadata_start,
                       static_cast<int>(boxes.size()));
    aggregate_profile.add();
  }
}

void pull_detections(AppRuntime& app, const AppConfig& cfg, AggregateProfile& aggregate_profile) {
  bool saw_detection = false;
  std::uint64_t total_pulls = 0;
  const int liveness_ms = app_liveness_ms();
  auto next_liveness = std::chrono::steady_clock::now() + std::chrono::milliseconds(liveness_ms);
  while (g_stop_requested == 0 && !g_runtime_stop_requested.load(std::memory_order_relaxed)) {
    simaai::neat::Sample detections;
    simaai::neat::PullError err;
    const int timeout_ms =
        saw_detection ? kDetectorResultTimeoutMs : cfg.initial_detection_timeout_ms;
    const auto status = app.run.pull("detections", timeout_ms, detections, &err);
    if (status == simaai::neat::PullStatus::Closed) {
      break;
    }
    if (status == simaai::neat::PullStatus::Timeout) {
      print_pull_liveness(app.sources,
                          saw_detection ? "timeout_after_detections"
                                        : "timeout_before_initial_detection",
                          total_pulls);
      throw std::runtime_error(saw_detection ? "timed out waiting for detections"
                                             : "timed out waiting for initial detection");
    }
    if (status == simaai::neat::PullStatus::Error) {
      throw std::runtime_error("failed to pull detections: " + err.message);
    }
    if (status != simaai::neat::PullStatus::Ok) {
      continue;
    }

    saw_detection = true;
    ++total_pulls;
    const int stream_index =
        stream_index_from_detection(detections, static_cast<int>(app.sources.size()));
    auto& source = app.sources[static_cast<std::size_t>(stream_index)];
    if (cfg.bypass_model || cfg.bypass_preprocess) {
      complete_passthrough_sample(source, cfg, aggregate_profile);
    } else {
      complete_detection(source, cfg, aggregate_profile, detections);
    }
    if (target_reached(app.sources)) {
      break;
    }
    if (liveness_ms > 0) {
      const auto now = std::chrono::steady_clock::now();
      if (now >= next_liveness) {
        print_pull_liveness(app.sources, "heartbeat", total_pulls);
        next_liveness = now + std::chrono::milliseconds(liveness_ms);
      }
    }
  }
  if (g_runtime_stop_requested.load(std::memory_order_relaxed)) {
    rethrow_encoded_video_error(app);
    throw std::runtime_error("encoded video delivery stopped");
  }
}

void flush_and_print_runtime_stats(AggregateProfile& aggregate_profile,
                                   std::vector<SourceRuntime>& sources) {
  aggregate_profile.flush();
  for (auto& source : sources) {
    source.profile.flush();
    const auto video = source.video_snapshot();
    const auto metadata_stats = source.metadata_sender ? source.metadata_sender->stats()
                                                       : simaai::neat::MetadataSenderStats{};
    std::cout << "[stream " << source.index << "] processed=" << source.processed
              << " video_cached=" << video.cached << " video_match_ok=" << video.match_ok
              << " video_match_miss=" << video.match_miss << " video_push_ok=" << video.push_ok
              << " video_queue_reject=" << video.queue_reject
              << " video_try_busy=" << video.try_busy
              << " video_try_busy_streak=" << video.try_busy_streak
              << " video_try_busy_streak_max=" << video.try_busy_streak_max
              << " metadata_send_ok=" << video.metadata_ok
              << " metadata_send_fail=" << video.metadata_fail
              << " metadata_would_block=" << metadata_stats.would_block
              << " metadata_no_buffer_space=" << metadata_stats.no_buffer_space
              << " metadata_send_max_ns=" << metadata_stats.max_send_duration_ns
              << " video_pending=" << video.pending
              << " video_pending_frames=" << video.pending_frames
              << " video_pending_metadata=" << video.pending_metadata
              << " video_frame_high_water=" << video.frame_count_high_water
              << " video_byte_high_water=" << video.frame_bytes_high_water
              << " last_video_pts_ns=" << video.video_pts_ns
              << " last_detection_pts_ns=" << video.detection_pts_ns
              << " video_pts_epoch=" << video.frame_epoch
              << " metadata_pts_epoch=" << video.metadata_epoch
              << " video_pts_epoch_resets=" << video.frame_epoch_resets
              << " metadata_pts_epoch_resets=" << video.metadata_epoch_resets
              << " metadata_pts_epoch_unresolved=" << video.metadata_epoch_unresolved << "\n";
  }
}

void run_app(const AppConfig& cfg) {
  g_stop_requested = 0;
  g_runtime_stop_requested.store(false, std::memory_order_relaxed);
  auto previous_sigint = std::signal(SIGINT, request_stop);

  const auto labels = (cfg.bypass_model || cfg.bypass_preprocess) ? std::vector<std::string>{}
                                                                  : load_labels(cfg.labels_path);
  AggregateProfile aggregate_profile;
  aggregate_profile.enabled = cfg.profile;
  aggregate_profile.stream_count = static_cast<int>(cfg.rtsp_urls.size());

  // The selected fused realtime fan-in has one pending slot per stream. Latest
  // mode replaces that slot; every-frame mode backpressures only that stream.
  // This small global depth only decouples shared model stages and does not
  // create another decoded-frame queue per camera. GraphOptions must be set on
  // the final composition owner rather than a nested model fragment.
  AppRuntime app(cfg.internal_queue_depth);
  app.sources.reserve(cfg.rtsp_urls.size());
  std::cout << "[detector] fan_in_policy=" << fan_in_policy_name(cfg.fan_in_policy)
            << " max_inflight_per_stream=" << cfg.max_inflight_per_stream << "\n";
  try {
    auto detector_graph = build_detector_graph(cfg, app.model);
    for (std::size_t index = 0; index < cfg.rtsp_urls.size(); ++index) {
      app.sources.push_back(make_source_runtime(cfg, static_cast<int>(index), labels));
      connect_source_graph(app, cfg, app.sources.back(), detector_graph);
    }
    validate_worker_pool_geometry(app.sources);

    if (cfg.profile && app_print_backend()) {
      std::cout << "Application backend:\n" << app.graph.describe_backend() << "\n";
    }

    // The sender pipelines only packetize already-encoded CPU buffers, so they
    // allocate no EV encoder surfaces. Build them in channel order and install
    // the tap before starting RTSP; every channel then sees its source from the
    // first AU while the decoder paths stay in the fused C++ mux pipeline.
    start_encoded_video_senders(app, cfg);
    install_encoded_video_frame_tap(app, cfg);
    start_encoded_video_dispatch(app);
    app.run = app.graph.build_fused_realtime_sources(realtime_options(
        cfg.queue_depth, simaai::neat::OverflowPolicy::KeepLatest, cfg.copy_input));

    pull_detections(app, cfg, aggregate_profile);
  } catch (...) {
    simaai::neat::clear_latest_by_stream_encoded_frame_callback();
    stop_encoded_video_dispatch(app);
    stop_encoded_video_senders(app);
    app.run.close();
    flush_and_print_runtime_stats(aggregate_profile, app.sources);
    std::signal(SIGINT, previous_sigint);
    throw;
  }

  simaai::neat::clear_latest_by_stream_encoded_frame_callback();
  stop_encoded_video_dispatch(app);
  stop_encoded_video_senders(app);
  app.run.close();
  flush_and_print_runtime_stats(aggregate_profile, app.sources);
  std::signal(SIGINT, previous_sigint);
}

} // namespace

int main(int argc, char** argv) {
  try {
    const CliOptions cli = parse_args(argc, argv);
    if (!fs::exists(cli.config_path)) {
      std::cerr << "Error: config file not found: " << cli.config_path << "\n";
      return 2;
    }

    const AppConfig cfg = load_app_config(cli.config_path);
    if (cli.validate_config_only) {
      const int visible_streams = effective_insight_visible_streams(cfg);
      const int video_port_last = cfg.video_port_base + std::max(0, visible_streams - 1);
      const int metadata_port_last = cfg.metadata_port_base + std::max(0, visible_streams - 1);
      std::cout << "Config validated: " << cli.config_path << " (streams=" << cfg.rtsp_urls.size()
                << ", model_path=" << cfg.model_path << ", labels_path=" << cfg.labels_path.string()
                << ", workers=" << cfg.workers << ", queue_depth=" << cfg.queue_depth
                << ", internal_queue_depth=" << cfg.internal_queue_depth
                << ", inference_async=" << (kInferenceAsync ? "true" : "false")
                << ", max_inflight_per_stream=" << cfg.max_inflight_per_stream
                << ", fan_in_policy=" << fan_in_policy_name(cfg.fan_in_policy)
                << ", input=" << cfg.input_width << "x" << cfg.input_height << "@" << cfg.input_fps
                << ", insight_visible_streams=" << visible_streams
                << ", video_ports=" << cfg.video_port_base << "-" << video_port_last
                << ", metadata_ports=" << cfg.metadata_port_base << "-" << metadata_port_last
                << ", sync_delay_ms=" << cfg.insight_video_sync_delay_ms
                << ", bypass_model=" << (cfg.bypass_model ? "true" : "false")
                << ", bypass_preprocess=" << (cfg.bypass_preprocess ? "true" : "false")
                << ", copy_input=" << (cfg.copy_input ? "true" : "false")
                << ", decoder_admission=core)\n";
      return 0;
    }
    run_app(cfg);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[ERR] " << e.what() << "\n";
    return 1;
  }
}
