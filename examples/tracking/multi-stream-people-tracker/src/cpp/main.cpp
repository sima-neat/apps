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
#include "examples/tracking/multi-stream-people-tracker/src/cpp/utils/camera_motion_api.cpp"
#include "examples/tracking/multi-stream-people-tracker/src/cpp/utils/tracker_api.cpp"
#include "examples/tracking/multi-stream-people-tracker/src/cpp/utils/tracker_overlay_api.cpp"
#include "support/object_detection/obj_detection_utils.h"
#include "support/runtime/config_utils.h"
#include "support/runtime/example_utils.h"

#include <nodes/groups/VideoSender.h>
#include <nodes/io/MetadataSender.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using multi_stream_people_tracker::Detection;
using multi_stream_people_tracker::draw_tracks_bgr;
using multi_stream_people_tracker::FrameCameraMotionEstimator;
using multi_stream_people_tracker::ObjectTracker;
using multi_stream_people_tracker::TrackedDetection;
using multi_stream_people_tracker::TrackerConfig;

namespace {

volatile std::sig_atomic_t g_stop_requested = 0;

void request_stop(int) {
  g_stop_requested = 1;
}

enum class RuntimeOverflowPolicy {
  KeepLatest,
  Block,
};

struct AppConfig {
  std::string model_path;
  std::vector<std::string> rtsp_urls;
  /// Encoded RTSP path used for every stream in this application.
  simaai::neat::nodes::groups::RtspCodec codec = simaai::neat::nodes::groups::RtspCodec::H264;
  int latency_ms = 100;
  bool tcp = true;
  int frames = 0;
  int fps = 0;
  int max_inflight_per_stream = 4;
  int max_inflight_total = 16;
  int num_classes = 80;
  int target_class_id = 0;
  std::string target_label = "person";
  double min_score = 0.55;
  double nms_iou = 0.60;
  int max_detections = 50;
  bool profile = false;
  int warmup_frames = 30;
  RuntimeOverflowPolicy overflow_policy = RuntimeOverflowPolicy::KeepLatest;
  double tracker_high_score = 0.55;
  double tracker_new_track_score = 0.55;
  float tracker_iou_threshold = 0.30f;
  float tracker_max_center_distance = 2.5f;
  float tracker_velocity_momentum = 0.80f;
  float tracker_box_smoothing_alpha = 1.0f;
  int tracker_max_missing = 15;
  int tracker_min_confirmed_hits = 1;
  int tracker_max_prediction_frames = 0;
  bool tracker_center_distance_enabled = false;
  bool tracker_camera_motion_compensation = false;
  bool tracker_covariance_motion_enabled = true;
  float tracker_overlap_threshold = 0.20f;
  int tracker_max_occlusion_frames = 10;
  int tracker_max_active_tracks = 128;
  std::string insight_host = "127.0.0.1";
  int video_port_base = 9000;
  int metadata_port_base = 9100;
  bool video_enabled = true;
  fs::path save_dir;
  int save_every = 0;
  fs::path replay_dir;
};

class DebugFrameWriter {
public:
  DebugFrameWriter() : worker_([this] { run(); }) {}

  ~DebugFrameWriter() {
    {
      std::lock_guard lock(mutex_);
      stopping_ = true;
    }
    ready_.notify_one();
    if (worker_.joinable()) {
      worker_.join();
    }
  }

  DebugFrameWriter(const DebugFrameWriter&) = delete;
  DebugFrameWriter& operator=(const DebugFrameWriter&) = delete;

  void enqueue(fs::path path, cv::Mat frame) {
    {
      std::lock_guard lock(mutex_);
      if (stopping_) {
        return;
      }
      // Debug capture must never apply backpressure to live inference. Keep
      // the freshest bounded sample when storage cannot keep up.
      if (queue_.size() >= kMaximumQueuedFrames) {
        queue_.pop_front();
      }
      queue_.push_back(Task{std::move(path), std::move(frame)});
    }
    ready_.notify_one();
  }

private:
  struct Task {
    fs::path path;
    cv::Mat frame;
  };

  void run() {
    while (true) {
      Task task;
      {
        std::unique_lock lock(mutex_);
        ready_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
        if (queue_.empty()) {
          if (stopping_) {
            return;
          }
          continue;
        }
        task = std::move(queue_.front());
        queue_.pop_front();
      }
      if (!cv::imwrite(task.path.string(), task.frame)) {
        std::cerr << "[warn] failed to write output frame: " << task.path << "\n";
      }
    }
  }

  static constexpr std::size_t kMaximumQueuedFrames = 16;
  std::mutex mutex_;
  std::condition_variable ready_;
  std::deque<Task> queue_;
  bool stopping_ = false;
  std::thread worker_;
};

struct CameraMotionResult {
  multi_stream_people_tracker::CameraTransform transform;
  double compute_ms = 0.0;
  bool chain_discontinuous = false;
};

// Camera motion is ordered within a stream, but independent of MLA inference.
// Submitting decoded frames here hides sparse-flow work under the detector
// branch without racing the estimator's previous-frame state.
class AsyncCameraMotionEstimator {
public:
  AsyncCameraMotionEstimator() : worker_([this] { run(); }) {}

  ~AsyncCameraMotionEstimator() {
    {
      std::lock_guard lock(mutex_);
      stopping_ = true;
    }
    ready_.notify_one();
    if (worker_.joinable()) {
      worker_.join();
    }
  }

  AsyncCameraMotionEstimator(const AsyncCameraMotionEstimator&) = delete;
  AsyncCameraMotionEstimator& operator=(const AsyncCameraMotionEstimator&) = delete;

  std::future<CameraMotionResult> submit(cv::Mat gray, std::vector<Detection> object_mask) {
    Task task;
    task.gray = std::move(gray);
    task.object_mask = std::move(object_mask);
    auto result = task.result.get_future();
    {
      std::lock_guard lock(mutex_);
      if (stopping_) {
        throw std::runtime_error("camera-motion worker is stopping");
      }
      if (queue_.size() >= kMaximumQueuedFrames) {
        discard_pending_locked();
        task.reset_before_update = true;
      }
      queue_.push_back(std::move(task));
    }
    ready_.notify_one();
    return result;
  }

  void reset() {
    Task task;
    task.reset = true;
    auto result = task.result.get_future();
    {
      std::lock_guard lock(mutex_);
      if (stopping_) {
        return;
      }
      discard_pending_locked();
      queue_.push_back(std::move(task));
    }
    ready_.notify_one();
    // A missing side-branch frame breaks the transform chain. This uncommon
    // path waits for earlier work before returning, so no stale transform can
    // be consumed by a later frame.
    (void)result.get();
  }

private:
  struct Task {
    cv::Mat gray;
    std::vector<Detection> object_mask;
    bool reset = false;
    bool reset_before_update = false;
    std::promise<CameraMotionResult> result;
  };

  void discard_pending_locked() {
    for (auto& pending : queue_) {
      CameraMotionResult discarded;
      discarded.chain_discontinuous = true;
      pending.result.set_value(discarded);
    }
    queue_.clear();
  }

  void run() {
    while (true) {
      Task task;
      {
        std::unique_lock lock(mutex_);
        ready_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
        if (queue_.empty()) {
          if (stopping_) {
            return;
          }
          continue;
        }
        task = std::move(queue_.front());
        queue_.pop_front();
      }
      try {
        CameraMotionResult result;
        if (task.reset) {
          estimator_.reset();
        } else {
          if (task.reset_before_update) {
            estimator_.reset();
            result.chain_discontinuous = true;
          }
          const double start = sima_examples::time_ms();
          result.transform = estimator_.update(task.gray, task.object_mask);
          result.compute_ms = sima_examples::time_ms() - start;
        }
        task.result.set_value(result);
      } catch (...) {
        task.result.set_exception(std::current_exception());
      }
    }
  }

  FrameCameraMotionEstimator estimator_;
  std::mutex mutex_;
  std::condition_variable ready_;
  std::deque<Task> queue_;
  // At most one not-yet-started full-resolution frame is retained. If newer
  // input overtakes it, submit() replaces it and makes the replacement a new
  // optical-flow baseline instead of estimating across a missing frame.
  static constexpr std::size_t kMaximumQueuedFrames = 1;
  bool stopping_ = false;
  std::thread worker_;
};

std::string lower_copy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

simaai::neat::nodes::groups::RtspCodec parse_input_codec(const std::string& value) {
  const std::string lowered = lower_copy(value);
  if (lowered == "h264" || lowered == "avc" || lowered == "h.264") {
    return simaai::neat::nodes::groups::RtspCodec::H264;
  }
  if (lowered == "h265" || lowered == "hevc" || lowered == "h.265") {
    return simaai::neat::nodes::groups::RtspCodec::H265;
  }
  throw std::runtime_error("input.codec must be h264/avc or h265/hevc");
}

RuntimeOverflowPolicy parse_overflow_policy(const std::string& value) {
  const std::string lowered = lower_copy(value);
  if (lowered == "keep_latest" || lowered == "keep-latest") {
    return RuntimeOverflowPolicy::KeepLatest;
  }
  if (lowered == "block") {
    return RuntimeOverflowPolicy::Block;
  }
  throw std::runtime_error("runtime.overflow_policy must be keep_latest or block");
}

const char* overflow_policy_name(RuntimeOverflowPolicy policy) {
  return policy == RuntimeOverflowPolicy::Block ? "block" : "keep_latest";
}

struct CliOptions {
  fs::path config_path;
  std::optional<fs::path> benchmark_image;
  bool validate_config_only = false;
};

struct ProfileWindow {
  bool enabled = false;
  int stream_index = 0;
  int interval = 100;
  int frames = 0;
  int boxes = 0;
  double start_ms = 0.0;
  double detection_pull_ms = 0.0;
  double box_parse_ms = 0.0;
  double debug_frame_wait_ms = 0.0;
  double camera_motion_compute_ms = 0.0;
  double camera_motion_wait_ms = 0.0;
  double tracker_ms = 0.0;
  double metadata_send_ms = 0.0;
  double overlay_ms = 0.0;

  void add(double detection_pull, double box_parse, double debug_frame_wait,
           double camera_motion_compute, double camera_motion_wait, double tracker,
           double metadata_send, double overlay, int box_count) {
    if (!enabled)
      return;
    if (frames == 0)
      start_ms = sima_examples::time_ms();
    ++frames;
    boxes += box_count;
    detection_pull_ms += detection_pull;
    box_parse_ms += box_parse;
    debug_frame_wait_ms += debug_frame_wait;
    camera_motion_compute_ms += camera_motion_compute;
    camera_motion_wait_ms += camera_motion_wait;
    tracker_ms += tracker;
    metadata_send_ms += metadata_send;
    overlay_ms += overlay;
    if (frames >= interval)
      flush();
  }

  void flush() {
    if (!enabled || frames == 0)
      return;
    const double elapsed = sima_examples::time_ms() - start_ms;
    const double output_fps = elapsed > 0.0 ? static_cast<double>(frames) * 1000.0 / elapsed : 0.0;
    const auto avg = [this](double value) { return value / static_cast<double>(frames); };
    std::cout << "[profile stream=" << stream_index << "] frames=" << frames
              << " output_fps=" << output_fps << " avg_detection_pull_ms=" << avg(detection_pull_ms)
              << " avg_box_parse_ms=" << avg(box_parse_ms) << " avg_tracker_ms=" << avg(tracker_ms)
              << " avg_debug_frame_wait_ms=" << avg(debug_frame_wait_ms)
              << " avg_camera_motion_compute_ms=" << avg(camera_motion_compute_ms)
              << " avg_camera_motion_wait_ms=" << avg(camera_motion_wait_ms)
              << " avg_metadata_send_ms=" << avg(metadata_send_ms)
              << " avg_overlay_ms=" << avg(overlay_ms)
              << " avg_tracks=" << static_cast<double>(boxes) / static_cast<double>(frames) << "\n";
    frames = 0;
    boxes = 0;
    start_ms = 0.0;
    detection_pull_ms = 0.0;
    box_parse_ms = 0.0;
    debug_frame_wait_ms = 0.0;
    camera_motion_compute_ms = 0.0;
    camera_motion_wait_ms = 0.0;
    tracker_ms = 0.0;
    metadata_send_ms = 0.0;
    overlay_ms = 0.0;
  }
};

struct DebugFrameSample {
  int64_t frame_id = -1;
  int64_t input_seq = -1;
  int64_t orig_input_seq = -1;
  int64_t pts_ns = -1;
  cv::Mat gray;
  cv::Mat bgr;
  std::future<CameraMotionResult> camera_motion;
};

struct DebugFrameQueue {
  std::mutex mutex;
  std::condition_variable ready;
  std::condition_variable space_available;
  std::deque<DebugFrameSample> frames;
  std::vector<Detection> object_mask;
  bool chain_discontinuous = false;
};

struct StreamRuntime {
  int index = 0;
  std::string url;
  simaai::neat::nodes::groups::RtspDecodedInputOptions source_options;
  std::unique_ptr<simaai::neat::MetadataSender> metadata_sender;
  std::unique_ptr<DebugFrameWriter> debug_writer;
  std::unique_ptr<std::ofstream> replay_writer;
  ObjectTracker tracker;
  std::unique_ptr<AsyncCameraMotionEstimator> camera_motion;
  ProfileWindow profile;
  std::vector<std::uint8_t> payload;
  std::vector<objdet::Box> boxes;
  std::vector<Detection> detections;
  std::vector<TrackedDetection> tracks;
  std::vector<sima_examples::MetadataBox> metadata_tracks;
  std::unique_ptr<DebugFrameQueue> debug_frames;
  int frame_w = 0;
  int frame_h = 0;
  int output_fps = 0;
  int video_port = 0;
  int processed = 0;
  int tracking_frame_index = -1;
  int64_t last_source_sequence = -1;
  int64_t last_tracking_pts_ns = -1;
  double benchmark_first_completed_ms = 0.0;
  double benchmark_last_completed_ms = 0.0;
  int benchmark_completed_frames = 0;
  bool benchmark_mode = false;
  bool has_processed_debug_frame = false;
  bool closed = false;
};

struct AppRuntime {
  simaai::neat::Graph graph;
  simaai::neat::Run run;
  std::unique_ptr<simaai::neat::Model> model;
  std::vector<StreamRuntime> streams;
};

CliOptions parse_args(int argc, char** argv) {
  CliOptions options;
  options.config_path = sima_examples::default_config_path(SIMANEAT_APPS_EXAMPLE_SOURCE_DIR);
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--config") {
      if (i + 1 >= argc) {
        throw std::runtime_error("--config requires a path");
      }
      options.config_path = argv[++i];
    } else if (arg == "--benchmark-image") {
      if (i + 1 >= argc) {
        throw std::runtime_error("--benchmark-image requires a path");
      }
      options.benchmark_image = fs::path(argv[++i]);
    } else if (arg == "--validate-config-only") {
      options.validate_config_only = true;
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: " << argv[0]
                << " [--config <path>] [--benchmark-image <path>]"
                   " [--validate-config-only]\n"
                << "  --benchmark-image  Repeatedly feed one image without pacing or Insight"
                   " output. Preprocess, MLA inference, BoxDecode, camera motion, and tracking"
                   " remain enabled.\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  return options;
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

    int indent = 0;
    while (indent < static_cast<int>(line_without_comment.size()) &&
           (line_without_comment[static_cast<std::size_t>(indent)] == ' ' ||
            line_without_comment[static_cast<std::size_t>(indent)] == '\t')) {
      ++indent;
    }
    const std::string line = sima_examples::trim_copy(line_without_comment);

    if (in_streams && indent <= streams_indent && line.rfind("- ", 0) != 0) {
      in_streams = false;
    }
    if (!in_streams && line == "streams:") {
      in_streams = true;
      streams_indent = indent;
      continue;
    }
    if (in_streams && line.rfind("- ", 0) == 0) {
      const std::string value = unquote(line.substr(2));
      if (value.empty()) {
        throw std::runtime_error("streams entries must be non-empty strings");
      }
      streams.push_back(value);
    }
  }
  if (streams.empty()) {
    throw std::runtime_error("streams must be a non-empty list");
  }
  return streams;
}

void validate_config(const AppConfig& cfg) {
  sima_examples::require(!cfg.model_path.empty(), "model.path must be set");
  sima_examples::require(!cfg.rtsp_urls.empty(), "streams must be set");
  sima_examples::require(cfg.rtsp_urls.size() <= 4, "this phase supports up to four streams");
  sima_examples::require(
      cfg.overflow_policy != RuntimeOverflowPolicy::Block || cfg.rtsp_urls.size() == 1,
      "runtime.overflow_policy=block requires exactly one stream because shared detector fan-in "
      "uses latest-frame scheduling");
  sima_examples::require(!cfg.insight_host.empty(), "output.insight.host must be set");
  sima_examples::require(cfg.latency_ms >= 0, "input.latency_ms must be >= 0");
  sima_examples::require(cfg.frames >= 0, "inference.frames must be >= 0");
  sima_examples::require(cfg.fps >= 0, "inference.fps must be >= 0");
  sima_examples::require(cfg.max_inflight_per_stream == -1 || cfg.max_inflight_per_stream > 0,
                         "inference.max_inflight_per_stream must be -1 or > 0");
  sima_examples::require(cfg.max_inflight_total == -1 || cfg.max_inflight_total > 0,
                         "inference.max_inflight_total must be -1 or > 0");
  sima_examples::require(cfg.num_classes > 0, "inference.num_classes must be > 0");
  sima_examples::require(cfg.target_class_id >= 0, "inference.target_class_id must be >= 0");
  sima_examples::require(cfg.target_class_id < cfg.num_classes,
                         "inference.target_class_id (" + std::to_string(cfg.target_class_id) +
                             ") must be less than inference.num_classes (" +
                             std::to_string(cfg.num_classes) + ")");
  sima_examples::require(std::any_of(cfg.target_label.begin(), cfg.target_label.end(),
                                     [](unsigned char c) { return std::isspace(c) == 0; }),
                         "inference.target_label must be set");
  sima_examples::require(std::isfinite(cfg.min_score) && cfg.min_score >= 0.0 &&
                             cfg.min_score <= 1.0,
                         "inference.min_score must be between 0 and 1");
  sima_examples::require(std::isfinite(cfg.nms_iou) && cfg.nms_iou >= 0.0 && cfg.nms_iou <= 1.0,
                         "inference.nms_iou must be between 0 and 1");
  sima_examples::require(cfg.max_detections > 0, "inference.max_detections must be > 0");
  sima_examples::require(cfg.warmup_frames >= 0, "runtime.warmup_frames must be >= 0");
  sima_examples::require(std::isfinite(cfg.tracker_iou_threshold) &&
                             cfg.tracker_iou_threshold >= 0.0f && cfg.tracker_iou_threshold <= 1.0f,
                         "tracking.match_iou_threshold must be between 0 and 1");
  sima_examples::require(std::isfinite(cfg.tracker_high_score) &&
                             cfg.tracker_high_score >= cfg.min_score &&
                             cfg.tracker_high_score <= 1.0f,
                         "tracking.high_score_threshold must be in [inference.min_score, 1]");
  sima_examples::require(std::isfinite(cfg.tracker_new_track_score) &&
                             cfg.tracker_new_track_score >= cfg.tracker_high_score &&
                             cfg.tracker_new_track_score <= 1.0f,
                         "tracking.new_track_threshold must be in [high_score_threshold, 1]");
  sima_examples::require(std::isfinite(cfg.tracker_max_center_distance) &&
                             cfg.tracker_max_center_distance >= 0.0f,
                         "tracking.max_center_distance must be >= 0");
  sima_examples::require(std::isfinite(cfg.tracker_velocity_momentum) &&
                             cfg.tracker_velocity_momentum >= 0.0f &&
                             cfg.tracker_velocity_momentum < 1.0f,
                         "tracking.velocity_momentum must be in [0, 1)");
  sima_examples::require(std::isfinite(cfg.tracker_box_smoothing_alpha) &&
                             cfg.tracker_box_smoothing_alpha > 0.0f &&
                             cfg.tracker_box_smoothing_alpha <= 1.0f,
                         "tracking.box_smoothing_alpha must be in (0, 1]");
  sima_examples::require(cfg.tracker_max_missing >= 0, "tracking.max_missing_frames must be >= 0");
  sima_examples::require(cfg.tracker_min_confirmed_hits >= 1,
                         "tracking.min_confirmed_hits must be >= 1");
  sima_examples::require(cfg.tracker_max_prediction_frames >= 0 &&
                             cfg.tracker_max_prediction_frames <= cfg.tracker_max_missing,
                         "tracking.max_prediction_frames must be in [0, max_missing_frames]");
  sima_examples::require(std::isfinite(cfg.tracker_overlap_threshold) &&
                             cfg.tracker_overlap_threshold >= 0.0f &&
                             cfg.tracker_overlap_threshold <= 1.0f,
                         "tracking.overlap_threshold must be between 0 and 1");
  sima_examples::require(cfg.tracker_max_occlusion_frames >= 0 &&
                             cfg.tracker_max_occlusion_frames <= cfg.tracker_max_missing,
                         "tracking.max_occlusion_frames must be in [0, max_missing_frames]");
  sima_examples::require(cfg.tracker_max_active_tracks >= cfg.max_detections,
                         "tracking.max_active_tracks must be >= inference.max_detections");
  sima_examples::require(cfg.video_port_base > 0, "output.insight.video_port_base must be > 0");
  sima_examples::require(cfg.metadata_port_base > 0,
                         "output.insight.metadata_port_base must be > 0");
  sima_examples::require(cfg.save_every >= 0, "output.save_every must be >= 0");
}

AppConfig load_app_config(const fs::path& config_path) {
  const auto raw = sima_examples::ScalarConfig::load(config_path);

  AppConfig cfg;
  cfg.model_path = raw.string_or("model.path", "");
  cfg.rtsp_urls = parse_streams(config_path);
  cfg.codec = parse_input_codec(raw.string_or("input.codec", "h264"));
  cfg.tcp = raw.bool_or("input.tcp", true);
  cfg.latency_ms = raw.int_or("input.latency_ms", 100);
  cfg.frames = raw.int_or("inference.frames", 0);
  cfg.fps = raw.int_or("inference.fps", 0);
  cfg.max_inflight_per_stream = raw.int_or("inference.max_inflight_per_stream", 4);
  cfg.max_inflight_total = raw.int_or("inference.max_inflight_total", 16);
  cfg.num_classes = raw.int_or("inference.num_classes", 80);
  cfg.target_class_id =
      raw.int_or("inference.target_class_id", raw.int_or("inference.person_class_id", 0));
  cfg.target_label = raw.string_or("inference.target_label", "person");
  cfg.min_score = raw.double_or("inference.min_score", 0.55);
  cfg.nms_iou = raw.double_or("inference.nms_iou", 0.60);
  cfg.max_detections = raw.int_or("inference.max_detections", 50);
  cfg.profile = raw.bool_or("runtime.profile", false);
  cfg.warmup_frames = raw.int_or("runtime.warmup_frames", 30);
  cfg.overflow_policy =
      parse_overflow_policy(raw.string_or("runtime.overflow_policy", "keep_latest"));
  cfg.tracker_high_score = raw.double_or("tracking.high_score_threshold", cfg.min_score);
  cfg.tracker_new_track_score =
      raw.double_or("tracking.new_track_threshold", cfg.tracker_high_score);
  const bool has_new_iou_threshold = raw.string_value("tracking.match_iou_threshold").has_value();
  const bool has_center_distance = raw.string_value("tracking.max_center_distance").has_value();
  cfg.tracker_center_distance_enabled = has_center_distance || has_new_iou_threshold;
  const double default_iou_threshold = cfg.tracker_center_distance_enabled ? 0.10 : 0.30;
  cfg.tracker_iou_threshold = static_cast<float>(
      raw.double_or("tracking.match_iou_threshold",
                    raw.double_or("tracking.iou_threshold", default_iou_threshold)));
  cfg.tracker_max_center_distance =
      static_cast<float>(raw.double_or("tracking.max_center_distance", 2.5));
  cfg.tracker_velocity_momentum =
      static_cast<float>(raw.double_or("tracking.velocity_momentum", 0.80));
  cfg.tracker_box_smoothing_alpha =
      static_cast<float>(raw.double_or("tracking.box_smoothing_alpha", 1.0));
  cfg.tracker_max_missing = raw.int_or("tracking.max_missing_frames", 15);
  cfg.tracker_min_confirmed_hits = raw.int_or("tracking.min_confirmed_hits", 1);
  cfg.tracker_max_prediction_frames = raw.int_or("tracking.max_prediction_frames", 0);
  cfg.tracker_camera_motion_compensation =
      raw.bool_or("tracking.camera_motion_compensation", false);
  cfg.tracker_covariance_motion_enabled =
      raw.bool_or("tracking.covariance_motion_enabled", cfg.tracker_center_distance_enabled);
  cfg.tracker_overlap_threshold =
      static_cast<float>(raw.double_or("tracking.overlap_threshold", 0.20));
  cfg.tracker_max_occlusion_frames =
      raw.int_or("tracking.max_occlusion_frames", cfg.tracker_center_distance_enabled ? 10 : 0);
  cfg.tracker_max_active_tracks =
      raw.int_or("tracking.max_active_tracks", std::max(128, cfg.max_detections));
  cfg.insight_host = raw.string_or("output.insight.host", "");
  cfg.video_port_base = raw.int_or("output.insight.video_port_base", 9000);
  cfg.metadata_port_base = raw.int_or("output.insight.metadata_port_base", 9100);
  cfg.video_enabled = raw.bool_or("output.video_enabled", true);
  cfg.save_dir = raw.string_or("output.debug_dir", "");
  cfg.save_every = raw.int_or("output.save_every", 0);
  cfg.replay_dir = raw.string_or("output.replay_dir", "");
  validate_config(cfg);
  return cfg;
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

void filter_target_class_into(const std::vector<objdet::Box>& boxes, int target_class_id,
                              std::vector<Detection>& detections) {
  detections.clear();
  detections.reserve(boxes.size());
  for (const auto& box : boxes) {
    if (box.class_id != target_class_id) {
      continue;
    }
    detections.push_back(Detection{box.x1, box.y1, box.x2, box.y2, box.score, box.class_id});
  }
}

void build_metadata_tracks_into(const std::vector<TrackedDetection>& tracks, int frame_w,
                                int frame_h, const std::string& target_label,
                                std::vector<sima_examples::MetadataBox>& metadata_boxes) {
  metadata_boxes.clear();
  metadata_boxes.reserve(tracks.size());
  for (const auto& track : tracks) {
    const int x1 = std::clamp(static_cast<int>(std::lround(track.x1)), 0, frame_w);
    const int y1 = std::clamp(static_cast<int>(std::lround(track.y1)), 0, frame_h);
    const int x2 = std::clamp(static_cast<int>(std::lround(track.x2)), 0, frame_w);
    const int y2 = std::clamp(static_cast<int>(std::lround(track.y2)), 0, frame_h);

    sima_examples::MetadataBox obj;
    obj.id = std::to_string(track.track_id);
    obj.label = target_label;
    obj.confidence = track.score;
    obj.x = static_cast<float>(x1);
    obj.y = static_cast<float>(y1);
    obj.w = static_cast<float>(std::max(0, x2 - x1));
    obj.h = static_cast<float>(std::max(0, y2 - y1));
    metadata_boxes.push_back(obj);
  }
}

simaai::neat::nodes::groups::RtspDecodedInputOptions
build_source_options(const AppConfig& cfg, const std::string& url, int& fps_out, int& width_out,
                     int& height_out) {
  sima_examples::RtspStreamInfo probe;
  sima_examples::RtspProbeOptions probe_options;
  probe_options.payload_type = 96;
  probe_options.latency_ms = cfg.latency_ms;
  probe_options.rtsp_tcp = cfg.tcp;
  probe_options.debug = cfg.profile;
  (void)sima_examples::probe_rtsp_stream_info(url, probe_options, probe);

  simaai::neat::nodes::groups::RtspDecodedInputOptions opt;
  opt.url = url;
  opt.latency_ms = cfg.latency_ms;
  opt.tcp = cfg.tcp;
  opt.payload_type = 96;
  opt.insert_queue = true;
  opt.out_format = "NV12";
  opt.decoder_name = "decoder";
  opt.decoder_raw_output = true;
  opt.auto_caps_from_stream = true;
  opt.codec = cfg.codec;
  if (probe.width > 0 && probe.height > 0) {
    opt.dec_width = probe.width;
    opt.dec_height = probe.height;
    if (cfg.codec == simaai::neat::nodes::groups::RtspCodec::H264) {
      opt.fallback_h264_width = probe.width;
      opt.fallback_h264_height = probe.height;
    }
    width_out = probe.width;
    height_out = probe.height;
  }
  if (probe.fps > 0) {
    opt.source_fps = probe.fps;
    fps_out = probe.fps;
  }
  if (width_out > 0 && height_out > 0 && fps_out > 0) {
    opt.output_caps.enable = true;
    opt.output_caps.format = "NV12";
    opt.output_caps.width = width_out;
    opt.output_caps.height = height_out;
    opt.output_caps.fps = fps_out;
    opt.output_caps.memory = simaai::neat::CapsMemory::Any;
  }
  return opt;
}

bool output_caps_enabled(
    const simaai::neat::nodes::groups::RtspDecodedInputOptions::OutputCaps& caps) {
  return caps.enable || caps.width > 0 || caps.height > 0 || caps.fps > 0;
}

simaai::neat::FormatTag encoded_format_tag(simaai::neat::nodes::groups::RtspCodec codec) {
  return codec == simaai::neat::nodes::groups::RtspCodec::H265 ? simaai::neat::FormatTag::H265
                                                               : simaai::neat::FormatTag::H264;
}

simaai::neat::InputOptions
encoded_decode_input_options(simaai::neat::nodes::groups::RtspCodec codec) {
  simaai::neat::InputOptions opt;
  opt.payload_type = simaai::neat::PayloadType::Encoded;
  opt.format = encoded_format_tag(codec);
  opt.memory_policy = simaai::neat::InputMemoryPolicy::Ev74;
  return opt;
}

simaai::neat::InputOptions
encoded_video_input_options(simaai::neat::nodes::groups::RtspCodec codec) {
  simaai::neat::InputOptions opt;
  opt.payload_type = simaai::neat::PayloadType::Encoded;
  opt.format = encoded_format_tag(codec);
  opt.memory_policy = simaai::neat::InputMemoryPolicy::SystemMemory;
  return opt;
}

simaai::neat::Graph
build_encoded_source_graph(const simaai::neat::nodes::groups::RtspDecodedInputOptions& opt) {
  simaai::neat::Graph source("rtsp_encoded_source");

  simaai::neat::nodes::groups::RtspEncodedInputOptions encoded_opt;
  encoded_opt.url = opt.url;
  encoded_opt.codec = opt.codec;
  encoded_opt.latency_ms = opt.latency_ms;
  encoded_opt.tcp = opt.tcp;
  encoded_opt.source_fps = opt.source_fps;
  if (opt.codec == simaai::neat::nodes::groups::RtspCodec::H264) {
    encoded_opt.fallback_h264_width = opt.fallback_h264_width;
    encoded_opt.fallback_h264_height = opt.fallback_h264_height;
  }
  source.add(simaai::neat::nodes::groups::RtspEncodedInput(encoded_opt));
  return source;
}

simaai::neat::Graph
build_decode_graph(const std::string& input_name,
                   const simaai::neat::nodes::groups::RtspDecodedInputOptions& opt) {
  simaai::neat::Graph decode("decode");
  const bool use_h265 = opt.codec == simaai::neat::nodes::groups::RtspCodec::H265;

  simaai::neat::SimaDecodeOptions dec;
  dec.type = use_h265 ? simaai::neat::SimaDecodeType::H265 : simaai::neat::SimaDecodeType::H264;
  dec.sima_allocator_type = opt.sima_allocator_type;
  dec.out_format = opt.out_format;
  dec.decoder_name = opt.decoder_name;
  dec.raw_output = opt.decoder_raw_output;
  dec.next_element = opt.decoder_next_element;
  dec.dec_width = opt.dec_width;
  dec.dec_height = opt.dec_height;
  dec.dec_fps = opt.source_fps;
  dec.num_buffers = opt.num_buffers;
  dec.input_buffers = opt.decoder_input_buffers;
  dec.decoder_tuning = opt.decoder_tuning;
  dec.memory_opt = opt.decoder_memory_opt;

  decode.connect(simaai::neat::nodes::Input(input_name, encoded_decode_input_options(opt.codec)),
                 simaai::neat::nodes::SimaDecode(dec));
  if (opt.use_videoconvert) {
    decode.add(simaai::neat::nodes::VideoConvert());
  }
  if (opt.use_videoscale) {
    decode.add(simaai::neat::nodes::VideoScale());
  }
  if (output_caps_enabled(opt.output_caps)) {
    const auto& caps = opt.output_caps;
    decode.add(
        simaai::neat::nodes::CapsRaw(caps.format, caps.width, caps.height, caps.fps, caps.memory));
  }
  if (!opt.extra_fragment.empty()) {
    decode.add(simaai::neat::nodes::Custom(opt.extra_fragment));
  }
  return decode;
}

simaai::neat::Graph
build_video_sender_graph(const std::string& input_name,
                         simaai::neat::nodes::groups::RtspCodec codec,
                         const simaai::neat::nodes::groups::VideoSenderOptions& video_options) {
  simaai::neat::Graph video("video_sender");
  video.connect(simaai::neat::nodes::Input(input_name, encoded_video_input_options(codec)),
                simaai::neat::nodes::groups::VideoSender(video_options));
  return video;
}

std::unique_ptr<simaai::neat::Model> build_model(const AppConfig& cfg) {
  simaai::neat::Model::Options model_opt;
  model_opt.preprocess.kind = simaai::neat::InputKind::Image;
  model_opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::NV12;
  model_opt.preprocess.preset = simaai::neat::NormalizePreset::COCO_YOLO;
  model_opt.decode_type = simaai::neat::BoxDecodeType::YoloV26;
  model_opt.num_classes = cfg.num_classes;
  model_opt.score_threshold = cfg.min_score;
  model_opt.nms_iou_threshold = cfg.nms_iou;
  model_opt.top_k = cfg.max_detections;
  return std::make_unique<simaai::neat::Model>(cfg.model_path, model_opt);
}

simaai::neat::RunOptions build_run_options() {
  simaai::neat::RunOptions run_options;
  run_options.preset = simaai::neat::RunPreset::Realtime;
  return run_options;
}

bool save_frames_enabled(const AppConfig& cfg) {
  return !cfg.save_dir.empty() && cfg.save_every > 0;
}

bool tracking_frames_enabled(const AppConfig& cfg) {
  return save_frames_enabled(cfg) || cfg.tracker_camera_motion_compensation;
}

std::string stream_id_for(int stream_index) {
  return "stream" + std::to_string(stream_index);
}

int stream_index_from_sample(const simaai::neat::Sample& sample, int stream_count) {
  const std::string prefix = "stream";
  if (sample.stream_id.rfind(prefix, 0) != 0) {
    if (stream_count == 1) {
      return 0;
    }
    throw std::runtime_error("detection sample missing stream id: " + sample.stream_id);
  }
  const std::string suffix = sample.stream_id.substr(prefix.size());
  if (suffix.empty() || !std::all_of(suffix.begin(), suffix.end(),
                                     [](unsigned char c) { return std::isdigit(c) != 0; })) {
    throw std::runtime_error("invalid detection stream id: " + sample.stream_id);
  }
  const int index = std::stoi(suffix);
  if (index < 0 || index >= stream_count) {
    throw std::runtime_error("detection stream id out of range: " + sample.stream_id);
  }
  return index;
}

simaai::neat::GraphLinkOptions stream_link(const AppConfig& cfg, int stream_index, int queue_depth,
                                           int max_inflight_per_stream = -1,
                                           int max_inflight_total = -1) {
  simaai::neat::GraphLinkOptions link;
  link.policy = cfg.overflow_policy == RuntimeOverflowPolicy::Block
                    ? simaai::neat::GraphLinkPolicy::Default
                    : simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
  link.queue_depth = queue_depth;
  link.max_inflight_per_stream = max_inflight_per_stream;
  link.max_inflight_total = max_inflight_total;
  link.stream_id = stream_id_for(stream_index);
  return link;
}

simaai::neat::Graph build_detector_graph(const AppConfig& cfg,
                                         std::unique_ptr<simaai::neat::Model>& model) {
  model = build_model(cfg);
  auto input_options = model->input_appsrc_options(false);
  input_options.block = true;

  simaai::neat::Graph detector("detector");
  detector.connect(simaai::neat::nodes::Input("detector_frame", input_options), *model);
  return detector;
}

simaai::neat::Graph build_detections_graph() {
  simaai::neat::Graph detections("detections");
  detections.add(
      simaai::neat::nodes::Output("detections", simaai::neat::OutputOptions::EveryFrame(4)));
  return detections;
}

simaai::neat::Graph build_debug_frame_graph(int stream_index) {
  simaai::neat::Graph frames("debug_frame");
  frames.connect(simaai::neat::nodes::Input("debug_frame"),
                 simaai::neat::nodes::Output("debug_frame_" + std::to_string(stream_index),
                                             simaai::neat::OutputOptions::EveryFrame(4)));
  return frames;
}

simaai::neat::nodes::groups::VideoSenderOptions make_video_options(const AppConfig& cfg,
                                                                   int stream_index) {
  auto video_options = simaai::neat::nodes::groups::VideoSenderOptions::Passthrough(cfg.codec);
  video_options.host = cfg.insight_host;
  video_options.channel = stream_index;
  video_options.video_port_base = cfg.video_port_base;
  video_options.async = true;
  return video_options;
}

StreamRuntime make_stream_runtime(const AppConfig& cfg, int stream_index, std::string url) {
  StreamRuntime runtime;
  runtime.index = stream_index;
  runtime.url = std::move(url);
  runtime.debug_frames = std::make_unique<DebugFrameQueue>();
  runtime.tracker = ObjectTracker(TrackerConfig{
      static_cast<float>(cfg.tracker_high_score),
      static_cast<float>(cfg.tracker_new_track_score),
      cfg.tracker_iou_threshold,
      cfg.tracker_max_center_distance,
      cfg.tracker_velocity_momentum,
      cfg.tracker_box_smoothing_alpha,
      cfg.tracker_max_missing,
      cfg.tracker_min_confirmed_hits,
      cfg.tracker_max_prediction_frames,
      cfg.tracker_center_distance_enabled,
      cfg.tracker_camera_motion_compensation,
      cfg.tracker_covariance_motion_enabled,
      cfg.tracker_overlap_threshold,
      cfg.tracker_max_occlusion_frames,
      cfg.tracker_max_active_tracks,
  });
  if (cfg.tracker_camera_motion_compensation) {
    runtime.camera_motion = std::make_unique<AsyncCameraMotionEstimator>();
  }
  if (!cfg.replay_dir.empty()) {
    const fs::path replay_path =
        cfg.replay_dir / ("stream_" + std::to_string(stream_index) + ".jsonl");
    runtime.replay_writer =
        std::make_unique<std::ofstream>(replay_path, std::ios::out | std::ios::trunc);
    sima_examples::require(runtime.replay_writer->is_open(),
                           "failed to open output.replay_dir file: " + replay_path.string());
  }
  runtime.profile.enabled = cfg.profile;
  runtime.profile.stream_index = stream_index;
  if (save_frames_enabled(cfg)) {
    runtime.debug_writer = std::make_unique<DebugFrameWriter>();
  }
  return runtime;
}

StreamRuntime build_stream_runtime(const AppConfig& cfg, int stream_index, const std::string& url) {
  StreamRuntime runtime = make_stream_runtime(cfg, stream_index, url);
  const auto source_options =
      build_source_options(cfg, url, runtime.output_fps, runtime.frame_w, runtime.frame_h);
  sima_examples::require(runtime.frame_w > 0 && runtime.frame_h > 0,
                         "failed to probe RTSP frame dimensions");
  sima_examples::require(runtime.output_fps > 0, "failed to probe RTSP frame rate");
  if (cfg.fps > 0) {
    runtime.output_fps = cfg.fps;
  }

  runtime.source_options = source_options;
  if (cfg.video_enabled) {
    runtime.video_port = make_video_options(cfg, stream_index).video_port();
  }

  simaai::neat::MetadataSenderOptions metadata_options;
  metadata_options.host = cfg.insight_host;
  metadata_options.channel = stream_index;
  metadata_options.metadata_port_base = cfg.metadata_port_base;
  std::string metadata_err;
  runtime.metadata_sender =
      std::make_unique<simaai::neat::MetadataSender>(metadata_options, &metadata_err);
  sima_examples::require(runtime.metadata_sender->ok(), metadata_err);

  std::cout << "[stream " << stream_index << "] rtsp=" << url << " stream=" << runtime.frame_w
            << "x" << runtime.frame_h << "@" << runtime.output_fps
            << " insight=" << cfg.insight_host << " video=";
  if (cfg.video_enabled) {
    std::cout << runtime.video_port;
  } else {
    std::cout << "disabled";
  }
  std::cout << " metadata=" << runtime.metadata_sender->metadata_port() << "\n";
  return runtime;
}

StreamRuntime build_benchmark_stream_runtime(const AppConfig& cfg, int width, int height,
                                             const fs::path& image_path) {
  StreamRuntime runtime = make_stream_runtime(cfg, 0, "image:" + image_path.string());
  runtime.frame_w = width;
  runtime.frame_h = height;
  // PTS expresses a stable logical frame interval to the tracker. It does not
  // pace input; the producer is governed only by graph backpressure.
  runtime.output_fps = cfg.fps > 0 ? cfg.fps : 30;
  runtime.benchmark_mode = true;
  std::cout << "[stream 0] benchmark_image=" << image_path << " stream=" << width << "x" << height
            << " logical_fps=" << runtime.output_fps
            << " pacing=disabled video=disabled metadata=disabled overlay=disabled\n";
  return runtime;
}

void connect_stream_graph(AppRuntime& app, const AppConfig& cfg, const StreamRuntime& stream,
                          const simaai::neat::Graph& detector_graph) {
  auto source = build_encoded_source_graph(stream.source_options);
  auto decoder = build_decode_graph("decode_h264", stream.source_options);

  if (cfg.video_enabled) {
    auto encoded_branch = simaai::neat::graphs::Branch("encoded", {"decode_h264", "video_h264"});
    app.graph.connect(source, encoded_branch);
    app.graph.connect(encoded_branch, decoder, stream_link(cfg, stream.index, 3));

    const auto video_options = make_video_options(cfg, stream.index);
    app.graph.connect(encoded_branch,
                      build_video_sender_graph("video_h264", cfg.codec, video_options),
                      stream_link(cfg, stream.index, 3));
  } else {
    app.graph.connect(source, decoder, stream_link(cfg, stream.index, 3));
  }

  const bool save_debug_frames = tracking_frames_enabled(cfg);
  auto decoded_branch =
      save_debug_frames ? simaai::neat::graphs::Branch("decoded", {"detector_frame", "debug_frame"})
                        : simaai::neat::graphs::Branch("decoded", {"detector_frame"});
  app.graph.connect(decoder, decoded_branch);
  app.graph.connect(
      decoded_branch, detector_graph,
      stream_link(cfg, stream.index, 4, cfg.max_inflight_per_stream, cfg.max_inflight_total));
  if (save_debug_frames) {
    app.graph.connect(decoded_branch, build_debug_frame_graph(stream.index),
                      stream_link(cfg, stream.index, 4));
  }
}

void connect_benchmark_image_graph(AppRuntime& app, const AppConfig& cfg,
                                   const StreamRuntime& stream,
                                   const simaai::neat::Graph& detector_graph) {
  auto input_options = app.model->input_appsrc_options(false);
  input_options.block = true;
  simaai::neat::Graph source("benchmark_image");
  source.add(simaai::neat::nodes::Input("benchmark_frame", input_options));

  const bool tracking_frames = tracking_frames_enabled(cfg);
  auto branch =
      tracking_frames
          ? simaai::neat::graphs::Branch("benchmark_decoded", {"detector_frame", "debug_frame"})
          : simaai::neat::graphs::Branch("benchmark_decoded", {"detector_frame"});
  app.graph.connect(source, branch);
  app.graph.connect(
      branch, detector_graph,
      stream_link(cfg, stream.index, 4, cfg.max_inflight_per_stream, cfg.max_inflight_total));
  if (tracking_frames) {
    app.graph.connect(branch, build_debug_frame_graph(stream.index), stream_link(cfg, 0, 4));
  }
}

void send_metadata(StreamRuntime& stream, const AppConfig& cfg, const simaai::neat::Sample& sample,
                   const std::vector<TrackedDetection>& tracks) {
  build_metadata_tracks_into(tracks, stream.frame_w, stream.frame_h, cfg.target_label,
                             stream.metadata_tracks);
  const std::string data_json =
      sima_examples::metadata_boxes_data_json("tracks", stream.metadata_tracks);
  const int64_t timestamp_ms = sample.pts_ns >= 0 ? sample.pts_ns / 1'000'000 : -1;
  const std::string frame_id = sample.frame_id >= 0 ? std::to_string(sample.frame_id) : "";
  std::string err;
  if (!stream.metadata_sender->send_metadata("tracking", data_json, timestamp_ms, frame_id, &err)) {
    std::cerr << "[warn] stream " << stream.index << " metadata send failed: " << err << "\n";
  }
}

void write_replay_record(StreamRuntime& stream, int tracking_frame_index,
                         const simaai::neat::Sample& sample,
                         const multi_stream_people_tracker::CameraTransform& camera_transform) {
  if (stream.replay_writer == nullptr) {
    return;
  }
  auto& output = *stream.replay_writer;
  output << std::setprecision(9) << "{\"frame_index\":" << tracking_frame_index
         << ",\"processed_index\":" << stream.processed << ",\"frame_id\":" << sample.frame_id
         << ",\"pts_ns\":" << sample.pts_ns << ",\"camera_transform\":";
  if (camera_transform.valid) {
    output << '[' << camera_transform.a << ',' << camera_transform.b << ',' << camera_transform.tx
           << ',' << camera_transform.c << ',' << camera_transform.d << ',' << camera_transform.ty
           << ']';
  } else {
    output << "null";
  }
  output << ",\"camera_diagnostics\":{\"valid\":" << (camera_transform.valid ? "true" : "false")
         << ",\"confidence\":" << camera_transform.confidence
         << ",\"reprojection_error\":" << camera_transform.reprojection_error
         << ",\"inliers\":" << camera_transform.inliers << "},\"detections\":[";
  for (std::size_t index = 0; index < stream.detections.size(); ++index) {
    const auto& detection = stream.detections[index];
    if (index != 0) {
      output << ',';
    }
    output << "{\"x1\":" << detection.x1 << ",\"y1\":" << detection.y1 << ",\"x2\":" << detection.x2
           << ",\"y2\":" << detection.y2 << ",\"score\":" << detection.score
           << ",\"class_id\":" << detection.class_id << '}';
  }
  output << "],\"tracks\":[";
  for (std::size_t index = 0; index < stream.tracks.size(); ++index) {
    const auto& track = stream.tracks[index];
    if (index != 0) {
      output << ',';
    }
    output << "{\"id\":\"" << track.track_id << "\",\"bbox\":[" << track.x1 << ',' << track.y1
           << ',' << track.x2 - track.x1 << ',' << track.y2 - track.y1
           << "],\"confidence\":" << track.score << ",\"class_id\":" << track.class_id
           << ",\"predicted\":" << (track.predicted ? "true" : "false")
           << ",\"occluded\":" << (track.occluded ? "true" : "false")
           << ",\"association_confidence\":" << track.association_confidence << '}';
  }
  output << "]}\n";
}

void maybe_save_debug_frame(const AppConfig& cfg, StreamRuntime& stream, const cv::Mat* frame,
                            const std::vector<TrackedDetection>& tracks) {
  if (cfg.save_dir.empty() || cfg.save_every <= 0 || stream.processed % cfg.save_every != 0) {
    return;
  }
  if (frame == nullptr || frame->empty()) {
    return;
  }

  cv::Mat bgr = frame->clone();
  draw_tracks_bgr(bgr, tracks, static_cast<float>(cfg.min_score));
  const auto out_path = cfg.save_dir / ("stream_" + std::to_string(stream.index) + "_frame_" +
                                        std::to_string(stream.processed) + ".jpg");
  if (stream.debug_writer != nullptr) {
    stream.debug_writer->enqueue(out_path, std::move(bgr));
  }
}

bool all_streams_done(const std::vector<StreamRuntime>& streams, int frame_limit) {
  if (frame_limit <= 0) {
    return false;
  }
  return std::all_of(streams.begin(), streams.end(), [frame_limit](const StreamRuntime& stream) {
    return stream.processed >= frame_limit || stream.closed;
  });
}

bool samples_identify_same_frame(const simaai::neat::Sample& sample,
                                 const DebugFrameSample& frame) {
  if (sample.pts_ns >= 0 && frame.pts_ns >= 0) {
    return sample.pts_ns == frame.pts_ns;
  }
  if (sample.orig_input_seq >= 0 && frame.orig_input_seq >= 0) {
    return sample.orig_input_seq == frame.orig_input_seq;
  }
  if (sample.input_seq >= 0 && frame.input_seq >= 0) {
    return sample.input_seq == frame.input_seq;
  }
  return sample.frame_id >= 0 && frame.frame_id >= 0 && sample.frame_id == frame.frame_id;
}

int tracker_frame_index(StreamRuntime& stream, const simaai::neat::Sample& sample) {
  const int64_t source_sequence = sample.frame_id >= 0         ? sample.frame_id
                                  : sample.orig_input_seq >= 0 ? sample.orig_input_seq
                                  : sample.input_seq >= 0      ? sample.input_seq
                                                               : -1;
  if (stream.tracking_frame_index < 0) {
    stream.tracking_frame_index = 0;
  } else {
    int64_t elapsed_frames = 1;
    if (source_sequence >= 0 && stream.last_source_sequence >= 0 &&
        source_sequence > stream.last_source_sequence) {
      elapsed_frames = source_sequence - stream.last_source_sequence;
    } else if (sample.pts_ns >= 0 && stream.last_tracking_pts_ns >= 0 &&
               sample.pts_ns > stream.last_tracking_pts_ns && stream.output_fps > 0) {
      const double elapsed = static_cast<double>(sample.pts_ns - stream.last_tracking_pts_ns) *
                             static_cast<double>(stream.output_fps) / 1.0e9;
      elapsed_frames = std::max<int64_t>(1, static_cast<int64_t>(std::llround(elapsed)));
    }
    const int64_t available =
        static_cast<int64_t>(std::numeric_limits<int>::max()) - stream.tracking_frame_index;
    sima_examples::require(elapsed_frames <= available, "tracker frame index overflow");
    stream.tracking_frame_index += static_cast<int>(elapsed_frames);
  }
  if (source_sequence >= 0) {
    stream.last_source_sequence = source_sequence;
  }
  if (sample.pts_ns >= 0) {
    stream.last_tracking_pts_ns = sample.pts_ns;
  }
  return stream.tracking_frame_index;
}

multi_stream_people_tracker::CameraTransform
compose_camera_transforms(const multi_stream_people_tracker::CameraTransform& previous_to_middle,
                          const multi_stream_people_tracker::CameraTransform& middle_to_current) {
  using multi_stream_people_tracker::CameraTransform;
  if (!previous_to_middle.valid || !middle_to_current.valid) {
    return {};
  }
  const float confidence = std::clamp(previous_to_middle.confidence, 0.0f, 1.0f) *
                           std::clamp(middle_to_current.confidence, 0.0f, 1.0f);
  return CameraTransform{
      middle_to_current.a * previous_to_middle.a + middle_to_current.b * previous_to_middle.c,
      middle_to_current.a * previous_to_middle.b + middle_to_current.b * previous_to_middle.d,
      middle_to_current.a * previous_to_middle.tx + middle_to_current.b * previous_to_middle.ty +
          middle_to_current.tx,
      middle_to_current.c * previous_to_middle.a + middle_to_current.d * previous_to_middle.c,
      middle_to_current.c * previous_to_middle.b + middle_to_current.d * previous_to_middle.d,
      middle_to_current.c * previous_to_middle.tx + middle_to_current.d * previous_to_middle.ty +
          middle_to_current.ty,
      true,
      confidence,
      std::hypot(previous_to_middle.reprojection_error, middle_to_current.reprojection_error),
      std::min(previous_to_middle.inliers, middle_to_current.inliers)};
}

void process_output_sample(StreamRuntime& stream, const AppConfig& cfg,
                           const simaai::neat::Sample& sample, double detection_pull_ms) {
  if (cfg.frames > 0 && stream.processed >= cfg.frames) {
    return;
  }

  std::string err;
  if (!extract_bbox_payload(sample, stream.payload, err)) {
    throw std::runtime_error("stream " + std::to_string(stream.index) +
                             " bbox extract failed: " + err);
  }
  const double box_parse_start = sima_examples::time_ms();
  objdet::parse_boxes_strict_into(stream.payload, stream.frame_w, stream.frame_h,
                                  cfg.max_detections, false, stream.boxes);
  filter_target_class_into(stream.boxes, cfg.target_class_id, stream.detections);
  const double box_parse_end = sima_examples::time_ms();

  std::optional<DebugFrameSample> frame;
  std::vector<std::future<CameraMotionResult>> camera_motion_chain;
  bool camera_motion_chain_discontinuous = false;
  const double debug_frame_wait_start = sima_examples::time_ms();
  if (tracking_frames_enabled(cfg)) {
    auto& frames = *stream.debug_frames;
    std::unique_lock lock(frames.mutex);
    frames.object_mask = stream.detections;
    const auto find_matching = [&] {
      return std::find_if(frames.frames.begin(), frames.frames.end(),
                          [&sample](const DebugFrameSample& candidate) {
                            return samples_identify_same_frame(sample, candidate);
                          });
    };
    constexpr auto kMaximumSideBranchWait = std::chrono::milliseconds(50);
    frames.ready.wait_for(lock, kMaximumSideBranchWait,
                          [&] { return find_matching() != frames.frames.end(); });
    const auto matching = find_matching();
    if (matching != frames.frames.end()) {
      camera_motion_chain.reserve(
          static_cast<std::size_t>(std::distance(frames.frames.begin(), std::next(matching))));
      for (auto current = frames.frames.begin(); current != std::next(matching); ++current) {
        if (current->camera_motion.valid()) {
          camera_motion_chain.push_back(std::move(current->camera_motion));
        }
      }
      frame.emplace(std::move(*matching));
      frames.frames.erase(frames.frames.begin(), std::next(matching));
      frames.space_available.notify_one();
      camera_motion_chain_discontinuous = frames.chain_discontinuous;
      frames.chain_discontinuous = false;
    }
  }
  const double debug_frame_wait_ms = sima_examples::time_ms() - debug_frame_wait_start;
  double camera_motion_compute_ms = 0.0;
  double camera_motion_wait_ms = 0.0;
  multi_stream_people_tracker::CameraTransform camera_transform;
  if (cfg.tracker_camera_motion_compensation) {
    if (frame.has_value()) {
      const double camera_wait_start = sima_examples::time_ms();
      bool complete_chain = !camera_motion_chain_discontinuous;
      bool have_transform = false;
      for (auto& future : camera_motion_chain) {
        CameraMotionResult result = future.get();
        camera_motion_compute_ms += result.compute_ms;
        if (result.chain_discontinuous) {
          complete_chain = false;
        }
        if (!result.transform.valid) {
          complete_chain = false;
          continue;
        }
        camera_transform = have_transform
                               ? compose_camera_transforms(camera_transform, result.transform)
                               : result.transform;
        have_transform = true;
      }
      camera_motion_wait_ms = sima_examples::time_ms() - camera_wait_start;
      if (!stream.has_processed_debug_frame || !complete_chain || !have_transform) {
        camera_transform = {};
      }
      stream.has_processed_debug_frame = true;
    } else {
      // The estimator remains ordered independently. Mark the next matched
      // decoded frame as a new baseline; detection consensus handles this
      // isolated gap without consuming a stale one-frame transform.
      stream.has_processed_debug_frame = false;
    }
  }
  const int tracking_frame = tracker_frame_index(stream, sample);
  const double tracker_start = sima_examples::time_ms();
  stream.tracker.update_into(stream.detections, tracking_frame, stream.tracks, camera_transform);
  const double tracker_end = sima_examples::time_ms();
  write_replay_record(stream, tracking_frame, sample, camera_transform);

  ++stream.processed;
  const bool warming_up = stream.processed <= cfg.warmup_frames;
  if (!warming_up) {
    double metadata_ms = 0.0;
    if (stream.metadata_sender != nullptr) {
      const double metadata_start = sima_examples::time_ms();
      send_metadata(stream, cfg, sample, stream.tracks);
      metadata_ms = sima_examples::time_ms() - metadata_start;
    }
    double overlay_ms = 0.0;
    if (save_frames_enabled(cfg)) {
      const double overlay_start = sima_examples::time_ms();
      if (frame.has_value()) {
        maybe_save_debug_frame(cfg, stream, &frame->bgr, stream.tracks);
      } else {
        std::cerr << "[warn] stream " << stream.index << " has no decoded frame matching detection"
                  << " frame_id=" << sample.frame_id << " pts_ns=" << sample.pts_ns << "\n";
      }
      overlay_ms = sima_examples::time_ms() - overlay_start;
    }
    stream.profile.add(detection_pull_ms, box_parse_end - box_parse_start, debug_frame_wait_ms,
                       camera_motion_compute_ms, camera_motion_wait_ms, tracker_end - tracker_start,
                       metadata_ms, overlay_ms, static_cast<int>(stream.tracks.size()));
    if (stream.benchmark_mode) {
      const double completed_ms = sima_examples::time_ms();
      if (stream.benchmark_completed_frames == 0) {
        stream.benchmark_first_completed_ms = completed_ms;
      }
      stream.benchmark_last_completed_ms = completed_ms;
      ++stream.benchmark_completed_frames;
    }
  }
}

std::string debug_frame_output_name(int stream_index) {
  return "debug_frame_" + std::to_string(stream_index);
}

bool pull_debug_frame(AppRuntime& app, StreamRuntime& stream, const AppConfig& cfg,
                      int timeout_ms) {
  const std::string output_name = debug_frame_output_name(stream.index);
  simaai::neat::Sample sample;
  simaai::neat::PullError pull_error;
  const auto status = app.run.pull(output_name, timeout_ms, sample, &pull_error);
  if (status == simaai::neat::PullStatus::Timeout || status == simaai::neat::PullStatus::Closed) {
    return false;
  }
  if (status != simaai::neat::PullStatus::Ok) {
    throw std::runtime_error("failed to pull " + output_name + ": " + pull_error.message);
  }
  const auto tensors = simaai::neat::tensors_from_sample(sample, false);
  if (tensors.empty()) {
    return true;
  }
  const auto& tensor = tensors.front();
  const int width = tensor.width() > 0 ? tensor.width() : stream.frame_w;
  const int height = tensor.height() > 0 ? tensor.height() : stream.frame_h;
  if (!tensor.is_nv12() || width <= 0 || height <= 0) {
    std::cerr << "[warn] failed to prepare tracking frame: expected a sized NV12 tensor\n";
    return true;
  }
  std::vector<std::uint8_t> nv12 = tensor.copy_nv12_contiguous();
  if (nv12.size() < static_cast<std::size_t>(width * height * 3 / 2)) {
    std::cerr << "[warn] failed to prepare tracking frame: NV12 copy is incomplete\n";
    return true;
  }
  cv::Mat yuv(height + height / 2, width, CV_8UC1, nv12.data());
  cv::Mat gray = yuv(cv::Rect(0, 0, width, height)).clone();
  cv::Mat bgr;
  if (save_frames_enabled(cfg)) {
    cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_NV12);
  }
  std::vector<Detection> object_mask;
  {
    std::lock_guard lock(stream.debug_frames->mutex);
    object_mask = stream.debug_frames->object_mask;
  }
  std::future<CameraMotionResult> camera_motion;
  if (cfg.tracker_camera_motion_compensation) {
    // The latest accepted detections are a conservative foreground mask. The
    // targets occupy very little of this scene, and the sparse-flow estimator
    // still validates every point forward and backward before RANSAC.
    camera_motion = stream.camera_motion->submit(gray, std::move(object_mask));
  }
  {
    std::unique_lock lock(stream.debug_frames->mutex);
    constexpr std::size_t kMaxBufferedDebugFrames = 16;
    if (stream.benchmark_mode) {
      // An uncapped image source can outrun MLA on the side branch. Preserve
      // frame pairing by applying bounded backpressure instead of evicting the
      // exact frames whose detections are still in flight.
      stream.debug_frames->space_available.wait_for(lock, std::chrono::milliseconds(100), [&] {
        return stream.debug_frames->frames.size() < kMaxBufferedDebugFrames;
      });
    }
    if (stream.debug_frames->frames.size() >= kMaxBufferedDebugFrames) {
      stream.debug_frames->frames.pop_front();
      stream.debug_frames->chain_discontinuous = true;
    }
    stream.debug_frames->frames.push_back(
        DebugFrameSample{sample.frame_id, sample.input_seq, sample.orig_input_seq, sample.pts_ns,
                         std::move(gray), std::move(bgr), std::move(camera_motion)});
  }
  stream.debug_frames->ready.notify_one();
  return true;
}

bool process_run_once(AppRuntime& app, const AppConfig& cfg, const std::string& output_name) {
  constexpr int kPullTimeoutMs = 50;
  const double pull_start = sima_examples::time_ms();
  simaai::neat::Sample sample;
  simaai::neat::PullError pull_error;
  const auto status = app.run.pull(output_name, kPullTimeoutMs, sample, &pull_error);
  const double pull_end = sima_examples::time_ms();
  if (status == simaai::neat::PullStatus::Timeout || status == simaai::neat::PullStatus::Closed) {
    return false;
  }
  if (status != simaai::neat::PullStatus::Ok) {
    throw std::runtime_error("failed to pull " + output_name + ": " + pull_error.message);
  }
  const int stream_index = stream_index_from_sample(sample, static_cast<int>(app.streams.size()));
  process_output_sample(app.streams[static_cast<std::size_t>(stream_index)], cfg, sample,
                        pull_end - pull_start);
  return true;
}

simaai::neat::Tensor load_benchmark_image_nv12(const fs::path& image_path) {
  const cv::Mat bgr = cv::imread(image_path.string(), cv::IMREAD_COLOR);
  sima_examples::require(!bgr.empty(), "failed to read benchmark image: " + image_path.string());
  sima_examples::require(bgr.cols > 0 && bgr.rows > 0 && bgr.cols % 2 == 0 && bgr.rows % 2 == 0,
                         "benchmark image width and height must be positive even values");

  cv::Mat i420;
  cv::cvtColor(bgr, i420, cv::COLOR_BGR2YUV_I420);
  sima_examples::require(i420.isContinuous(), "benchmark image conversion is not contiguous");

  const int width = bgr.cols;
  const int height = bgr.rows;
  const std::size_t y_bytes = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  const std::size_t chroma_plane_bytes = y_bytes / 4U;
  std::vector<std::uint8_t> nv12(y_bytes + chroma_plane_bytes * 2U);

  const auto* source = i420.ptr<std::uint8_t>();
  auto* destination = nv12.data();
  std::memcpy(destination, source, y_bytes);
  const auto* source_u = source + y_bytes;
  const auto* source_v = source_u + chroma_plane_bytes;
  auto* destination_uv = destination + y_bytes;
  for (std::size_t index = 0; index < chroma_plane_bytes; ++index) {
    destination_uv[index * 2U] = source_u[index];
    destination_uv[index * 2U + 1U] = source_v[index];
  }

  // Transfer once before timing. Reusing an EV74-backed tensor keeps the
  // producer loop free of the compatibility CPU-to-device copy that the
  // production device-visible input route deliberately rejects.
  simaai::neat::Tensor tensor = simaai::neat::Tensor::from_vector(nv12, {height * 3 / 2, width},
                                                                  simaai::neat::TensorMemory::EV74);
  tensor.dtype = simaai::neat::TensorDType::UInt8;
  tensor.layout = simaai::neat::TensorLayout::HW;
  tensor.shape = {height, width};
  tensor.axis_semantics = {simaai::neat::TensorAxisSemantic::H,
                           simaai::neat::TensorAxisSemantic::W};
  tensor.read_only = true;
  tensor.semantic.image = simaai::neat::ImageSpec{simaai::neat::ImageSpec::PixelFormat::NV12, ""};

  simaai::neat::Plane y;
  y.role = simaai::neat::PlaneRole::Y;
  y.shape = {height, width};
  y.strides_bytes = {width, 1};
  y.byte_offset = 0;
  simaai::neat::Plane uv;
  uv.role = simaai::neat::PlaneRole::UV;
  uv.shape = {height / 2, width};
  uv.strides_bytes = {width, 1};
  uv.byte_offset = static_cast<int64_t>(y_bytes);
  tensor.planes = {std::move(y), std::move(uv)};
  return tensor;
}

simaai::neat::Sample make_benchmark_sample(const simaai::neat::Tensor& tensor, int64_t frame_id,
                                           int logical_fps) {
  simaai::neat::Sample sample;
  sample.kind = simaai::neat::SampleKind::TensorSet;
  sample.tensors = {tensor};
  sample.payload_type = simaai::neat::PayloadType::Image;
  sample.media_type = "video/x-raw";
  sample.payload_tag = "NV12";
  sample.format = "NV12";
  sample.frame_id = frame_id;
  sample.stream_id = stream_id_for(0);
  sample.stream_label = sample.stream_id;
  const int64_t frame_duration_ns = 1'000'000'000LL / std::max(logical_fps, 1);
  sample.pts_ns = frame_id * frame_duration_ns;
  sample.duration_ns = frame_duration_ns;
  return sample;
}

void run_app(AppConfig cfg, const std::optional<fs::path>& benchmark_image) {
  const bool benchmark_mode = benchmark_image.has_value();
  if (benchmark_mode) {
    sima_examples::require(cfg.rtsp_urls.size() == 1,
                           "--benchmark-image requires a single-stream configuration");
    sima_examples::require(cfg.frames > cfg.warmup_frames + 1,
                           "--benchmark-image requires inference.frames to exceed "
                           "runtime.warmup_frames by at least two frames");
    // The benchmark measures every submitted frame and intentionally performs
    // no visualization or output I/O. Keep the detector and tracker settings
    // identical while making admission lossless.
    cfg.overflow_policy = RuntimeOverflowPolicy::Block;
    cfg.video_enabled = false;
    cfg.save_dir.clear();
    cfg.save_every = 0;
    cfg.replay_dir.clear();
    // The source tensor is immutable and already device-backed. Keep Graph's
    // defensive queue fallback from trying to clone the shared NV12 holder to
    // CPU, which is both invalid for this route and outside the workload being
    // measured. GStreamer creates writable metadata-only views per push while
    // retaining the same device payload.
    setenv("SIMA_GRAPH_ZERO_COPY_BACKPRESSURE_CAP", "0", 1);
  }
  g_stop_requested = 0;
  auto previous_sigint = std::signal(SIGINT, request_stop);
  if (cfg.profile) {
    setenv("SIMA_GST_ELEMENT_TIMINGS", "1", 0);
    setenv("SIMA_GST_FLOW_DEBUG", "1", 0);
    setenv("SIMA_GST_BOUNDARY_PROBES", "1", 0);
  }
  if (save_frames_enabled(cfg)) {
    fs::create_directories(cfg.save_dir);
  }
  if (!cfg.replay_dir.empty()) {
    fs::create_directories(cfg.replay_dir);
  }

  AppRuntime app;
  app.streams.reserve(benchmark_mode ? 1U : cfg.rtsp_urls.size());
  auto detector_graph = build_detector_graph(cfg, app.model);
  auto detections_graph = build_detections_graph();
  std::optional<simaai::neat::Tensor> benchmark_tensor;

  if (benchmark_mode) {
    benchmark_tensor = load_benchmark_image_nv12(*benchmark_image);
    const int height = static_cast<int>(benchmark_tensor->shape.at(0));
    const int width = static_cast<int>(benchmark_tensor->shape.at(1));
    app.streams.push_back(build_benchmark_stream_runtime(cfg, width, height, *benchmark_image));
    connect_benchmark_image_graph(app, cfg, app.streams.back(), detector_graph);
  } else {
    for (std::size_t index = 0; index < cfg.rtsp_urls.size(); ++index) {
      app.streams.push_back(
          build_stream_runtime(cfg, static_cast<int>(index), cfg.rtsp_urls[index]));
      connect_stream_graph(app, cfg, app.streams.back(), detector_graph);
    }
  }
  app.graph.connect(detector_graph, detections_graph);

  if (cfg.profile) {
    std::cout << "Backend:\n" << app.graph.describe_backend() << "\n";
  }

  app.run = app.graph.build(build_run_options());
  std::atomic_bool stop_debug_pullers = false;
  std::mutex debug_error_mutex;
  std::exception_ptr debug_error;
  std::vector<std::thread> debug_pullers;
  if (tracking_frames_enabled(cfg)) {
    debug_pullers.reserve(app.streams.size());
    for (auto& stream : app.streams) {
      StreamRuntime* const stream_ptr = &stream;
      debug_pullers.emplace_back(
          [&app, &cfg, stream_ptr, &stop_debug_pullers, &debug_error_mutex, &debug_error] {
            try {
              while (!stop_debug_pullers.load(std::memory_order_relaxed)) {
                (void)pull_debug_frame(app, *stream_ptr, cfg, 50);
              }
            } catch (...) {
              {
                std::lock_guard lock(debug_error_mutex);
                if (debug_error == nullptr) {
                  debug_error = std::current_exception();
                }
              }
              stop_debug_pullers.store(true, std::memory_order_relaxed);
            }
          });
    }
  }

  std::atomic_bool stop_input_producer = false;
  std::atomic_bool input_producer_failed = false;
  std::mutex input_error_mutex;
  std::exception_ptr input_error;
  std::thread input_producer;
  if (benchmark_mode) {
    const int logical_fps = app.streams.front().output_fps;
    input_producer = std::thread([&app, &cfg, &benchmark_tensor, logical_fps, &stop_input_producer,
                                  &input_producer_failed, &input_error_mutex, &input_error] {
      try {
        for (int64_t frame_id = 0;
             frame_id < cfg.frames && !stop_input_producer.load(std::memory_order_relaxed) &&
             g_stop_requested == 0;
             ++frame_id) {
          auto sample = make_benchmark_sample(*benchmark_tensor, frame_id, logical_fps);
          if (!app.run.push("benchmark_frame", sample)) {
            if (!stop_input_producer.load(std::memory_order_relaxed)) {
              throw std::runtime_error("benchmark image input closed before all frames were sent");
            }
            break;
          }
        }
      } catch (...) {
        {
          std::lock_guard lock(input_error_mutex);
          input_error = std::current_exception();
        }
        input_producer_failed.store(true, std::memory_order_relaxed);
      }
    });
  }

  const auto stop_and_join_input = [&] {
    stop_input_producer.store(true, std::memory_order_relaxed);
    if (benchmark_mode) {
      try {
        app.run.close_input();
      } catch (...) {
        // Preserve the workload failure, if any. Run::close() below remains
        // the final unconditional shutdown boundary.
      }
    }
    if (input_producer.joinable()) {
      input_producer.join();
    }
  };
  try {
    while (g_stop_requested == 0 && !stop_debug_pullers.load(std::memory_order_relaxed) &&
           !input_producer_failed.load(std::memory_order_relaxed) &&
           !all_streams_done(app.streams, cfg.frames)) {
      (void)process_run_once(app, cfg, "detections");
    }
  } catch (...) {
    stop_and_join_input();
    stop_debug_pullers.store(true, std::memory_order_relaxed);
    for (auto& puller : debug_pullers) {
      puller.join();
    }
    app.run.close();
    throw;
  }
  stop_and_join_input();
  stop_debug_pullers.store(true, std::memory_order_relaxed);
  for (auto& puller : debug_pullers) {
    puller.join();
  }
  app.run.close();
  {
    std::lock_guard lock(debug_error_mutex);
    if (debug_error != nullptr) {
      std::rethrow_exception(debug_error);
    }
  }
  {
    std::lock_guard lock(input_error_mutex);
    if (input_error != nullptr) {
      std::rethrow_exception(input_error);
    }
  }

  for (auto& stream : app.streams) {
    stream.profile.flush();
    std::cout << "[stream " << stream.index << "] processed=" << stream.processed << "\n";
    if (stream.benchmark_mode) {
      const int intervals = std::max(0, stream.benchmark_completed_frames - 1);
      const double elapsed_ms =
          stream.benchmark_last_completed_ms - stream.benchmark_first_completed_ms;
      const double throughput_fps =
          elapsed_ms > 0.0 ? static_cast<double>(intervals) * 1000.0 / elapsed_ms : 0.0;
      std::cout << std::fixed << std::setprecision(3)
                << "[benchmark] measured_frames=" << stream.benchmark_completed_frames
                << " elapsed_ms=" << elapsed_ms << " throughput_fps=" << throughput_fps
                << " route=NV12->Preprocess->MLA->BoxDecode->CameraMotion->Tracker"
                   " video=off metadata=off overlay=off\n";
    }
  }

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
    if (cli.benchmark_image.has_value() && !fs::is_regular_file(*cli.benchmark_image)) {
      std::cerr << "Error: benchmark image not found: " << *cli.benchmark_image << "\n";
      return 2;
    }
    if (cli.validate_config_only) {
      std::cout << "Config validated: " << cli.config_path << " (streams=" << cfg.rtsp_urls.size()
                << ", max_inflight_per_stream=" << cfg.max_inflight_per_stream
                << ", max_inflight_total=" << cfg.max_inflight_total
                << ", overflow_policy=" << overflow_policy_name(cfg.overflow_policy)
                << ", min_score=" << cfg.min_score
                << ", match_iou_threshold=" << cfg.tracker_iou_threshold
                << ", center_distance_enabled="
                << (cfg.tracker_center_distance_enabled ? "true" : "false")
                << ", camera_motion_compensation="
                << (cfg.tracker_camera_motion_compensation ? "true" : "false")
                << ", max_prediction_frames=" << cfg.tracker_max_prediction_frames << ")\n";
      return 0;
    }
    run_app(cfg, cli.benchmark_image);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[ERR] " << e.what() << "\n";
    return 1;
  }
}
