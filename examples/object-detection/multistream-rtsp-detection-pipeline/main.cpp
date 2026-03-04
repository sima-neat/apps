#include "support/object_detection/obj_detection_utils.h"
#include "support/runtime/example_utils.h"

#include "neat/models.h"
#include "neat/node_groups.h"
#include "neat/nodes.h"
#include "neat/session.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

using Clock = std::chrono::steady_clock;
using TimePoint = std::chrono::time_point<Clock>;

struct Config {
  std::vector<std::string> rtsp_urls;
  std::string model_path;
  std::string output_dir;
  std::string labels_file = "examples/object-detection/multistream-rtsp-detection-pipeline/coco_label.txt";

  int frames = 100;
  bool debug = false;
  bool profile = false;
  int profile_every = 50;

  bool tcp = false;
  int latency_ms = 200;
  int width = 1280;
  int height = 720;
  int fps = 30;
  int sample_every = 1;
  int save_every = 10;
  int run_queue_depth = 4;
  simaai::neat::OverflowPolicy overflow_policy = simaai::neat::OverflowPolicy::KeepLatest;
  simaai::neat::OutputMemory output_memory = simaai::neat::OutputMemory::Owned;

  int pull_timeout_ms = 50;
  int max_idle_ms = 15000;
  int reconnect_miss = 3;

  float min_score = 0.52f;
  float nms_iou = 0.50f;
  int max_det = 100;
  int model_timeout_ms = 3000;

  int frame_queue = 4;
  int result_queue = 4;
};

struct StreamState {
  int idx = -1;
  std::string url;
  fs::path out_dir;
  simaai::neat::Session session;
  simaai::neat::Run run;
  int processed = 0;
  int pulled = 0;
  int frame_q_dropped = 0;
  int result_q_dropped = 0;
  bool producer_done = false;
  bool infer_done = false;
};

struct FramePacket {
  int stream_idx = -1;
  cv::Mat frame;
  TimePoint pulled_ts{};
};

struct ResultPacket {
  int stream_idx = -1;
  cv::Mat frame;
  std::vector<objdet::Box> boxes;
  TimePoint pulled_ts{};
};

struct TimingAgg {
  int count = 0;
  double total_ms = 0.0;
  double max_ms = 0.0;

  void add(double dt_ms) {
    ++count;
    total_ms += dt_ms;
    if (dt_ms > max_ms)
      max_ms = dt_ms;
  }
};

struct RtspRuntime {
  simaai::neat::Session session;
  simaai::neat::Run run;
  std::string pipeline;
};

struct InferRuntime {
  simaai::neat::Model model;
  simaai::neat::Session session;
  simaai::neat::Run run;
  std::string pipeline;

  InferRuntime(simaai::neat::Model m, simaai::neat::Session s, simaai::neat::Run r, std::string p)
      : model(std::move(m)), session(std::move(s)), run(std::move(r)), pipeline(std::move(p)) {}
};

struct StreamSnapshot {
  int idx = -1;
  int processed = 0;
  int pulled = 0;
  int frame_q_dropped = 0;
  int result_q_dropped = 0;
};

template <typename T> class KeepLatestQueue {
public:
  explicit KeepLatestQueue(std::size_t capacity) : capacity_(std::max<std::size_t>(1, capacity)) {}

  std::pair<int, std::size_t> push_keep_latest(T item) {
    std::unique_lock<std::mutex> lock(mu_);
    if (closed_) {
      return {0, q_.size()};
    }
    int dropped = 0;
    while (q_.size() >= capacity_) {
      q_.pop_front();
      ++dropped;
    }
    q_.push_back(std::move(item));
    const std::size_t qsize = q_.size();
    lock.unlock();
    cv_.notify_one();
    return {dropped, qsize};
  }

  bool pop_wait(T& out, int timeout_ms) {
    std::unique_lock<std::mutex> lock(mu_);
    const auto ready = [&] { return closed_ || !q_.empty(); };
    if (timeout_ms < 0) {
      cv_.wait(lock, ready);
    } else if (!cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), ready)) {
      return false;
    }
    if (q_.empty()) {
      return false;
    }
    // Keep-latest semantics on dequeue: consume newest and drop stale backlog.
    out = std::move(q_.back());
    q_.clear();
    return true;
  }

  std::size_t size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return q_.size();
  }

  bool empty() const {
    std::lock_guard<std::mutex> lock(mu_);
    return q_.empty();
  }

  void close() {
    std::lock_guard<std::mutex> lock(mu_);
    closed_ = true;
    cv_.notify_all();
  }

private:
  std::size_t capacity_ = 1;
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::deque<T> q_;
  bool closed_ = false;
};

class ProfileTracker {
public:
  explicit ProfileTracker(bool enabled, int every)
      : enabled_(enabled), every_(std::max(1, every)) {}

  static std::string stage_key(const std::string& name, int stream_idx) {
    return name + ":s" + std::to_string(stream_idx);
  }

  void add_time(const std::string& key, double dt_ms) {
    if (!enabled_) {
      return;
    }
    std::lock_guard<std::mutex> lock(mu_);
    timing_window_[key].add(dt_ms);
  }


  void mark_processing_started(int done_frames) {
    std::lock_guard<std::mutex> lock(mu_);
    if (started_) {
      return;
    }
    started_ = true;
    processing_start_ts_ = Clock::now();
    processing_start_done_ = std::max(0, done_frames);
    window_start_ts_ = processing_start_ts_;
    window_start_frame_ = processing_start_done_;
  }

  int processed_since_start(int done_frames) const {
    std::lock_guard<std::mutex> lock(mu_);
    if (!started_) {
      return std::max(0, done_frames);
    }
    return std::max(0, done_frames - processing_start_done_);
  }

  double elapsed_since_start_s(TimePoint now) const {
    std::lock_guard<std::mutex> lock(mu_);
    if (!started_) {
      return 0.0;
    }
    return std::chrono::duration<double>(now - processing_start_ts_).count();
  }

  void maybe_report(int done_frames, const std::vector<StreamSnapshot>& stream_snapshot,
                    std::mutex& log_mu) {
    if (!enabled_ || done_frames <= 0 || (done_frames % every_) != 0) {
      return;
    }

    std::lock_guard<std::mutex> lock(mu_);
    if (!started_) {
      return;
    }
    const auto now = Clock::now();
    const int window_frames = done_frames - window_start_frame_;
    if (window_frames <= 0) {
      return;
    }
    const double window_elapsed_s =
        std::max(1e-6, std::chrono::duration<double>(now - window_start_ts_).count());
    const double fps = static_cast<double>(window_frames) / window_elapsed_s;

    std::lock_guard<std::mutex> log_lock(log_mu);
    std::cout << "--------------------------\n";
    std::cout << "[PROGRESS] frames=" << done_frames << " fps=" << std::fixed << std::setprecision(2)
              << fps << " (last " << window_frames << " in " << window_elapsed_s << "s)\n";

    for (const auto& s : stream_snapshot) {
      const int idx = s.idx;
      const int total_dropped = s.frame_q_dropped + s.result_q_dropped;
      std::cout << "[PROFILE][stream " << idx << "] processed=" << s.processed
                << " pulled=" << s.pulled << " dropped=" << total_dropped << "\n";
      print_stage_header();
      print_combined_stage(idx, "pull_call", "to_numpy", "rtsp_decode");
      print_combined_stage(idx, "model_run", "decode_boxes", "model");
      print_combined_stage(idx, "draw_boxes", "imwrite", "write");
      print_stage(idx, "end_to_end", "e2e");
    }
    std::cout << std::defaultfloat;

    timing_window_.clear();
    window_start_ts_ = now;
    window_start_frame_ = done_frames;
  }

private:

  void print_stage_header() const {
    std::cout << "[PROFILE]   stage         avg(ms)   max(ms)   n\n";
  }

  void print_stage(int stream_idx, const std::string& key, const std::string& pretty) const {
    auto it = timing_window_.find(stage_key(key, stream_idx));
    if (it == timing_window_.end() || it->second.count <= 0) {
      return;
    }
    const TimingAgg& agg = it->second;
    const double avg_ms = agg.total_ms / static_cast<double>(agg.count);
    std::cout << "[PROFILE]   " << std::left << std::setw(12) << pretty << std::right << std::setw(8)
              << std::fixed << std::setprecision(1) << avg_ms << " " << std::setw(8) << agg.max_ms
              << " " << std::setw(4) << agg.count << "\n";
  }

  void print_combined_stage(int stream_idx, const std::string& key_a, const std::string& key_b,
                            const std::string& pretty) const {
    auto it_a = timing_window_.find(stage_key(key_a, stream_idx));
    auto it_b = timing_window_.find(stage_key(key_b, stream_idx));
    const bool has_a = it_a != timing_window_.end() && it_a->second.count > 0;
    const bool has_b = it_b != timing_window_.end() && it_b->second.count > 0;
    if (!has_a && !has_b) {
      return;
    }
    double total_ms = 0.0;
    double max_ms = 0.0;
    int count = 0;
    if (has_a) {
      total_ms += it_a->second.total_ms;
      if (it_a->second.max_ms > max_ms) max_ms = it_a->second.max_ms;
      count = it_a->second.count;
    }
    if (has_b) {
      total_ms += it_b->second.total_ms;
      if (it_b->second.max_ms > max_ms) max_ms = it_b->second.max_ms;
      if (it_b->second.count > count) count = it_b->second.count;
    }
    const double avg_ms = (count > 0) ? total_ms / static_cast<double>(count) : 0.0;
    std::cout << "[PROFILE]   " << std::left << std::setw(12) << pretty << std::right << std::setw(8)
              << std::fixed << std::setprecision(1) << avg_ms << " " << std::setw(8) << max_ms
              << " " << std::setw(4) << count << "\n";
  }

  bool enabled_ = false;
  int every_ = 50;
  mutable std::mutex mu_;
  std::unordered_map<std::string, TimingAgg> timing_window_;
  bool started_ = false;
  TimePoint processing_start_ts_{};
  int processing_start_done_ = 0;
  TimePoint window_start_ts_{};
  int window_start_frame_ = 0;
};

[[noreturn]] void die(const std::string& msg) {
  throw std::runtime_error(msg);
}

std::string lower_ascii(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return s;
}

simaai::neat::OverflowPolicy parse_overflow_policy(const std::string& value) {
  const std::string v = lower_ascii(value);
  if (v == "block") {
    return simaai::neat::OverflowPolicy::Block;
  }
  if (v == "keep-latest") {
    return simaai::neat::OverflowPolicy::KeepLatest;
  }
  if (v == "drop-incoming") {
    return simaai::neat::OverflowPolicy::DropIncoming;
  }
  die("unsupported --overflow-policy: " + value);
}

simaai::neat::OutputMemory parse_output_memory(const std::string& value) {
  const std::string v = lower_ascii(value);
  if (v == "auto") {
    return simaai::neat::OutputMemory::Auto;
  }
  if (v == "zero-copy") {
    return simaai::neat::OutputMemory::ZeroCopy;
  }
  if (v == "owned") {
    return simaai::neat::OutputMemory::Owned;
  }
  die("unsupported --output-memory: " + value);
}

void usage() {
  std::cerr << "Usage:\n";
  std::cerr << "  multistream-rtsp-detection-pipeline --model <mpk.tar.gz> --output <dir>\n";
  std::cerr << "    --rtsp <url0> [--rtsp <url1> ...]\n";
  std::cerr << "    [--labels-file examples/object-detection/multistream-rtsp-detection-pipeline/coco_label.txt]\n";
  std::cerr << "    [--frames 100] [--fps 30] [--width 1280] [--height 720]\n";
  std::cerr << "    [--frame-queue 128] [--result-queue 128]\n";
  std::cerr << "    [--pull-timeout-ms 50] [--max-idle-ms 15000] [--reconnect-miss 3]\n";
  std::cerr << "    [--min-score 0.52] [--nms-iou 0.50] [--max-det 100]\n";
  std::cerr << "    [--model-timeout-ms 3000]\n";
  std::cerr << "    [--run-queue-depth 4] [--overflow-policy keep-latest]\n";
  std::cerr << "    [--output-memory owned] [--sample-every 1] [--save-every 10]\n";
  std::cerr << "    [--tcp] [--latency-ms 200]\n";
  std::cerr << "    [--debug] [--profile] [--profile-every 50]\n";
}

Config parse_config(int argc, char** argv) {
  Config cfg;

  auto need_value = [&](int& i, const std::string& key) -> std::string {
    if (i + 1 >= argc) {
      die("missing value for " + key);
    }
    return argv[++i];
  };

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      usage();
      std::exit(0);
    }
    if (arg == "--rtsp") {
      cfg.rtsp_urls.push_back(need_value(i, arg));
      continue;
    }
    if (arg == "--model") {
      cfg.model_path = need_value(i, arg);
      continue;
    }
    if (arg == "--output") {
      cfg.output_dir = need_value(i, arg);
      continue;
    }
    if (arg == "--labels-file") {
      cfg.labels_file = need_value(i, arg);
      continue;
    }
    if (arg == "--frames") {
      cfg.frames = std::stoi(need_value(i, arg));
      continue;
    }
    if (arg == "--fps") {
      cfg.fps = std::stoi(need_value(i, arg));
      continue;
    }
    if (arg == "--width") {
      cfg.width = std::stoi(need_value(i, arg));
      continue;
    }
    if (arg == "--height") {
      cfg.height = std::stoi(need_value(i, arg));
      continue;
    }
    if (arg == "--sample-every") {
      cfg.sample_every = std::stoi(need_value(i, arg));
      continue;
    }
    if (arg == "--save-every") {
      cfg.save_every = std::stoi(need_value(i, arg));
      continue;
    }
    if (arg == "--run-queue-depth") {
      cfg.run_queue_depth = std::stoi(need_value(i, arg));
      continue;
    }
    if (arg == "--overflow-policy") {
      cfg.overflow_policy = parse_overflow_policy(need_value(i, arg));
      continue;
    }
    if (arg == "--output-memory") {
      cfg.output_memory = parse_output_memory(need_value(i, arg));
      continue;
    }
    if (arg == "--pull-timeout-ms") {
      cfg.pull_timeout_ms = std::stoi(need_value(i, arg));
      continue;
    }
    if (arg == "--max-idle-ms") {
      cfg.max_idle_ms = std::stoi(need_value(i, arg));
      continue;
    }
    if (arg == "--reconnect-miss") {
      cfg.reconnect_miss = std::stoi(need_value(i, arg));
      continue;
    }
    if (arg == "--min-score") {
      cfg.min_score = std::stof(need_value(i, arg));
      continue;
    }
    if (arg == "--nms-iou") {
      cfg.nms_iou = std::stof(need_value(i, arg));
      continue;
    }
    if (arg == "--max-det") {
      cfg.max_det = std::stoi(need_value(i, arg));
      continue;
    }
    if (arg == "--model-timeout-ms") {
      cfg.model_timeout_ms = std::stoi(need_value(i, arg));
      continue;
    }
    if (arg == "--frame-queue") {
      cfg.frame_queue = std::stoi(need_value(i, arg));
      continue;
    }
    if (arg == "--result-queue") {
      cfg.result_queue = std::stoi(need_value(i, arg));
      continue;
    }
    if (arg == "--latency-ms") {
      cfg.latency_ms = std::stoi(need_value(i, arg));
      continue;
    }
    if (arg == "--profile-every") {
      cfg.profile_every = std::stoi(need_value(i, arg));
      continue;
    }
    if (arg == "--tcp") {
      cfg.tcp = true;
      continue;
    }
    if (arg == "--debug") {
      cfg.debug = true;
      continue;
    }
    if (arg == "--profile") {
      cfg.profile = true;
      continue;
    }
    die("unknown argument: " + arg);
  }

  if (cfg.model_path.empty()) {
    die("--model is required");
  }
  if (cfg.output_dir.empty()) {
    die("--output is required");
  }
  if (cfg.rtsp_urls.empty()) {
    die("at least one --rtsp URL is required");
  }
  if (cfg.frames <= 0) {
    die("--frames must be > 0");
  }
  if (cfg.width <= 0 || cfg.height <= 0) {
    die("--width/--height must be > 0");
  }
  if (cfg.fps <= 0) {
    die("--fps must be > 0");
  }
  if (cfg.sample_every <= 0) {
    die("--sample-every must be > 0");
  }
  if (cfg.save_every <= 0) {
    die("--save-every must be > 0");
  }
  if (cfg.run_queue_depth <= 0) {
    die("--run-queue-depth must be > 0");
  }
  if (cfg.pull_timeout_ms <= 0) {
    die("--pull-timeout-ms must be > 0");
  }
  if (cfg.max_idle_ms <= 0) {
    die("--max-idle-ms must be > 0");
  }
  if (cfg.reconnect_miss <= 0) {
    die("--reconnect-miss must be > 0");
  }
  if (!(cfg.min_score > 0.0f && cfg.min_score < 1.0f)) {
    die("--min-score must be in (0, 1)");
  }
  if (!(cfg.nms_iou > 0.0f && cfg.nms_iou < 1.0f)) {
    die("--nms-iou must be in (0, 1)");
  }
  if (cfg.max_det <= 0) {
    die("--max-det must be > 0");
  }
  if (cfg.model_timeout_ms <= 0) {
    die("--model-timeout-ms must be > 0");
  }
  if (cfg.frame_queue <= 0 || cfg.result_queue <= 0) {
    die("--frame-queue/--result-queue must be > 0");
  }
  if (cfg.profile_every <= 0) {
    die("--profile-every must be > 0");
  }
  return cfg;
}

std::vector<std::string> load_labels(const std::string& path) {
  std::vector<std::string> labels;
  if (path.empty()) {
    return labels;
  }
  std::ifstream in(path);
  if (!in.good()) {
    return labels;
  }
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty()) {
      labels.push_back(line);
    }
  }
  return labels;
}

std::string label_for_class(const std::vector<std::string>& labels, int class_id) {
  if (class_id >= 0 && class_id < static_cast<int>(labels.size())) {
    return labels[static_cast<std::size_t>(class_id)];
  }
  return std::to_string(class_id);
}

cv::Scalar class_color(int cls_id) {
  return cv::Scalar((37 * cls_id + 17) % 256, (97 * cls_id + 73) % 256, (53 * cls_id + 191) % 256);
}

void draw_boxes(cv::Mat& frame, const std::vector<objdet::Box>& boxes,
                const std::vector<std::string>& labels) {
  for (const auto& b : boxes) {
    const int x1 = std::max(0, std::min(frame.cols - 1, static_cast<int>(std::round(b.x1))));
    const int y1 = std::max(0, std::min(frame.rows - 1, static_cast<int>(std::round(b.y1))));
    const int x2 = std::max(0, std::min(frame.cols - 1, static_cast<int>(std::round(b.x2))));
    const int y2 = std::max(0, std::min(frame.rows - 1, static_cast<int>(std::round(b.y2))));
    if (x2 <= x1 || y2 <= y1) {
      continue;
    }

    const cv::Scalar color = class_color(b.class_id);
    const std::string text = label_for_class(labels, b.class_id) + " " + cv::format("%.2f", b.score);
    cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);

    int baseline = 0;
    const cv::Size tsz = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    const int y_text_top = std::max(0, y1 - tsz.height - 4);
    const int y_text_bottom = std::max(0, y1);
    cv::rectangle(frame, cv::Point(x1, y_text_top), cv::Point(x1 + tsz.width, y_text_bottom), color,
                  cv::FILLED);
    cv::putText(frame, text, cv::Point(x1, std::max(0, y1 - 2)), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
  }
}

cv::Mat tensor_to_bgr_mat(const simaai::neat::Tensor& t) {
  cv::Mat bgr = t.to_cv_mat_copy(simaai::neat::ImageSpec::PixelFormat::BGR);
  if (bgr.empty()) {
    die("received empty frame from RTSP run");
  }
  if (bgr.type() != CV_8UC3) {
    cv::Mat out;
    bgr.convertTo(out, CV_8UC3);
    bgr = out;
  }
  return bgr;
}

RtspRuntime build_rtsp_runtime(const Config& cfg, const std::string& url, int sima_allocator_type,
                               const sima_examples::RtspStreamInfo& probe) {
  simaai::neat::nodes::groups::RtspDecodedInputOptions ro;
  ro.url = url;
  ro.latency_ms = cfg.latency_ms;
  ro.tcp = cfg.tcp;
  ro.payload_type = 96;
  ro.insert_queue = true;
  ro.out_format = "BGR";
  ro.decoder_raw_output = false;
  ro.use_videoconvert = false;
  ro.use_videoscale = true;
  ro.sima_allocator_type = sima_allocator_type;
  ro.output_caps.enable = true;
  ro.output_caps.format = "BGR";
  ro.output_caps.width = (probe.width > 0) ? probe.width : cfg.width;
  ro.output_caps.height = (probe.height > 0) ? probe.height : cfg.height;
  ro.output_caps.fps = (probe.fps > 0) ? probe.fps : cfg.fps;
  ro.output_caps.memory = simaai::neat::CapsMemory::SystemMemory;

  simaai::neat::Session session;
  session.add(simaai::neat::nodes::groups::RtspDecodedInput(ro));
  session.add(simaai::neat::nodes::Output(simaai::neat::OutputOptions::EveryFrame(cfg.sample_every)));

  simaai::neat::RunOptions run_opt;
  run_opt.queue_depth = std::max(1, cfg.run_queue_depth);
  run_opt.overflow_policy = cfg.overflow_policy;
  run_opt.output_memory = cfg.output_memory;
  simaai::neat::Run run = session.build(run_opt);
  const std::string pipeline = session.last_pipeline();
  return RtspRuntime{std::move(session), std::move(run), pipeline};
}

RtspRuntime build_rtsp_runtime_with_fallback(const Config& cfg, const std::string& url,
                                             bool& used_fallback_allocator) {
  used_fallback_allocator = false;
  sima_examples::RtspProbeOptions probe_opt;
  probe_opt.payload_type = 96;
  probe_opt.latency_ms = cfg.latency_ms;
  probe_opt.rtsp_tcp = cfg.tcp;
  probe_opt.debug = cfg.debug;
  sima_examples::RtspStreamInfo probe;
  (void)sima_examples::probe_rtsp_stream_info(url, probe_opt, probe);
  try {
    return build_rtsp_runtime(cfg, url, /*sima_allocator_type=*/2, probe);
  } catch (const std::exception& first_ex) {
    try {
      RtspRuntime rtsp = build_rtsp_runtime(cfg, url, /*sima_allocator_type=*/1, probe);
      used_fallback_allocator = true;
      return rtsp;
    } catch (const std::exception& second_ex) {
      throw std::runtime_error(std::string("rtsp build failed (alloc=2): ") + first_ex.what() +
                               " | retry (alloc=1) failed: " + second_ex.what());
    }
  }
}

InferRuntime build_infer_runtime(const Config& cfg) {
  simaai::neat::Model::Options mopt;
  mopt.media_type = "video/x-raw";
  mopt.format = "BGR";
  mopt.input_max_width = cfg.width;
  mopt.input_max_height = cfg.height;
  mopt.input_max_depth = 3;
  simaai::neat::Model model(cfg.model_path, mopt);

  simaai::neat::Session yolo;
  simaai::neat::InputOptions in_opt = model.input_appsrc_options(false);
  in_opt.format = "BGR";
  in_opt.width = cfg.width;
  in_opt.height = cfg.height;
  yolo.add(simaai::neat::nodes::Input(in_opt));
  yolo.add(simaai::neat::nodes::groups::Preprocess(model));
  yolo.add(simaai::neat::nodes::groups::Infer(model));
  yolo.add(simaai::neat::nodes::SimaBoxDecode(model, "yolov8", cfg.width, cfg.height, cfg.min_score,
                                               cfg.nms_iou, cfg.max_det));
  yolo.add(simaai::neat::nodes::Output());

  simaai::neat::RunOptions run_opt;
  run_opt.queue_depth = std::max(1, cfg.run_queue_depth);
  run_opt.overflow_policy = cfg.overflow_policy;
  run_opt.output_memory = cfg.output_memory;
  cv::Mat dummy(cfg.height, cfg.width, CV_8UC3, cv::Scalar(0, 0, 0));
  simaai::neat::Run run = yolo.build(dummy, simaai::neat::RunMode::Async, run_opt);
  const std::string pipeline = yolo.last_pipeline();
  return InferRuntime(std::move(model), std::move(yolo), std::move(run), pipeline);
}

double elapsed_ms(TimePoint a, TimePoint b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}


} // namespace

// Orchestrates producer/infer/overlay workers while keeping NEAT pipeline construction explicit.
// NEAT usage (C++):
// - build_rtsp_runtime_with_fallback(...) creates per-stream simaai::neat::Session + Run for RTSP decode.
// - build_infer_runtime(...) creates model session:
//   Input -> Preprocess -> Infer -> SimaBoxDecode -> Output.
class PipelineApp {
public:
  explicit PipelineApp(Config cfg)
      : cfg_(std::move(cfg)), labels_(load_labels(cfg_.labels_file)),
        out_root_(cfg_.output_dir), profiler_(cfg_.profile, cfg_.profile_every) {}

  int run() {
    fs::create_directories(out_root_);
    setup_streams();
    total_target_ = cfg_.frames * static_cast<int>(streams_.size());
    t0_ = Clock::now();
    start_producer_threads();
    setup_infer_runtimes();
    start_infer_threads();
    start_overlay_threads();

    // Release producers only after infer/overlay workers are live to avoid startup pull backlog.
    start_producers_.store(true, std::memory_order_release);
    monitor_until_done();
    close_queues();
    join_threads();
    print_summary();
    close_runs();
    return 0;
  }

private:
  void setup_streams() {
    streams_.reserve(cfg_.rtsp_urls.size());
    frame_queues_.reserve(cfg_.rtsp_urls.size());
    result_queues_.reserve(cfg_.rtsp_urls.size());

    for (std::size_t i = 0; i < cfg_.rtsp_urls.size(); ++i) {
      const std::string& url = cfg_.rtsp_urls[i];
      bool used_fallback_allocator = false;
      RtspRuntime rtsp = build_rtsp_runtime_with_fallback(cfg_, url, used_fallback_allocator);

      auto st = std::make_shared<StreamState>();
      st->idx = static_cast<int>(i);
      st->url = url;
      st->session = std::move(rtsp.session);
      st->run = std::move(rtsp.run);
      st->out_dir = out_root_ / ("stream_" + std::to_string(i));
      fs::create_directories(st->out_dir);

      frame_queues_.push_back(
          std::make_shared<KeepLatestQueue<FramePacket>>(static_cast<std::size_t>(cfg_.frame_queue)));
      result_queues_.push_back(std::make_shared<KeepLatestQueue<ResultPacket>>(
          static_cast<std::size_t>(cfg_.result_queue)));
      streams_.push_back(st);

      std::lock_guard<std::mutex> log_lock(log_mu_);
      std::cout << "[stream " << i << "] started: " << url << " -> " << st->out_dir.string() << "/\n";
      if (used_fallback_allocator) {
        std::cout << "[stream " << i << "] decoder allocator fallback: using sima_allocator_type=1\n";
      }
      if (cfg_.debug) {
        std::cout << "[PIPE][rtsp " << i << "] " << rtsp.pipeline << "\n";
      }
    }
  }

  void setup_infer_runtimes() {
    infer_runtimes_.reserve(streams_.size());
    // Build infer runtimes sequentially to avoid shared model-config race during startup.
    for (const auto& st : streams_) {
      try {
        auto infer = std::make_unique<InferRuntime>(build_infer_runtime(cfg_));
        if (cfg_.debug) {
          std::lock_guard<std::mutex> log_lock(log_mu_);
          std::cout << "[PIPE][infer " << st->idx << "] " << infer->pipeline << "\n";
        }
        infer_runtimes_.push_back(std::move(infer));
      } catch (const std::exception& ex) {
        std::lock_guard<std::mutex> log_lock(log_mu_);
        std::cerr << "[stream " << st->idx << "] infer worker setup failed: " << ex.what() << "\n";
        infer_runtimes_.push_back(nullptr);
        {
          std::lock_guard<std::mutex> stats_lock(stats_mu_);
          st->infer_done = true;
        }
      }
    }
  }

  void start_producer_threads() {
    for (const auto& st : streams_) {
      const int idx = st->idx;
      auto frame_q = frame_queues_[static_cast<std::size_t>(idx)];
      producer_threads_.emplace_back([this, st, frame_q]() { producer_worker(st, frame_q); });
    }
  }

  void start_infer_threads() {
    for (const auto& st : streams_) {
      const int idx = st->idx;
      if (!infer_runtimes_[static_cast<std::size_t>(idx)]) {
        continue;
      }
      auto frame_q = frame_queues_[static_cast<std::size_t>(idx)];
      auto result_q = result_queues_[static_cast<std::size_t>(idx)];
      auto infer_ptr = std::move(infer_runtimes_[static_cast<std::size_t>(idx)]);
      infer_threads_.emplace_back([this, st, frame_q, result_q, infer_ptr = std::move(infer_ptr)]() mutable {
        infer_worker(st, frame_q, result_q, std::move(infer_ptr));
      });
    }
  }

  void start_overlay_threads() {
    for (const auto& st : streams_) {
      const int idx = st->idx;
      auto result_q = result_queues_[static_cast<std::size_t>(idx)];
      overlay_threads_.emplace_back([this, st, result_q]() { overlay_worker(st, result_q); });
    }
  }

  void producer_worker(const std::shared_ptr<StreamState>& st,
                       const std::shared_ptr<KeepLatestQueue<FramePacket>>& frame_q) {
    try {
      while (!stop_event_.load(std::memory_order_relaxed) &&
             !start_producers_.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      const int miss_limit = std::max(1, cfg_.max_idle_ms / std::max(1, cfg_.pull_timeout_ms));
      int miss_count = 0;
      while (!stop_event_.load()) {
        {
          std::lock_guard<std::mutex> stats_lock(stats_mu_);
          if (st->processed >= cfg_.frames) {
            break;
          }
        }

        const auto t_pull0 = Clock::now();
        auto frame_opt = st->run.pull_tensor(cfg_.pull_timeout_ms);
        const auto t_pull1 = Clock::now();
        profiler_.add_time(ProfileTracker::stage_key("pull_call", st->idx), elapsed_ms(t_pull0, t_pull1));

        if (!frame_opt.has_value()) {
          ++miss_count;
          bool running = false;
          try {
            running = st->run.running();
          } catch (...) {
            running = false;
          }

          if (!running && miss_count >= std::max(1, cfg_.reconnect_miss)) {
            try {
              st->run.close();
            } catch (...) {
            }
            try {
              bool used_fallback_allocator = false;
              RtspRuntime rtsp_new =
                  build_rtsp_runtime_with_fallback(cfg_, st->url, used_fallback_allocator);
              st->session = std::move(rtsp_new.session);
              st->run = std::move(rtsp_new.run);
              miss_count = 0;
              std::lock_guard<std::mutex> log_lock(log_mu_);
              std::cout << "[stream " << st->idx << "] reconnecting RTSP run\n";
              if (used_fallback_allocator) {
                std::cout << "[stream " << st->idx
                          << "] decoder allocator fallback: using sima_allocator_type=1\n";
              }
              if (cfg_.debug) {
                std::cout << "[PIPE][rtsp " << st->idx << "] " << rtsp_new.pipeline << "\n";
              }
            } catch (const std::exception& ex) {
              std::lock_guard<std::mutex> log_lock(log_mu_);
              std::cerr << "[stream " << st->idx << "] reconnect failed: " << ex.what() << "\n";
            }
          }

          if (miss_count >= miss_limit) {
            std::lock_guard<std::mutex> log_lock(log_mu_);
            std::cerr << "[stream " << st->idx << "] no frames for ~" << cfg_.max_idle_ms
                      << "ms; closing stream\n";
            break;
          }
          continue;
        }

        miss_count = 0;
        const auto pulled_ts = Clock::now();
        const auto t_np0 = Clock::now();
        cv::Mat bgr = tensor_to_bgr_mat(*frame_opt);
        const auto t_np1 = Clock::now();
        profiler_.add_time(ProfileTracker::stage_key("to_numpy", st->idx), elapsed_ms(t_np0, t_np1));

        {
          std::lock_guard<std::mutex> stats_lock(stats_mu_);
          st->pulled += 1;
        }

        auto [dropped, qsize] =
            frame_q->push_keep_latest(FramePacket{st->idx, std::move(bgr), pulled_ts});
        (void)qsize;
        if (dropped > 0) {
          std::lock_guard<std::mutex> stats_lock(stats_mu_);
          st->frame_q_dropped += dropped;
        }
      }
    } catch (const std::exception& ex) {
      std::lock_guard<std::mutex> log_lock(log_mu_);
      std::cerr << "[stream " << st->idx << "] pull error: " << ex.what() << "\n";
    }
    {
      std::lock_guard<std::mutex> stats_lock(stats_mu_);
      st->producer_done = true;
    }
  }

  void infer_worker(const std::shared_ptr<StreamState>& st,
                    const std::shared_ptr<KeepLatestQueue<FramePacket>>& frame_q,
                    const std::shared_ptr<KeepLatestQueue<ResultPacket>>& result_q,
                    std::unique_ptr<InferRuntime> infer_ptr) {
    try {
      InferRuntime& infer = *infer_ptr;
      const int qdepth = std::max(1, cfg_.run_queue_depth);
      std::deque<FramePacket> pending;
      bool input_done = false;

      // Helper: pull one result from the model and post-process it.
      auto pull_and_process = [&]() -> bool {
        const auto t_model0 = Clock::now();
        simaai::neat::Sample sample;
        simaai::neat::PullError perr;
        const auto pst = infer.run.pull(cfg_.model_timeout_ms, sample, &perr);
        const auto t_model1 = Clock::now();
        if (pst != simaai::neat::PullStatus::Ok) {
          if (pst == simaai::neat::PullStatus::Timeout) {
            return false;
          }
          if (!perr.message.empty()) {
            throw std::runtime_error("model pull failed: " + perr.message);
          }
          if (pst == simaai::neat::PullStatus::Closed) {
            throw std::runtime_error("model run closed");
          }
          throw std::runtime_error("model pull failed");
        }
        profiler_.add_time(ProfileTracker::stage_key("model_run", st->idx),
                           elapsed_ms(t_model0, t_model1));

        FramePacket fpkt = std::move(pending.front());
        pending.pop_front();

        const auto t_post0 = Clock::now();
        std::vector<uint8_t> payload;
        std::string bbox_err;
        std::vector<objdet::Box> boxes;
        if (objdet::extract_bbox_payload(sample, payload, bbox_err)) {
          try {
            boxes = objdet::parse_boxes_strict(payload, fpkt.frame.cols, fpkt.frame.rows,
                                               cfg_.max_det, false);
          } catch (const std::exception& ex) {
            boxes =
                objdet::parse_boxes_lenient(payload, fpkt.frame.cols, fpkt.frame.rows, cfg_.max_det);
            if (cfg_.debug) {
              std::lock_guard<std::mutex> log_lock(log_mu_);
              std::cerr << "[stream " << st->idx
                        << "] strict bbox parse failed (" << ex.what()
                        << "), using lenient parser\n";
            }
          }
        } else if (cfg_.debug) {
          std::lock_guard<std::mutex> log_lock(log_mu_);
          std::cerr << "[stream " << st->idx << "] bbox payload missing: " << bbox_err << "\n";
        }
        const auto t_post1 = Clock::now();
        profiler_.add_time(ProfileTracker::stage_key("decode_boxes", st->idx),
                           elapsed_ms(t_post0, t_post1));

        auto [dropped, qsize] = result_q->push_keep_latest(
            ResultPacket{fpkt.stream_idx, std::move(fpkt.frame), std::move(boxes), fpkt.pulled_ts});
        (void)qsize;
        if (dropped > 0) {
          std::lock_guard<std::mutex> stats_lock(stats_mu_);
          st->result_q_dropped += dropped;
        }
        return true;
      };

      while (true) {
        if (stop_event_.load() && frame_q->empty() && pending.empty()) {
          break;
        }

        // Push phase: fill pipeline up to qdepth.
        while (static_cast<int>(pending.size()) < qdepth && !input_done) {
          FramePacket pkt;
          const auto t_wait0 = Clock::now();
          const bool got = frame_q->pop_wait(pkt, 50);
          const auto t_wait1 = Clock::now();
          if (!got) {
            bool producer_done = false;
            {
              std::lock_guard<std::mutex> stats_lock(stats_mu_);
              producer_done = st->producer_done;
            }
            if (producer_done && frame_q->empty()) {
              input_done = true;
            }
            break; // No frame available right now, move to pull phase.
          }

          profiler_.add_time(ProfileTracker::stage_key("infer_q_wait", st->idx), elapsed_ms(t_wait0, t_wait1));

          {
            std::lock_guard<std::mutex> stats_lock(stats_mu_);
            if (streams_[static_cast<std::size_t>(pkt.stream_idx)]->processed >= cfg_.frames) {
              input_done = true;
              break;
            }
          }

          if (!infer.run.push(pkt.frame)) {
            std::lock_guard<std::mutex> log_lock(log_mu_);
            std::cerr << "[stream " << st->idx << "] model push failed\n";
            break;
          }
          pending.push_back(std::move(pkt));
        }

        // Pull phase: drain one result if anything is in-flight.
        if (!pending.empty()) {
          try {
            pull_and_process();
          } catch (const std::exception& ex) {
            std::lock_guard<std::mutex> log_lock(log_mu_);
            std::cerr << "[stream " << st->idx << "] inference error: " << ex.what() << "\n";
            pending.pop_front(); // Drop the failed frame.
          }
        } else if (input_done) {
          break;
        }

        if (stop_event_.load() && pending.empty()) {
          break;
        }
      }

      // Drain phase: pull remaining in-flight results.
      while (!pending.empty()) {
        try {
          if (!pull_and_process()) {
            pending.pop_front(); // Timeout, skip.
          }
        } catch (const std::exception& ex) {
          std::lock_guard<std::mutex> log_lock(log_mu_);
          std::cerr << "[stream " << st->idx << "] drain error: " << ex.what() << "\n";
          pending.pop_front();
        }
      }

      infer.run.close();
    } catch (const std::exception& ex) {
      std::lock_guard<std::mutex> log_lock(log_mu_);
      std::cerr << "[stream " << st->idx << "] infer worker failed: " << ex.what() << "\n";
    }

    {
      std::lock_guard<std::mutex> stats_lock(stats_mu_);
      st->infer_done = true;
    }
  }

  void overlay_worker(const std::shared_ptr<StreamState>& st,
                      const std::shared_ptr<KeepLatestQueue<ResultPacket>>& result_q) {
    while (true) {
      if (stop_event_.load() && result_q->empty()) {
        break;
      }

      ResultPacket pkt;
      const auto t_wait0 = Clock::now();
      const bool got = result_q->pop_wait(pkt, 50);
      const auto t_wait1 = Clock::now();
      if (!got) {
        bool infer_done = false;
        {
          std::lock_guard<std::mutex> stats_lock(stats_mu_);
          infer_done = st->infer_done;
        }
        if (infer_done && result_q->empty()) {
          break;
        }
        continue;
      }

      profiler_.add_time(ProfileTracker::stage_key("overlay_q_wait", st->idx),
                         elapsed_ms(t_wait0, t_wait1));

      int frame_idx = -1;
      int pulled_now = 0;
      {
        std::lock_guard<std::mutex> stats_lock(stats_mu_);
        if (st->processed >= cfg_.frames) {
          continue;
        }
        frame_idx = st->processed;
        st->processed += 1;
        pulled_now = st->pulled;
      }

      profiler_.mark_processing_started(total_done_.load());

      const bool should_save = (frame_idx % cfg_.save_every) == 0;
      if (should_save) {
        const auto t_draw0 = Clock::now();
        draw_boxes(pkt.frame, pkt.boxes, labels_);
        const auto t_draw1 = Clock::now();
        profiler_.add_time(ProfileTracker::stage_key("draw_boxes", st->idx),
                           elapsed_ms(t_draw0, t_draw1));

        const fs::path out_path = st->out_dir / ("frame_" + cv::format("%06d.jpg", frame_idx));
        const auto t_wr0 = Clock::now();
        cv::imwrite(out_path.string(), pkt.frame);
        const auto t_wr1 = Clock::now();
        profiler_.add_time(ProfileTracker::stage_key("imwrite", st->idx), elapsed_ms(t_wr0, t_wr1));
        profiler_.add_time(ProfileTracker::stage_key("end_to_end", st->idx),
                           elapsed_ms(pkt.pulled_ts, t_wr1));
      }

      if (cfg_.debug || ((frame_idx + 1) % 10 == 0)) {
        std::lock_guard<std::mutex> log_lock(log_mu_);
        std::cout << "[stream " << st->idx << "] frames=" << (frame_idx + 1) << " pulled=" << pulled_now
                  << " det=" << pkt.boxes.size() << "\n";
      }

      const int done_now = total_done_.fetch_add(1) + 1;
      if (cfg_.profile && (done_now % std::max(1, cfg_.profile_every) == 0)) {
        std::vector<StreamSnapshot> snapshot;
        snapshot.reserve(streams_.size());
        {
          std::lock_guard<std::mutex> stats_lock(stats_mu_);
          for (const auto& s : streams_) {
            snapshot.push_back(StreamSnapshot{s->idx, s->processed, s->pulled,
                                              s->frame_q_dropped, s->result_q_dropped});
          }
        }
        profiler_.maybe_report(done_now, snapshot, log_mu_);
      }

      if (done_now >= total_target_) {
        stop_event_.store(true);
        break;
      }
    }
  }

  void monitor_until_done() {
    while (!stop_event_.load()) {
      const int done = total_done_.load();
      if (done >= total_target_) {
        stop_event_.store(true);
        break;
      }

      bool all_infer_done = true;
      {
        std::lock_guard<std::mutex> stats_lock(stats_mu_);
        for (const auto& s : streams_) {
          if (!s->infer_done) {
            all_infer_done = false;
            break;
          }
        }
      }
      bool result_empty = true;
      for (const auto& q : result_queues_) {
        if (!q->empty()) {
          result_empty = false;
          break;
        }
      }
      if (all_infer_done && result_empty) {
        stop_event_.store(true);
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
  }

  void close_queues() {
    for (const auto& q : frame_queues_) {
      q->close();
    }
    for (const auto& q : result_queues_) {
      q->close();
    }
  }

  void join_threads() {
    for (auto& t : producer_threads_) {
      t.join();
    }
    for (auto& t : infer_threads_) {
      t.join();
    }
    for (auto& t : overlay_threads_) {
      t.join();
    }
  }

  void print_summary() {
    const auto t_end = Clock::now();
    const int done = total_done_.load();
    const int done_since_start = profiler_.processed_since_start(done);
    const double elapsed_from_start_s = profiler_.elapsed_since_start_s(t_end);
    const double elapsed_s =
        (done_since_start > 0 && elapsed_from_start_s > 0.0)
            ? std::max(1e-6, elapsed_from_start_s)
            : std::max(1e-6, std::chrono::duration<double>(t_end - t0_).count());
    {
      std::lock_guard<std::mutex> log_lock(log_mu_);
      std::cout << "Processed " << done << " outputs in " << std::fixed << std::setprecision(2)
                << elapsed_s << "s avg_fps="
                << (static_cast<double>(std::max(0, done_since_start)) / elapsed_s) << "\n";
      for (const auto& s : streams_) {
        std::cout << "stream[" << s->idx << "] processed=" << s->processed << " pulled=" << s->pulled
                  << " saved to " << s->out_dir.string() << "/\n";
      }
    }
  }

  void close_runs() {
    for (const auto& s : streams_) {
      try {
        s->run.close();
      } catch (...) {
      }
    }
  }

private:
  Config cfg_;
  std::vector<std::string> labels_;
  fs::path out_root_;
  std::mutex log_mu_;
  std::mutex stats_mu_;
  std::atomic<bool> stop_event_{false};
  std::atomic<bool> start_producers_{false};
  ProfileTracker profiler_;
  std::vector<std::shared_ptr<StreamState>> streams_;
  std::vector<std::shared_ptr<KeepLatestQueue<FramePacket>>> frame_queues_;
  std::vector<std::shared_ptr<KeepLatestQueue<ResultPacket>>> result_queues_;
  int total_target_ = 0;
  std::atomic<int> total_done_{0};
  TimePoint t0_{};
  std::vector<std::thread> producer_threads_;
  std::vector<std::thread> infer_threads_;
  std::vector<std::thread> overlay_threads_;
  std::vector<std::unique_ptr<InferRuntime>> infer_runtimes_;
};

int main(int argc, char** argv) {
  try {
    PipelineApp app(parse_config(argc, argv));
    return app.run();
  } catch (const std::exception& ex) {
    std::cerr << "[ERR] " << ex.what() << "\n";
    usage();
    return 1;
  }
}
