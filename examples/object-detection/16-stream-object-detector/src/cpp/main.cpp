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

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr int kStreamLimit = 16;
constexpr int kInitialFrameTimeoutMs = 30000;
constexpr int kSourceDrainPullTimeoutMs = 250;
constexpr int kDetectorResultTimeoutMs = 5000;

volatile std::sig_atomic_t g_stop_requested = 0;

void request_stop(int) {
  g_stop_requested = 1;
}

struct AppConfig {
  std::string model_path;
  std::string decode_type = "yolo26";
  fs::path labels_path;
  std::vector<std::string> rtsp_urls;
  int workers = 4;
  int latency_ms = 100;
  bool tcp = true;
  double min_score = 0.55;
  double nms_iou = 0.60;
  int max_detections = 50;
  bool profile = false;
  int warmup_frames = 30;
  std::string insight_host = "127.0.0.1";
  int video_port_base = 9000;
  int metadata_port_base = 9100;
  bool video_enabled = true;
  fs::path save_dir;
  int save_every = 0;
};

simaai::neat::BoxDecodeType parse_box_decode_type(const std::string& token) {
  if (token == "yolo26" || token == "yolov26")
    return simaai::neat::BoxDecodeType::YoloV26;
  if (token == "yolov8")
    return simaai::neat::BoxDecodeType::YoloV8;
  throw std::runtime_error("model.decode_type must be one of: yolo26, yolov8");
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
  std::uint64_t latest_frame_overwrites = 0;
  int dropped_busy = 0;
  int detector_timeouts = 0;
  double start_ms = 0.0;
  double source_pull_ms = 0.0;
  double detector_ms = 0.0;
  double parse_ms = 0.0;
  double metadata_send_ms = 0.0;

  void add(double source_pull, double detector, double parse, double metadata_send, int box_count) {
    if (!enabled)
      return;
    if (frames == 0)
      start_ms = sima_examples::time_ms();
    ++frames;
    boxes += box_count;
    source_pull_ms += source_pull;
    detector_ms += detector;
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
              << " metadata_fps=" << metadata_fps << " avg_source_pull_ms=" << avg(source_pull_ms)
              << " avg_detector_ms=" << avg(detector_ms) << " avg_parse_ms=" << avg(parse_ms)
              << " avg_metadata_send_ms=" << avg(metadata_send_ms)
              << " avg_boxes=" << static_cast<double>(boxes) / static_cast<double>(frames)
              << " latest_frame_overwrites=" << latest_frame_overwrites
              << " dropped_busy=" << dropped_busy << " detector_timeouts=" << detector_timeouts
              << "\n";
    frames = 0;
    boxes = 0;
    latest_frame_overwrites = 0;
    dropped_busy = 0;
    detector_timeouts = 0;
    start_ms = 0.0;
    source_pull_ms = 0.0;
    detector_ms = 0.0;
    parse_ms = 0.0;
    metadata_send_ms = 0.0;
  }
};

class LatestFrameMailbox {
public:
  void store(simaai::neat::Sample frame) {
    std::lock_guard<std::mutex> lock(mu_);
    if (closed_) {
      return;
    }
    if (frame_.has_value()) {
      ++overwrites_;
    }
    frame_.emplace(std::move(frame));
  }

  std::optional<simaai::neat::Sample> take(std::uint64_t& overwrites) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!frame_.has_value()) {
      overwrites = 0;
      return std::nullopt;
    }
    overwrites = overwrites_;
    overwrites_ = 0;
    auto frame = std::move(frame_);
    frame_.reset();
    return frame;
  }

  void close() {
    std::lock_guard<std::mutex> lock(mu_);
    closed_ = true;
  }

  bool done() const {
    std::lock_guard<std::mutex> lock(mu_);
    return closed_ && !frame_.has_value();
  }

private:
  mutable std::mutex mu_;
  std::optional<simaai::neat::Sample> frame_;
  std::uint64_t overwrites_ = 0;
  bool closed_ = false;
};

struct SourceRuntime {
  int index = 0;
  std::string url;
  simaai::neat::Graph graph;
  simaai::neat::Run run;
  std::shared_ptr<LatestFrameMailbox> latest_frame = std::make_shared<LatestFrameMailbox>();
  std::unique_ptr<simaai::neat::MetadataSender> metadata_sender;
  std::vector<std::string> labels;
  StreamProfile profile;
  int frame_w = 0;
  int frame_h = 0;
  int source_fps = 0;
  int video_port = 0;
  int processed = 0;
};

struct DetectorWorker {
  int index = 0;
  std::vector<int> stream_indices;
  std::unique_ptr<simaai::neat::Model> model;
  simaai::neat::Graph graph;
  simaai::neat::Run run;
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
  sima_examples::require(!cfg.model_path.empty(), "model.path must be set");
  (void)parse_box_decode_type(cfg.decode_type);
  sima_examples::require(!cfg.labels_path.empty(), "model.labels must be set");
  sima_examples::require(!cfg.rtsp_urls.empty(), "streams must be set");
  sima_examples::require(cfg.rtsp_urls.size() <= kStreamLimit,
                         "this example supports up to 16 streams");
  sima_examples::require(cfg.workers > 0, "inference.workers must be > 0");
  sima_examples::require(cfg.workers <= static_cast<int>(cfg.rtsp_urls.size()),
                         "inference.workers must be <= stream count");
  sima_examples::require(!cfg.insight_host.empty(), "output.insight.host must be set");
  sima_examples::require(cfg.latency_ms >= 0, "input.latency_ms must be >= 0");
  sima_examples::require(cfg.min_score >= 0.0 && cfg.min_score <= 1.0,
                         "inference.min_score must be between 0 and 1");
  sima_examples::require(cfg.nms_iou >= 0.0 && cfg.nms_iou <= 1.0,
                         "inference.nms_iou must be between 0 and 1");
  sima_examples::require(cfg.max_detections > 0, "inference.max_detections must be > 0");
  sima_examples::require(cfg.warmup_frames >= 0, "runtime.warmup_frames must be >= 0");
  sima_examples::require(cfg.video_port_base > 0, "output.insight.video_port_base must be > 0");
  sima_examples::require(cfg.metadata_port_base > 0,
                         "output.insight.metadata_port_base must be > 0");
  sima_examples::require(cfg.save_every >= 0, "output.save_every must be >= 0");
}

AppConfig load_app_config(const fs::path& config_path) {
  const auto raw = sima_examples::ScalarConfig::load(config_path);
  sima_examples::require(!raw.string_value("inference.fps").has_value(),
                         "inference.fps is not supported; set stream FPS at the RTSP source");
  sima_examples::require(
      !raw.string_value("inference.target_fps").has_value(),
      "inference.target_fps is not supported; set stream FPS at the RTSP source");
  const auto default_labels =
      fs::path(SIMANEAT_APPS_EXAMPLE_SOURCE_DIR).parent_path() / "common" / "coco_label.txt";

  AppConfig cfg;
  cfg.model_path = raw.string_or("model.path", "");
  cfg.decode_type = raw.string_or("model.decode_type", "yolo26");
  cfg.labels_path = raw.string_or("model.labels", default_labels.string());
  cfg.rtsp_urls = parse_streams(config_path);
  cfg.tcp = raw.bool_or("input.tcp", true);
  cfg.latency_ms = raw.int_or("input.latency_ms", 100);
  cfg.workers = raw.int_or("inference.workers", std::min<int>(4, cfg.rtsp_urls.size()));
  cfg.min_score = raw.double_or("inference.min_score", 0.55);
  cfg.nms_iou = raw.double_or("inference.nms_iou", 0.60);
  cfg.max_detections = raw.int_or("inference.max_detections", 50);
  cfg.profile = raw.bool_or("runtime.profile", false);
  cfg.warmup_frames = raw.int_or("runtime.warmup_frames", 30);
  cfg.insight_host = raw.string_or("output.insight.host", "");
  cfg.video_port_base = raw.int_or("output.insight.video_port_base", 9000);
  cfg.metadata_port_base = raw.int_or("output.insight.metadata_port_base", 9100);
  cfg.video_enabled = raw.bool_or("output.video_enabled", true);
  cfg.save_dir = raw.string_or("output.debug_dir", "");
  cfg.save_every = raw.int_or("output.save_every", 0);
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

std::vector<std::vector<int>> assign_streams_to_workers(int stream_count, int worker_count) {
  std::vector<std::vector<int>> assignments(static_cast<std::size_t>(worker_count));
  for (int stream_index = 0; stream_index < stream_count; ++stream_index) {
    assignments[static_cast<std::size_t>(stream_index % worker_count)].push_back(stream_index);
  }
  return assignments;
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

int64_t fallback_timestamp_ms() {
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  return std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
}

simaai::neat::RunOptions realtime_options() {
  simaai::neat::RunOptions run_options;
  run_options.preset = simaai::neat::RunPreset::Realtime;
  run_options.queue_depth = 3;
  run_options.overflow_policy = simaai::neat::OverflowPolicy::KeepLatest;
  run_options.output_memory = simaai::neat::OutputMemory::ZeroCopy;
  return run_options;
}

simaai::neat::nodes::groups::RtspDecodedInputOptions
make_source_options(const AppConfig& cfg, const std::string& url, int& fps_out, int& width_out,
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
  opt.decoder_next_element = "CVU";
  opt.auto_caps_from_stream = true;
  if (probe.width > 0 && probe.height > 0) {
    opt.fallback_h264_width = probe.width;
    opt.fallback_h264_height = probe.height;
    opt.output_caps.width = probe.width;
    opt.output_caps.height = probe.height;
    width_out = probe.width;
    height_out = probe.height;
  }
  if (probe.fps > 0) {
    opt.fallback_h264_fps = probe.fps;
    opt.output_caps.fps = probe.fps;
    fps_out = probe.fps;
  }
  opt.output_caps.enable = true;
  opt.output_caps.format = simaai::neat::FormatTag::NV12;
  opt.output_caps.memory = simaai::neat::CapsMemory::Any;
  return opt;
}

std::unique_ptr<simaai::neat::Model> make_model(const AppConfig& cfg) {
  simaai::neat::Model::Options model_opt;
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
  auto video_options = simaai::neat::nodes::groups::VideoSenderOptions::H264RtpUdpFromRaw(
      source.frame_w, source.frame_h, source.source_fps);
  video_options.host = cfg.insight_host;
  video_options.channel = source.index;
  video_options.video_port_base = cfg.video_port_base;
  video_options.encoder.bitrate_kbps = 1000;
  return video_options;
}

SourceRuntime build_source_runtime(const AppConfig& cfg, int stream_index,
                                   const std::vector<std::string>& labels) {
  SourceRuntime source_runtime;
  source_runtime.index = stream_index;
  source_runtime.url = cfg.rtsp_urls[static_cast<std::size_t>(stream_index)];
  const auto source_options =
      make_source_options(cfg, source_runtime.url, source_runtime.source_fps,
                          source_runtime.frame_w, source_runtime.frame_h);
  sima_examples::require(source_runtime.frame_w > 0 && source_runtime.frame_h > 0,
                         "failed to probe RTSP frame dimensions");
  sima_examples::require(source_runtime.source_fps > 0, "failed to probe RTSP frame rate");
  source_runtime.labels = labels;
  source_runtime.profile.enabled = cfg.profile;
  source_runtime.profile.stream_index = stream_index;

  auto rtsp = simaai::neat::nodes::groups::RtspDecodedInput(source_options);
  std::vector<std::string> outputs = {"detector_frame"};
  if (cfg.video_enabled) {
    outputs.push_back("video");
  }
  auto branch = simaai::neat::graphs::Branch("source", outputs);

  source_runtime.graph.connect(rtsp, branch);
  if (cfg.video_enabled) {
    auto video_options = make_video_options(cfg, source_runtime);
    source_runtime.video_port = video_options.video_port();
    simaai::neat::Graph video_graph("video");
    video_graph.connect(simaai::neat::nodes::Input("video"),
                        simaai::neat::nodes::groups::VideoSender(video_options));
    source_runtime.graph.connect(branch, video_graph);
  }

  simaai::neat::Graph detector_frame_graph("detector_frame");
  detector_frame_graph.add(
      simaai::neat::nodes::Output("detector_frame", simaai::neat::OutputOptions::Latest()));
  source_runtime.graph.connect(branch, detector_frame_graph);

  if (cfg.profile) {
    std::cout << "Source backend stream=" << stream_index << ":\n"
              << source_runtime.graph.describe_backend() << "\n";
  }
  source_runtime.run = source_runtime.graph.build(realtime_options());

  simaai::neat::MetadataSenderOptions metadata_options;
  metadata_options.host = cfg.insight_host;
  metadata_options.channel = stream_index;
  metadata_options.metadata_port_base = cfg.metadata_port_base;
  std::string metadata_err;
  source_runtime.metadata_sender =
      std::make_unique<simaai::neat::MetadataSender>(metadata_options, &metadata_err);
  sima_examples::require(source_runtime.metadata_sender->ok(), metadata_err);

  std::cout << "[stream " << stream_index << "] rtsp=" << source_runtime.url
            << " stream=" << source_runtime.frame_w << "x" << source_runtime.frame_h << "@"
            << source_runtime.source_fps << " insight=" << cfg.insight_host << " video=";
  if (cfg.video_enabled) {
    std::cout << source_runtime.video_port;
  } else {
    std::cout << "disabled";
  }
  std::cout << " metadata=" << source_runtime.metadata_sender->metadata_port() << "\n";
  return source_runtime;
}

void validate_worker_pool_geometry(const std::vector<SourceRuntime>& sources) {
  if (sources.empty()) {
    return;
  }
  const int frame_w = sources.front().frame_w;
  const int frame_h = sources.front().frame_h;
  for (const auto& source : sources) {
    if (source.frame_w != frame_w || source.frame_h != frame_h) {
      throw std::runtime_error("detector worker pool requires all streams to share frame size");
    }
  }
}

void record_first_error(std::exception_ptr& first_error, std::mutex& error_mutex) {
  std::lock_guard<std::mutex> lock(error_mutex);
  if (!first_error)
    first_error = std::current_exception();
}

std::exception_ptr first_error_snapshot(std::exception_ptr& first_error, std::mutex& error_mutex) {
  std::lock_guard<std::mutex> lock(error_mutex);
  return first_error;
}

simaai::neat::Sample wait_for_seed_sample(std::vector<SourceRuntime>& sources,
                                          std::atomic_bool& stop_requested,
                                          std::exception_ptr& first_error,
                                          std::mutex& error_mutex) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(kInitialFrameTimeoutMs);
  while (std::chrono::steady_clock::now() < deadline && !stop_requested.load() &&
         g_stop_requested == 0) {
    if (auto error = first_error_snapshot(first_error, error_mutex)) {
      std::rethrow_exception(error);
    }
    for (auto& source : sources) {
      std::uint64_t ignored_overwrites = 0;
      auto sample = source.latest_frame->take(ignored_overwrites);
      if (sample.has_value()) {
        return std::move(*sample);
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  throw std::runtime_error("timed out waiting for initial detector frame");
}

simaai::neat::TensorList detector_input_tensors_from_frame(const simaai::neat::Sample& frame) {
  const auto tensors = simaai::neat::tensors_from_sample(frame, true);
  return simaai::neat::TensorList{tensors.front().cvu()};
}

DetectorWorker build_detector_worker(const AppConfig& cfg, int worker_index,
                                     std::vector<int> stream_indices,
                                     const simaai::neat::Sample& seed) {
  DetectorWorker worker;
  worker.index = worker_index;
  worker.stream_indices = std::move(stream_indices);
  worker.model = make_model(cfg);

  simaai::neat::Graph input_graph("frame");
  input_graph.add(simaai::neat::nodes::Input("frame", worker.model->input_appsrc_options(false)));

  simaai::neat::Graph model_graph = worker.model->graph();

  simaai::neat::Graph detections_graph("detections");
  detections_graph.add(
      simaai::neat::nodes::Output("detections", simaai::neat::OutputOptions::EveryFrame(4)));

  worker.graph.connect(input_graph, model_graph);
  worker.graph.connect(model_graph, detections_graph);
  if (cfg.profile) {
    std::cout << "Detector worker backend worker=" << worker_index << ":\n"
              << worker.graph.describe_backend() << "\n";
  }
  worker.run = worker.graph.build(detector_input_tensors_from_frame(seed), realtime_options());
  return worker;
}

void send_metadata(SourceRuntime& source, const simaai::neat::Sample& frame,
                   const std::vector<objdet::Box>& boxes) {
  const auto metadata_boxes =
      build_metadata_boxes(boxes, source.labels, source.frame_w, source.frame_h);
  const std::string data_json = sima_examples::metadata_boxes_data_json("objects", metadata_boxes);
  const int64_t timestamp_ms = frame.pts_ns >= 0 ? frame.pts_ns / 1000000 : fallback_timestamp_ms();
  const int64_t frame_id = frame.frame_id >= 0 ? frame.frame_id : 0;
  std::string err;
  if (!source.metadata_sender->send_metadata("object-detection", data_json, timestamp_ms,
                                             std::to_string(frame_id), &err)) {
    std::cerr << "[warn] stream " << source.index << " metadata send failed: " << err << "\n";
  }
}

void maybe_save_debug_frame(const AppConfig& cfg, const SourceRuntime& source,
                            const simaai::neat::Sample& frame,
                            const std::vector<objdet::Box>& boxes) {
  if (cfg.save_dir.empty() || cfg.save_every <= 0 || source.processed % cfg.save_every != 0) {
    return;
  }

  const auto tensors = simaai::neat::tensors_from_sample(frame, false);
  if (tensors.empty()) {
    return;
  }

  cv::Mat bgr;
  std::string err;
  if (!sima_examples::nv12_to_bgr(tensors.front(), bgr, err)) {
    std::cerr << "[warn] failed to prepare output frame: " << err << "\n";
    return;
  }
  objdet::draw_boxes(bgr, boxes, cfg.min_score, cv::Scalar(0, 255, 0), "");
  const auto out_path = cfg.save_dir / ("stream_" + std::to_string(source.index) + "_frame_" +
                                        std::to_string(source.processed) + ".jpg");
  if (!cv::imwrite(out_path.string(), bgr)) {
    std::cerr << "[warn] failed to write output frame: " << out_path.string() << "\n";
  }
}

bool process_stream_once(DetectorWorker& worker, SourceRuntime& source, const AppConfig& cfg) {
  const double source_pull_start = sima_examples::time_ms();
  std::uint64_t latest_frame_overwrites = 0;
  auto frame = source.latest_frame->take(latest_frame_overwrites);
  const double source_pull_end = sima_examples::time_ms();
  if (!frame.has_value()) {
    return false;
  }
  source.profile.latest_frame_overwrites += latest_frame_overwrites;

  const double detector_start = sima_examples::time_ms();
  const auto detector_input = detector_input_tensors_from_frame(*frame);
  if (!worker.run.try_push("frame", detector_input)) {
    ++source.profile.dropped_busy;
    return true;
  }

  simaai::neat::Sample detections;
  simaai::neat::PullError detector_error;
  const auto detector_status =
      worker.run.pull("detections", kDetectorResultTimeoutMs, detections, &detector_error);
  const double detector_end = sima_examples::time_ms();
  if (detector_status == simaai::neat::PullStatus::Timeout) {
    ++source.profile.detector_timeouts;
    throw std::runtime_error("worker " + std::to_string(worker.index) +
                             " timed out waiting for detections");
  }
  if (detector_status != simaai::neat::PullStatus::Ok) {
    throw std::runtime_error("worker " + std::to_string(worker.index) +
                             " failed to pull detections: " + detector_error.message);
  }

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
  const bool warming_up = source.processed <= cfg.warmup_frames;
  if (!warming_up) {
    const double metadata_start = sima_examples::time_ms();
    send_metadata(source, *frame, boxes);
    const double metadata_end = sima_examples::time_ms();
    maybe_save_debug_frame(cfg, source, *frame, boxes);
    source.profile.add(source_pull_end - source_pull_start, detector_end - detector_start,
                       parse_end - parse_start, metadata_end - metadata_start,
                       static_cast<int>(boxes.size()));
  }
  return true;
}

bool assigned_streams_closed(const DetectorWorker& worker,
                             const std::vector<SourceRuntime>& sources) {
  for (int stream_index : worker.stream_indices) {
    if (!sources[static_cast<std::size_t>(stream_index)].latest_frame->done()) {
      return false;
    }
  }
  return true;
}

void run_source_drain(SourceRuntime& source, std::atomic_bool& stop_requested,
                      std::exception_ptr& first_error, std::mutex& error_mutex) {
  try {
    while (!stop_requested.load() && g_stop_requested == 0) {
      simaai::neat::Sample frame;
      simaai::neat::PullError err;
      const auto status = source.run.pull("detector_frame", kSourceDrainPullTimeoutMs, frame, &err);
      if (status == simaai::neat::PullStatus::Ok) {
        source.latest_frame->store(std::move(frame));
      } else if (status == simaai::neat::PullStatus::Closed) {
        break;
      } else if (status == simaai::neat::PullStatus::Error) {
        throw std::runtime_error("stream " + std::to_string(source.index) +
                                 " failed to drain detector frame: " + err.message);
      }
    }
  } catch (...) {
    record_first_error(first_error, error_mutex);
    stop_requested.store(true);
  }
  source.latest_frame->close();
}

void run_worker(DetectorWorker& worker, std::vector<SourceRuntime>& sources, const AppConfig& cfg,
                std::atomic_bool& stop_requested, std::exception_ptr& first_error,
                std::mutex& error_mutex) {
  try {
    while (!stop_requested.load() && g_stop_requested == 0 &&
           !assigned_streams_closed(worker, sources)) {
      bool did_work = false;
      for (int stream_index : worker.stream_indices) {
        auto& source = sources[static_cast<std::size_t>(stream_index)];
        if (source.latest_frame->done()) {
          continue;
        }
        did_work = process_stream_once(worker, source, cfg) || did_work;
      }
      if (!did_work) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }
  } catch (...) {
    record_first_error(first_error, error_mutex);
    stop_requested.store(true);
  }
}

void run_app(const AppConfig& cfg) {
  g_stop_requested = 0;
  auto previous_sigint = std::signal(SIGINT, request_stop);
  if (cfg.profile) {
    setenv("SIMA_GST_ELEMENT_TIMINGS", "1", 0);
    setenv("SIMA_GST_FLOW_DEBUG", "1", 0);
    setenv("SIMA_GST_BOUNDARY_PROBES", "1", 0);
  }
  if (!cfg.save_dir.empty()) {
    fs::create_directories(cfg.save_dir);
  }

  std::atomic_bool stop_requested{false};
  std::exception_ptr first_error;
  std::mutex error_mutex;

  const auto labels = load_labels(cfg.labels_path);
  std::vector<SourceRuntime> sources;
  // Source drain threads hold references to SourceRuntime entries; reserve prevents relocation.
  sources.reserve(cfg.rtsp_urls.size());
  std::vector<std::thread> source_threads;
  source_threads.reserve(cfg.rtsp_urls.size());
  try {
    for (std::size_t index = 0; index < cfg.rtsp_urls.size(); ++index) {
      sources.push_back(build_source_runtime(cfg, static_cast<int>(index), labels));
      source_threads.emplace_back(run_source_drain, std::ref(sources.back()),
                                  std::ref(stop_requested), std::ref(first_error),
                                  std::ref(error_mutex));
    }
    validate_worker_pool_geometry(sources);
  } catch (...) {
    record_first_error(first_error, error_mutex);
    stop_requested.store(true);
  }

  const auto assignments = assign_streams_to_workers(static_cast<int>(sources.size()), cfg.workers);
  std::vector<DetectorWorker> workers;
  workers.reserve(assignments.size());
  if (!first_error_snapshot(first_error, error_mutex)) {
    try {
      const simaai::neat::Sample seed =
          wait_for_seed_sample(sources, stop_requested, first_error, error_mutex);
      for (std::size_t index = 0; index < assignments.size(); ++index) {
        workers.push_back(
            build_detector_worker(cfg, static_cast<int>(index), assignments[index], seed));
      }
    } catch (...) {
      record_first_error(first_error, error_mutex);
      stop_requested.store(true);
    }
  }

  std::vector<std::thread> worker_threads;
  worker_threads.reserve(workers.size());
  if (!first_error_snapshot(first_error, error_mutex)) {
    for (auto& worker : workers) {
      worker_threads.emplace_back(run_worker, std::ref(worker), std::ref(sources), std::cref(cfg),
                                  std::ref(stop_requested), std::ref(first_error),
                                  std::ref(error_mutex));
    }
  }

  for (auto& worker_thread : worker_threads) {
    if (worker_thread.joinable())
      worker_thread.join();
  }
  stop_requested.store(true);
  for (auto& source_thread : source_threads) {
    if (source_thread.joinable())
      source_thread.join();
  }

  for (auto& worker : workers) {
    worker.run.close();
  }
  for (auto& source : sources) {
    source.profile.flush();
    source.run.close();
    std::cout << "[stream " << source.index << "] processed=" << source.processed << "\n";
  }

  std::signal(SIGINT, previous_sigint);
  if (first_error)
    std::rethrow_exception(first_error);
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
      std::cout << "Config validated: " << cli.config_path << " (streams=" << cfg.rtsp_urls.size()
                << ", workers=" << cfg.workers << ")\n";
      return 0;
    }
    run_app(cfg);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[ERR] " << e.what() << "\n";
    return 1;
  }
}
