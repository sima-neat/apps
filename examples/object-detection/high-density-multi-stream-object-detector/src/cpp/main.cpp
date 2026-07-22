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
#include "detection_egress.h"
#include "detection_watchdog.h"
#include "support/object_detection/obj_detection_utils.h"
#include "support/runtime/config_utils.h"
#include "support/runtime/example_utils.h"

#include <nodes/groups/VideoSender.h>
#include <nodes/io/MetadataSender.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <csignal>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr int kStreamLimit = 80;
constexpr int kDefaultInitialDetectionTimeoutMs = 30000;
constexpr int kDetectorResultTimeoutMs = 5000;
constexpr int kDefaultQueueDepth = 4;
constexpr int kDefaultInternalQueueDepth = 1;
constexpr int kDefaultMaxInflightPerStream = 4;
constexpr int kDefaultMaxInflightTotal = 8;
// Complete MLA work asynchronously so a busy shared model stage cannot starve
// a subset of decoder streams.
constexpr bool kInferenceAsync = true;
constexpr int kDefaultDecoderBuffers = 16;
constexpr int kDefaultDecoderInputBuffers = 2;
constexpr int kAllInsightStreams = -1;

volatile std::sig_atomic_t g_stop_requested = 0;

void request_stop(int) {
  g_stop_requested = 1;
}

struct AppConfig {
  std::string model_path;
  std::string decode_type = "yolo26";
  fs::path labels_path;
  std::vector<std::string> rtsp_urls;
  bool use_h265 = false;
  int workers = 1;
  int queue_depth = kDefaultQueueDepth;
  int internal_queue_depth = kDefaultInternalQueueDepth;
  int max_inflight_per_stream = kDefaultMaxInflightPerStream;
  int max_inflight_total = kDefaultMaxInflightTotal;
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
  bool profile = false;
  int warmup_frames = 30;
  int initial_detection_timeout_ms = kDefaultInitialDetectionTimeoutMs;
  std::string insight_host = "127.0.0.1";
  int video_port_base = 9000;
  int metadata_port_base = 9100;
  int insight_visible_streams = kAllInsightStreams;
  bool video_enabled = true;
};

bool parse_use_h265(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (value == "h264" || value == "avc" || value == "h.264")
    return false;
  if (value == "h265" || value == "hevc" || value == "h.265")
    return true;
  throw std::runtime_error("input.codec must be h264/avc or h265/hevc");
}

void apply_source_caps(simaai::neat::nodes::groups::RtspDecodedInputOptions& opt, bool use_h265,
                       int width, int height, int fps, int& width_out, int& height_out,
                       int& fps_out) {
  if (width > 0 && height > 0) {
    if (use_h265) {
      opt.dec_width = width;
      opt.dec_height = height;
    } else {
      opt.fallback_h264_width = width;
      opt.fallback_h264_height = height;
    }
    opt.output_caps.width = width;
    opt.output_caps.height = height;
    width_out = width;
    height_out = height;
  }
  if (fps > 0) {
    if (use_h265) {
      opt.source_fps = fps;
    } else {
      opt.fallback_h264_fps = fps;
    }
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
  std::vector<objdet::Box> parsed_boxes;
  int frame_w = 0;
  int frame_h = 0;
  int source_fps = 0;
  int video_port = 0;
  int processed = 0;
  std::uint64_t metadata_send_ok = 0;
  std::uint64_t metadata_send_fail = 0;
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
  sima_examples::require(!cfg.model_path.empty(), "model.path must be set");
  (void)parse_box_decode_type(cfg.decode_type);
  sima_examples::require(!cfg.labels_path.empty(), "model.labels must be set");
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
  sima_examples::require(cfg.max_inflight_total > 0, "inference.max_inflight_total must be > 0");
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
  sima_examples::require(
      !raw.string_value("inference.fan_in_policy").has_value(),
      "inference.fan_in_policy was removed; remove it because ordinary connect()/build() now "
      "selects realtime fan-in automatically");
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
  cfg.labels_path = raw.string_or("model.labels", "coco_label.txt");
  cfg.rtsp_urls = parse_streams(config_path);
  cfg.use_h265 = parse_use_h265(raw.string_or("input.codec", "h264"));
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
  cfg.max_inflight_total = raw.int_or("inference.max_inflight_total", kDefaultMaxInflightTotal);
  cfg.min_score = raw.double_or("inference.min_score", 0.55);
  cfg.nms_iou = raw.double_or("inference.nms_iou", 0.60);
  cfg.max_detections = raw.int_or("inference.max_detections", 50);
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

const simaai::neat::Tensor* find_bbox_tensor(const simaai::neat::Sample& sample, std::string& err) {
  if (sample.kind == simaai::neat::SampleKind::Bundle) {
    for (const auto& field : sample.fields) {
      if (const auto* tensor = find_bbox_tensor(field, err))
        return tensor;
    }
    err = "bundle missing BBOX field";
    return nullptr;
  }

  const simaai::neat::Tensor* tensor = nullptr;
  if (sample.kind == simaai::neat::SampleKind::Tensor && sample.tensor.has_value()) {
    tensor = &*sample.tensor;
  } else if (sample.kind == simaai::neat::SampleKind::TensorSet && !sample.tensors.empty()) {
    tensor = &sample.tensors.front();
  } else {
    err = sample.kind == simaai::neat::SampleKind::Tensor ? "capture_missing_tensor"
                                                          : "capture_expected_tensor";
    return nullptr;
  }

  std::string format = sample.payload_tag;
  if (format.empty() && !sample.format.empty())
    format = sample.format;
  if (format.empty() && tensor->semantic.tess.has_value())
    format = tensor->semantic.tess->format;
  const std::string format_upper = objdet::upper_ascii_copy(format);
  if (!format_upper.empty() && format_upper != "BBOX") {
    err = "capture_expected_bbox format=" + format_upper;
    return nullptr;
  }
  return tensor;
}

struct BboxPayloadView {
  simaai::neat::Mapping mapping;
  std::vector<std::uint8_t> owned;
  const std::uint8_t* data = nullptr;
  std::size_t size = 0;

  [[nodiscard]] std::span<const std::uint8_t> bytes() const {
    return {data, size};
  }
};

bool map_bbox_payload(const simaai::neat::Sample& sample, BboxPayloadView& payload,
                      std::string& err) {
  const auto* tensor = find_bbox_tensor(sample, err);
  if (!tensor)
    return false;

  try {
    const std::size_t tight_bytes = tensor->dense_bytes_tight();
    if (tight_bytes > 0 && tensor->is_contiguous()) {
      payload.mapping = tensor->map_read();
      if (payload.mapping.data && payload.mapping.size_bytes >= tight_bytes) {
        payload.data = static_cast<const std::uint8_t*>(payload.mapping.data);
        payload.size = tight_bytes;
      }
    }
    if (!payload.data) {
      payload.mapping = {};
      payload.owned = tensor->copy_payload_bytes();
      payload.data = payload.owned.data();
      payload.size = payload.owned.size();
    }
  } catch (const std::exception& ex) {
    err = "capture_payload_failed err=";
    err += ex.what();
    return false;
  }

  if (payload.size == 0) {
    err = "capture_empty_payload";
    return false;
  }
  return true;
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
  return env_bool("HIGH_DENSITY_DETECTOR_VERBOSE", false);
}

bool app_print_backend() {
  return app_verbose() || env_bool("HIGH_DENSITY_DETECTOR_PRINT_BACKEND", false);
}

int app_liveness_ms() {
  return std::max(0, env_int("HIGH_DENSITY_DETECTOR_LIVENESS_MS", 0));
}

void print_pull_liveness(const std::vector<SourceRuntime>& sources, const char* reason,
                         std::uint64_t total_pulls) {
  if (sources.empty()) {
    std::cerr << "[detector][liveness] reason=" << (reason ? reason : "snapshot")
              << " streams=0 total_pulls=" << total_pulls << "\n";
    return;
  }

  int min_processed = sources.front().processed;
  int max_processed = sources.front().processed;
  int zero_streams = 0;
  std::uint64_t metadata_send_ok = 0;
  std::uint64_t metadata_send_fail = 0;
  for (const auto& source : sources) {
    min_processed = std::min(min_processed, source.processed);
    max_processed = std::max(max_processed, source.processed);
    if (source.processed == 0) {
      ++zero_streams;
    }
    metadata_send_ok += source.metadata_send_ok;
    metadata_send_fail += source.metadata_send_fail;
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

  std::cerr << "[detector][liveness] reason=" << (reason ? reason : "snapshot")
            << " streams=" << sources.size() << " total_pulls=" << total_pulls
            << " min_processed=" << min_processed << " max_processed=" << max_processed
            << " zero_streams=" << zero_streams << " metadata_send_ok=" << metadata_send_ok
            << " metadata_send_fail=" << metadata_send_fail << " low_cutoff=" << low_cutoff
            << " low=";
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
  return std::max(0, env_int("HIGH_DENSITY_DETECTOR_FRAMES_PER_STREAM", 0));
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
    simaai::neat::OverflowPolicy overflow_policy = simaai::neat::OverflowPolicy::KeepLatest) {
  simaai::neat::RunOptions run_options;
  run_options.preset = simaai::neat::RunPreset::Realtime;
  run_options.queue_depth = queue_depth;
  run_options.overflow_policy = overflow_policy;
  run_options.output_memory = simaai::neat::OutputMemory::ZeroCopy;
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
  opt.codec = cfg.use_h265 ? simaai::neat::nodes::groups::RtspCodec::H265
                           : simaai::neat::nodes::groups::RtspCodec::H264;
  apply_source_caps(opt, cfg.use_h265, probe.width, probe.height, probe.fps, width_out, height_out,
                    fps_out);
  apply_source_caps(opt, cfg.use_h265, cfg.input_width, cfg.input_height, cfg.input_fps, width_out,
                    height_out, fps_out);
  opt.output_caps.enable = true;
  opt.output_caps.format = simaai::neat::FormatTag::NV12;
  opt.output_caps.memory = simaai::neat::CapsMemory::Any;
  return opt;
}

simaai::neat::Graph
make_rtsp_encoded_input(const simaai::neat::nodes::groups::RtspDecodedInputOptions& opt) {
  simaai::neat::nodes::groups::RtspEncodedInputOptions encoded;
  encoded.url = opt.url;
  encoded.codec = opt.codec;
  encoded.latency_ms = opt.latency_ms;
  encoded.tcp = opt.tcp;
  encoded.drop_on_latency = opt.drop_on_latency;
  encoded.buffer_mode = opt.buffer_mode;
  encoded.insert_queue = opt.insert_queue;
  encoded.sync_mode = opt.sync_mode;
  encoded.auto_caps_from_stream = opt.auto_caps_from_stream;
  if (opt.codec == simaai::neat::nodes::groups::RtspCodec::H265) {
    encoded.h265_payload_type = opt.payload_type;
    encoded.source_fps = opt.source_fps;
  } else {
    encoded.h264_payload_type = opt.payload_type;
    encoded.h264_parse_config_interval = opt.h264_parse_config_interval;
    encoded.h264_fps = opt.h264_fps;
    encoded.h264_width = opt.h264_width;
    encoded.h264_height = opt.h264_height;
    encoded.fallback_h264_fps = opt.fallback_h264_fps;
    encoded.fallback_h264_width = opt.fallback_h264_width;
    encoded.fallback_h264_height = opt.fallback_h264_height;
  }
  return simaai::neat::nodes::groups::RtspEncodedInput(encoded);
}

simaai::neat::Graph make_decoder(const simaai::neat::nodes::groups::RtspDecodedInputOptions& opt,
                                 int decoder_buffers) {
  const bool use_h265 = opt.codec == simaai::neat::nodes::groups::RtspCodec::H265;
  const int dec_w =
      use_h265 ? opt.dec_width : ((opt.h264_width > 0) ? opt.h264_width : opt.fallback_h264_width);
  const int dec_h = use_h265 ? opt.dec_height
                             : ((opt.h264_height > 0) ? opt.h264_height : opt.fallback_h264_height);
  const int dec_fps =
      use_h265 ? opt.source_fps : ((opt.h264_fps > 0) ? opt.h264_fps : opt.fallback_h264_fps);

  simaai::neat::Graph graph("decoder");
  simaai::neat::SimaDecodeOptions decode;
  decode.type = use_h265 ? simaai::neat::SimaDecodeType::H265 : simaai::neat::SimaDecodeType::H264;
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
  graph.add(simaai::neat::nodes::Output("detector_frame"));
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
  auto video_options =
      cfg.use_h265 ? simaai::neat::nodes::groups::VideoSenderOptions::H265RtpUdpFromEncoded()
                   : simaai::neat::nodes::groups::VideoSenderOptions::H264RtpUdpFromEncoded();
  video_options.host = cfg.insight_host;
  video_options.channel = source.index;
  video_options.video_port_base = cfg.video_port_base;
  // This sender is fused into the same live pipeline as the decoder fan-in.
  // It must not take part in preroll, otherwise every asynchronous UDP sink
  // can hold the shared pipeline in PAUSED before all decoder branches start.
  video_options.async = false;
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
  source.parsed_boxes.reserve(static_cast<std::size_t>(cfg.max_detections));

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

void connect_source_graph(AppRuntime& app, const AppConfig& cfg, SourceRuntime& source,
                          const simaai::neat::Graph& detector_graph) {
  simaai::neat::GraphLinkOptions detector_link;
  detector_link.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
  detector_link.queue_depth = cfg.queue_depth;
  detector_link.stream_id = stream_id_for(source.index);
  detector_link.max_inflight_per_stream = cfg.max_inflight_per_stream;
  detector_link.max_inflight_total = cfg.max_inflight_total;

  // Keep the encoded RTSP producer explicit in the public topology. Core
  // lowers this fan-out internally: the decoder branch remains fused with the
  // shared detector, while VideoSender consumes the same read-only encoded AU
  // before the decoder. This keeps video delivery off the application pull
  // path and avoids retaining decoded EV buffers.
  auto rtsp = make_rtsp_encoded_input(source.source_options);
  auto decoder = make_decoder(source.source_options, cfg.decoder_buffers);
  app.graph.connect(rtsp, decoder);
  app.graph.connect(decoder, detector_graph, detector_link);

  if (cfg.video_enabled && is_insight_visible_stream(cfg, source.index)) {
    const auto video_options = make_video_options(cfg, source);
    source.video_port = video_options.video_port();
    auto video_sender = simaai::neat::nodes::groups::VideoSender(video_options);
    video_sender.set_name("encoded_insight_video_sender_" + std::to_string(source.index));
    simaai::neat::GraphLinkOptions video_link;
    video_link.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
    app.graph.connect(rtsp, video_sender, video_link);
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
  if (sent) {
    ++source.metadata_send_ok;
    return;
  }
  const std::uint64_t failures = ++source.metadata_send_fail;
  if (should_log_failure_count(failures)) {
    std::cerr << "[warn] stream " << source.index
              << " nonblocking metadata send failed (count=" << failures << "): " << err << "\n";
  }
}

void send_metadata(SourceRuntime& source, const simaai::neat::Sample& frame,
                   const std::vector<objdet::Box>& boxes) {
  if (!source.metadata_sender) {
    return;
  }
  high_density::detection_egress::FrameMetadata metadata;
  metadata.stream_index = source.index;
  const std::string fallback_stream_id =
      frame.stream_id.empty() ? stream_id_for(source.index) : std::string{};
  metadata.stream_id = frame.stream_id.empty() ? std::string_view(fallback_stream_id)
                                               : std::string_view(frame.stream_id);
  metadata.frame_id = frame.frame_id;
  metadata.pts_ns = frame.pts_ns;
  metadata.dts_ns = frame.dts_ns;
  metadata.duration_ns = frame.duration_ns;
  metadata.input_seq = frame.input_seq;
  metadata.orig_input_seq = frame.orig_input_seq;
  if (frame.pts_ns >= 0)
    metadata.rtp_timestamp = rtp_timestamp_from_pts_ns(frame.pts_ns);

  std::string payload;
  try {
    payload = high_density::detection_egress::serialize(boxes, source.labels, source.frame_w,
                                                        source.frame_h, metadata);
  } catch (const std::exception& ex) {
    std::cerr << "[warn] stream " << source.index << " metadata JSON build failed: " << ex.what()
              << "\n";
    return;
  }

  send_metadata_nonblocking(source, payload);
}

void complete_detection(SourceRuntime& source, const AppConfig& cfg,
                        AggregateProfile& aggregate_profile,
                        const simaai::neat::Sample& detections) {
  const double parse_start = sima_examples::time_ms();
  {
    BboxPayloadView payload;
    std::string err;
    if (!map_bbox_payload(detections, payload, err)) {
      throw std::runtime_error("stream " + std::to_string(source.index) +
                               " bbox extract failed: " + err);
    }
    objdet::parse_boxes_strict_into(payload.bytes(), source.frame_w, source.frame_h,
                                    cfg.max_detections, false, source.parsed_boxes);
  }
  const double parse_end = sima_examples::time_ms();

  ++source.processed;
  const bool warming_up = source.processed <= cfg.warmup_frames;

  if (!warming_up) {
    const double metadata_start = sima_examples::time_ms();
    send_metadata(source, detections, source.parsed_boxes);
    const double metadata_end = sima_examples::time_ms();
    source.profile.add(parse_end - parse_start, metadata_end - metadata_start,
                       static_cast<int>(source.parsed_boxes.size()));
    aggregate_profile.add();
  }
}

void pull_detections(AppRuntime& app, const AppConfig& cfg, AggregateProfile& aggregate_profile) {
  std::uint64_t total_pulls = 0;
  const int liveness_ms = app_liveness_ms();
  auto now = std::chrono::steady_clock::now();
  high_density::DetectionWatchdog watchdog(
      app.sources.size(), std::chrono::milliseconds(cfg.initial_detection_timeout_ms),
      std::chrono::milliseconds(kDetectorResultTimeoutMs), now);
  auto next_liveness = now + std::chrono::milliseconds(liveness_ms);
  while (g_stop_requested == 0) {
    bool did_work = false;
    constexpr int kMaxDetectionsPerRound = 64;
    for (int drained = 0; drained < kMaxDetectionsPerRound; ++drained) {
      simaai::neat::Sample detections;
      simaai::neat::PullError err;
      const auto status = app.run.pull("detections", /*timeout_ms=*/0, detections, &err);
      if (status == simaai::neat::PullStatus::Closed) {
        const std::string run_error = app.run.last_error();
        throw std::runtime_error("detections output closed unexpectedly" +
                                 (run_error.empty() ? std::string{} : ": " + run_error));
      }
      if (status == simaai::neat::PullStatus::Timeout) {
        break;
      }
      if (status == simaai::neat::PullStatus::Error) {
        throw std::runtime_error("failed to pull detections: " + err.message);
      }
      if (status != simaai::neat::PullStatus::Ok) {
        continue;
      }

      did_work = true;
      ++total_pulls;
      const int stream_index =
          stream_index_from_detection(detections, static_cast<int>(app.sources.size()));
      watchdog.observe(static_cast<std::size_t>(stream_index));
      auto& source = app.sources[static_cast<std::size_t>(stream_index)];
      complete_detection(source, cfg, aggregate_profile, detections);
      if (target_reached(app.sources)) {
        return;
      }
    }

    now = std::chrono::steady_clock::now();
    const auto expired = watchdog.expired_streams(now);
    if (!expired.empty()) {
      const bool startup_complete = watchdog.startup_complete();
      print_pull_liveness(app.sources,
                          startup_complete ? "stream_detection_timeout"
                                           : "initial_stream_detection_timeout",
                          total_pulls);
      std::string stream_list;
      for (const auto index : expired) {
        if (!stream_list.empty()) {
          stream_list += ",";
        }
        stream_list += std::to_string(index);
      }
      throw std::runtime_error(std::string("timed out waiting for ") +
                               (startup_complete ? "detections" : "initial detections") +
                               " from streams: " + stream_list);
    }
    if (liveness_ms > 0) {
      if (now >= next_liveness) {
        print_pull_liveness(app.sources, "heartbeat", total_pulls);
        next_liveness = now + std::chrono::milliseconds(liveness_ms);
      }
    }
    if (!did_work) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
}

void flush_and_print_runtime_stats(AggregateProfile& aggregate_profile,
                                   std::vector<SourceRuntime>& sources) {
  aggregate_profile.flush();
  for (auto& source : sources) {
    source.profile.flush();
    const auto metadata_stats = source.metadata_sender ? source.metadata_sender->stats()
                                                       : simaai::neat::MetadataSenderStats{};
    std::cout << "[stream " << source.index << "] processed=" << source.processed
              << " metadata_send_ok=" << source.metadata_send_ok
              << " metadata_send_fail=" << source.metadata_send_fail
              << " metadata_would_block=" << metadata_stats.would_block
              << " metadata_no_buffer_space=" << metadata_stats.no_buffer_space
              << " metadata_send_max_ns=" << metadata_stats.max_send_duration_ns << "\n";
  }
}

void run_app(const AppConfig& cfg) {
  g_stop_requested = 0;
  auto previous_sigint = std::signal(SIGINT, request_stop);

  const auto labels = load_labels(cfg.labels_path);
  AggregateProfile aggregate_profile;
  aggregate_profile.enabled = cfg.profile;
  aggregate_profile.stream_count = static_cast<int>(cfg.rtsp_urls.size());

  // The realtime fan-in retains the latest pending frame per stream. This small
  // global depth only decouples shared model stages and does not create another
  // decoded-frame queue per camera. GraphOptions must be set on the final
  // composition owner rather than a nested model fragment.
  AppRuntime app(cfg.internal_queue_depth);
  app.sources.reserve(cfg.rtsp_urls.size());
  std::cout << "[detector] max_inflight_per_stream=" << cfg.max_inflight_per_stream
            << " max_inflight_total=" << cfg.max_inflight_total << "\n";
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

    // Ordinary build() lowers the explicit encoded VideoSender and decoder
    // fan-out into the fused realtime source pipeline.
    app.run = app.graph.build(
        realtime_options(cfg.queue_depth, simaai::neat::OverflowPolicy::KeepLatest));

    pull_detections(app, cfg, aggregate_profile);
  } catch (...) {
    app.run.close();
    flush_and_print_runtime_stats(aggregate_profile, app.sources);
    std::signal(SIGINT, previous_sigint);
    throw;
  }

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
                << ", max_inflight_total=" << cfg.max_inflight_total
                << ", input=" << cfg.input_width << "x" << cfg.input_height << "@" << cfg.input_fps
                << ", insight_visible_streams=" << visible_streams
                << ", video_ports=" << cfg.video_port_base << "-" << video_port_last
                << ", metadata_ports=" << cfg.metadata_port_base << "-" << metadata_port_last
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
