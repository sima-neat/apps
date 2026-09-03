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

#include "app_config.h"

#include "support/runtime/config_utils.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace pcie_high_density {
namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
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

std::vector<std::string> parse_streams(const std::filesystem::path& config_path) {
  std::ifstream input(config_path);
  if (!input.is_open()) {
    throw std::runtime_error("failed to open config file: " + config_path.string());
  }

  std::vector<std::string> streams;
  bool in_streams = false;
  int streams_indent = -1;
  std::string raw_line;
  while (std::getline(input, raw_line)) {
    const std::string without_comment = strip_inline_comment(raw_line);
    if (sima_examples::trim_copy(without_comment).empty()) {
      continue;
    }

    const int indent = leading_indent(without_comment);
    const std::string line = sima_examples::trim_copy(without_comment);
    if (in_streams && indent <= streams_indent && line.rfind("- ", 0) != 0) {
      in_streams = false;
    }
    if (!in_streams && line == "streams:") {
      in_streams = true;
      streams_indent = indent;
      continue;
    }
    if (!in_streams || line.rfind("- ", 0) != 0) {
      continue;
    }

    std::string url = unquote(line.substr(2));
    require(!url.empty(), "streams entries must be non-empty strings");
    streams.push_back(std::move(url));
  }
  return streams;
}

std::filesystem::path resolve_path(const std::filesystem::path& config_path,
                                   const std::string& value) {
  std::filesystem::path path(value);
  if (path.empty() || path.is_absolute()) {
    return path;
  }
  const std::filesystem::path config_relative =
      (config_path.parent_path() / path).lexically_normal();
  if (std::filesystem::exists(config_relative)) {
    return config_relative;
  }
  return path.lexically_normal();
}

void validate(const AppConfig& config) {
  require(config.stream_count == 16 || config.stream_count == 24 || config.stream_count == 48,
          "streams must contain exactly 16, 24, or 48 RTSP URLs");
  require(config.decode_type == "yolo26" || config.decode_type == "yolov8",
          "model.decode_type must be yolo26 or yolov8");
  require(config.queue >= 0 && config.queue <= 5, "card.queue must be in [0, 5]");
  require(config.card_id >= 0, "card.card_id must be >= 0");
  require(config.input_width > 0 && config.input_height > 0 && config.input_fps > 0,
          "input width, height, and fps must be > 0");
  require(config.startup_stagger_ms >= 0, "input.startup_stagger_ms must be >= 0");
  require(config.decoder_buffers > 0 && config.decoder_buffers <= 64,
          "input.decoder_buffers must be in [1, 64]");
  require(config.decoder_input_buffers > 0, "input.decoder_input_buffers must be > 0");
  require(config.max_inflight_per_stream > 0, "pcie.max_inflight_per_stream must be > 0");
  require(config.max_inflight_total > 0, "pcie.max_inflight_total must be > 0");
  require(config.max_inflight_per_stream <= config.max_inflight_total,
          "pcie.max_inflight_per_stream must be <= pcie.max_inflight_total");
  require(config.pcie_queue_size >= config.max_inflight_total,
          "pcie.queue_size must be >= pcie.max_inflight_total");
  require(config.pcie_buffer_size > 0, "pcie.buffer_size must be > 0");
  require(config.pcie_pool_size >= 2 && config.pcie_pool_size <= 120,
          "pcie.pool_size must be in [2, 120]");
  require(config.decoder_input_buffers >= config.pcie_pool_size,
          "input.decoder_input_buffers must be >= pcie.pool_size");
  require(config.result_timeout_ms > 0, "pcie.result_timeout_ms must be > 0");
  require(config.correlation_cache_size > 0, "pcie.correlation_cache_size must be > 0");
  require(config.min_score >= 0.0 && config.min_score <= 1.0,
          "inference.min_score must be between 0 and 1");
  require(config.nms_iou >= 0.0 && config.nms_iou <= 1.0,
          "inference.nms_iou must be between 0 and 1");
  require(config.max_detections > 0, "inference.max_detections must be > 0");
  require(config.inference_queue_depth > 0 && config.inference_queue_depth <= 32,
          "inference.queue_depth must be in [1, 32]");
  require(config.inference_internal_queue_depth >= 0 && config.inference_internal_queue_depth <= 32,
          "inference.internal_queue_depth must be in [0, 32]");
  require(config.inference_max_inflight_per_stream > 0,
          "inference.max_inflight_per_stream must be > 0");
  require(config.inference_max_inflight_total > 0, "inference.max_inflight_total must be > 0");
  require(config.inference_max_inflight_per_stream <= config.inference_max_inflight_total,
          "inference.max_inflight_per_stream must be <= inference.max_inflight_total");
  require(config.warmup_frames >= 0, "runtime.warmup_frames must be >= 0");
  require(config.stall_timeout_ms >= 0, "runtime.stall_timeout_ms must be >= 0");
  require(config.stall_timeout_ms == 0 || config.stall_timeout_ms >= config.result_timeout_ms,
          "runtime.stall_timeout_ms must be 0 or >= pcie.result_timeout_ms");
  require(config.teardown_timeout_ms > 0, "runtime.teardown_timeout_ms must be > 0");
  require(!config.insight_host.empty(), "output.insight.host must be set");
  require(config.video_port_base > 0 && config.video_port_base + config.stream_count - 1 <= 65535,
          "output.insight video port range is invalid");
  require(config.metadata_port_base > 0 &&
              config.metadata_port_base + config.stream_count - 1 <= 65535,
          "output.insight metadata port range is invalid");
  require(config.video_port_base + config.stream_count <= config.metadata_port_base ||
              config.metadata_port_base + config.stream_count <= config.video_port_base,
          "output.insight video and metadata port ranges overlap");
}

} // namespace

AppConfig load_config(const std::filesystem::path& path) {
  require(std::filesystem::exists(path), "config file not found: " + path.string());
  const auto raw = sima_examples::ScalarConfig::load(path);

  AppConfig config;
  config.profile_name = raw.string_or("profile.name", "pcie-high-density");
  config.model_path = resolve_path(path, raw.string_or("model.path", ""));
  config.labels_path = resolve_path(path, raw.string_or("model.labels", "coco_label.txt"));
  config.decode_type = raw.string_or("model.decode_type", "yolo26");
  config.card_id = raw.int_or("card.card_id", 0);
  config.queue = raw.int_or("card.queue", 0);
  config.rtsp_urls = parse_streams(path);
  config.stream_count = static_cast<int>(config.rtsp_urls.size());
  require(raw.string_or("input.codec", "h264") == "h264",
          "the first PCIe implementation supports input.codec=h264 only");
  config.rtsp_tcp = raw.bool_or("input.tcp", true);
  config.rtsp_drop_on_latency = raw.bool_or("input.drop_on_latency", false);
  config.latency_ms = raw.int_or("input.latency_ms", 100);
  config.startup_stagger_ms = raw.int_or("input.startup_stagger_ms", 50);
  config.input_width = raw.int_or("input.width", 1280);
  config.input_height = raw.int_or("input.height", 720);
  config.input_fps = raw.int_or("input.fps", 25);
  config.decoder_buffers = raw.int_or("input.decoder_buffers", 8);
  config.decoder_input_buffers = raw.int_or("input.decoder_input_buffers", 8);
  config.decoder_tuning = raw.string_or("input.decoder_tuning", "auto");
  config.pcie_queue_size = raw.int_or("pcie.queue_size", 256);
  config.pcie_buffer_size = raw.int_or("pcie.buffer_size", 4 * 1024 * 1024);
  config.pcie_pool_size = raw.int_or("pcie.pool_size", 4);
  config.max_inflight_per_stream = raw.int_or("pcie.max_inflight_per_stream", 1);
  config.max_inflight_total = raw.int_or("pcie.max_inflight_total", 8);
  config.result_timeout_ms = raw.int_or("pcie.result_timeout_ms", 30000);
  config.correlation_cache_size = raw.int_or("pcie.correlation_cache_size", 256);
  config.min_score = raw.double_or("inference.min_score", 0.30);
  config.nms_iou = raw.double_or("inference.nms_iou", 0.60);
  config.max_detections = raw.int_or("inference.max_detections", 50);
  config.inference_queue_depth = raw.int_or("inference.queue_depth", 16);
  config.inference_internal_queue_depth = raw.int_or("inference.internal_queue_depth", 1);
  config.inference_max_inflight_per_stream = raw.int_or("inference.max_inflight_per_stream", 1);
  config.inference_max_inflight_total = raw.int_or("inference.max_inflight_total", 8);
  config.profile = raw.bool_or("runtime.profile", false);
  config.warmup_frames = raw.int_or("runtime.warmup_frames", 30);
  // 0 selects the default of twice pcie.result_timeout_ms once the config is validated.
  config.stall_timeout_ms = raw.int_or("runtime.stall_timeout_ms", 0);
  config.teardown_timeout_ms = raw.int_or("runtime.teardown_timeout_ms", 10000);
  config.insight_host = raw.string_or("output.insight.host", "");
  config.video_port_base = raw.int_or("output.insight.video_port_base", 9000);
  config.metadata_port_base = raw.int_or("output.insight.metadata_port_base", 9100);
  config.video_enabled = raw.bool_or("output.video_enabled", true);

  validate(config);
  if (config.stall_timeout_ms == 0) {
    config.stall_timeout_ms = 2 * config.result_timeout_ms;
  }
  return config;
}

std::vector<std::string> load_labels(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input.is_open()) {
    throw std::runtime_error("failed to open labels file: " + path.string());
  }
  std::vector<std::string> labels;
  std::string line;
  while (std::getline(input, line)) {
    if (!line.empty()) {
      labels.push_back(line);
    }
  }
  require(!labels.empty(), "labels file is empty: " + path.string());
  return labels;
}

std::string config_summary(const AppConfig& config) {
  std::ostringstream out;
  out << "profile=" << config.profile_name << " streams=" << config.stream_count
      << " input=" << config.input_width << 'x' << config.input_height << '@' << config.input_fps
      << " rtsp_startup_stagger_ms=" << config.startup_stagger_ms << " card=" << config.card_id
      << " queue=" << config.queue << " pcie_pool_size=" << config.pcie_pool_size
      << " pcie_max_inflight_per_stream=" << config.max_inflight_per_stream
      << " correlation_cache_size=" << config.correlation_cache_size
      << " pcie_max_inflight_total=" << config.max_inflight_total
      << " decoder_input_buffers=" << config.decoder_input_buffers
      << " inference_queue_depth=" << config.inference_queue_depth
      << " inference_internal_queue_depth=" << config.inference_internal_queue_depth
      << " inference_max_inflight_per_stream=" << config.inference_max_inflight_per_stream
      << " inference_max_inflight_total=" << config.inference_max_inflight_total
      << " stall_timeout_ms=" << config.stall_timeout_ms
      << " teardown_timeout_ms=" << config.teardown_timeout_ms
      << " insight=" << config.insight_host;
  return out.str();
}

} // namespace pcie_high_density
