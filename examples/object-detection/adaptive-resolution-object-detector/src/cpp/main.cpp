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

// Multi-stream RTSP YOLO26 detector for Insight, with three topologies.
//
// Every stream is detected at its own resolution and frame rate, so 720p,
// 1080p, and 4K sources can run side by side. `mode` in the config picks the
// topology:
//
//   live   One graph per stream. Edit streams.sources while the application
//          runs and the change is applied on the next poll; the other streams
//          keep running. Each stream re-encodes its video for Insight.
//
//   scale  One graph for every stream, fanning into a single shared detector,
//          with each source's H.264 passed through to Insight untouched.
//          Highest stream count; changing the stream set needs a restart.
//
//   group  The scale topology run as several independent processes, each
//          owning `group.size` streams. A failure stops only its own group.

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

#include <fcntl.h>
#include <poll.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace {

volatile std::sig_atomic_t g_stop_requested = 0;

void request_stop(int) {
  g_stop_requested = 1;
}

bool stop_requested() {
  return g_stop_requested != 0;
}

// ── configuration ─────────────────────────────────────────────────────────────

struct StreamSource {
  std::string id;
  std::string url;
};

struct AppConfig {
  std::string mode;
  std::string model_path;
  fs::path labels_path;
  std::vector<StreamSource> sources;
  fs::path config_path;
  int max_streams = 16;
  int group_size = 4;
  std::string source_type = "rtsp";
  std::string source_codec = "h264";
  bool tcp = true;
  int latency_ms = 100;
  int source_fps = 0;
  int frames = 0;
  int fps = 0;
  double min_score = 0.30;
  double nms_iou = 0.60;
  int max_detections = 50;
  bool profile = false;
  int warmup_frames = 30;
  double config_watch_seconds = 1.0;
  std::string insight_host;
  int video_port = 9000;
  int metadata_port = 9100;
  bool video_enabled = true;
  fs::path save_dir;
  int save_every = 0;
  // Insight channel of this process's first stream. Not a config key: group
  // mode sets it per worker so every group publishes on its own channel range.
  int channel_base = 0;
};

struct CliOptions {
  fs::path config_path;
  bool validate_config_only = false;
};

std::string lower_copy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

std::string parse_mode(const std::string& value) {
  const std::string lowered = lower_copy(sima_examples::trim_copy(value));
  if (lowered == "live" || lowered == "scale" || lowered == "group") {
    return lowered;
  }
  if (lowered.empty()) {
    throw std::runtime_error("mode must be set to one of: live, scale, group");
  }
  throw std::runtime_error("mode must be one of: live, scale, group (got '" + value + "')");
}

std::string parse_source_type(const std::string& value) {
  if (lower_copy(value) == "rtsp") {
    return "rtsp";
  }
  throw std::runtime_error("source.type must be rtsp");
}

std::string parse_source_codec(const std::string& value) {
  // The scale and group topologies forward each source's H.264 to Insight
  // untouched, so H.264 is the only codec every mode can carry.
  const std::string lowered = lower_copy(value);
  if (lowered == "h264" || lowered == "avc" || lowered == "h.264") {
    return "h264";
  }
  throw std::runtime_error("source.codec must be h264");
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

bool starts_with(const std::string& value, const std::string& prefix) {
  return value.rfind(prefix, 0) == 0;
}

// streams.sources: [{id, url}, ...]. The shared ScalarConfig reader skips list
// blocks, so the list is read here. Entries are validated exactly as the
// Python implementation does, naming the offending index.
std::vector<StreamSource> parse_stream_sources(const fs::path& config_path) {
  std::ifstream input(config_path);
  if (!input.is_open()) {
    throw std::runtime_error("failed to open config file: " + config_path.string());
  }

  struct Item {
    bool mapping = false;
    std::string id;
    std::string url;
  };
  std::vector<Item> items;
  int streams_indent = -1;
  int sources_indent = -1;
  bool in_streams = false;
  bool in_sources = false;

  std::string raw_line;
  while (std::getline(input, raw_line)) {
    const std::string without_comment = strip_inline_comment(raw_line);
    const std::string line = sima_examples::trim_copy(without_comment);
    if (line.empty()) {
      continue;
    }
    const int indent = leading_indent(without_comment);
    const bool list_item = starts_with(line, "- ") || line == "-";

    if (in_sources && indent <= sources_indent && !(list_item && indent == sources_indent)) {
      in_sources = false;
    }
    if (in_streams && indent <= streams_indent) {
      if (list_item) {
        throw std::runtime_error("streams must be a mapping");
      }
      in_streams = false;
    }

    if (!in_streams && indent == 0 && starts_with(line, "streams:")) {
      const std::string rest = sima_examples::trim_copy(line.substr(8));
      if (!rest.empty()) {
        throw std::runtime_error("streams must be a mapping");
      }
      in_streams = true;
      streams_indent = indent;
      continue;
    }
    if (!in_streams) {
      continue;
    }

    if (!in_sources) {
      if (list_item) {
        throw std::runtime_error("streams must be a mapping");
      }
      if (starts_with(line, "sources:")) {
        const std::string rest = sima_examples::trim_copy(line.substr(8));
        if (rest == "[]") {
          continue;
        }
        if (!rest.empty()) {
          throw std::runtime_error("streams.sources must be a list");
        }
        in_sources = true;
        sources_indent = indent;
      }
      continue;
    }

    std::string field = line;
    if (list_item) {
      items.push_back({});
      field = line.size() > 1 ? sima_examples::trim_copy(line.substr(1)) : "";
      if (field.empty()) {
        continue;
      }
    } else if (items.empty()) {
      throw std::runtime_error("streams.sources must be a list");
    }
    Item& item = items.back();
    const std::size_t colon = field.find(':');
    // A scalar entry, e.g. `- rtsp://host/stream`. Its own colon follows the
    // scheme, so only a `key:` with a plain key name counts as a mapping.
    const std::string key = colon == std::string::npos ? "" : field.substr(0, colon);
    const bool is_key = !key.empty() && std::all_of(key.begin(), key.end(), [](unsigned char c) {
      return std::isalnum(c) != 0 || c == '_';
    });
    if (!is_key || (colon + 1 < field.size() && field[colon + 1] == '/')) {
      if (list_item) {
        item.mapping = false;
        continue;
      }
      throw std::runtime_error("invalid streams.sources line: " + field);
    }
    item.mapping = true;
    const std::string value = unquote(field.substr(colon + 1));
    if (key == "id") {
      item.id = value;
    } else if (key == "url") {
      item.url = value;
    }
  }

  std::vector<StreamSource> sources;
  for (std::size_t index = 0; index < items.size(); ++index) {
    const std::string at = "streams.sources[" + std::to_string(index) + "]";
    const Item& item = items[index];
    if (!item.mapping) {
      throw std::runtime_error(at + " must be a mapping with id and url");
    }
    // The id is what live mode diffs a config edit by, so it cannot be derived
    // from the position: removing the first entry would otherwise rename every
    // stream after it and restart them all.
    if (sima_examples::trim_copy(item.id).empty()) {
      throw std::runtime_error(at + ".id must be set");
    }
    if (sima_examples::trim_copy(item.url).empty()) {
      throw std::runtime_error(at + ".url must be set");
    }
    sources.push_back({sima_examples::trim_copy(item.id), sima_examples::trim_copy(item.url)});
  }
  return sources;
}

void validate_sources(const std::vector<StreamSource>& sources, int max_streams) {
  sima_examples::require(max_streams >= 1, "streams.max_streams must be >= 1");
  sima_examples::require(static_cast<int>(sources.size()) <= max_streams,
                         "streams.sources has " + std::to_string(sources.size()) +
                             " entries, more than streams.max_streams (" +
                             std::to_string(max_streams) + ")");
  std::set<std::string> ids;
  for (const auto& source : sources) {
    ids.insert(source.id);
  }
  sima_examples::require(ids.size() == sources.size(), "streams.sources ids must be unique");
}

void validate_config(const AppConfig& cfg) {
  sima_examples::require(!cfg.model_path.empty(), "model.path must be set");
  sima_examples::require(!cfg.labels_path.empty(), "model.labels must be set");
  validate_sources(cfg.sources, cfg.max_streams);
  // Live mode may start empty and wait for a config edit to add streams; the
  // other modes build their graph from the startup list and need one.
  sima_examples::require(cfg.mode == "live" || !cfg.sources.empty(),
                         "streams.sources must not be empty in " + cfg.mode + " mode");
  sima_examples::require(cfg.group_size >= 1, "group.size must be >= 1");
  sima_examples::require(!cfg.insight_host.empty(), "output.insight.host must be set");
  sima_examples::require(cfg.latency_ms >= 0, "source.latency_ms must be >= 0");
  sima_examples::require(cfg.source_fps >= 0, "source.fps must be >= 0");
  sima_examples::require(cfg.frames >= 0, "inference.frames must be >= 0");
  sima_examples::require(cfg.fps >= 0, "inference.fps must be >= 0");
  sima_examples::require(cfg.min_score >= 0.0 && cfg.min_score <= 1.0,
                         "inference.min_score must be between 0 and 1");
  sima_examples::require(cfg.nms_iou >= 0.0 && cfg.nms_iou <= 1.0,
                         "inference.nms_iou must be between 0 and 1");
  sima_examples::require(cfg.max_detections > 0, "inference.max_detections must be > 0");
  sima_examples::require(cfg.warmup_frames >= 0, "runtime.warmup_frames must be >= 0");
  sima_examples::require(cfg.config_watch_seconds > 0.0,
                         "runtime.config_watch_seconds must be > 0");
  sima_examples::require(cfg.video_port > 0, "output.insight.video_port must be > 0");
  sima_examples::require(cfg.metadata_port > 0, "output.insight.metadata_port must be > 0");
  sima_examples::require(cfg.save_every >= 0, "output.save_every must be >= 0");
}

AppConfig load_app_config(const fs::path& config_path) {
  const auto raw = sima_examples::ScalarConfig::load(config_path);
  const auto default_labels =
      fs::path(SIMANEAT_APPS_EXAMPLE_SOURCE_DIR).parent_path() / "common" / "coco_label.txt";

  AppConfig cfg;
  cfg.config_path = config_path;
  cfg.mode = parse_mode(raw.string_or("mode", ""));
  cfg.model_path = raw.string_or("model.path", "");
  cfg.labels_path = raw.string_or("model.labels", default_labels.string());
  cfg.sources = parse_stream_sources(config_path);
  cfg.max_streams = raw.int_or("streams.max_streams", 16);
  cfg.group_size = raw.int_or("group.size", 4);
  cfg.source_type = parse_source_type(raw.string_or("source.type", "rtsp"));
  cfg.source_codec = parse_source_codec(raw.string_or("source.codec", "h264"));
  cfg.tcp = raw.bool_or("source.tcp", true);
  cfg.latency_ms = raw.int_or("source.latency_ms", 100);
  cfg.source_fps = raw.int_or("source.fps", 0);
  cfg.frames = raw.int_or("inference.frames", 0);
  cfg.fps = raw.int_or("inference.fps", 0);
  cfg.min_score = raw.double_or("inference.min_score", 0.30);
  cfg.nms_iou = raw.double_or("inference.nms_iou", 0.60);
  cfg.max_detections = raw.int_or("inference.max_detections", 50);
  cfg.profile = raw.bool_or("runtime.profile", false);
  cfg.warmup_frames = raw.int_or("runtime.warmup_frames", 30);
  cfg.config_watch_seconds = raw.double_or("runtime.config_watch_seconds", 1.0);
  cfg.insight_host = raw.string_or("output.insight.host", "");
  cfg.video_port = raw.int_or("output.insight.video_port", 9000);
  cfg.metadata_port = raw.int_or("output.insight.metadata_port", 9100);
  cfg.video_enabled = raw.bool_or("output.video_enabled", true);
  cfg.save_dir = raw.string_or("output.save_dir", "");
  cfg.save_every = raw.int_or("output.save_every", 0);
  validate_config(cfg);
  return cfg;
}

// Parse a live config edit with the same checks as startup. `capacity` is the
// channel count fixed at startup from the ORIGINAL max_streams; raising
// max_streams in a live edit cannot add channels, so the reload is checked
// against both. Throwing makes run_live ignore the whole edit instead of
// applying the part that fits.
std::vector<StreamSource> reload_sources(const fs::path& config_path, int capacity) {
  const auto raw = sima_examples::ScalarConfig::load(config_path);
  auto sources = parse_stream_sources(config_path);
  validate_sources(sources, raw.int_or("streams.max_streams", 16));
  if (static_cast<int>(sources.size()) > capacity) {
    throw std::runtime_error("streams.sources has " + std::to_string(sources.size()) +
                             " entries, more than the " + std::to_string(capacity) +
                             " channels this run started with; restart to raise "
                             "streams.max_streams");
  }
  return sources;
}

// One scale config per group of `group_size` streams, in config order. Group g
// owns Insight channels g*size .. g*size+size-1, so its streams land on the
// same channels they would have in a single scale run.
std::vector<AppConfig> split_groups(const AppConfig& cfg) {
  std::vector<AppConfig> groups;
  for (std::size_t start = 0; start < cfg.sources.size();
       start += static_cast<std::size_t>(cfg.group_size)) {
    AppConfig group = cfg;
    const std::size_t end =
        std::min(cfg.sources.size(), start + static_cast<std::size_t>(cfg.group_size));
    group.sources.assign(cfg.sources.begin() + static_cast<std::ptrdiff_t>(start),
                         cfg.sources.begin() + static_cast<std::ptrdiff_t>(end));
    group.channel_base = static_cast<int>(start);
    groups.push_back(std::move(group));
  }
  return groups;
}

// Apply a config edit to the current groups without moving any stream. A stream
// stays in the group it was placed in, so removing one changes only its own
// group. A new stream fills the first group with a free slot, or starts a new
// group when every group is full. A group left empty keeps its index, so the
// groups after it keep their channels.
std::vector<AppConfig> regroup(const AppConfig& cfg, const std::vector<AppConfig>& groups,
                               const std::vector<StreamSource>& sources) {
  std::map<std::string, StreamSource> wanted;
  for (const auto& source : sources) {
    wanted[source.id] = source;
  }
  std::vector<std::vector<StreamSource>> members;
  std::set<std::string> placed;
  for (const auto& group : groups) {
    std::vector<StreamSource> kept;
    for (const auto& source : group.sources) {
      const auto it = wanted.find(source.id);
      if (it != wanted.end()) {
        kept.push_back(it->second);
        placed.insert(source.id);
      }
    }
    members.push_back(std::move(kept));
  }
  const auto size = static_cast<std::size_t>(cfg.group_size);
  for (const auto& source : sources) {
    if (placed.count(source.id) != 0) {
      continue;
    }
    std::size_t index = 0;
    while (index < members.size() && members[index].size() >= size) {
      ++index;
    }
    if (index == members.size()) {
      members.emplace_back();
    }
    members[index].push_back(source);
  }
  while (!members.empty() && members.back().empty()) {
    members.pop_back();
  }
  std::vector<AppConfig> result;
  for (std::size_t index = 0; index < members.size(); ++index) {
    AppConfig group = cfg;
    group.sources = std::move(members[index]);
    group.channel_base = static_cast<int>(index * size);
    result.push_back(std::move(group));
  }
  return result;
}

// ── shared detection helpers ──────────────────────────────────────────────────

void enable_profile_env(const AppConfig& cfg) {
  if (cfg.profile) {
    setenv("SIMA_GST_ELEMENT_TIMINGS", "1", 0);
    setenv("SIMA_GST_FLOW_DEBUG", "1", 0);
    setenv("SIMA_GST_BOUNDARY_PROBES", "1", 0);
  }
}

bool save_frames_enabled(const AppConfig& cfg) {
  return !cfg.save_dir.empty() && cfg.save_every > 0;
}

std::vector<std::string> load_labels(const fs::path& labels_path) {
  std::ifstream in(labels_path);
  if (!in.good()) {
    throw std::runtime_error("labels file does not exist: " + labels_path.string());
  }
  std::vector<std::string> labels;
  std::string line;
  while (std::getline(in, line)) {
    line = sima_examples::trim_copy(line);
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

// Labels come from the model's label file, so only the two characters that can
// break a JSON string literal need escaping.
std::string json_escape(const std::string& value) {
  std::string out;
  out.reserve(value.size());
  for (char c : value) {
    if (c == '"' || c == '\\') {
      out.push_back('\\');
    }
    out.push_back(c);
  }
  return out;
}

// Compact object list: only the fields Insight's overlay reads.
//
// Insight drops metadata datagrams larger than the ~1500-byte MTU instead of
// reassembling them. Leaving out a per-box id, rounding confidence, and using
// integer pixels takes a box from ~95 to ~59 bytes, so ~25 detections fit one
// datagram instead of ~15.
std::string metadata_objects_json(const std::vector<objdet::Box>& boxes,
                                  const std::vector<std::string>& labels, int frame_w,
                                  int frame_h) {
  std::string out = "[";
  for (std::size_t i = 0; i < boxes.size(); ++i) {
    const auto& box = boxes[i];
    const int x = std::max(0, static_cast<int>(box.x1));
    const int y = std::max(0, static_cast<int>(box.y1));
    int w = std::max(0, static_cast<int>(box.x2 - box.x1));
    int h = std::max(0, static_cast<int>(box.y2 - box.y1));
    if (x + w > frame_w) {
      w = frame_w - x;
    }
    if (y + h > frame_h) {
      h = frame_h - y;
    }
    const std::string label = (box.class_id >= 0 && box.class_id < static_cast<int>(labels.size()))
                                  ? labels[static_cast<std::size_t>(box.class_id)]
                                  : "unknown";
    char confidence[16];
    std::snprintf(confidence, sizeof(confidence), "%.2f", box.score);
    if (i > 0) {
      out.push_back(',');
    }
    out += "{\"label\":\"" + json_escape(label) + "\",\"confidence\":" + confidence +
           ",\"bbox\":[" + std::to_string(x) + "," + std::to_string(y) + "," +
           std::to_string(std::max(0, w)) + "," + std::to_string(std::max(0, h)) + "]}";
  }
  out += "]";
  return out;
}

struct ProbedStream {
  int width = 0;
  int height = 0;
  int fps = 0;
};

ProbedStream probe_stream(const AppConfig& cfg, const StreamSource& source) {
  sima_examples::RtspStreamInfo probe;
  sima_examples::RtspProbeOptions probe_options;
  probe_options.payload_type = 96;
  probe_options.latency_ms = cfg.latency_ms;
  probe_options.rtsp_tcp = cfg.tcp;
  probe_options.debug = cfg.profile;
  (void)sima_examples::probe_rtsp_stream_info(source.url, probe_options, probe);
  sima_examples::require(probe.width > 0 && probe.height > 0,
                         "failed to probe RTSP frame size for stream " + source.id);
  return ProbedStream{probe.width, probe.height, std::max(0, probe.fps)};
}

std::unique_ptr<simaai::neat::Model> make_model(const AppConfig& cfg, int input_max_w,
                                                int input_max_h) {
  simaai::neat::Model::Options model_opt;
  model_opt.preprocess.kind = simaai::neat::InputKind::Image;
  model_opt.preprocess.enable = simaai::neat::AutoFlag::On;
  model_opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::NV12;
  model_opt.preprocess.preset = simaai::neat::NormalizePreset::COCO_YOLO;
  // The preprocess envelope defaults to 1920x1080, so a larger frame (a 4K
  // source) fails the graph compile with "input width N exceeds configured
  // capacity". Size it to the frame this model actually receives.
  model_opt.preprocess.input_max_width = input_max_w;
  model_opt.preprocess.input_max_height = input_max_h;
  model_opt.decode_type = simaai::neat::BoxDecodeType::YoloV26;
  model_opt.score_threshold = cfg.min_score;
  model_opt.nms_iou_threshold = cfg.nms_iou;
  model_opt.top_k = cfg.max_detections;
  return std::make_unique<simaai::neat::Model>(cfg.model_path, model_opt);
}

struct ProfileWindow {
  bool enabled = false;
  std::string stream_label;
  int interval = 100;
  int frames = 0;
  int boxes = 0;
  double start_ms = 0.0;
  double detection_pull_ms = 0.0;
  double metadata_send_ms = 0.0;

  void add(double detection_pull, double metadata_send, int box_count) {
    if (!enabled) {
      return;
    }
    if (frames == 0) {
      start_ms = sima_examples::time_ms();
    }
    ++frames;
    boxes += box_count;
    detection_pull_ms += detection_pull;
    metadata_send_ms += metadata_send;
    if (frames >= interval) {
      flush();
    }
  }

  void flush() {
    if (!enabled || frames == 0) {
      return;
    }
    const double elapsed = sima_examples::time_ms() - start_ms;
    const double output_fps = elapsed > 0.0 ? static_cast<double>(frames) * 1000.0 / elapsed : 0.0;
    const auto avg = [this](double value) { return value / static_cast<double>(frames); };
    std::cout << "[profile stream=" << stream_label << "] frames=" << frames
              << " output_fps=" << output_fps << " avg_detection_pull_ms=" << avg(detection_pull_ms)
              << " avg_metadata_send_ms=" << avg(metadata_send_ms)
              << " avg_boxes=" << avg(static_cast<double>(boxes)) << "\n";
    frames = 0;
    boxes = 0;
    start_ms = 0.0;
    detection_pull_ms = 0.0;
    metadata_send_ms = 0.0;
  }
};

void save_debug_frame(const AppConfig& cfg, int channel, int processed, cv::Mat frame,
                      const std::vector<objdet::Box>& boxes) {
  if (frame.empty() || processed % cfg.save_every != 0) {
    return;
  }
  objdet::draw_boxes(frame, boxes, static_cast<float>(cfg.min_score), cv::Scalar(0, 255, 0), "");
  const auto out_path = cfg.save_dir / ("stream_" + std::to_string(channel) + "_frame_" +
                                        std::to_string(processed) + ".jpg");
  if (!cv::imwrite(out_path.string(), frame)) {
    std::cerr << "[warn] failed to write output frame: " << out_path.string() << "\n";
  }
}

// PTS -> the 32-bit 90 kHz RTP timestamp Insight matches metadata on. Seconds
// are split from the remainder before scaling: pts_ns * 90000 overflows int64
// at ~28 h of stream time, and a wrong key is worse than none.
int64_t rtp_timestamp_from_pts_ns(int64_t pts_ns) {
  if (pts_ns < 0) {
    return 0;
  }
  const int64_t seconds = pts_ns / 1000000000;
  const int64_t remainder = pts_ns % 1000000000;
  return (seconds * 90000 + (remainder * 90000) / 1000000000) & 0xFFFFFFFF;
}

int64_t wall_clock_ms() {
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  return std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
}

std::string describe_current_exception() {
  try {
    throw;
  } catch (const std::exception& e) {
    return e.what();
  } catch (...) {
    return "unknown error";
  }
}

// ── live mode: one graph per stream ───────────────────────────────────────────

// Serializes MLA model loads across stream threads. Concurrent pipeline builds
// (initial builds and reconnects) otherwise race on MLA/RPMsg setup.
std::mutex g_model_build_mutex;

struct LiveStream {
  int channel = 0;
  std::string id;
  std::string url;
  std::vector<std::string> labels;
  int width = 0;
  int height = 0;
  int output_fps = 0;
  std::unique_ptr<simaai::neat::MetadataSender> metadata_sender; // persists across rebuilds
  std::unique_ptr<simaai::neat::Model> model;                    // rebuilt on reconnect
  std::unique_ptr<simaai::neat::Graph> graph;                    // rebuilt on reconnect
  simaai::neat::Run run;                                         // rebuilt on reconnect
  std::string output_name = "detections";
  int video_port = 0;
  int processed = 0;
  ProfileWindow profile;
};

simaai::neat::nodes::groups::RtspDecodedInputOptions
make_live_source_options(const AppConfig& cfg, const LiveStream& stream) {
  simaai::neat::nodes::groups::RtspDecodedInputOptions opt;
  opt.url = stream.url;
  opt.latency_ms = cfg.latency_ms;
  opt.tcp = cfg.tcp;
  opt.payload_type = 96;
  opt.insert_queue = true;
  opt.out_format = "NV12";
  opt.decoder_name = "decoder";
  opt.decoder_raw_output = true;
  opt.auto_caps_from_stream = true;
  opt.fallback_h264_width = stream.width;
  opt.fallback_h264_height = stream.height;
  // Leave the decoder fps unspecified (-1) rather than pinning it: a fixed
  // dec-fps rejects streams whose real rate differs (high/variable-rate or
  // mis-probed sources) with a framerate mismatch. inference.fps still caps
  // how often detections are processed.
  opt.fallback_h264_fps = -1;
  // No output caps: the stream is detected and delivered at its native size,
  // which keeps the zero-copy DMA path.
  opt.output_caps.enable = false;
  return opt;
}

// RTSP -> decode -> {model, re-encoded video, optional debug frames}.
//
// Video and detections hang off the same decoded branch, so both carry the
// same restamped PTS and the metadata's rtp_timestamp matches the video
// Insight receives (measured: 100% of boxes rendered up to 6 streams).
// Assumes the stream's run, graph, and model have been cleared.
void build_live_pipeline_once(const AppConfig& cfg, LiveStream& stream) {
  const bool save_frames = save_frames_enabled(cfg);
  stream.model = make_model(cfg, stream.width, stream.height);
  stream.graph = std::make_unique<simaai::neat::Graph>();
  auto& graph = *stream.graph;

  const std::string model_name = "model";
  const std::string video_name = "video";
  const std::string debug_frame_name = "debug_frame";
  const std::string detections_name = "detections";
  const std::string debug_output_name = "debug_output";

  simaai::neat::GraphLinkOptions live_link;
  live_link.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;

  std::vector<std::string> outputs = {model_name};
  if (cfg.video_enabled) {
    outputs.push_back(video_name);
  }
  if (save_frames) {
    outputs.push_back(debug_frame_name);
  }
  auto source =
      simaai::neat::nodes::groups::RtspDecodedInput(make_live_source_options(cfg, stream));
  auto branch = simaai::neat::graphs::Branch("source", outputs);
  graph.connect(source, branch);

  stream.video_port = 0;
  if (cfg.video_enabled) {
    auto video_options = simaai::neat::nodes::groups::VideoSenderOptions::H264RtpUdpFromRaw(
        stream.width, stream.height, stream.output_fps);
    video_options.encoder.bitrate_kbps = 1000;
    video_options.host = cfg.insight_host;
    video_options.channel = stream.channel;
    video_options.video_port_base = cfg.video_port;
    stream.video_port = video_options.video_port();
    simaai::neat::Graph video_graph(video_name);
    video_graph.connect(simaai::neat::nodes::Input(video_name),
                        simaai::neat::nodes::groups::VideoSender(video_options));
    graph.connect(branch, video_graph, live_link);
  }

  simaai::neat::Graph model_graph(model_name);
  model_graph.connect(simaai::neat::nodes::Input(model_name), *stream.model);
  graph.connect(branch, model_graph, live_link);

  simaai::neat::Graph detections_graph(detections_name);
  detections_graph.add(
      simaai::neat::nodes::Output(detections_name, simaai::neat::OutputOptions::EveryFrame(4)));
  graph.connect(model_graph, detections_graph);

  if (save_frames) {
    simaai::neat::Graph frames(debug_frame_name);
    frames.add(
        simaai::neat::nodes::Output(debug_frame_name, simaai::neat::OutputOptions::EveryFrame(4)));
    auto debug_join =
        simaai::neat::graphs::Combine({debug_frame_name, detections_name}, debug_output_name,
                                      simaai::neat::CombinePolicy::ByFrame);
    graph.connect(branch, frames, live_link);
    graph.connect(frames, debug_join);
    graph.connect(detections_graph, debug_join);
  }

  simaai::neat::RunOptions run_options;
  run_options.preset = simaai::neat::RunPreset::Realtime;
  run_options.queue_depth = 3;
  run_options.overflow_policy = simaai::neat::OverflowPolicy::KeepLatest;
  run_options.output_memory = simaai::neat::OutputMemory::ZeroCopy;
  stream.run = graph.build(run_options);
  stream.output_name = save_frames ? debug_output_name : detections_name;
}

// (Re)build one stream's graph. The metadata sender and probe persist.
//
// Model loads are serialized across streams and retried: concurrent rebuilds
// otherwise race on MLA/RPMsg setup ("Unable to load model").
void build_live_pipeline(const AppConfig& cfg, LiveStream& stream) {
  constexpr int kBuildAttempts = 3;
  for (int attempt = 1;; ++attempt) {
    try {
      std::lock_guard<std::mutex> build_lock(g_model_build_mutex);
      // Tear down (run -> graph -> model) and rebuild under the same lock, so
      // two streams never tear down and load MLA pipelines at the same time.
      const bool rebuilding = static_cast<bool>(stream.model);
      if (rebuilding) {
        try {
          stream.run.close();
        } catch (...) { // teardown is best-effort
        }
      }
      stream.run = simaai::neat::Run{};
      stream.graph.reset();
      stream.model.reset();
      // The MLA pipeline tears down asynchronously; building the next one
      // before it is released can segfault when several streams rebuild at once.
      if (rebuilding) {
        std::this_thread::sleep_for(std::chrono::milliseconds(800));
      }
      build_live_pipeline_once(cfg, stream);
      return;
    } catch (const std::exception& e) {
      stream.run = simaai::neat::Run{};
      stream.graph.reset();
      stream.model.reset();
      if (attempt >= kBuildAttempts) {
        throw;
      }
      std::cerr << "[warn] stream " << stream.id << " pipeline build attempt " << attempt
                << " failed (" << e.what() << "); retrying\n";
      std::this_thread::sleep_for(std::chrono::milliseconds(750));
    }
  }
}

void init_live_stream(const AppConfig& cfg, int channel, const StreamSource& source,
                      const std::vector<std::string>& labels, LiveStream& stream) {
  const ProbedStream probe = probe_stream(cfg, source);
  // Not every source reports its rate (high or variable-rate streams); the live
  // topology only needs one for the video encoder, so fall back to 30.
  const int source_fps = cfg.source_fps > 0 ? cfg.source_fps : (probe.fps > 0 ? probe.fps : 30);
  stream.channel = channel;
  stream.id = source.id;
  stream.url = source.url;
  stream.labels = labels;
  stream.width = probe.width;
  stream.height = probe.height;
  stream.output_fps = cfg.fps > 0 ? cfg.fps : source_fps;
  stream.profile.enabled = cfg.profile;
  stream.profile.stream_label = source.id;

  simaai::neat::MetadataSenderOptions metadata_options;
  metadata_options.host = cfg.insight_host;
  metadata_options.channel = channel;
  metadata_options.metadata_port_base = cfg.metadata_port;
  std::string metadata_err;
  stream.metadata_sender =
      std::make_unique<simaai::neat::MetadataSender>(metadata_options, &metadata_err);
  sima_examples::require(stream.metadata_sender->ok(), metadata_err);

  build_live_pipeline(cfg, stream);
  std::cout << "[stream " << source.id << "] channel=" << channel << " rtsp=" << source.url
            << " size=" << stream.width << "x" << stream.height << "@" << stream.output_fps
            << " video=" << (cfg.video_enabled ? std::to_string(stream.video_port) : "disabled")
            << " metadata=" << stream.metadata_sender->metadata_port() << "\n"
            << std::flush;
}

// Publish detections keyed to the frame's RTP timestamp.
//
// Insight matches metadata to a held video frame by an exact 32-bit
// `rtp_timestamp`. send_metadata() cannot carry that field, so the full
// envelope is built here and sent with send_raw_json().
void send_live_metadata(LiveStream& stream, const simaai::neat::Sample& sample,
                        const std::vector<objdet::Box>& boxes, int stream_count) {
  const int64_t pts_ns = sample.pts_ns;
  std::string payload = "{\"type\":\"object-detection\",\"data\":{\"objects\":";
  payload += metadata_objects_json(boxes, stream.labels, stream.width, stream.height);
  payload += ",\"stream_count\":" + std::to_string(stream_count);
  payload += ",\"stream_id\":\"" + json_escape(stream.id) + "\"}";
  payload += ",\"timestamp\":" + std::to_string(pts_ns >= 0 ? pts_ns / 1000000 : wall_clock_ms());
  payload += ",\"frame_id\":\"" +
             (sample.frame_id >= 0 ? std::to_string(sample.frame_id) : std::string{}) + "\"";
  payload += ",\"stream_id\":\"" + json_escape(stream.id) + "\"";
  payload += ",\"stream_index\":" + std::to_string(stream.channel);
  payload += ",\"pts_ns\":" + std::to_string(pts_ns);
  payload += ",\"dts_ns\":" + std::to_string(sample.dts_ns);
  payload += ",\"duration_ns\":" + std::to_string(sample.duration_ns);
  payload += ",\"input_seq\":" + std::to_string(sample.input_seq);
  if (pts_ns >= 0) {
    payload += ",\"rtp_timestamp\":" + std::to_string(rtp_timestamp_from_pts_ns(pts_ns));
  }
  payload += "}";

  // A dropped datagram must never stall detection.
  std::string err;
  (void)stream.metadata_sender->send_raw_json(payload, &err);
}

struct ManagedStream {
  std::string id;
  std::string url;
  int channel = 0;
  std::atomic_bool stop{false};
  std::atomic_bool finished{false};
  std::thread thread;
};

// Owns the live set of streams, channel allocation, and shared stop state.
//
// A stream failure is isolated to that stream: its worker retires its own slot
// and the rest keep running. stop_ means the whole application is stopping,
// never that one stream broke.
class StreamManager {
public:
  StreamManager(const AppConfig& cfg, std::vector<std::string> labels)
      : cfg_(cfg), labels_(std::move(labels)) {
    for (int channel = 0; channel < cfg.max_streams; ++channel) {
      free_channels_.push_back(channel);
    }
  }

  int active_count() {
    std::lock_guard<std::mutex> lock(mtx_);
    return static_cast<int>(streams_.size());
  }

  bool add(const StreamSource& source) {
    ManagedStream* raw = nullptr;
    {
      std::lock_guard<std::mutex> lock(mtx_);
      if (streams_.count(source.id) != 0) {
        return false;
      }
      if (free_channels_.empty()) {
        std::cerr << "[warn] cannot add stream " << source.id << ": no free channel\n";
        return false;
      }
      auto managed = std::make_unique<ManagedStream>();
      managed->id = source.id;
      managed->url = source.url;
      managed->channel = free_channels_.front();
      free_channels_.erase(free_channels_.begin());
      raw = managed.get();
      streams_[source.id] = std::move(managed);
      failed_.erase(source.id);
    }
    raw->thread = std::thread(&StreamManager::consume, this, raw);
    return true;
  }

  void remove(const std::string& id) {
    std::unique_ptr<ManagedStream> managed;
    {
      std::lock_guard<std::mutex> lock(mtx_);
      const auto it = streams_.find(id);
      if (it == streams_.end()) {
        return;
      }
      managed = std::move(it->second);
      streams_.erase(it);
    }
    managed->stop.store(true);
    // Joined before the channel is released, so the next add cannot bind the
    // same Insight ports while this worker is still running.
    if (managed->thread.joinable()) {
      managed->thread.join();
    }
    {
      std::lock_guard<std::mutex> lock(mtx_);
      free_channels_.push_back(managed->channel);
      std::sort(free_channels_.begin(), free_channels_.end());
    }
    std::cout << "[stream " << id << "] removed (channel " << managed->channel << " released)\n";
  }

  void apply_sources(const std::vector<StreamSource>& sources) {
    std::map<std::string, std::string> current;
    {
      std::lock_guard<std::mutex> lock(mtx_);
      for (const auto& [id, managed] : streams_) {
        current[id] = managed->url;
      }
    }
    std::map<std::string, StreamSource> wanted;
    for (const auto& source : sources) {
      wanted[source.id] = source;
    }
    // A changed URL under the same id is a replacement, not a no-op.
    std::set<std::string> removed;
    for (const auto& [id, url] : current) {
      const auto it = wanted.find(id);
      if (it == wanted.end() || it->second.url != url) {
        remove(id);
        removed.insert(id);
      }
    }
    {
      // A source dropped from the config stops counting as failed.
      std::lock_guard<std::mutex> lock(mtx_);
      for (auto it = failed_.begin(); it != failed_.end();) {
        it = wanted.count(it->first) != 0 ? std::next(it) : failed_.erase(it);
      }
    }
    for (const auto& source : sources) {
      if (current.count(source.id) == 0 || removed.count(source.id) != 0) {
        add(source);
      }
    }
  }

  // True when every stream is gone and each one left because it failed.
  bool all_failed() {
    std::lock_guard<std::mutex> lock(mtx_);
    return !failed_.empty() && streams_.empty();
  }

  bool all_done() {
    std::lock_guard<std::mutex> lock(mtx_);
    if (streams_.empty()) {
      return false;
    }
    return std::all_of(streams_.begin(), streams_.end(),
                       [](const auto& kv) { return kv.second->finished.load(); });
  }

  std::map<std::string, std::string> failed() {
    std::lock_guard<std::mutex> lock(mtx_);
    return failed_;
  }

  bool stopping() const {
    return stop_.load() || stop_requested();
  }

  void shutdown() {
    stop_.store(true);
    std::vector<std::unique_ptr<ManagedStream>> drained;
    {
      std::lock_guard<std::mutex> lock(mtx_);
      for (auto& [id, managed] : streams_) {
        drained.push_back(std::move(managed));
      }
      streams_.clear();
      for (auto& managed : retired_) {
        drained.push_back(std::move(managed));
      }
      retired_.clear();
    }
    for (auto& managed : drained) {
      managed->stop.store(true);
    }
    for (auto& managed : drained) {
      if (managed->thread.joinable()) {
        managed->thread.join();
      }
    }
  }

  void rethrow_first_error() {
    std::lock_guard<std::mutex> lock(error_mtx_);
    if (first_error_) {
      std::rethrow_exception(first_error_);
    }
  }

private:
  bool stopping(const ManagedStream* managed) const {
    return managed->stop.load() || stopping();
  }

  // Release a failed stream's slot from inside its own worker. Cannot go
  // through remove(), which joins the thread we are running on. The entry moves
  // to retired_, where shutdown() joins it. The identity check keeps a late
  // failure from evicting a same-id stream that has since been re-added.
  void retire(ManagedStream* managed, const std::string& reason) {
    {
      std::lock_guard<std::mutex> lock(mtx_);
      const auto it = streams_.find(managed->id);
      if (it != streams_.end() && it->second.get() == managed) {
        free_channels_.push_back(managed->channel);
        std::sort(free_channels_.begin(), free_channels_.end());
        retired_.push_back(std::move(it->second));
        streams_.erase(it);
      }
      failed_[managed->id] = reason;
    }
    std::cerr << "[warn] stream " << managed->id << " failed (" << reason << "); channel "
              << managed->channel << " released, other streams keep running\n";
  }

  // Rebuild a stream after its source closed (RTSP drop, EOS, timeout).
  // Returns true once reconnected, false when the stream should stop.
  bool reconnect(ManagedStream* managed, LiveStream& stream) {
    constexpr int kMaxReconnects = 30;
    for (int attempt = 1; attempt <= kMaxReconnects; ++attempt) {
      if (stopping(managed)) {
        return false;
      }
      const int backoff_ms = std::min(500 * attempt, 5000);
      std::cerr << "[warn] stream " << managed->id << " source closed; reconnecting (attempt "
                << attempt << "/" << kMaxReconnects << " in " << backoff_ms << "ms)\n";
      std::this_thread::sleep_for(std::chrono::milliseconds(backoff_ms));
      if (stopping(managed)) {
        return false;
      }
      try {
        build_live_pipeline(cfg_, stream);
        std::cout << "[stream " << managed->id << "] reconnected\n";
        return true;
      } catch (const std::exception& e) {
        std::cerr << "[warn] stream " << managed->id << " reconnect attempt " << attempt
                  << " failed (" << e.what() << ")\n";
      }
    }
    std::cerr << "[warn] stream " << managed->id
              << " reconnect budget exhausted; stopping stream\n";
    return false;
  }

  void consume(ManagedStream* managed) {
    LiveStream stream;
    bool built = false;
    // Set when this stream ends because it broke, rather than because it was
    // asked to stop or finished its frame budget.
    std::string failure;
    // Desynchronize identical streams so a fleet-wide recovery rolls through
    // instead of every stream rebuilding at once.
    const int stagger_ms = managed->channel * 350;
    int transient_recoveries = 0;
    constexpr int kMaxTransientRecoveries = 10;
    int consecutive_transient = 0;
    constexpr int kTransientRetries = 8;
    // inference.fps caps how often this stream processes and publishes a
    // detection. The runtime already keeps only the newest frame.
    const double min_interval_ms = cfg_.fps > 0 ? 1000.0 / static_cast<double>(cfg_.fps) : 0.0;
    double last_process_ms = -1e12;
    try {
      init_live_stream(cfg_, managed->channel, {managed->id, managed->url}, labels_, stream);
      built = true;
      while (!stopping(managed) && (cfg_.frames <= 0 || stream.processed < cfg_.frames)) {
        if (min_interval_ms > 0.0) {
          const double now = sima_examples::time_ms();
          const double due = last_process_ms + min_interval_ms;
          if (now < due) {
            std::this_thread::sleep_for(
                std::chrono::milliseconds(std::min(static_cast<int>(due - now) + 1, 15)));
            continue;
          }
        }
        simaai::neat::Sample sample;
        simaai::neat::PullError pull_error;
        const double pull_start = sima_examples::time_ms();
        const auto status = stream.run.pull(stream.output_name, 50, sample, &pull_error);
        const double pull_end = sima_examples::time_ms();
        if (status == simaai::neat::PullStatus::Timeout) {
          continue;
        }
        if (status == simaai::neat::PullStatus::Closed) {
          if (stopping(managed)) {
            break;
          }
          if (!reconnect(managed, stream)) {
            if (!stopping(managed)) {
              failure = "source closed and the reconnect budget ran out";
            }
            break;
          }
          last_process_ms = -1e12;
          consecutive_transient = 0;
          continue;
        }
        if (status != simaai::neat::PullStatus::Ok) {
          // Accelerator errors under churn are transient: the MLA model load is
          // deferred to run start, so a rebuild can surface here as a caps or
          // load error, and closing another stream's run (a live edit) can fail
          // this stream's accelerator stage. Rebuilding this stream recovers it.
          const std::string& message = pull_error.message;
          const bool transient = message.find("Unable to load model") != std::string::npos ||
                                 message.find("misconfig.caps") != std::string::npos ||
                                 message.find("accelerator_execution_failed") != std::string::npos;
          if (transient && !stopping(managed)) {
            // Most blips clear on their own; rebuilding at once cascades into
            // a rebuild storm.
            if (++consecutive_transient < kTransientRetries) {
              std::this_thread::sleep_for(std::chrono::milliseconds(25));
              continue;
            }
            if (transient_recoveries < kMaxTransientRecoveries) {
              ++transient_recoveries;
              consecutive_transient = 0;
              std::cerr << "[warn] stream " << managed->id
                        << " persistent MLA error; rebuilding pipeline (recovery "
                        << transient_recoveries << "/" << kMaxTransientRecoveries << ")\n";
              std::this_thread::sleep_for(std::chrono::milliseconds(300 + stagger_ms));
              build_live_pipeline(cfg_, stream);
              last_process_ms = -1e12;
              continue;
            }
          }
          // Closing the other streams' runs during shutdown can fail this
          // one's accelerator stage; that is not a stream failure.
          if (stopping(managed)) {
            break;
          }
          throw std::runtime_error("stream " + managed->id + " runtime error: " + message);
        }

        std::vector<std::uint8_t> payload;
        std::string err;
        if (!extract_bbox_payload(sample, payload, err)) {
          throw std::runtime_error("stream " + managed->id + " bbox extract failed: " + err);
        }
        const auto boxes = objdet::parse_boxes_strict(payload, stream.width, stream.height,
                                                      cfg_.max_detections, false);
        ++stream.processed;
        transient_recoveries = 0;
        consecutive_transient = 0;
        last_process_ms = sima_examples::time_ms();
        if (stream.processed <= cfg_.warmup_frames) {
          continue;
        }

        const double metadata_start = sima_examples::time_ms();
        send_live_metadata(stream, sample, boxes, active_count());
        const double metadata_end = sima_examples::time_ms();
        if (save_frames_enabled(cfg_) && stream.processed % cfg_.save_every == 0) {
          const auto tensors = simaai::neat::tensors_from_sample(sample, false);
          cv::Mat frame;
          std::string frame_err;
          if (!tensors.empty() && sima_examples::nv12_to_bgr(tensors.front(), frame, frame_err)) {
            save_debug_frame(cfg_, stream.channel, stream.processed, frame, boxes);
          } else if (!frame_err.empty()) {
            std::cerr << "[warn] failed to prepare output frame: " << frame_err << "\n";
          }
        }
        stream.profile.add(pull_end - pull_start, metadata_end - metadata_start,
                           static_cast<int>(boxes.size()));
      }
    } catch (...) {
      {
        std::lock_guard<std::mutex> lock(error_mtx_);
        if (!first_error_) {
          first_error_ = std::current_exception();
        }
      }
      failure = describe_current_exception();
    }
    if (built) {
      stream.profile.flush();
      try {
        stream.run.close();
      } catch (...) { // teardown is best-effort
      }
      std::cout << "[stream " << managed->id << "] processed=" << stream.processed << "\n";
    }
    managed->finished.store(true);
    if (!failure.empty()) {
      retire(managed, failure);
    }
  }

  const AppConfig& cfg_;
  std::vector<std::string> labels_;
  std::mutex mtx_;
  std::map<std::string, std::unique_ptr<ManagedStream>> streams_;
  std::vector<int> free_channels_;
  std::atomic_bool stop_{false};
  // Failed streams, kept until shutdown(): each owns the thread that retired it,
  // so it cannot be destroyed from inside that thread.
  std::vector<std::unique_ptr<ManagedStream>> retired_;
  // id -> reason for streams that died on their own. Kept after the slot is
  // released so the run can report them, and so all_failed() can tell "every
  // stream broke" apart from "the config lists no streams".
  std::map<std::string, std::string> failed_;
  std::mutex error_mtx_;
  std::exception_ptr first_error_;
};

void sleep_unless_stopping(double seconds, const StreamManager& manager) {
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(static_cast<int>(seconds * 1000.0));
  while (std::chrono::steady_clock::now() < deadline && !manager.stopping()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
}

void run_live(const AppConfig& cfg) {
  const auto labels = load_labels(cfg.labels_path);
  StreamManager manager(cfg, labels);
  std::error_code ec;
  auto last_write = fs::last_write_time(cfg.config_path, ec);

  for (const auto& source : cfg.sources) {
    manager.add(source);
  }
  std::cout << "[app] live mode: " << cfg.sources.size() << " stream(s); watching "
            << cfg.config_path.string() << "\n"
            << std::flush;

  while (!manager.stopping()) {
    if (manager.all_done()) {
      break;
    }
    // A finite run has nothing left to wait for once every stream has failed.
    // A continuous run keeps going: the config may sit at zero streams until an
    // edit adds one.
    if (cfg.frames > 0 && manager.all_failed()) {
      break;
    }
    sleep_unless_stopping(cfg.config_watch_seconds, manager);
    const auto now = fs::last_write_time(cfg.config_path, ec);
    if (ec || now == last_write) {
      continue;
    }
    last_write = now;
    try {
      const auto sources = reload_sources(cfg.config_path, cfg.max_streams);
      std::cout << "[config] reload: " << sources.size() << " stream(s)\n";
      manager.apply_sources(sources);
    } catch (const std::exception& e) {
      std::cerr << "[warn] ignoring invalid config reload: " << e.what() << "\n";
    }
  }

  // Sampled before shutdown(), which drains every stream and would make any run
  // look like a total failure.
  const bool nothing_survived = manager.all_failed();
  manager.shutdown();
  for (const auto& [id, reason] : manager.failed()) {
    std::cerr << "[warn] stream " << id << " did not run: " << reason << "\n";
  }
  // Individual stream failures are reported, not fatal. The run fails only
  // when no stream survived.
  if (nothing_survived) {
    manager.rethrow_first_error();
  }
}

// ── scale mode: one fused graph, one shared detector ──────────────────────────

// Largest upscale the hardware scaler performs. Past it the scaler emits
// nothing, with no error, so the shared detector geometry is clamped against
// it. Measured on Modalix: 2x works, 3x does not.
constexpr int kMaxUpscale = 2;
// The detector's own input size; clamping below it would discard detail the
// model uses anyway.
constexpr int kModelInput = 640;

// Decoder profile the scale and group modes were measured with at 16 streams.
// Lower kDecoderBuffers only past ~16 streams, where the memory saving is worth
// the risk of stutter under network jitter.
constexpr int kDecoderBuffers = 8;
constexpr int kDecoderInputBuffers = 2;
constexpr const char* kDecoderTuning = "auto";
// Declared decode rate cap for admission. A very high source rate times many
// streams exceeds the decoder core and the whole graph is rejected before it
// starts ("decoder processing capacity unavailable"). Above the cap the decode
// rate is left unpinned so admission passes and the decoder drops what it
// cannot process. Video to Insight is the untouched passthrough either way.
constexpr int kDecoderFpsCap = 30;

// Shared-detector admission limits, as measured per mode.
struct FusedProfile {
  int max_inflight_per_stream;
  int max_inflight_total;
};

constexpr FusedProfile kScaleProfile{4, 16};
// A group owns a few streams on one detector input port, so queue depth buys
// staleness rather than throughput.
constexpr FusedProfile kGroupProfile{1, 8};

struct FusedStream {
  int index = 0;
  int channel = 0;
  std::string id;
  std::string url;
  std::unique_ptr<simaai::neat::MetadataSender> metadata_sender;
  std::vector<std::string> labels;
  ProfileWindow profile;
  int frame_w = 0;
  int frame_h = 0;
  // Geometry the detector leg was normalized to. Detections come back in this
  // space and are scaled to frame_w/frame_h before publishing.
  int det_w = 0;
  int det_h = 0;
  int source_fps = 0;
  int video_port = 0;
  // Decoded frames awaiting their detection, keyed by frame id.
  std::map<int64_t, cv::Mat> debug_frames;
  int processed = 0;
  bool closed = false;
  bool debug_pairing_warned = false;
  double last_process_ms = -1e12;
};

// One input geometry for the shared detector, from every source's size.
//
// A shared model has one input port, so sources of different sizes are scaled
// to one geometry in front of it. The largest source keeps the detail small
// objects need, clamped so the smallest source is never upscaled past
// kMaxUpscale, and never below kModelInput unless every source is smaller.
void detector_geometry(const std::vector<ProbedStream>& probes, int& det_w, int& det_h) {
  int max_w = probes.front().width;
  int max_h = probes.front().height;
  int min_w = max_w;
  int min_h = max_h;
  for (const auto& p : probes) {
    max_w = std::max(max_w, p.width);
    max_h = std::max(max_h, p.height);
    min_w = std::min(min_w, p.width);
    min_h = std::min(min_h, p.height);
  }
  det_w = std::min(max_w, std::max(kModelInput, std::min(max_w, kMaxUpscale * min_w)));
  det_h = std::min(max_h, std::max(kModelInput, std::min(max_h, kMaxUpscale * min_h)));
  // NV12 needs even dimensions.
  det_w -= det_w % 2;
  det_h -= det_h % 2;
}

std::string stream_link_id(int stream_index) {
  return "stream" + std::to_string(stream_index);
}

int stream_index_from_sample(const simaai::neat::Sample& sample, int stream_count) {
  const std::string prefix = "stream";
  if (!starts_with(sample.stream_id, prefix)) {
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

simaai::neat::GraphLinkOptions realtime_link(int stream_index, int queue_depth,
                                             int max_inflight_per_stream = -1,
                                             int max_inflight_total = -1) {
  simaai::neat::GraphLinkOptions link;
  link.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
  link.queue_depth = queue_depth;
  link.max_inflight_per_stream = max_inflight_per_stream;
  link.max_inflight_total = max_inflight_total;
  link.stream_id = stream_link_id(stream_index);
  return link;
}

simaai::neat::InputOptions encoded_input_options(simaai::neat::InputMemoryPolicy memory_policy) {
  simaai::neat::InputOptions opt;
  opt.payload_type = simaai::neat::PayloadType::Encoded;
  opt.format = simaai::neat::FormatTag::H264;
  opt.memory_policy = memory_policy;
  return opt;
}

simaai::neat::Graph build_fused_source_graph(const AppConfig& cfg, const FusedStream& stream) {
  simaai::neat::nodes::groups::RtspEncodedInputOptions opt;
  opt.url = stream.url;
  opt.codec = simaai::neat::nodes::groups::RtspCodec::H264;
  opt.latency_ms = cfg.latency_ms;
  opt.tcp = cfg.tcp;
  opt.source_fps = stream.source_fps;
  opt.fallback_h264_width = stream.frame_w;
  opt.fallback_h264_height = stream.frame_h;
  simaai::neat::Graph source("rtsp_encoded_source");
  source.add(simaai::neat::nodes::groups::RtspEncodedInput(opt));
  return source;
}

// Decode, then scale to the shared detector geometry when it differs.
simaai::neat::Graph build_fused_decode_graph(const FusedStream& stream) {
  // The decoded RTSP group's defaults are the decoder settings Core expects in
  // front of a model; reuse them instead of restating them here.
  const simaai::neat::nodes::groups::RtspDecodedInputOptions defaults;
  simaai::neat::SimaDecodeOptions dec;
  dec.type = simaai::neat::SimaDecodeType::H264;
  dec.sima_allocator_type = defaults.sima_allocator_type;
  dec.out_format = "NV12";
  dec.decoder_name = "decoder";
  dec.raw_output = true;
  dec.next_element = defaults.decoder_next_element;
  dec.dec_width = stream.frame_w;
  dec.dec_height = stream.frame_h;
  // Pinning a high source rate makes admission reject the graph, and pinning a
  // lower rate than the stream fails caps negotiation ("framerate mismatch").
  // Above the cap the rate is left unspecified (-1).
  const bool capped = stream.source_fps > kDecoderFpsCap;
  dec.dec_fps = capped ? -1 : stream.source_fps;
  dec.num_buffers = kDecoderBuffers;
  dec.input_buffers = kDecoderInputBuffers;
  dec.decoder_tuning = kDecoderTuning;
  // Only the low-memory decoder tunings enable memory_opt; "auto" does not.
  dec.memory_opt = false;

  simaai::neat::Graph decode("decode");
  decode.connect(simaai::neat::nodes::Input(
                     "decode_h264", encoded_input_options(simaai::neat::InputMemoryPolicy::Ev74)),
                 simaai::neat::nodes::SimaDecode(dec));
  if (stream.det_w != stream.frame_w || stream.det_h != stream.frame_h) {
    decode.add(simaai::neat::nodes::VideoScale());
  }
  decode.add(simaai::neat::nodes::CapsRaw("NV12", stream.det_w, stream.det_h,
                                          capped ? 0 : stream.source_fps,
                                          simaai::neat::CapsMemory::Any));
  return decode;
}

std::string debug_frame_output_name(int stream_index) {
  return "debug_frame_" + std::to_string(stream_index);
}

void connect_fused_stream(simaai::neat::Graph& graph, const AppConfig& cfg,
                          const FusedProfile& profile, const FusedStream& stream,
                          const simaai::neat::Graph& detector_graph) {
  auto source = build_fused_source_graph(cfg, stream);
  auto decoder = build_fused_decode_graph(stream);

  if (cfg.video_enabled) {
    // The encoded access unit feeds both the decoder and Insight, so the video
    // reaches Insight without a re-encode. Hardware encoders are what run out
    // first at high stream counts.
    auto encoded_branch = simaai::neat::graphs::Branch("encoded", {"decode_h264", "video_h264"});
    graph.connect(source, encoded_branch);
    graph.connect(encoded_branch, decoder, realtime_link(stream.index, 3));

    auto video_options = simaai::neat::nodes::groups::VideoSenderOptions::Passthrough(
        simaai::neat::nodes::groups::RtspCodec::H264);
    video_options.host = cfg.insight_host;
    video_options.channel = stream.channel;
    video_options.video_port_base = cfg.video_port;
    video_options.async = true;
    simaai::neat::Graph video("video_sender");
    video.connect(
        simaai::neat::nodes::Input(
            "video_h264", encoded_input_options(simaai::neat::InputMemoryPolicy::SystemMemory)),
        simaai::neat::nodes::groups::VideoSender(video_options));
    graph.connect(encoded_branch, video, realtime_link(stream.index, 3));
  } else {
    graph.connect(source, decoder, realtime_link(stream.index, 3));
  }

  const bool save_frames = save_frames_enabled(cfg);
  auto decoded_branch =
      save_frames ? simaai::neat::graphs::Branch("decoded", {"detector_frame", "debug_frame"})
                  : simaai::neat::graphs::Branch("decoded", {"detector_frame"});
  graph.connect(decoder, decoded_branch);
  graph.connect(
      decoded_branch, detector_graph,
      realtime_link(stream.index, 4, profile.max_inflight_per_stream, profile.max_inflight_total));
  if (save_frames) {
    simaai::neat::Graph frames("debug_frame");
    frames.connect(simaai::neat::nodes::Input("debug_frame"),
                   simaai::neat::nodes::Output(debug_frame_output_name(stream.index),
                                               simaai::neat::OutputOptions::EveryFrame(4)));
    graph.connect(decoded_branch, frames, realtime_link(stream.index, 4));
  }
}

simaai::neat::RunOptions fused_run_options(int stream_count) {
  simaai::neat::RunOptions run_options;
  run_options.preset = simaai::neat::RunPreset::Realtime;
  // One ingress queue slot per stream, floor 4, as the 16-stream high-density
  // profile does.
  run_options.queue_depth = std::max(4, std::min(32, stream_count));
  run_options.overflow_policy = simaai::neat::OverflowPolicy::KeepLatest;
  run_options.output_memory = simaai::neat::OutputMemory::ZeroCopy;
  return run_options;
}

// Arrival-order metadata, without rtp_timestamp. Exact matching needs the
// passthrough payloader pinned to a known RTP base, which VideoSenderOptions
// does not expose, so a key derived from PTS would match nothing and Insight
// would render no boxes. Measured: 99% forwarded at 16 streams x 25 fps.
void send_fused_metadata(FusedStream& stream, const simaai::neat::Sample& sample,
                         const std::vector<objdet::Box>& boxes) {
  const std::string data_json =
      "{\"objects\":" +
      metadata_objects_json(boxes, stream.labels, stream.frame_w, stream.frame_h) + "}";
  const int64_t timestamp_ms = sample.pts_ns >= 0 ? sample.pts_ns / 1000000 : -1;
  const std::string frame_id = sample.frame_id >= 0 ? std::to_string(sample.frame_id) : "";
  std::string err;
  (void)stream.metadata_sender->send_metadata("object-detection", data_json, timestamp_ms, frame_id,
                                              &err);
}

// Decoded frames held per stream while their detection catches up. Inference is
// a few frames behind the decoder; past that the detection is never coming.
constexpr std::size_t kDebugFrameCache = 16;

// Collect decoded frames keyed by frame id. The decoded and detector branches
// run on independent queues, so the newest decoded frame is not the one the
// next detection describes.
void drain_debug_frames(simaai::neat::Run& run, const AppConfig& cfg,
                        std::vector<FusedStream>& streams) {
  if (!save_frames_enabled(cfg)) {
    return;
  }
  for (auto& stream : streams) {
    const std::string output_name = debug_frame_output_name(stream.index);
    for (;;) {
      simaai::neat::Sample sample;
      simaai::neat::PullError pull_error;
      const auto status = run.pull(output_name, 0, sample, &pull_error);
      if (status == simaai::neat::PullStatus::Timeout ||
          status == simaai::neat::PullStatus::Closed) {
        break;
      }
      if (status != simaai::neat::PullStatus::Ok) {
        throw std::runtime_error("failed to pull " + output_name + ": " + pull_error.message);
      }
      const auto tensors = simaai::neat::tensors_from_sample(sample, false);
      if (tensors.empty()) {
        continue;
      }
      cv::Mat bgr;
      std::string err;
      if (!sima_examples::nv12_to_bgr(tensors.front(), bgr, err)) {
        std::cerr << "[warn] failed to prepare debug frame: " << err << "\n";
        continue;
      }
      stream.debug_frames[sample.frame_id] = std::move(bgr);
      while (stream.debug_frames.size() > kDebugFrameCache) {
        stream.debug_frames.erase(stream.debug_frames.begin());
      }
    }
  }
}

// The decoded frame this detection came from, or an empty Mat if nothing is
// held. A match also clears everything older: those frames' detections have
// already gone past, so they can never be claimed.
cv::Mat take_debug_frame(FusedStream& stream, int64_t frame_id) {
  const auto it = stream.debug_frames.find(frame_id);
  if (it != stream.debug_frames.end()) {
    cv::Mat frame = std::move(it->second);
    stream.debug_frames.erase(stream.debug_frames.begin(), std::next(it));
    return frame;
  }
  if (stream.debug_frames.empty()) {
    return {};
  }
  if (!stream.debug_pairing_warned) {
    stream.debug_pairing_warned = true;
    std::cerr << "[warn] stream " << stream.id
              << " debug frames carry no id matching their detections; saved images fall back"
                 " to the newest frame\n";
  }
  const auto newest = std::prev(stream.debug_frames.end());
  cv::Mat frame = std::move(newest->second);
  stream.debug_frames.erase(newest);
  return frame;
}

void process_fused_sample(FusedStream& stream, const AppConfig& cfg,
                          const simaai::neat::Sample& sample, double detection_pull_ms) {
  if (cfg.frames > 0 && stream.processed >= cfg.frames) {
    return;
  }
  // inference.fps caps how often this stream publishes; the shared detector
  // has already run on the sample, and the video stays at the source rate.
  const double now = sima_examples::time_ms();
  if (cfg.fps > 0 && now < stream.last_process_ms + 1000.0 / static_cast<double>(cfg.fps)) {
    return;
  }
  stream.last_process_ms = now;

  std::vector<std::uint8_t> payload;
  std::string err;
  if (!extract_bbox_payload(sample, payload, err)) {
    throw std::runtime_error("stream " + stream.id + " bbox extract failed: " + err);
  }
  // Boxes come back in the shared detector geometry. Clamp there, then scale to
  // the stream's native size so they line up with the passthrough video.
  auto boxes =
      objdet::parse_boxes_strict(payload, stream.det_w, stream.det_h, cfg.max_detections, false);
  // Debug frames are tapped after the scaler, so they need the boxes in the
  // detector geometry.
  const std::vector<objdet::Box> debug_boxes =
      save_frames_enabled(cfg) ? boxes : std::vector<objdet::Box>{};
  if (stream.det_w != stream.frame_w || stream.det_h != stream.frame_h) {
    // The scale to the detector geometry is a stretch, not a letterbox: the
    // output caps pin width and height and leave the pixel aspect ratio free,
    // so a plain per-axis ratio maps the boxes back.
    const float sx = static_cast<float>(stream.frame_w) / static_cast<float>(stream.det_w);
    const float sy = static_cast<float>(stream.frame_h) / static_cast<float>(stream.det_h);
    for (auto& box : boxes) {
      box.x1 *= sx;
      box.x2 *= sx;
      box.y1 *= sy;
      box.y2 *= sy;
    }
  }

  ++stream.processed;
  if (stream.processed <= cfg.warmup_frames) {
    return;
  }
  const double metadata_start = sima_examples::time_ms();
  send_fused_metadata(stream, sample, boxes);
  const double metadata_end = sima_examples::time_ms();
  if (save_frames_enabled(cfg) && stream.processed % cfg.save_every == 0) {
    save_debug_frame(cfg, stream.channel, stream.processed,
                     take_debug_frame(stream, sample.frame_id), debug_boxes);
  }
  stream.profile.add(detection_pull_ms, metadata_end - metadata_start,
                     static_cast<int>(boxes.size()));
}

// Pull and handle one detection. Returns false once the shared output has
// closed.
bool process_fused_run_once(simaai::neat::Run& run, const AppConfig& cfg,
                            std::vector<FusedStream>& streams) {
  drain_debug_frames(run, cfg, streams);
  simaai::neat::Sample sample;
  simaai::neat::PullError pull_error;
  const double pull_start = sima_examples::time_ms();
  const auto status = run.pull("detections", 50, sample, &pull_error);
  const double pull_end = sima_examples::time_ms();
  if (status == simaai::neat::PullStatus::Timeout) {
    return true;
  }
  if (status == simaai::neat::PullStatus::Closed) {
    // The shared graph is gone; no later pull can succeed.
    std::cerr << "[warn] detection output closed; ending run\n";
    for (auto& stream : streams) {
      stream.closed = true;
    }
    return false;
  }
  if (status != simaai::neat::PullStatus::Ok) {
    throw std::runtime_error("runtime error: " + pull_error.message);
  }
  auto& stream = streams[static_cast<std::size_t>(
      stream_index_from_sample(sample, static_cast<int>(streams.size())))];
  process_fused_sample(stream, cfg, sample, pull_end - pull_start);
  return true;
}

bool all_streams_done(const std::vector<FusedStream>& streams, int frame_limit) {
  if (streams.empty()) {
    return true;
  }
  return std::all_of(streams.begin(), streams.end(), [frame_limit](const FusedStream& stream) {
    return stream.closed || (frame_limit > 0 && stream.processed >= frame_limit);
  });
}

// Run every stream of `cfg` through one graph and one shared detector.
// `on_running`, when set, is called once the graph is running; group mode uses
// it to start its workers one at a time.
void run_scale(const AppConfig& cfg, const FusedProfile& profile,
               const std::function<void()>& on_running = {}) {
  const auto labels = load_labels(cfg.labels_path);
  std::vector<ProbedStream> probes;
  probes.reserve(cfg.sources.size());
  for (const auto& source : cfg.sources) {
    probes.push_back(probe_stream(cfg, source));
    sima_examples::require(cfg.source_fps > 0 || probes.back().fps > 0,
                           "stream " + source.id + " did not report a frame rate; set source.fps");
  }

  int det_w = 0;
  int det_h = 0;
  detector_geometry(probes, det_w, det_h);
  // Raising the clamp to kModelInput can re-cross the upscale limit (320x180
  // beside 1080p), and no geometry satisfies both. Name the stream that will
  // deliver video but no detections rather than let it go quiet.
  double worst = 0.0;
  const ProbedStream* smallest = &probes.front();
  std::set<std::pair<int, int>> sizes;
  for (const auto& p : probes) {
    worst = std::max(
        {worst, static_cast<double>(det_w) / p.width, static_cast<double>(det_h) / p.height});
    if (p.width * p.height < smallest->width * smallest->height) {
      smallest = &p;
    }
    sizes.insert({p.width, p.height});
  }
  if (worst > kMaxUpscale) {
    std::cerr << "[warn] detector geometry " << det_w << "x" << det_h << " needs a " << worst
              << "x upscale from the smallest source (" << smallest->width << "x"
              << smallest->height << "), past the " << kMaxUpscale
              << "x the scaler performs - that stream will deliver video but no detections. Use "
                 "sources within "
              << kMaxUpscale << "x of each other.\n";
  }
  if (sizes.size() > 1) {
    std::cout << "[detector] mixed source sizes -> detector input " << det_w << "x" << det_h
              << " (video stays native per stream)\n";
  }

  auto model = make_model(cfg, det_w, det_h);
  auto detector_input = model->input_appsrc_options(false);
  detector_input.block = true;
  simaai::neat::Graph detector_graph("detector");
  detector_graph.connect(simaai::neat::nodes::Input("detector_frame", detector_input), *model);
  simaai::neat::Graph detections_graph("detections");
  detections_graph.add(
      simaai::neat::nodes::Output("detections", simaai::neat::OutputOptions::EveryFrame(4)));

  simaai::neat::Graph graph;
  std::vector<FusedStream> streams;
  streams.reserve(cfg.sources.size());
  for (std::size_t index = 0; index < cfg.sources.size(); ++index) {
    const auto& source = cfg.sources[index];
    const auto& probe = probes[index];
    FusedStream stream;
    stream.index = static_cast<int>(index);
    stream.channel = cfg.channel_base + static_cast<int>(index);
    stream.id = source.id;
    stream.url = source.url;
    stream.labels = labels;
    stream.profile.enabled = cfg.profile;
    stream.profile.stream_label = source.id;
    stream.frame_w = probe.width;
    stream.frame_h = probe.height;
    stream.det_w = det_w;
    stream.det_h = det_h;
    stream.source_fps = cfg.source_fps > 0 ? cfg.source_fps : probe.fps;
    stream.video_port = cfg.video_enabled ? cfg.video_port + stream.channel : 0;

    simaai::neat::MetadataSenderOptions metadata_options;
    metadata_options.host = cfg.insight_host;
    metadata_options.channel = stream.channel;
    metadata_options.metadata_port_base = cfg.metadata_port;
    std::string metadata_err;
    stream.metadata_sender =
        std::make_unique<simaai::neat::MetadataSender>(metadata_options, &metadata_err);
    sima_examples::require(stream.metadata_sender->ok(), metadata_err);

    connect_fused_stream(graph, cfg, profile, stream, detector_graph);
    std::cout << "[stream " << source.id << "] channel=" << stream.channel << " rtsp=" << source.url
              << " size=" << stream.frame_w << "x" << stream.frame_h << "@" << stream.source_fps
              << " video=" << (cfg.video_enabled ? std::to_string(stream.video_port) : "disabled")
              << " metadata=" << stream.metadata_sender->metadata_port() << "\n";
    streams.push_back(std::move(stream));
  }
  graph.connect(detector_graph, detections_graph);

  if (cfg.profile) {
    std::cout << "Backend:\n" << graph.describe_backend() << "\n";
  }
  simaai::neat::Run run = graph.build(fused_run_options(static_cast<int>(streams.size())));
  std::cout << "[app] graph running: " << streams.size() << " stream(s)\n" << std::flush;
  if (on_running) {
    on_running();
  }

  bool output_closed = false;
  try {
    while (!stop_requested() && !all_streams_done(streams, cfg.frames)) {
      if (!process_fused_run_once(run, cfg, streams)) {
        output_closed = true;
      }
    }
  } catch (...) {
    run.close();
    throw;
  }
  run.close();
  for (auto& stream : streams) {
    stream.profile.flush();
    std::cout << "[stream " << stream.id << "] processed=" << stream.processed << "\n";
  }

  // A continuous run has no natural end, so a closed output is always a
  // failure; a finite run fails only if it closed short of its budget.
  if (output_closed && !stop_requested()) {
    if (cfg.frames <= 0) {
      throw std::runtime_error(
          "detection output closed unexpectedly; the run produced no further metadata");
    }
    std::string detail;
    for (const auto& stream : streams) {
      if (stream.processed < cfg.frames) {
        detail += (detail.empty() ? "" : ", ") + std::string("stream ") + stream.id + " got " +
                  std::to_string(stream.processed) + "/" + std::to_string(cfg.frames);
      }
    }
    if (!detail.empty()) {
      throw std::runtime_error(
          "detection output closed before every stream reached inference.frames (" + detail + ")");
    }
  }
}

// ── group mode: several independent scale processes ───────────────────────────

// Set by the group supervisor on each worker it starts, as
// "<group>:<ready fd>:<channel base>:" followed by each stream's id and url.
// Internal to this application, not a user setting.
constexpr const char* kGroupWorkerEnv = "ADAPTIVE_DETECTOR_GROUP_WORKER";

struct GroupWorkerRole {
  int index = -1;
  int ready_fd = -1;
  int channel_base = -1;
  std::vector<StreamSource> sources;
};

// Each id and url is written as "<length>:<text>", so they may hold any character.
void append_role_field(std::string& role, const std::string& text) {
  role += std::to_string(text.size()) + ":" + text;
}

std::string read_role_field(const std::string& role, std::size_t& pos) {
  const std::size_t colon = role.find(':', pos);
  sima_examples::require(colon != std::string::npos && colon > pos, "missing field length");
  const std::size_t length = std::stoul(role.substr(pos, colon - pos));
  sima_examples::require(colon + 1 + length <= role.size(), "truncated field");
  pos = colon + 1 + length;
  return role.substr(colon + 1, length);
}

std::string group_worker_role_value(int index, int ready_fd, const AppConfig& group) {
  std::string role = std::to_string(index) + ":" + std::to_string(ready_fd) + ":" +
                     std::to_string(group.channel_base) + ":";
  for (const auto& source : group.sources) {
    append_role_field(role, source.id);
    append_role_field(role, source.url);
  }
  return role;
}

std::optional<GroupWorkerRole> group_worker_role() {
  const char* raw = std::getenv(kGroupWorkerEnv);
  if (raw == nullptr || *raw == '\0') {
    return std::nullopt;
  }
  const std::string text(raw);
  GroupWorkerRole role;
  int consumed = 0;
  try {
    sima_examples::require(std::sscanf(raw, "%d:%d:%d:%n", &role.index, &role.ready_fd,
                                       &role.channel_base, &consumed) == 3 &&
                               consumed > 0 && role.index >= 0 && role.ready_fd >= 0 &&
                               role.channel_base >= 0,
                           "bad header");
    std::size_t pos = static_cast<std::size_t>(consumed);
    while (pos < text.size()) {
      StreamSource source;
      source.id = read_role_field(text, pos);
      source.url = read_role_field(text, pos);
      role.sources.push_back(std::move(source));
    }
    sima_examples::require(!role.sources.empty(), "no streams");
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string("invalid ") + kGroupWorkerEnv + " (" + e.what() +
                             "): " + raw);
  }
  return role;
}

// One group's process: a scale run over the streams its supervisor assigned it.
int run_group_worker(const AppConfig& cfg, const GroupWorkerRole& role) {
  AppConfig group = cfg;
  group.sources = role.sources;
  group.channel_base = role.channel_base;
  try {
    sima_examples::require(cfg.mode == "group", "config is no longer in group mode");
    run_scale(group, kGroupProfile, [&role]() {
      const char ready = 1;
      (void)!write(role.ready_fd, &ready, 1);
    });
  } catch (const std::exception& e) {
    std::cerr << "[ERR] group " << role.index << ": " << e.what() << "\n";
    close(role.ready_fd);
    return 1;
  }
  close(role.ready_fd);
  return 0;
}

struct GroupWorker {
  int index = 0;
  pid_t pid = -1;
  int exit_code = -1;                                       // set once reaped
  std::vector<std::pair<std::string, std::string>> streams; // (id, url) it runs
  bool running() const {
    return pid > 0 && exit_code < 0;
  }
};

int exit_code_from_status(int status) {
  if (WIFEXITED(status)) {
    return WEXITSTATUS(status);
  }
  return WIFSIGNALED(status) ? 128 + WTERMSIG(status) : 1;
}

// Reap a worker that has exited, without blocking.
void poll_worker(GroupWorker& worker) {
  if (!worker.running()) {
    return;
  }
  int status = 0;
  if (waitpid(worker.pid, &status, WNOHANG) == worker.pid) {
    worker.exit_code = exit_code_from_status(status);
  }
}

// Start one group as a fresh process running this same executable and config.
// A forked copy of this process cannot be used directly: the Neat runtime sets
// up process-wide state when it loads, and a forked child inherits that state
// without the threads behind it, so its output and teardown break.
GroupWorker start_group_worker(const fs::path& config_path, int index, const AppConfig& group,
                               int& ready_read_fd) {
  char exe[4096] = {0};
  const ssize_t exe_len = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
  sima_examples::require(exe_len > 0, "failed to locate this executable");
  int fds[2];
  sima_examples::require(pipe2(fds, O_CLOEXEC) == 0, "failed to create a group ready pipe");

  // Everything the child needs is built before fork(): only async-signal-safe
  // calls are allowed between fork() and execve().
  const std::string config = config_path.string();
  const std::string role =
      std::string(kGroupWorkerEnv) + "=" + group_worker_role_value(index, fds[1], group);
  std::vector<char*> argv = {exe, const_cast<char*>("--config"), const_cast<char*>(config.c_str()),
                             nullptr};
  std::vector<char*> envp;
  for (char** env = environ; *env != nullptr; ++env) {
    if (std::strncmp(*env, kGroupWorkerEnv, std::strlen(kGroupWorkerEnv)) != 0) {
      envp.push_back(*env);
    }
  }
  envp.push_back(const_cast<char*>(role.c_str()));
  envp.push_back(nullptr);

  std::cout << std::flush;
  std::cerr << std::flush;
  const pid_t pid = fork();
  if (pid == 0) {
    fcntl(fds[1], F_SETFD, 0); // the ready pipe survives execve()
    execve(exe, argv.data(), envp.data());
    _exit(127);
  }
  close(fds[1]);
  if (pid < 0) {
    close(fds[0]);
    throw std::runtime_error("failed to start group " + std::to_string(index));
  }
  ready_read_fd = fds[0];
  return GroupWorker{index, pid, -1, {}};
}

// Block until the worker reports its graph running, exits, or a stop arrives.
void wait_until_running(GroupWorker& worker, int ready_fd) {
  pollfd fd{ready_fd, POLLIN, 0};
  while (!stop_requested()) {
    const int ready = poll(&fd, 1, 500);
    if (ready > 0) {
      return; // a byte, or EOF because the worker exited
    }
    if (ready < 0 && errno != EINTR) {
      return;
    }
    poll_worker(worker);
    if (!worker.running()) {
      return;
    }
  }
}

// What a group runs; a group restarts only when this changes.
using GroupStreams = std::vector<std::pair<std::string, std::string>>;

GroupStreams group_streams(const AppConfig& group) {
  GroupStreams streams;
  for (const auto& source : group.sources) {
    streams.emplace_back(source.id, source.url);
  }
  return streams;
}

// Indexes of groups to stop, start, or restart to go from `running` to `wanted`.
std::vector<int> changed_groups(const std::map<int, GroupStreams>& running,
                                const std::vector<AppConfig>& wanted) {
  std::map<int, GroupStreams> wanted_streams;
  for (std::size_t index = 0; index < wanted.size(); ++index) {
    if (!wanted[index].sources.empty()) {
      wanted_streams[static_cast<int>(index)] = group_streams(wanted[index]);
    }
  }
  std::set<int> indexes;
  for (const auto& [index, streams] : running) {
    indexes.insert(index);
  }
  for (const auto& [index, streams] : wanted_streams) {
    indexes.insert(index);
  }
  std::vector<int> changed;
  for (const int index : indexes) {
    const auto have = running.find(index);
    const auto want = wanted_streams.find(index);
    if (have == running.end() || want == wanted_streams.end() || have->second != want->second) {
      changed.push_back(index);
    }
  }
  return changed;
}

// Runs one scale process per group and applies config edits group by group.
//
// A group whose streams change is restarted alone; the other groups keep
// running. Groups start one at a time because concurrent model loads race on
// the MLA, and a group that fails to start does not block the next one.
class GroupSupervisor {
public:
  explicit GroupSupervisor(const AppConfig& cfg) : cfg_(cfg) {}

  void apply(const std::vector<AppConfig>& groups) {
    std::map<int, GroupStreams> running;
    for (const auto& [index, worker] : workers_) {
      running[index] = worker.streams;
    }
    const auto changed = changed_groups(running, groups);
    const auto keep = [&groups](int index) {
      return index < static_cast<int>(groups.size()) &&
             !groups[static_cast<std::size_t>(index)].sources.empty();
    };
    for (const int index : changed) {
      if (workers_.count(index) != 0) {
        std::cout << "[group " << index << "] "
                  << (keep(index) ? "streams changed; restarting" : "removed") << "\n"
                  << std::flush;
        stop(index);
      }
    }
    for (const int index : changed) {
      if (keep(index) && !stop_requested()) {
        start(index, groups[static_cast<std::size_t>(index)]);
      }
    }
  }

  // Note groups that exited non-zero on their own.
  void record_exits() {
    for (auto& [index, worker] : workers_) {
      poll_worker(worker);
      if (!worker.running() && worker.exit_code != 0 && failed_.count(index) == 0) {
        failed_[index] = worker.exit_code;
        std::cerr << "[warn] group " << index << " exited with code " << worker.exit_code
                  << "; other groups keep running\n";
      }
    }
  }

  bool any_running() {
    return std::any_of(workers_.begin(), workers_.end(),
                       [](const auto& kv) { return kv.second.running(); });
  }

  void shutdown() {
    while (!workers_.empty()) {
      stop(workers_.begin()->first);
    }
  }

  const std::map<int, int>& failed() const {
    return failed_;
  }

private:
  void start(int index, const AppConfig& group) {
    int ready_fd = -1;
    GroupWorker worker = start_group_worker(cfg_.config_path, index, group, ready_fd);
    worker.streams = group_streams(group);
    failed_.erase(index);
    std::cout << "[group " << index << "] pid=" << worker.pid << " streams=";
    for (std::size_t s = 0; s < group.sources.size(); ++s) {
      std::cout << (s == 0 ? "" : ",") << group.sources[s].id;
    }
    std::cout << " channels=" << group.channel_base << "-"
              << group.channel_base + static_cast<int>(group.sources.size()) - 1 << "\n"
              << std::flush;
    wait_until_running(worker, ready_fd);
    close(ready_fd);
    workers_[index] = worker;
  }

  // Stopped by the supervisor (a config edit or shutdown), so its exit is not
  // a group failure.
  void stop(int index) {
    GroupWorker worker = workers_.at(index);
    workers_.erase(index);
    // SIGTERM runs the group's normal shutdown, which releases its pools.
    if (worker.running()) {
      kill(worker.pid, SIGTERM);
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
    while (worker.running() && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      poll_worker(worker);
    }
    if (worker.running()) {
      std::cerr << "[warn] group " << index << " did not stop after SIGTERM; killing it\n";
      kill(worker.pid, SIGKILL);
      int status = 0;
      waitpid(worker.pid, &status, 0);
    }
  }

  const AppConfig& cfg_;
  std::map<int, GroupWorker> workers_;
  // Groups that exited non-zero on their own, by index: the exit code.
  std::map<int, int> failed_;
};

void run_group(const AppConfig& cfg) {
  auto groups = split_groups(cfg);
  std::cout << "[app] group mode: " << cfg.sources.size() << " stream(s) in " << groups.size()
            << " group(s) of up to " << cfg.group_size << "; watching " << cfg.config_path.string()
            << "\n"
            << std::flush;

  GroupSupervisor supervisor(cfg);
  std::error_code ec;
  auto last_write = fs::last_write_time(cfg.config_path, ec);
  supervisor.apply(groups);
  while (!stop_requested()) {
    supervisor.record_exits();
    // A finite run ends once every group has finished. A continuous run keeps
    // going: a config edit can still restart a failed group.
    if (cfg.frames > 0 && !supervisor.any_running()) {
      break;
    }
    const auto deadline =
        std::chrono::steady_clock::now() +
        std::chrono::milliseconds(static_cast<int>(cfg.config_watch_seconds * 1000.0));
    while (std::chrono::steady_clock::now() < deadline && !stop_requested()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    const auto now = fs::last_write_time(cfg.config_path, ec);
    if (ec || now == last_write || stop_requested()) {
      continue;
    }
    last_write = now;
    try {
      const auto sources = reload_sources(cfg.config_path, cfg.max_streams);
      std::cout << "[config] reload: " << sources.size() << " stream(s)\n" << std::flush;
      groups = regroup(cfg, groups, sources);
      supervisor.apply(groups);
    } catch (const std::exception& e) {
      std::cerr << "[warn] ignoring invalid config reload: " << e.what() << "\n";
    }
  }

  supervisor.record_exits();
  const bool stopping = stop_requested();
  supervisor.shutdown();
  if (stopping) {
    return;
  }
  std::string failed;
  for (const auto& [index, code] : supervisor.failed()) {
    failed += (failed.empty() ? "" : ", ") + std::to_string(index) + " (exit " +
              std::to_string(code) + ")";
  }
  if (!failed.empty()) {
    throw std::runtime_error("group(s) " + failed + " failed; see the [ERR] lines above");
  }
}

// ── entry point ───────────────────────────────────────────────────────────────

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
      std::cout << "Usage: " << argv[0] << " [--config <path>] [--validate-config-only]\n\n"
                << "Multi-stream RTSP YOLO26 detector for Insight. Set `mode` (live, scale, or "
                   "group) in the config file.\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  return options;
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
      std::cout << "Config validated: " << cli.config_path.string() << " (mode=" << cfg.mode
                << ", streams=" << cfg.sources.size() << ")\n";
      return 0;
    }

    // SIGINT and SIGTERM both run the normal shutdown: without a handler the
    // process dies before each run is closed, and the decoder and CVU pools it
    // holds stay allocated. Group workers run this same path.
    g_stop_requested = 0;
    std::signal(SIGINT, request_stop);
    std::signal(SIGTERM, request_stop);
    enable_profile_env(cfg);
    if (save_frames_enabled(cfg)) {
      fs::create_directories(cfg.save_dir);
    }
    if (const auto role = group_worker_role()) {
      return run_group_worker(cfg, *role);
    }
    if (cfg.mode == "group") {
      run_group(cfg);
    } else if (cfg.mode == "scale") {
      run_scale(cfg, kScaleProfile);
    } else {
      run_live(cfg);
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[ERR] " << e.what() << "\n";
    return 1;
  }
}
