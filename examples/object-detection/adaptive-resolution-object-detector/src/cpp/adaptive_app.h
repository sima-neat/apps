// Adaptive-resolution detector: ONE GRAPH PER STREAM (--mode adaptive).
//
// Definitions live in a header because the Apps build compiles ONLY main.cpp
// for an example (cmake/ExampleModule.cmake picks main.cpp when it exists), so
// the whole program has to be one translation unit. The named namespace keeps
// this implementation's helpers from colliding with the other mode's, which
// were both written in anonymous namespaces.
#pragma once

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

// Live multi-stream YOLO26 detector for Insight - ONE GRAPH PER STREAM.
//
// Each stream owns its own graph, so a stream can be built or torn down while
// the others keep running: edit streams.sources in the config and the app picks
// the change up on its next poll. Every stream is decoded and detected at its
// native size and publishes H.264 video plus detection metadata to Insight on
// its own channel. Mirrors src/python/adaptive_app.py.

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
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace adaptive_app {

namespace fs = std::filesystem;

namespace {

volatile std::sig_atomic_t g_stop_requested = 0;

void request_stop(int) { g_stop_requested = 1; }

// Serializes MLA model loads across stream threads. Concurrent pipeline builds
// (initial builds and reconnects) otherwise race on MLA/RPMsg setup.
std::mutex g_model_build_mutex;

// ── configuration ────────────────────────────────────────────────────────────

struct StreamSource {
  std::string id;
  std::string rtsp_url;
};

struct AppConfig {
  std::string model_path;
  fs::path labels_path;
  int max_streams = 8;
  std::vector<StreamSource> sources;
  fs::path config_path;
  int latency_ms = 100;
  bool tcp = true;
  int frames = 0;
  int fps = 0;
  double min_score = 0.30;
  double nms_iou = 0.60;
  int max_detections = 50;
  bool profile = false;
  int warmup_frames = 30;
  double config_watch_seconds = 1.0;
  // This implementation has NO encoded-passthrough topology: it always builds a
  // decoded source plus a raw video sender, i.e. it re-encodes every stream,
  // unlike src/python/adaptive_app.py which honours the flag. Rather than
  // silently do the opposite of what a config asks, an explicit `true` is
  // rejected in validate_config(). Defaults to false so a config that omits the
  // key - including the shipped src/common/config.yaml - runs exactly as before.
  bool encoded_passthrough = false;
  std::string insight_host = "127.0.0.1";
  int video_port_base = 9000;
  int metadata_port_base = 9100;
  // "auto" | "on" | "off". Sibling of `insight:` under `output:`, not nested
  // inside it - see _config_live() in pipelines/pipeline-live/pipeline.py and
  // adaptive_app.py's string_or(output, ...). Mirrors src/python/adaptive_app.py
  // - the panel writes this key into the config both languages read, so
  // ignoring it here left C++ runs silently matching by arrival order while
  // Python matched exactly.
  std::string metadata_rtp_timestamp = "auto";
  bool video_enabled = true;
  fs::path save_dir;
  int save_every = 0;


};

struct CliOptions {
  fs::path config_path;
  bool validate_config_only = false;
};

// ── small YAML helpers (no YAML lib in the C++ path, matching sibling examples) ─

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

bool starts_with(const std::string& value, const std::string& prefix) {
  return value.rfind(prefix, 0) == 0;
}

int leading_indent(const std::string& line) {
  int indent = 0;
  while (indent < static_cast<int>(line.size()) && (line[indent] == ' ' || line[indent] == '\t')) {
    ++indent;
  }
  return indent;
}

void extract_ints(const std::string& text, std::vector<int>& out) {
  std::string token;
  bool negative = false;
  for (std::size_t i = 0; i <= text.size(); ++i) {
    const char c = i < text.size() ? text[i] : '\0';
    if (std::isdigit(static_cast<unsigned char>(c))) {
      token.push_back(c);
    } else {
      if (!token.empty()) {
        out.push_back((negative ? -1 : 1) * std::stoi(token));
        token.clear();
      }
      negative = (c == '-');
    }
  }
}


void apply_kv_or_scalar(const std::string& rest, std::string& cur_id, std::string& cur_url) {
  if (starts_with(rest, "id:")) {
    cur_id = unquote(sima_examples::trim_copy(rest.substr(3)));
  } else if (starts_with(rest, "rtsp_url:")) {
    cur_url = unquote(sima_examples::trim_copy(rest.substr(9)));
  } else if (!rest.empty()) {
    cur_url = unquote(rest);
  }
}

void apply_kv(const std::string& t, std::string& cur_id, std::string& cur_url) {
  if (starts_with(t, "id:")) {
    cur_id = unquote(sima_examples::trim_copy(t.substr(3)));
  } else if (starts_with(t, "rtsp_url:")) {
    cur_url = unquote(sima_examples::trim_copy(t.substr(9)));
  }
}

// Parse `streams`, accepting the rich (streams.sources: [{id, rtsp_url}]) and the
// bare-list (streams: [url, ...]) forms. Auto-assigns cam-<n> ids where absent.
std::vector<StreamSource> parse_stream_sources(const fs::path& config_path) {
  std::ifstream input(config_path);
  if (!input.is_open()) {
    throw std::runtime_error("failed to open config file: " + config_path.string());
  }

  std::vector<std::pair<std::string, std::string>> items;
  bool in_streams = false;
  int streams_indent = -1;
  bool have_item = false;
  std::string cur_id;
  std::string cur_url;
  const auto flush = [&]() {
    if (have_item)
      items.emplace_back(cur_id, cur_url);
    have_item = false;
    cur_id.clear();
    cur_url.clear();
  };

  std::string raw_line;
  while (std::getline(input, raw_line)) {
    const std::string line_wo = strip_inline_comment(raw_line);
    const std::string t = sima_examples::trim_copy(line_wo);
    if (t.empty())
      continue;
    const int indent = leading_indent(line_wo);

    if (in_streams && indent <= streams_indent && !starts_with(t, "- ")) {
      flush();
      in_streams = false;
    }

    if (!in_streams && starts_with(t, "streams:")) {
      in_streams = true;
      streams_indent = indent;
      const std::string rest = sima_examples::trim_copy(t.substr(std::string("streams:").size()));
      if (!rest.empty()) { // inline value, e.g. `streams: []` or `streams: [a, b]`
        std::string body = rest;
        if (body.front() == '[' && body.back() == ']')
          body = body.substr(1, body.size() - 2);
        std::string token;
        const auto push_token = [&]() {
          const std::string url = unquote(sima_examples::trim_copy(token));
          if (!url.empty())
            items.emplace_back(std::string(), url);
          token.clear();
        };
        for (char c : body) {
          if (c == ',')
            push_token();
          else
            token.push_back(c);
        }
        push_token();
        in_streams = false;
      }
      continue;
    }

    if (in_streams) {
      if (t == "sources:" || starts_with(t, "max_streams:"))
        continue;
      if (starts_with(t, "- ") || t == "-") {
        flush();
        have_item = true;
        const std::string rest = t.size() > 2 ? sima_examples::trim_copy(t.substr(2)) : "";
        apply_kv_or_scalar(rest, cur_id, cur_url);
      } else {
        apply_kv(t, cur_id, cur_url);
      }
    }
  }
  flush();

  std::vector<StreamSource> sources;
  for (std::size_t i = 0; i < items.size(); ++i) {
    std::string id = items[i].first;
    const std::string& url = items[i].second;
    if (url.empty())
      throw std::runtime_error("streams entry has an empty rtsp_url");
    if (id.empty())
      id = "cam-" + std::to_string(i + 1);
    sources.push_back({id, url});
  }
  if (sources.empty())
    throw std::runtime_error("streams must be a non-empty list");
  return sources;
}

void validate_config(const AppConfig& cfg) {
  sima_examples::require(!cfg.model_path.empty(),
                         "model.path must be set");
  sima_examples::require(!cfg.labels_path.empty(), "model.labels must be set");
  sima_examples::require(!cfg.sources.empty(), "streams must be a non-empty list");
  sima_examples::require(cfg.max_streams >= 1, "streams.max_streams must be >= 1");
  sima_examples::require(static_cast<int>(cfg.sources.size()) <= cfg.max_streams,
                         "streams count exceeds streams.max_streams");
  std::set<std::string> ids;
  for (const auto& s : cfg.sources)
    ids.insert(s.id);
  sima_examples::require(ids.size() == cfg.sources.size(), "streams.sources ids must be unique");
  sima_examples::require(!cfg.insight_host.empty(), "output.insight.host must be set");
  sima_examples::require(cfg.latency_ms >= 0, "input.latency_ms must be >= 0");
  sima_examples::require(cfg.frames >= 0, "inference.frames must be >= 0");
  sima_examples::require(cfg.fps >= 0, "inference.fps must be >= 0");
  sima_examples::require(cfg.min_score >= 0.0 && cfg.min_score <= 1.0,
                         "inference.min_score must be between 0 and 1");
  sima_examples::require(cfg.nms_iou >= 0.0 && cfg.nms_iou <= 1.0,
                         "inference.nms_iou must be between 0 and 1");
  sima_examples::require(cfg.max_detections > 0, "inference.max_detections must be > 0");
  sima_examples::require(cfg.warmup_frames >= 0, "runtime.warmup_frames must be >= 0");
  sima_examples::require(cfg.config_watch_seconds > 0.0, "runtime.config_watch_seconds must be > 0");
  sima_examples::require(cfg.video_port_base > 0, "output.insight.video_port must be > 0");
  sima_examples::require(cfg.metadata_port_base > 0, "output.insight.metadata_port must be > 0");
  sima_examples::require(
      !cfg.encoded_passthrough,
      "output.encoded_passthrough: true is not supported by the C++ implementation - "
      "it re-encodes every stream. Use --mode adaptive with the Python "
      "implementation for passthrough, or set output.encoded_passthrough: false.");
  sima_examples::require(cfg.metadata_rtp_timestamp == "auto" ||
                             cfg.metadata_rtp_timestamp == "on" ||
                             cfg.metadata_rtp_timestamp == "off",
                         "output.metadata_rtp_timestamp must be auto, on or off");
  sima_examples::require(cfg.save_every >= 0, "output.save_every must be >= 0");
}

AppConfig load_app_config(const fs::path& config_path) {
  const auto raw = sima_examples::ScalarConfig::load(config_path);
  const auto default_labels =
      fs::path(SIMANEAT_APPS_EXAMPLE_SOURCE_DIR).parent_path() / "common" / "coco_label.txt";

  AppConfig cfg;
  cfg.config_path = config_path;
  cfg.model_path = raw.string_or("model.path", "");
  cfg.labels_path = raw.string_or("model.labels", default_labels.string());
  cfg.max_streams = raw.int_or("streams.max_streams", 8);
  cfg.sources = parse_stream_sources(config_path);
  cfg.tcp = raw.bool_or("input.tcp", true);
  cfg.latency_ms = raw.int_or("input.latency_ms", 100);
  cfg.frames = raw.int_or("inference.frames", 0);
  cfg.fps = raw.int_or("inference.fps", 0);
  cfg.min_score = raw.double_or("inference.min_score", 0.30);
  cfg.nms_iou = raw.double_or("inference.nms_iou", 0.60);
  cfg.max_detections = raw.int_or("inference.max_detections", 50);
  cfg.profile = raw.bool_or("runtime.profile", false);
  cfg.warmup_frames = raw.int_or("runtime.warmup_frames", 30);
  cfg.config_watch_seconds = raw.double_or("runtime.config_watch_seconds", 1.0);
  cfg.encoded_passthrough = raw.bool_or("output.encoded_passthrough", false);
  cfg.insight_host = raw.string_or("output.insight.host", "");
  cfg.metadata_rtp_timestamp = raw.string_or("output.metadata_rtp_timestamp", "auto");
  cfg.video_port_base =
      raw.int_or("output.insight.video_port_base", raw.int_or("output.insight.video_port", 9000));
  cfg.metadata_port_base = raw.int_or("output.insight.metadata_port_base",
                                      raw.int_or("output.insight.metadata_port", 9100));
  cfg.video_enabled = raw.bool_or("output.video_enabled", true);
  cfg.save_dir = raw.string_or("output.debug_dir", "");
  cfg.save_every = raw.int_or("output.save_every", 0);
  validate_config(cfg);
  return cfg;
}

std::vector<std::string> load_labels(const fs::path& labels_path) {
  std::ifstream in(labels_path);
  if (!in.good())
    throw std::runtime_error("labels file does not exist: " + labels_path.string());
  std::vector<std::string> labels;
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty())
      labels.push_back(line);
  }
  if (labels.empty())
    throw std::runtime_error("labels file is empty: " + labels_path.string());
  return labels;
}

// ── detections / metadata ────────────────────────────────────────────────────

bool extract_bbox_payload(const simaai::neat::Sample& sample, std::vector<std::uint8_t>& payload,
                          std::string& err) {
  if (sample.kind == simaai::neat::SampleKind::Bundle) {
    for (const auto& field : sample.fields) {
      if (extract_bbox_payload(field, payload, err))
        return true;
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


std::string json_escape(const std::string& value) {
  std::string out;
  out.reserve(value.size());
  for (char c : value) {
    if (c == '"' || c == '\\')
      out.push_back('\\');
    out.push_back(c);
  }
  return out;
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

// Compact object-detection metadata: only the fields Insight's overlay reads
// (label, confidence, bbox), plus the stream count. The "id" is
// dropped, confidence is rounded to 2 decimals, and bboxes are integer pixels, so
// ~2x more detections fit in a single UDP datagram (Insight drops fragmented
// metadata packets larger than the ~1500-byte MTU, since it does not reassemble).
std::string metadata_payload(const std::vector<objdet::Box>& boxes,
                             const std::vector<std::string>& labels, int frame_w, int frame_h,
                             int stream_count, const std::string& stream_id) {
  const auto mb = build_metadata_boxes(boxes, labels, frame_w, frame_h);
  std::string s = "{\"objects\":[";
  for (std::size_t i = 0; i < mb.size(); ++i) {
    char conf[16];
    std::snprintf(conf, sizeof(conf), "%.2f", mb[i].confidence);
    if (i)
      s.push_back(',');
    s += "{\"label\":\"" + json_escape(mb[i].label) + "\",\"confidence\":" + conf + ",\"bbox\":[" +
         std::to_string(static_cast<int>(mb[i].x)) + "," +
         std::to_string(static_cast<int>(mb[i].y)) + "," +
         std::to_string(static_cast<int>(mb[i].w)) + "," +
         std::to_string(static_cast<int>(mb[i].h)) + "]}";
  }
  s += "],\"stream_count\":" + std::to_string(stream_count) + ",\"stream_id\":\"" +
       json_escape(stream_id) + "\"}";
  return s;
}

int64_t fallback_timestamp_ms() {
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  return std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
}

// ── per-stream pipeline ──────────────────────────────────────────────────────

struct ProfileWindow {
  bool enabled = false;
  std::string stream_id;
  int interval = 100;
  int frames = 0;
  int boxes = 0;
  double start_ms = 0.0;
  double pull_ms = 0.0;

  void add(double pull, int box_count) {
    if (!enabled)
      return;
    if (frames == 0)
      start_ms = sima_examples::time_ms();
    ++frames;
    boxes += box_count;
    pull_ms += pull;
    if (frames >= interval)
      flush();
  }

  void flush() {
    if (!enabled || frames == 0)
      return;
    const double elapsed = sima_examples::time_ms() - start_ms;
    const double output_fps = elapsed > 0.0 ? frames * 1000.0 / elapsed : 0.0;
    std::cout << "[profile stream=" << stream_id << "] frames=" << frames
              << " output_fps=" << output_fps << " avg_pull_ms=" << pull_ms / frames
              << " avg_boxes=" << static_cast<double>(boxes) / frames << "\n";
    frames = 0;
    boxes = 0;
    start_ms = 0.0;
    pull_ms = 0.0;
  }
};

struct StreamRuntime {
  int channel = 0;
  std::string id;
  std::string url;
  std::vector<std::string> labels;
  int frame_w = 0;
  int frame_h = 0;
  int output_fps = 0;
  std::unique_ptr<simaai::neat::MetadataSender> metadata_sender; // persists across rebuilds
  std::unique_ptr<simaai::neat::Model> model;                    // rebuilt on reconnect
  std::unique_ptr<simaai::neat::Graph> graph;                    // rebuilt on reconnect
  simaai::neat::Run run;                                         // rebuilt on reconnect
  std::string output_name;
  int video_port = 0;
  int processed = 0;
  // False when the lowered graph restamps PTS: an rtp_timestamp that cannot
  // match is worse than none, because Insight stops falling back to arrival
  // order the moment the field is present. Decided in build_pipeline().
  bool emit_rtp_timestamp = true;
  ProfileWindow profile;
};

simaai::neat::nodes::groups::RtspDecodedInputOptions
make_source_options(const AppConfig& cfg, const StreamRuntime& rt) {
  simaai::neat::nodes::groups::RtspDecodedInputOptions opt;
  opt.url = rt.url;
  opt.latency_ms = cfg.latency_ms;
  opt.tcp = cfg.tcp;
  opt.payload_type = 96;
  opt.insert_queue = true;
  opt.out_format = "NV12";
  opt.decoder_name = "decoder";
  opt.decoder_raw_output = true;
  opt.auto_caps_from_stream = true;
  opt.fallback_h264_width = rt.frame_w;
  opt.fallback_h264_height = rt.frame_h;
  // Leave the decoder fps unspecified (-1) rather than pinning it: a fixed dec-fps
  // rejects streams whose real rate differs (high/variable-rate sources, or a
  // mis-probed fps) with "Caps negotiation failed: framerate mismatch".
  // auto_caps_from_stream lets the decoder emit the stream's native rate, and
  // inference.fps still caps the output at the VideoSender.
  opt.fallback_h264_fps = -1;
  opt.output_caps.enable = false;
  return opt;
}

std::unique_ptr<simaai::neat::Model> make_model(const AppConfig& cfg, const std::string& model_path,
                                                int frame_w = 0, int frame_h = 0) {
  simaai::neat::Model::Options model_opt;
  model_opt.preprocess.kind = simaai::neat::InputKind::Image;
  model_opt.preprocess.enable = simaai::neat::AutoFlag::On;
  model_opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::NV12;
  model_opt.preprocess.preset = simaai::neat::NormalizePreset::COCO_YOLO;
  // The preprocess envelope defaults to 1920x1080; anything larger (a 4K source)
  // fails the graph compile with "input width N exceeds configured capacity".
  if (frame_w > 0 && frame_h > 0) {
    model_opt.preprocess.input_max_width = frame_w;
    model_opt.preprocess.input_max_height = frame_h;
  }
  model_opt.decode_type = simaai::neat::BoxDecodeType::YoloV26;
  model_opt.score_threshold = cfg.min_score;
  model_opt.nms_iou_threshold = cfg.nms_iou;
  model_opt.top_k = cfg.max_detections;
  return std::make_unique<simaai::neat::Model>(model_path, model_opt);
}

// One pipeline build attempt. Assumes rt.run/graph/model have been cleared.
void build_pipeline_once(const AppConfig& cfg, StreamRuntime& rt) {
  rt.model = make_model(cfg, cfg.model_path, rt.frame_w, rt.frame_h);
  rt.graph = std::make_unique<simaai::neat::Graph>();
  auto& graph = *rt.graph;

  const std::string model_name = "model";
  const std::string video_name = "video";
  const std::string debug_frame_name = "debug_frame";
  const std::string detections_name = "detections";
  const std::string debug_output_name = "debug_output";
  const bool save_debug_frames = !cfg.save_dir.empty() && cfg.save_every > 0;
  rt.output_name = save_debug_frames ? debug_output_name : detections_name;

  std::vector<std::string> outputs = {model_name};
  if (cfg.video_enabled)
    outputs.push_back(video_name);
  if (save_debug_frames)
    outputs.push_back(debug_frame_name);

  auto source = simaai::neat::nodes::groups::RtspDecodedInput(make_source_options(cfg, rt));
  auto branch = simaai::neat::graphs::Branch("source", outputs);

  simaai::neat::GraphLinkOptions live_link_options;
  live_link_options.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
  graph.connect(source, branch);

  if (cfg.video_enabled) {
    auto video_options = simaai::neat::nodes::groups::VideoSenderOptions::H264RtpUdpFromRaw(
        rt.frame_w, rt.frame_h, rt.output_fps);
    video_options.host = cfg.insight_host;
    video_options.channel = rt.channel;
    video_options.video_port_base = cfg.video_port_base;
    video_options.encoder.bitrate_kbps = 1000;
    rt.video_port = video_options.video_port();
    simaai::neat::Graph video_graph(video_name);
    video_graph.connect(simaai::neat::nodes::Input(video_name),
                        simaai::neat::nodes::groups::VideoSender(video_options));
    graph.connect(branch, video_graph, live_link_options);
  }

  simaai::neat::Graph model_graph(model_name);
  model_graph.connect(simaai::neat::nodes::Input(model_name), *rt.model);
  simaai::neat::Graph detections_graph(detections_name);
  detections_graph.add(
      simaai::neat::nodes::Output(detections_name, simaai::neat::OutputOptions::EveryFrame(4)));
  graph.connect(branch, model_graph, live_link_options);
  graph.connect(model_graph, detections_graph);

  if (save_debug_frames) {
    simaai::neat::Graph frames(debug_frame_name);
    frames.add(
        simaai::neat::nodes::Output(debug_frame_name, simaai::neat::OutputOptions::EveryFrame(4)));
    auto debug_join = simaai::neat::graphs::Combine({debug_frame_name, detections_name},
                                                    debug_output_name,
                                                    simaai::neat::CombinePolicy::ByFrame);
    graph.connect(branch, frames, live_link_options);
    graph.connect(frames, debug_join);
    graph.connect(detections_graph, debug_join);
  }

  simaai::neat::RunOptions run_options;
  run_options.preset = simaai::neat::RunPreset::Realtime;
  run_options.queue_depth = 3;
  run_options.overflow_policy = simaai::neat::OverflowPolicy::KeepLatest;
  run_options.output_memory = simaai::neat::OutputMemory::ZeroCopy;
  rt.run = graph.build(run_options);

  // Decide rtp_timestamp from what Core ACTUALLY lowered, not from hope. If a
  // bridging appsrc still restamps PTS, an rtp_timestamp can never match and
  // Insight discards every message silently; omitting the field instead lets the
  // browser fall back to arrival order, so boxes render with small bounded lag.
  std::string backend;
  try {
    backend = graph.describe_backend();
  } catch (...) {  // introspection is best-effort
  }
  const bool restamped = backend.find("do-timestamp=true") != std::string::npos;
  if (cfg.metadata_rtp_timestamp == "on") {
    rt.emit_rtp_timestamp = true;
  } else if (cfg.metadata_rtp_timestamp == "off") {
    rt.emit_rtp_timestamp = false;
  } else {
    rt.emit_rtp_timestamp = !restamped;
  }
  if (!rt.emit_rtp_timestamp) {
    std::cout << "[stream " << rt.id
              << "] metadata rtp_timestamp DISABLED (graph restamps PTS; Insight will"
                 " match by arrival order)\n";
  }
}

// (Re)build the RTSP -> {video, model} graph for one stream.
// The MetadataSender and probed geometry persist; only model/graph/run rebuild.
// Model loads are serialized across streams and retried: concurrent rebuilds
// otherwise race on MLA/RPMsg setup (observed "Unable to load model" under heavy
// concurrent rebuilds). The steady-state per-frame pull loop is unaffected.
void build_pipeline(const AppConfig& cfg, StreamRuntime& rt) {
  constexpr int kBuildAttempts = 3;
  constexpr int kRebuildSettleMs = 800;
  for (int attempt = 1;; ++attempt) {
    try {
      std::lock_guard<std::mutex> build_lock(g_model_build_mutex);
      // Tear down (run -> graph -> model) AND rebuild under the same lock, so two
      // streams never tear down / load MLA pipelines concurrently.
      const bool rebuilding = static_cast<bool>(rt.model);
      if (rebuilding) {
        try {
          rt.run.close();
        } catch (...) {
        }
      }
      rt.run = simaai::neat::Run{};
      rt.graph.reset();
      rt.model.reset();
      // The Neat MLA pipeline tears down asynchronously; give it time to fully
      // release before building the next one, or concurrent teardown/build in the
      // runtime can segfault when several streams rebuild at once.
      if (rebuilding)
        std::this_thread::sleep_for(std::chrono::milliseconds(kRebuildSettleMs));
      build_pipeline_once(cfg, rt);
      return;
    } catch (const std::exception& e) {
      rt.run = simaai::neat::Run{};
      rt.graph.reset();
      rt.model.reset();
      if (attempt >= kBuildAttempts)
        throw;
      std::cerr << "[warn] stream " << rt.id << " pipeline build attempt " << attempt
                << " failed (" << e.what() << "); retrying\n";
      std::this_thread::sleep_for(std::chrono::milliseconds(750));
    }
  }
}

StreamRuntime init_stream_runtime(const AppConfig& cfg, int channel, const StreamSource& source,
                                  const std::vector<std::string>& labels) {
  StreamRuntime rt;
  rt.channel = channel;
  rt.id = source.id;
  rt.url = source.rtsp_url;
  rt.labels = labels;
  rt.profile.enabled = cfg.profile;
  rt.profile.stream_id = source.id;

  sima_examples::RtspStreamInfo probe;
  sima_examples::RtspProbeOptions probe_options;
  probe_options.payload_type = 96;
  probe_options.latency_ms = cfg.latency_ms;
  probe_options.rtsp_tcp = cfg.tcp;
  probe_options.debug = cfg.profile;
  (void)sima_examples::probe_rtsp_stream_info(source.rtsp_url, probe_options, probe);
  sima_examples::require(probe.width > 0 && probe.height > 0,
                         "failed to probe RTSP frame dimensions for " + source.id);
  rt.frame_w = probe.width;
  rt.frame_h = probe.height;
  // RTSP fps is not always probeable (high/variable-rate sources report 0 or a
  // bogus value); fall back to inference.fps if set, else 30, rather than failing.
  const int probed_fps = probe.fps > 0 ? probe.fps : 30;
  rt.output_fps = cfg.fps > 0 ? cfg.fps : probed_fps;

  simaai::neat::MetadataSenderOptions metadata_options;
  metadata_options.host = cfg.insight_host;
  metadata_options.channel = channel;
  metadata_options.metadata_port_base = cfg.metadata_port_base;
  std::string metadata_err;
  rt.metadata_sender =
      std::make_unique<simaai::neat::MetadataSender>(metadata_options, &metadata_err);
  sima_examples::require(rt.metadata_sender->ok(), metadata_err);

  build_pipeline(cfg, rt);
  // The `native=` label is not decoration: pipelines/ parses this banner to show
  // each stream's resolution, and an unlabelled WxH cannot be told apart from
  // anything else on the line. src/python/adaptive_app.py prints the same shape.
  std::cout << "[stream " << source.id << "] channel=" << channel << " rtsp=" << source.rtsp_url
            << " native=" << rt.frame_w << "x" << rt.frame_h << "@" << rt.output_fps
            << " video=" << (cfg.video_enabled ? std::to_string(rt.video_port) : "disabled")
            << " metadata=" << rt.metadata_sender->metadata_port() << "\n";
  return rt;
}


// Insight's 32-bit RTP clock, at the 90 kHz rate an H.264 stream is stamped on.
int64_t rtp_timestamp_from_pts_ns(int64_t pts_ns) {
  if (pts_ns < 0) {
    return 0;
  }
  // Split seconds from the remainder before scaling. The direct pts_ns * 90000
  // overflows int64 at ~28 h of stream time, which a soak run reaches - and a
  // wrong rtp_timestamp is worse than none, since Insight then matches nothing.
  // Python gets this free from arbitrary-precision ints.
  const int64_t seconds = pts_ns / 1000000000;
  const int64_t remainder = pts_ns % 1000000000;
  return (seconds * 90000 + (remainder * 90000) / 1000000000) & 0xFFFFFFFF;
}

void send_metadata(StreamRuntime& rt, const simaai::neat::Sample& sample,
                   const std::vector<objdet::Box>& boxes, int stream_count) {
  // Publish detections on the frame's own source clock.
  //
  // Insight matches metadata to a held video frame by an EXACT 32-bit
  // `rtp_timestamp` key, with no arrival-order fallback once the field is
  // present - so a payload without it renders by arrival order, with small
  // bounded lag, while one with a WRONG value never renders a box at all. The
  // convenience send_metadata() API cannot carry that field (MetadataSender.h has
  // no such parameter), which is why this builds the full envelope and ships it
  // with send_raw_json, exactly as src/python/adaptive_app.py does.
  const std::string data_json =
      metadata_payload(boxes, rt.labels, rt.frame_w, rt.frame_h, stream_count, rt.id);
  const int64_t pts_ns = sample.pts_ns;
  const int64_t timestamp_ms = pts_ns >= 0 ? pts_ns / 1000000 : fallback_timestamp_ms();

  std::string payload = "{\"type\":\"object-detection\",\"data\":";
  payload += data_json;
  payload += ",\"timestamp\":" + std::to_string(timestamp_ms);
  payload += ",\"frame_id\":\"";
  if (sample.frame_id >= 0) {
    payload += std::to_string(sample.frame_id);
  }
  payload += "\",\"stream_id\":\"" + json_escape(rt.id) + "\"";
  payload += ",\"stream_index\":" + std::to_string(rt.channel);
  payload += ",\"pts_ns\":" + std::to_string(pts_ns);
  payload += ",\"dts_ns\":" + std::to_string(sample.dts_ns);
  payload += ",\"duration_ns\":" + std::to_string(sample.duration_ns);
  payload += ",\"input_seq\":" + std::to_string(sample.input_seq);
  if (pts_ns >= 0 && rt.emit_rtp_timestamp) {
    payload += ",\"rtp_timestamp\":" + std::to_string(rtp_timestamp_from_pts_ns(pts_ns));
  }
  payload += "}";

  std::string err;
  if (!rt.metadata_sender->send_raw_json(payload, &err)) {
    std::cerr << "[warn] stream " << rt.id << " metadata send failed: " << err << "\n";
  }
}

void maybe_save_debug_frame(const AppConfig& cfg, const StreamRuntime& rt,
                            const simaai::neat::Sample& sample, const std::vector<objdet::Box>& boxes,
                            int stream_count) {
  if (cfg.save_dir.empty() || cfg.save_every <= 0 || rt.processed % cfg.save_every != 0)
    return;
  const auto tensors = simaai::neat::tensors_from_sample(sample, false);
  if (tensors.empty())
    return;
  cv::Mat bgr;
  std::string err;
  if (!sima_examples::nv12_to_bgr(tensors.front(), bgr, err)) {
    std::cerr << "[warn] failed to prepare output frame: " << err << "\n";
    return;
  }
  objdet::draw_boxes(bgr, boxes, static_cast<float>(cfg.min_score), cv::Scalar(0, 255, 0), "");
  const std::string banner = rt.id +
                             "  streams=" + std::to_string(stream_count);
  cv::putText(bgr, banner, cv::Point(12, 28), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 4);
  cv::putText(bgr, banner, cv::Point(12, 28), cv::FONT_HERSHEY_SIMPLEX, 0.7,
              cv::Scalar(255, 255, 255), 2);
  const auto out_path = cfg.save_dir / ("stream_" + std::to_string(rt.channel) + "_frame_" +
                                        std::to_string(rt.processed) + ".jpg");
  if (!cv::imwrite(out_path.string(), bgr))
    std::cerr << "[warn] failed to write output frame: " << out_path.string() << "\n";
}

// ── stream registry (runtime add/remove) ─────────────────────────────────────

struct ManagedStream {
  std::string id;
  std::string url;
  int channel = 0;
  std::atomic_bool stop{false};
  std::atomic_bool finished{false};
  std::thread thread;
};

// Describes the exception being handled; only valid inside a catch block.
std::string describe_current_exception() {
  try {
    throw;
  } catch (const std::exception& e) {
    return e.what();
  } catch (...) {
    return "unknown error";
  }
}

// A stream failure is isolated to that stream: the worker retires its own slot
// and the rest keep running. stop_ means "the whole app is going down" (shutdown
// or signal), never "one stream broke" - a bad source added from the panel, or a
// bad one left in the config from a previous run, must not take the healthy
// streams with it.
class StreamManager {
public:
  StreamManager(const AppConfig& cfg, std::vector<std::string> labels)
      : cfg_(cfg), labels_(std::move(labels)) {
    for (int c = 0; c < cfg.max_streams; ++c)
      free_channels_.push_back(c);
  }

  int active_count() {
    std::lock_guard<std::mutex> lock(mtx_);
    return static_cast<int>(streams_.size());
  }

  bool stopped() const { return stop_.load(); }

  bool add(const StreamSource& source) {
    ManagedStream* raw = nullptr;
    {
      std::lock_guard<std::mutex> lock(mtx_);
      if (streams_.count(source.id))
        return false;
      if (static_cast<int>(streams_.size()) >= cfg_.max_streams || free_channels_.empty()) {
        std::cerr << "[warn] cannot add stream " << source.id << ": at max_streams\n";
        return false;
      }
      const int channel = free_channels_.front();
      free_channels_.erase(free_channels_.begin());
      auto managed = std::make_unique<ManagedStream>();
      managed->id = source.id;
      managed->url = source.rtsp_url;
      managed->channel = channel;
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
      auto it = streams_.find(id);
      if (it == streams_.end())
        return;
      managed = std::move(it->second);
      streams_.erase(it);
    }
    managed->stop.store(true);
    if (managed->thread.joinable())
      managed->thread.join();
    {
      std::lock_guard<std::mutex> lock(mtx_);
      free_channels_.push_back(managed->channel);
      std::sort(free_channels_.begin(), free_channels_.end());
    }
    std::cout << "[stream " << id << "] removed (channel " << managed->channel << " released)\n";
  }

  void apply_sources(const std::vector<StreamSource>& sources) {
    std::set<std::string> current;
    {
      std::lock_guard<std::mutex> lock(mtx_);
      for (const auto& kv : streams_)
        current.insert(kv.first);
    }
    std::map<std::string, StreamSource> by_id;
    for (const auto& s : sources)
      by_id[s.id] = s;
    for (const auto& id : current) {
      if (!by_id.count(id))
        remove(id);
    }
    {
      // A source dropped from the config stops counting as failed, so deleting
      // the bad stream in the panel clears the failed state.
      std::lock_guard<std::mutex> lock(mtx_);
      for (auto it = failed_.begin(); it != failed_.end();) {
        if (by_id.count(it->first))
          ++it;
        else
          it = failed_.erase(it);
      }
    }
    for (const auto& kv : by_id) {
      if (!current.count(kv.first))
        add(kv.second);
    }
  }

  // True when every stream is gone and each one left because it failed.
  bool all_failed() {
    std::lock_guard<std::mutex> lock(mtx_);
    return !failed_.empty() && streams_.empty();
  }

  std::map<std::string, std::string> failed() {
    std::lock_guard<std::mutex> lock(mtx_);
    return failed_;
  }

  bool all_done() {
    std::lock_guard<std::mutex> lock(mtx_);
    if (streams_.empty())
      return false;
    for (const auto& kv : streams_) {
      if (!kv.second->finished.load())
        return false;
    }
    return true;
  }

  void request_stop() { stop_.store(true); }

  void shutdown() {
    stop_.store(true);
    std::vector<std::unique_ptr<ManagedStream>> drained;
    {
      std::lock_guard<std::mutex> lock(mtx_);
      for (auto& kv : streams_)
        drained.push_back(std::move(kv.second));
      streams_.clear();
      for (auto& m : retired_)
        drained.push_back(std::move(m));
      retired_.clear();
    }
    for (auto& m : drained)
      m->stop.store(true);
    for (auto& m : drained) {
      if (m->thread.joinable())
        m->thread.join();
    }
  }

  void rethrow_if_error() {
    std::lock_guard<std::mutex> lock(error_mtx_);
    if (first_error_)
      std::rethrow_exception(first_error_);
  }

private:
  // Release a failed stream's slot from inside that stream's own worker. Cannot
  // go through remove(): that joins the thread we are running on, and dropping
  // the unique_ptr here would destroy that still-running thread. The entry moves
  // to retired_ instead, where shutdown() joins and frees it. The identity check
  // keeps a late failure from evicting a same-id stream that has since been
  // re-added.
  void retire(ManagedStream* managed, const std::string& reason) {
    {
      std::lock_guard<std::mutex> lock(mtx_);
      auto it = streams_.find(managed->id);
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

  // A source-side close (RTSP drop, EOS at a loop boundary, RTCP timeout) is not
  // fatal: rebuild this stream and rejoin, with bounded
  // exponential backoff. Returns true once reconnected; false only if we should
  // stop the stream (shutdown, stream removed, or reconnect budget exhausted).
  bool reconnect_stream(ManagedStream* managed, StreamRuntime& rt) {
    constexpr int kMaxReconnects = 30;
    for (int attempt = 1; attempt <= kMaxReconnects; ++attempt) {
      if (managed->stop.load() || stop_.load() || g_stop_requested != 0)
        return false;
      const int backoff_ms = std::min(500 * attempt, 5000);
      std::cerr << "[warn] stream " << managed->id << " source closed; reconnecting (attempt "
                << attempt << "/" << kMaxReconnects << " in " << backoff_ms << "ms)\n";
      std::this_thread::sleep_for(std::chrono::milliseconds(backoff_ms));
      if (managed->stop.load() || stop_.load() || g_stop_requested != 0)
        return false;
      try {
        build_pipeline(cfg_, rt);
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
    StreamRuntime rt;
    bool built = false;
    // Set when this stream ends because it broke rather than because it was asked
    // to stop or finished its frame budget; drives retire() below.
    std::string failure;
    int transient_recoveries = 0;
    // Desync identical streams so a fleet-wide recovery rolls through rather
    // than every stream rebuilding in the same instant.
    const double stagger_ms = static_cast<double>(managed->channel) * 350.0;
    constexpr int kMaxTransientRecoveries = 10;
    // Throttle: inference.fps caps the rate at which we PROCESS + emit (metadata,
    // video). <=0 means uncapped (run at the pipeline's max). The
    // runtime already drops stale decoded frames (OverflowPolicy::KeepLatest), so
    // this bounds MLA-output/network/CPU work, not the hardware decode rate (which
    // the source sets). Give an fps below the sustainable rate -> runs at exactly
    // that; give 0 or a high value -> runs at the max the pipeline sustains.
    const double min_process_interval_ms =
        cfg_.fps > 0 ? 1000.0 / static_cast<double>(cfg_.fps) : 0.0;
    double last_process_ms = -1e12;
    // Desync identical streams so a fleet-wide recovery rolls through
    // instead of storming: each stream waits a per-channel offset before committing.
    // Retry transient MLA blips a few times before paying for a full rebuild -- an
    // immediate rebuild is what turns one contention blip into a rebuild storm.
    int consecutive_transient = 0;
    constexpr int kTransientRetries = 8;
    try {
      rt = init_stream_runtime(cfg_, managed->channel, {managed->id, managed->url}, labels_);
      built = true;
      while (!managed->stop.load() && !stop_.load() && g_stop_requested == 0 &&
             (cfg_.frames <= 0 || rt.processed < cfg_.frames)) {
        // Throttle: pace processing to inference.fps. We keep pulling (KeepLatest
        // always has the freshest frame ready) but skip frames that arrive faster
        // than the target, capping metadata and video work per stream.
        if (min_process_interval_ms > 0.0) {
          const double due = last_process_ms + min_process_interval_ms;
          const double now = sima_examples::time_ms();
          if (now < due) {
            std::this_thread::sleep_for(
                std::chrono::milliseconds(std::min<int>(static_cast<int>(due - now) + 1, 15)));
            continue;
          }
        }
        simaai::neat::Sample sample;
        simaai::neat::PullError pull_error;
        const double pull_start = sima_examples::time_ms();
        const auto status = rt.run.pull(rt.output_name, 50, sample, &pull_error);
        const double pull_end = sima_examples::time_ms();
        if (status == simaai::neat::PullStatus::Timeout)
          continue;
        if (status == simaai::neat::PullStatus::Closed) {
          // The source ended. If we're shutting down or this stream was removed,
          // exit cleanly; otherwise it's a source drop -- reconnect and rejoin.
          if (managed->stop.load() || stop_.load() || g_stop_requested != 0)
            break;
          if (!reconnect_stream(managed, rt)) {
            if (!managed->stop.load() && !stop_.load() && g_stop_requested == 0)
              failure = "source closed and the reconnect budget ran out";
            break;
          }
          last_process_ms = -1e12;
          consecutive_transient = 0;
          continue;
        }
        if (status != simaai::neat::PullStatus::Ok) {
          // The MLA model load is deferred to run start, so an over-churned
          // rebuild can surface here as a transient caps/load error.
          const std::string& msg = pull_error.message;
          const bool transient = msg.find("Unable to load model") != std::string::npos ||
                                 msg.find("misconfig.caps") != std::string::npos;
          const bool stopping =
              managed->stop.load() || stop_.load() || g_stop_requested != 0;
          if (transient && !stopping) {
            // Retry the pull a few times first: most contention blips clear on
            // their own, and an immediate rebuild is what cascades into a storm.
            if (++consecutive_transient < kTransientRetries) {
              std::this_thread::sleep_for(std::chrono::milliseconds(25));
              continue;
            }
            // Persistent: rebuild, staggered by channel so identical streams don't
            // all rebuild in the same instant.
            if (transient_recoveries < kMaxTransientRecoveries) {
              ++transient_recoveries;
              consecutive_transient = 0;
              std::cerr << "[warn] stream " << managed->id
                        << " persistent MLA error; rebuilding pipeline (recovery "
                        << transient_recoveries << "/" << kMaxTransientRecoveries << ")\n";
              std::this_thread::sleep_for(
                  std::chrono::milliseconds(300 + static_cast<int>(stagger_ms)));
              build_pipeline(cfg_, rt);
              last_process_ms = -1e12;
              continue;
            }
          }
          throw std::runtime_error("stream " + managed->id +
                                   " failed to pull detections: " + msg);
        }

        std::vector<std::uint8_t> payload;
        std::string err;
        if (!extract_bbox_payload(sample, payload, err))
          throw std::runtime_error("stream " + managed->id + " bbox extract failed: " + err);
        const auto boxes = objdet::parse_boxes_strict(payload, rt.frame_w, rt.frame_h,
                                                      cfg_.max_detections, false);
        ++rt.processed;
        transient_recoveries = 0; // a successful pull clears the transient-error budget
        consecutive_transient = 0;
        last_process_ms = sima_examples::time_ms();
        if (rt.processed <= cfg_.warmup_frames)
          continue;

        send_metadata(rt, sample, boxes, active_count());
        maybe_save_debug_frame(cfg_, rt, sample, boxes, active_count());
        rt.profile.add(pull_end - pull_start, static_cast<int>(boxes.size()));
      }
    } catch (...) {
      // Deliberately NOT stop_: one stream failing to build or dying mid-run must
      // not tear down the streams around it.
      {
        std::lock_guard<std::mutex> lock(error_mtx_);
        if (!first_error_)
          first_error_ = std::current_exception();
      }
      failure = describe_current_exception();
    }
    if (built) {
      rt.profile.flush();
      rt.run.close();
      std::cout << "[stream " << managed->id << "] processed=" << rt.processed << "\n";
    }
    managed->finished.store(true);
    if (!failure.empty())
      retire(managed, failure);
  }

  const AppConfig& cfg_;
  std::vector<std::string> labels_;
  std::mutex mtx_;
  std::map<std::string, std::unique_ptr<ManagedStream>> streams_;
  std::vector<int> free_channels_;
  std::atomic_bool stop_{false};
  // Failed streams, kept alive until shutdown(): each one owns the thread that
  // retired it, so it cannot be destroyed from inside that thread.
  std::vector<std::unique_ptr<ManagedStream>> retired_;
  // id -> reason, for streams that died on their own. Kept after the slot is
  // released so the run can report them and so all_failed() can tell "every
  // stream broke" apart from "the panel is empty".
  std::map<std::string, std::string> failed_;
  std::mutex error_mtx_;
  std::exception_ptr first_error_;
};

void interruptible_sleep(double seconds, const StreamManager& manager) {
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(static_cast<int>(seconds * 1000.0));
  while (std::chrono::steady_clock::now() < deadline) {
    if (g_stop_requested != 0 || manager.stopped())
      return;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

void run_app(const AppConfig& cfg) {
  g_stop_requested = 0;
  auto previous_sigint = std::signal(SIGINT, request_stop);
  // SIGTERM is the NORMAL stop signal here: every pipelines/ panel and CLI
  // `down` sends it (see stop_app/stop_group). Without a handler the default
  // disposition terminates the process outright, so the teardown below never
  // runs and the decoder/CVU pools it would have released stay allocated in the
  // reserved region - the exact failure stop_app's docstring warns about, while
  // the caller sees the PID vanish and reports a clean stop.
  auto previous_sigterm = std::signal(SIGTERM, request_stop);
  if (cfg.profile) {
    setenv("SIMA_GST_ELEMENT_TIMINGS", "1", 0);
    setenv("SIMA_GST_FLOW_DEBUG", "1", 0);
    setenv("SIMA_GST_BOUNDARY_PROBES", "1", 0);
  }
  if (!cfg.save_dir.empty())
    fs::create_directories(cfg.save_dir);

  const auto labels = load_labels(cfg.labels_path);
  StreamManager manager(cfg, labels);

  std::error_code ec;
  auto last_write = fs::last_write_time(cfg.config_path, ec);

  for (const auto& source : cfg.sources)
    manager.add(source);

  while (!manager.stopped() && g_stop_requested == 0) {
    if (manager.all_done())
      break;
    // A batch run (--frames N) has nothing left to wait for once every stream has
    // failed. A live run keeps going: the panel is allowed to sit at zero streams
    // until the next config edit adds one.
    if (cfg.frames > 0 && manager.all_failed())
      break;
    interruptible_sleep(cfg.config_watch_seconds, manager);
    const auto now = fs::last_write_time(cfg.config_path, ec);
    if (ec)
      continue;
    if (now != last_write) {
      last_write = now;
      try {
        const auto sources = parse_stream_sources(cfg.config_path);
        // Same count check startup applies (see validate_config). Without it a
        // live edit was PARTIALLY applied - the manager started what fitted and
        // warned per stream about the rest - instead of the whole reload being
        // ignored, which is the documented behaviour for an invalid edit. The
        // channel count is fixed at startup, so max_streams cannot be raised
        // by a live edit either.
        if (static_cast<int>(sources.size()) > cfg.max_streams) {
          throw std::runtime_error(
              "streams count (" + std::to_string(sources.size()) + ") exceeds the " +
              std::to_string(cfg.max_streams) + " channels this run started with");
        }
        std::cout << "[config] reload: " << sources.size() << " stream(s)\n";
        manager.apply_sources(sources);
      } catch (const std::exception& e) {
        std::cerr << "[warn] ignoring invalid config reload: " << e.what() << "\n";
      }
    }
  }

  // Sampled before shutdown(), which drains streams_ and would make every run
  // look like a total failure.
  const bool nothing_survived = manager.all_failed();
  manager.shutdown();
  std::signal(SIGINT, previous_sigint);
  std::signal(SIGTERM, previous_sigterm);

  for (const auto& kv : manager.failed())
    std::cerr << "[warn] stream " << kv.first << " did not run: " << kv.second << "\n";

  // Individual stream failures are reported, not fatal - they are expected on a
  // live panel. The run only fails when no stream survived at all.
  if (nothing_survived)
    manager.rethrow_if_error();
}

// ── CLI ──────────────────────────────────────────────────────────────────────

CliOptions parse_args(int argc, char** argv) {
  CliOptions options;
  options.config_path = sima_examples::default_config_path(SIMANEAT_APPS_EXAMPLE_SOURCE_DIR);
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--config") {
      if (i + 1 >= argc)
        throw std::runtime_error("--config requires a path");
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

} // namespace

int run(int argc, char** argv) {
  try {
    const CliOptions cli = parse_args(argc, argv);
    if (!fs::exists(cli.config_path)) {
      std::cerr << "Error: config file not found: " << cli.config_path << "\n";
      return 2;
    }
    const AppConfig cfg = load_app_config(cli.config_path);
    if (cli.validate_config_only) {
      std::cout << "Config validated: " << cli.config_path << " (streams=" << cfg.sources.size()
                << ")\n";
      return 0;
    }
    run_app(cfg);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[ERR] " << e.what() << "\n";
    return 1;
  }
}

}  // namespace adaptive_app
