// Fused multi-stream detector: ONE GRAPH, ONE SHARED DETECTOR (--mode fused).
//
// H.264 only, matching src/python/fused_app.py - both languages must expose the
// same behaviour for the pipelines toggle to mean anything. (Upstream's newer
// copy of this program gained H.265 input; that needs Core APIs the board does
// not have, and nothing here streams H.265.)
//
// Definitions live in a header because the Apps build compiles ONLY main.cpp
// for an example, so the whole program is one translation unit. The named
// namespace keeps these helpers from colliding with the other mode's.
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
#include <cctype>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace fused_app {

namespace fs = std::filesystem;

namespace {

volatile std::sig_atomic_t g_stop_requested = 0;

void request_stop(int) {
  g_stop_requested = 1;
}

// Set when the shared detection output closes without anyone asking it to. A
// continuous run (frames == 0) that loses its output produces no further
// metadata at all, so finishing normally would hand supervisors a zero exit for
// a detector that had silently stopped detecting.
bool g_output_closed_unexpectedly = false;

// Decoder profile shared by the struct defaults and load_app_config()'s
// fallbacks, so the two cannot drift apart.
constexpr int kDefaultDecoderBuffers = 8;
constexpr int kDefaultDecoderInputBuffers = 2;
constexpr const char* kDefaultDecoderTuning = "auto";

struct AppConfig {
  std::string model_path;
  fs::path labels_path;
  std::vector<std::string> rtsp_urls;
  int latency_ms = 100;
  bool tcp = true;
  // Decoder pool sizing, tuned by the pipeline config for the advertised high
  // stream counts. The untuned element defaults exhaust the decoder pool well
  // before then.
  // 8/auto matches src/python/fused_app.py and what pipelines/ generates.
  // 4 + throughput-low-latency (which turns memory_opt ON) starves the decoder
  // output pool under network jitter and produces stutter/freezes.
  int decoder_buffers = kDefaultDecoderBuffers;
  int decoder_input_buffers = kDefaultDecoderInputBuffers;
  std::string decoder_tuning = kDefaultDecoderTuning;
  // DECODER-side fps cap for admission, not exposed via config (matches
  // src/python/fused_app.py). A source's native rate is declared to the
  // decoder for capacity admission; a very high rate (e.g. 500 fps) x many
  // streams exceeds the decoder core and the whole graph is REJECTED before it
  // starts ("decoder processing capacity unavailable"). See build_decode_graph.
  int decoder_fps_cap = 30;
  int frames = 0;
  int fps = 0;
  int max_inflight_per_stream = 4;
  int max_inflight_total = 16;
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

struct CliOptions {
  fs::path config_path;
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
  double metadata_send_ms = 0.0;

  void add(double detection_pull, double metadata_send, int box_count) {
    if (!enabled)
      return;
    if (frames == 0)
      start_ms = sima_examples::time_ms();
    ++frames;
    boxes += box_count;
    detection_pull_ms += detection_pull;
    metadata_send_ms += metadata_send;
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
              << " avg_metadata_send_ms=" << avg(metadata_send_ms)
              << " avg_boxes=" << static_cast<double>(boxes) / static_cast<double>(frames) << "\n";
    frames = 0;
    boxes = 0;
    start_ms = 0.0;
    detection_pull_ms = 0.0;
    metadata_send_ms = 0.0;
  }
};

struct StreamRuntime {
  int index = 0;
  std::string url;
  simaai::neat::nodes::groups::RtspDecodedInputOptions source_options;
  std::unique_ptr<simaai::neat::MetadataSender> metadata_sender;
  std::vector<std::string> labels;
  ProfileWindow profile;
  // Decoded frames awaiting their detection, keyed by frame id. See
  // drain_debug_frames(): the two branches are queued independently, so the
  // newest decoded frame is not the one the next detection describes. Ordered,
  // so evicting the oldest and reaching the newest are both cheap.
  std::map<int64_t, cv::Mat> debug_frames;
  int frame_w = 0;
  int frame_h = 0;
  int det_w = 0;
  int det_h = 0;
  int output_fps = 0;
  int video_port = 0;
  int processed = 0;
  bool closed = false;
  bool debug_pairing_warned = false;
  // Wall-clock throttle state for inference.fps. See process_output_sample():
  // this used to be set from the probed rate and then never consulted again,
  // so the cap only ever changed the startup banner.
  double last_process_ms = -1e12;
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
  sima_examples::require(!cfg.labels_path.empty(), "model.labels must be set");
  sima_examples::require(!cfg.rtsp_urls.empty(), "streams must be set");
  // The original limit was a development-phase scope guard ("this phase supports
  // up to four streams"), not a hardware one: this app's architecture - one fused
  // graph, one shared detector, encoded passthrough - is the same one the
  // high-density example uses for 16/24/48 streams. Raised so the real ceiling is
  // decided by measured decoder/pool capacity instead of a placeholder, and so
  // this matches src/python/fused_app.py, which made the same change.
  sima_examples::require(cfg.rtsp_urls.size() <= 64, "streams must not exceed 64");
  sima_examples::require(!cfg.insight_host.empty(), "output.insight.host must be set");
  sima_examples::require(cfg.latency_ms >= 0, "input.latency_ms must be >= 0");
  sima_examples::require(cfg.frames >= 0, "inference.frames must be >= 0");
  sima_examples::require(cfg.fps >= 0, "inference.fps must be >= 0");
  sima_examples::require(cfg.max_inflight_per_stream == -1 || cfg.max_inflight_per_stream > 0,
                         "inference.max_inflight_per_stream must be -1 or > 0");
  sima_examples::require(cfg.max_inflight_total == -1 || cfg.max_inflight_total > 0,
                         "inference.max_inflight_total must be -1 or > 0");
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
  const auto default_labels =
      fs::path(SIMANEAT_APPS_EXAMPLE_SOURCE_DIR).parent_path() / "common" / "coco_label.txt";

  AppConfig cfg;
  cfg.model_path = raw.string_or("model.path", "");
  cfg.labels_path = raw.string_or("model.labels", default_labels.string());
  cfg.rtsp_urls = parse_streams(config_path);
  cfg.tcp = raw.bool_or("input.tcp", true);
  cfg.latency_ms = raw.int_or("input.latency_ms", 100);
  cfg.decoder_buffers = raw.int_or("input.decoder_buffers", kDefaultDecoderBuffers);
  cfg.decoder_input_buffers =
      raw.int_or("input.decoder_input_buffers", kDefaultDecoderInputBuffers);
  cfg.decoder_tuning = raw.string_or("input.decoder_tuning", kDefaultDecoderTuning);
  cfg.frames = raw.int_or("inference.frames", 0);
  cfg.fps = raw.int_or("inference.fps", 0);
  cfg.max_inflight_per_stream = raw.int_or("inference.max_inflight_per_stream", 4);
  cfg.max_inflight_total = raw.int_or("inference.max_inflight_total", 16);
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

// Largest upscale the hardware scaler will actually perform. Past this it emits
// nothing at all - silently, with no error - so the shared detector geometry is
// clamped against it (see pick_detector_geometry). Measured on Modalix: 2x
// works, 3x does not.
constexpr int kMaxUpscale = 2;
// The detector's own input size; clamping below this would throw away detail
// the model is going to use anyway.
constexpr int kModelInput = 640;

struct ProbedStream {
  int width = 0;
  int height = 0;
  int fps = 0;
};

ProbedStream probe_stream(const AppConfig& cfg, const std::string& url) {
  sima_examples::RtspStreamInfo probe;
  sima_examples::RtspProbeOptions probe_options;
  probe_options.payload_type = 96;
  probe_options.latency_ms = cfg.latency_ms;
  probe_options.rtsp_tcp = cfg.tcp;
  probe_options.debug = cfg.profile;
  (void)sima_examples::probe_rtsp_stream_info(url, probe_options, probe);
  return ProbedStream{probe.width, probe.height, probe.fps};
}

// The shared detector needs ONE input geometry, so pick the largest native
// probed size and scale the other detector legs to it - largest (rather than
// smallest) avoids discarding detail small objects need. Video delivery is
// unaffected: it is the encoded passthrough, so it stays native per stream.
//
// ...but never ask the scaler for more than kMaxUpscale on the SMALLEST
// source; the hardware scaler silently produces nothing past ~2x upscale, so
// clamping to what it can actually do is what lets a small source share a
// detector with a large one. Downscaling is unconstrained. Keep the clamp at
// or above the model's own input so it never becomes the accuracy bottleneck,
// but never above the largest source either, or a single small stream would
// be upscaled for no reason. NV12 needs even dimensions.
void pick_detector_geometry(const std::vector<ProbedStream>& probes, int& det_w, int& det_h) {
  // Every entry is validated (width/height > 0) by the caller before this
  // runs, so the first probe is a safe seed for both extremes.
  int max_w = probes.front().width, max_h = probes.front().height;
  int min_w = probes.front().width, min_h = probes.front().height;
  for (const auto& p : probes) {
    max_w = std::max(max_w, p.width);
    max_h = std::max(max_h, p.height);
    min_w = std::min(min_w, p.width);
    min_h = std::min(min_h, p.height);
  }
  int capped_w = std::min(max_w, kMaxUpscale * min_w);
  int capped_h = std::min(max_h, kMaxUpscale * min_h);
  capped_w = std::min(max_w, std::max(kModelInput, capped_w));
  capped_h = std::min(max_h, std::max(kModelInput, capped_h));
  det_w = capped_w - (capped_w % 2);
  det_h = capped_h - (capped_h % 2);
}

simaai::neat::nodes::groups::RtspDecodedInputOptions
build_source_options(const AppConfig& cfg, const std::string& url, const ProbedStream& probe,
                     int det_w, int det_h, int& fps_out, int& width_out, int& height_out) {
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
  opt.num_buffers = cfg.decoder_buffers;
  opt.decoder_input_buffers = cfg.decoder_input_buffers;
  opt.decoder_tuning = cfg.decoder_tuning;
  opt.decoder_memory_opt =
      (cfg.decoder_tuning == "low-memory" || cfg.decoder_tuning == "throughput-low-latency");
  if (probe.width > 0 && probe.height > 0) {
    opt.fallback_h264_width = probe.width;
    opt.fallback_h264_height = probe.height;
    width_out = probe.width;
    height_out = probe.height;
  }
  if (probe.fps > 0) {
    opt.source_fps = probe.fps;
    fps_out = probe.fps;
  }
  if (width_out > 0 && height_out > 0 && fps_out > 0) {
    // A single shared model has a single input port, so streams of different
    // native sizes must be normalised to one geometry before it ("input spec
    // mismatch for port 'in'"). Only the detector leg is scaled - the video
    // sent to Insight is the encoded passthrough, which keeps each camera's
    // native resolution. Box coordinates stay correct because detections are
    // rescaled back to the native frame size before being sent (see
    // process_output_sample).
    const int scale_w = det_w > 0 ? det_w : width_out;
    const int scale_h = det_h > 0 ? det_h : height_out;
    if (scale_w != width_out || scale_h != height_out) {
      opt.use_videoscale = true;
    }
    opt.output_caps.enable = true;
    opt.output_caps.format = "NV12";
    opt.output_caps.width = scale_w;
    opt.output_caps.height = scale_h;
    // Do NOT pin the decoded framerate when the source will be admission-capped
    // (see build_decode_graph): a caps framerate that differs from the real
    // stream fails negotiation ("framerate mismatch"). 0 leaves it unconstrained
    // so the decoder emits whatever rate it achieves. Only pin fps for normal
    // (uncapped) sources.
    const bool capped = cfg.decoder_fps_cap > 0 && fps_out > cfg.decoder_fps_cap;
    opt.output_caps.fps = capped ? 0 : fps_out;
    opt.output_caps.memory = simaai::neat::CapsMemory::Any;
  }
  return opt;
}

bool output_caps_enabled(
    const simaai::neat::nodes::groups::RtspDecodedInputOptions::OutputCaps& caps) {
  return caps.enable || caps.width > 0 || caps.height > 0 || caps.fps > 0;
}

simaai::neat::InputOptions h264_decode_input_options() {
  simaai::neat::InputOptions opt;
  opt.payload_type = simaai::neat::PayloadType::Encoded;
  opt.format = simaai::neat::FormatTag::H264;
  opt.memory_policy = simaai::neat::InputMemoryPolicy::Ev74;
  return opt;
}

simaai::neat::InputOptions h264_video_input_options() {
  simaai::neat::InputOptions opt;
  opt.payload_type = simaai::neat::PayloadType::Encoded;
  opt.format = simaai::neat::FormatTag::H264;
  opt.memory_policy = simaai::neat::InputMemoryPolicy::SystemMemory;
  return opt;
}

simaai::neat::Graph
build_encoded_source_graph(const simaai::neat::nodes::groups::RtspDecodedInputOptions& opt) {
  simaai::neat::Graph source("rtsp_encoded_source");

  simaai::neat::nodes::groups::RtspEncodedInputOptions encoded_opt;
  encoded_opt.url = opt.url;
  encoded_opt.codec = simaai::neat::nodes::groups::RtspCodec::H264;
  encoded_opt.latency_ms = opt.latency_ms;
  encoded_opt.tcp = opt.tcp;
  encoded_opt.source_fps = opt.source_fps;
  encoded_opt.fallback_h264_width = opt.fallback_h264_width;
  encoded_opt.fallback_h264_height = opt.fallback_h264_height;
  source.add(simaai::neat::nodes::groups::RtspEncodedInput(encoded_opt));
  return source;
}

simaai::neat::Graph
build_decode_graph(const std::string& input_name,
                   const simaai::neat::nodes::groups::RtspDecodedInputOptions& opt,
                   const AppConfig& cfg) {
  simaai::neat::Graph decode("decode");
  const int dec_w = (opt.h264_width > 0) ? opt.h264_width : opt.fallback_h264_width;
  const int dec_h = (opt.h264_height > 0) ? opt.h264_height : opt.fallback_h264_height;

  simaai::neat::SimaDecodeOptions dec;
  dec.type = simaai::neat::SimaDecodeType::H264;
  dec.sima_allocator_type = opt.sima_allocator_type;
  dec.out_format = opt.out_format;
  dec.decoder_name = opt.decoder_name;
  dec.raw_output = opt.decoder_raw_output;
  dec.next_element = opt.decoder_next_element;
  dec.dec_width = dec_w;
  dec.dec_height = dec_h;
  // High-fps sources: leave dec_fps UNSPECIFIED (-1) rather than pinning a
  // value. Pinning the source rate (e.g. 500) makes admission reject the graph
  // ("processing capacity unavailable"); pinning a lower cap (e.g. 30) fails
  // caps negotiation ("framerate mismatch"). -1 lets admission use its default
  // (~30 fps) AND leaves caps unpinned, so the decoder admits and then processes
  // what it can, dropping the rest. Normal sources keep their exact rate.
  //
  // inference.fps deliberately does NOT feed this. Pinning a rate BELOW the
  // real stream is exactly what the paragraph above rules out, so there is no
  // safe way to make the decoder emit at a requested lower rate here - only the
  // source's true rate, or unpinned. inference.fps is applied after the pull
  // instead; see should_throttle_fps().
  const bool capped = cfg.decoder_fps_cap > 0 && opt.source_fps > cfg.decoder_fps_cap;
  dec.dec_fps = capped ? -1 : opt.source_fps;
  dec.num_buffers = opt.num_buffers;
  dec.input_buffers = opt.decoder_input_buffers;
  dec.decoder_tuning = opt.decoder_tuning;
  dec.memory_opt = opt.decoder_memory_opt;

  decode.connect(simaai::neat::nodes::Input(input_name, h264_decode_input_options()),
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
                         const simaai::neat::nodes::groups::VideoSenderOptions& video_options) {
  simaai::neat::Graph video("video_sender");
  video.connect(simaai::neat::nodes::Input(input_name, h264_video_input_options()),
                simaai::neat::nodes::groups::VideoSender(video_options));
  return video;
}

std::unique_ptr<simaai::neat::Model> build_model(const AppConfig& cfg, int det_w = 0,
                                                 int det_h = 0) {
  simaai::neat::Model::Options model_opt;
  model_opt.preprocess.kind = simaai::neat::InputKind::Image;
  model_opt.preprocess.enable = simaai::neat::AutoFlag::On;
  model_opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::NV12;
  model_opt.preprocess.preset = simaai::neat::NormalizePreset::COCO_YOLO;
  // The preprocess envelope defaults to 1920x1080; anything larger (a 4K source)
  // fails the graph compile with "input width N exceeds configured capacity".
  if (det_w > 0 && det_h > 0) {
    model_opt.preprocess.input_max_width = det_w;
    model_opt.preprocess.input_max_height = det_h;
  }
  model_opt.decode_type = simaai::neat::BoxDecodeType::YoloV26;
  model_opt.score_threshold = cfg.min_score;
  model_opt.nms_iou_threshold = cfg.nms_iou;
  model_opt.top_k = cfg.max_detections;
  return std::make_unique<simaai::neat::Model>(cfg.model_path, model_opt);
}

simaai::neat::RunOptions build_run_options() {
  simaai::neat::RunOptions run_options;
  run_options.preset = simaai::neat::RunPreset::Realtime;
  run_options.queue_depth = 4;
  run_options.overflow_policy = simaai::neat::OverflowPolicy::KeepLatest;
  run_options.output_memory = simaai::neat::OutputMemory::ZeroCopy;
  return run_options;
}

bool save_frames_enabled(const AppConfig& cfg) {
  return !cfg.save_dir.empty() && cfg.save_every > 0;
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

simaai::neat::GraphLinkOptions realtime_link(int stream_index, int queue_depth,
                                             int max_inflight_per_stream = -1,
                                             int max_inflight_total = -1) {
  simaai::neat::GraphLinkOptions link;
  link.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
  link.queue_depth = queue_depth;
  link.max_inflight_per_stream = max_inflight_per_stream;
  link.max_inflight_total = max_inflight_total;
  link.stream_id = stream_id_for(stream_index);
  return link;
}

simaai::neat::Graph build_detector_graph(const AppConfig& cfg, int det_w, int det_h,
                                         std::unique_ptr<simaai::neat::Model>& model) {
  model = build_model(cfg, det_w, det_h);
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

StreamRuntime build_stream_runtime(const AppConfig& cfg, int stream_index, const std::string& url,
                                   const std::vector<std::string>& labels,
                                   const ProbedStream& probe, int det_w, int det_h) {
  StreamRuntime runtime;
  runtime.index = stream_index;
  runtime.url = url;
  const auto source_options =
      build_source_options(cfg, url, probe, det_w, det_h, runtime.output_fps, runtime.frame_w,
                           runtime.frame_h);
  sima_examples::require(runtime.frame_w > 0 && runtime.frame_h > 0,
                         "failed to probe RTSP frame dimensions");
  sima_examples::require(runtime.output_fps > 0, "failed to probe RTSP frame rate");
  runtime.det_w = det_w > 0 ? det_w : runtime.frame_w;
  runtime.det_h = det_h > 0 ? det_h : runtime.frame_h;
  if (cfg.fps > 0) {
    runtime.output_fps = cfg.fps;
  }

  runtime.labels = labels;
  runtime.profile.enabled = cfg.profile;
  runtime.profile.stream_index = stream_index;
  runtime.source_options = source_options;
  if (cfg.video_enabled) {
    auto video_options = simaai::neat::nodes::groups::VideoSenderOptions::H264RtpUdpFromEncoded();
    video_options.host = cfg.insight_host;
    video_options.channel = stream_index;
    video_options.video_port_base = cfg.video_port_base;
    runtime.video_port = video_options.video_port();
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

void connect_stream_graph(AppRuntime& app, const AppConfig& cfg, const StreamRuntime& stream,
                          const simaai::neat::Graph& detector_graph) {
  auto source = build_encoded_source_graph(stream.source_options);
  auto decoder = build_decode_graph("decode_h264", stream.source_options, cfg);

  if (cfg.video_enabled) {
    auto encoded_branch = simaai::neat::graphs::Branch("encoded", {"decode_h264", "video_h264"});
    app.graph.connect(source, encoded_branch);
    app.graph.connect(encoded_branch, decoder, realtime_link(stream.index, 3));

    auto video_options = simaai::neat::nodes::groups::VideoSenderOptions::H264RtpUdpFromEncoded();
    video_options.host = cfg.insight_host;
    video_options.channel = stream.index;
    video_options.video_port_base = cfg.video_port_base;
    video_options.async = true;
    app.graph.connect(encoded_branch, build_video_sender_graph("video_h264", video_options),
                      realtime_link(stream.index, 3));
  } else {
    app.graph.connect(source, decoder, realtime_link(stream.index, 3));
  }

  const bool save_debug_frames = save_frames_enabled(cfg);
  auto decoded_branch =
      save_debug_frames ? simaai::neat::graphs::Branch("decoded", {"detector_frame", "debug_frame"})
                        : simaai::neat::graphs::Branch("decoded", {"detector_frame"});
  app.graph.connect(decoder, decoded_branch);
  app.graph.connect(
      decoded_branch, detector_graph,
      realtime_link(stream.index, 4, cfg.max_inflight_per_stream, cfg.max_inflight_total));
  if (save_debug_frames) {
    app.graph.connect(decoded_branch, build_debug_frame_graph(stream.index),
                      realtime_link(stream.index, 4));
  }
}

void send_metadata(StreamRuntime& stream, const simaai::neat::Sample& sample,
                   const std::vector<objdet::Box>& boxes) {
  const auto metadata_boxes =
      build_metadata_boxes(boxes, stream.labels, stream.frame_w, stream.frame_h);
  const std::string data_json = sima_examples::metadata_boxes_data_json("objects", metadata_boxes);
  const int64_t timestamp_ms = sample.pts_ns >= 0 ? sample.pts_ns / 1'000'000 : -1;
  const std::string frame_id = sample.frame_id >= 0 ? std::to_string(sample.frame_id) : "";
  std::string err;
  if (!stream.metadata_sender->send_metadata("object-detection", data_json, timestamp_ms, frame_id,
                                             &err)) {
    std::cerr << "[warn] stream " << stream.index << " metadata send failed: " << err << "\n";
  }
}

void maybe_save_debug_frame(const AppConfig& cfg, const StreamRuntime& stream, const cv::Mat* frame,
                            const std::vector<objdet::Box>& boxes) {
  if (cfg.save_dir.empty() || cfg.save_every <= 0 || stream.processed % cfg.save_every != 0) {
    return;
  }
  if (frame == nullptr || frame->empty()) {
    return;
  }

  cv::Mat bgr = frame->clone();
  objdet::draw_boxes(bgr, boxes, cfg.min_score, cv::Scalar(0, 255, 0), "");
  const auto out_path = cfg.save_dir / ("stream_" + std::to_string(stream.index) + "_frame_" +
                                        std::to_string(stream.processed) + ".jpg");
  if (!cv::imwrite(out_path.string(), bgr)) {
    std::cerr << "[warn] failed to write output frame: " << out_path.string() << "\n";
  }
}

bool all_streams_done(const std::vector<StreamRuntime>& streams, int frame_limit) {
  if (streams.empty()) {
    return true;
  }
  // A closed stream is done at any frame limit. Returning false outright for
  // frames<=0 meant a continuous run could never finish, so once the shared
  // output closed the loop spun on it forever.
  return std::all_of(streams.begin(), streams.end(), [frame_limit](const StreamRuntime& stream) {
    return stream.closed || (frame_limit > 0 && stream.processed >= frame_limit);
  });
}

// Decoded frames held per stream while their detection catches up. Inference is
// a few frames behind the decoder; past that the detection is never coming.
constexpr std::size_t kDebugFrameCache = 16;

// The decoded frame this detection came from, or nullptr if nothing is held. A
// match also clears everything older: those frames' detections have already gone
// past, so they can never be claimed.
cv::Mat* take_debug_frame(StreamRuntime& stream, int64_t frame_id) {
  auto it = stream.debug_frames.find(frame_id);
  if (it != stream.debug_frames.end()) {
    stream.debug_frames.erase(stream.debug_frames.begin(), it);
    return &stream.debug_frames.begin()->second;
  }
  if (stream.debug_frames.empty()) {
    return nullptr;
  }
  // Nothing carries this id. Rather than stop saving images altogether, fall
  // back to the newest frame - the old behaviour - but say so once, because
  // those images are the approximate pairing this function exists to avoid.
  if (!stream.debug_pairing_warned) {
    stream.debug_pairing_warned = true;
    std::cerr << "[warn] stream " << stream.index
              << " debug frames carry no id matching their detections; saved images"
                 " fall back to the newest frame\n";
  }
  return &stream.debug_frames.rbegin()->second;
}

// True when inference.fps says this stream must wait before processing again.
//
// Scope, stated plainly: this caps the rate at which THIS STREAM parses boxes
// and emits metadata and debug frames. It does NOT reduce MLA work - the shared
// detector has already inferred the sample by the time it is pulled, and the
// decoder cannot be asked for a lower rate (pinning one below the real stream
// fails caps negotiation; see build_decode_graph). Nor does it change the video
// Insight receives, which is the encoded passthrough at the source's own rate.
// Setting output_fps from cfg.fps at build time changed only the startup banner
// and nothing ever consulted it, so before this the setting did nothing at all
// (mirrors src/python/fused_app.py).
bool should_throttle_fps(const AppConfig& cfg, const StreamRuntime& stream, double now) {
  if (cfg.fps <= 0) {
    return false;
  }
  const double min_interval_ms = 1000.0 / static_cast<double>(cfg.fps);
  return now < stream.last_process_ms + min_interval_ms;
}

void process_output_sample(StreamRuntime& stream, const AppConfig& cfg,
                           const simaai::neat::Sample& sample, double detection_pull_ms) {
  if (cfg.frames > 0 && stream.processed >= cfg.frames) {
    return;
  }

  const double now = sima_examples::time_ms();
  if (should_throttle_fps(cfg, stream, now)) {
    return;
  }
  stream.last_process_ms = now;

  std::vector<std::uint8_t> payload;
  std::string err;
  if (!extract_bbox_payload(sample, payload, err)) {
    throw std::runtime_error("stream " + std::to_string(stream.index) +
                             " bbox extract failed: " + err);
  }
  // Detections are emitted in the detector's input geometry, which is shared
  // across streams. Clamp in that space, then scale to this stream's native
  // size so boxes line up with the natively-delivered video. This is a plain
  // per-axis ratio, not a letterbox correction: the output caps in
  // build_source_options pin width/height and leave pixel-aspect-ratio free,
  // so videoscale stretches the pixels instead of adding bars.
  auto boxes =
      objdet::parse_boxes_strict(payload, stream.det_w, stream.det_h, cfg.max_detections, false);
  // The debug frame is tapped off the decoded branch, which build_source_options
  // normalised to the shared detector geometry - so it needs the boxes in THAT
  // space. Copy them before the rescale below, or every non-native leg saves
  // images whose overlays are scaled wrong against a correct frame.
  const bool rescale_to_native =
      stream.det_w != stream.frame_w || stream.det_h != stream.frame_h;
  std::vector<objdet::Box> debug_boxes;
  if (rescale_to_native && save_frames_enabled(cfg)) {
    debug_boxes = boxes;
  }
  if (rescale_to_native) {
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
  const bool warming_up = stream.processed <= cfg.warmup_frames;
  if (!warming_up) {
    const double metadata_start = sima_examples::time_ms();
    send_metadata(stream, sample, boxes);
    const double metadata_end = sima_examples::time_ms();
    if (save_frames_enabled(cfg)) {
      maybe_save_debug_frame(cfg, stream, take_debug_frame(stream, sample.frame_id),
                             rescale_to_native ? debug_boxes : boxes);
    }
    stream.profile.add(detection_pull_ms, metadata_end - metadata_start,
                       static_cast<int>(boxes.size()));
  }
}

std::string debug_frame_output_name(int stream_index) {
  return "debug_frame_" + std::to_string(stream_index);
}

void drain_debug_frames(AppRuntime& app, const AppConfig& cfg) {
  if (!save_frames_enabled(cfg)) {
    return;
  }

  for (auto& stream : app.streams) {
    const std::string output_name = debug_frame_output_name(stream.index);
    for (;;) {
      simaai::neat::Sample sample;
      simaai::neat::PullError pull_error;
      const auto status = app.run.pull(output_name, 0, sample, &pull_error);
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

bool process_run_once(AppRuntime& app, const AppConfig& cfg, const std::string& output_name) {
  constexpr int kPullTimeoutMs = 50;
  drain_debug_frames(app, cfg);
  const double pull_start = sima_examples::time_ms();
  simaai::neat::Sample sample;
  simaai::neat::PullError pull_error;
  const auto status = app.run.pull(output_name, kPullTimeoutMs, sample, &pull_error);
  const double pull_end = sima_examples::time_ms();
  if (status == simaai::neat::PullStatus::Closed) {
    // The shared graph is gone, so no later pull can succeed. Mark every stream
    // closed and let the run loop end: treating this as a timeout left the
    // process alive and spinning at full CPU on an output that was never coming
    // back, reported as running but unable to emit another detection.
    std::cerr << "[warn] detection output closed; ending run\n";
    g_output_closed_unexpectedly = true;
    for (auto& stream : app.streams) {
      stream.closed = true;
    }
    return false;
  }
  if (status == simaai::neat::PullStatus::Timeout) {
    return false;
  }
  if (status != simaai::neat::PullStatus::Ok) {
    throw std::runtime_error("failed to pull " + output_name + ": " + pull_error.message);
  }
  const int stream_index = stream_index_from_sample(sample, static_cast<int>(app.streams.size()));
  process_output_sample(app.streams[static_cast<std::size_t>(stream_index)], cfg, sample,
                        pull_end - pull_start);
  drain_debug_frames(app, cfg);
  return true;
}

void run_app(const AppConfig& cfg) {
  g_stop_requested = 0;
  g_output_closed_unexpectedly = false;
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
  if (save_frames_enabled(cfg)) {
    fs::create_directories(cfg.save_dir);
  }

  const auto labels = load_labels(cfg.labels_path);

  // Probe every source first: the shared detector needs ONE input geometry
  // (see pick_detector_geometry), which has to be known before any stream's
  // source options - or the model's preprocess capacity - can be built.
  std::vector<ProbedStream> probes;
  probes.reserve(cfg.rtsp_urls.size());
  for (const auto& url : cfg.rtsp_urls) {
    probes.push_back(probe_stream(cfg, url));
  }
  for (const auto& p : probes) {
    sima_examples::require(p.width > 0 && p.height > 0, "failed to probe RTSP frame dimensions");
    sima_examples::require(p.fps > 0, "failed to probe RTSP frame rate");
  }

  int det_w = 0;
  int det_h = 0;
  pick_detector_geometry(probes, det_w, det_h);

  AppRuntime app;
  app.streams.reserve(cfg.rtsp_urls.size());
  auto detector_graph = build_detector_graph(cfg, det_w, det_h, app.model);
  auto detections_graph = build_detections_graph();

  for (std::size_t index = 0; index < cfg.rtsp_urls.size(); ++index) {
    app.streams.push_back(build_stream_runtime(cfg, static_cast<int>(index), cfg.rtsp_urls[index],
                                               labels, probes[index], det_w, det_h));
    connect_stream_graph(app, cfg, app.streams.back(), detector_graph);
  }
  app.graph.connect(detector_graph, detections_graph);

  if (cfg.profile) {
    std::cout << "Backend:\n" << app.graph.describe_backend() << "\n";
  }

  app.run = app.graph.build(build_run_options());
  // Printed ONLY after the shared graph builds - see the note in
  // src/python/fused_app.py. pipelines/'s wait_for_streams() requires this line
  // because the per-stream "] rtsp=" banners are emitted before the build.
  std::cout << "[app] graph running: " << app.streams.size() << " stream(s)\n" << std::flush;
  while (g_stop_requested == 0 && !all_streams_done(app.streams, cfg.frames)) {
    (void)process_run_once(app, cfg, "detections");
  }
  // Sampled before the teardown below, which is unconditional.
  const bool closed_unexpectedly =
      g_output_closed_unexpectedly && g_stop_requested == 0 && cfg.frames <= 0;

  app.run.close();

  for (auto& stream : app.streams) {
    stream.profile.flush();
    std::cout << "[stream " << stream.index << "] processed=" << stream.processed << "\n";
  }
  std::signal(SIGINT, previous_sigint);
  std::signal(SIGTERM, previous_sigterm);

  // A continuous run has no natural end, so getting here without a stop request
  // means the output went away underneath us. Exiting 0 would tell a supervisor
  // the experiment succeeded. Thrown after teardown so the graph still closes.
  if (closed_unexpectedly) {
    throw std::runtime_error(
        "detection output closed unexpectedly; the run produced no further metadata");
  }
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
      std::cout << "Config validated: " << cli.config_path << " (streams=" << cfg.rtsp_urls.size()
                << ", max_inflight_per_stream=" << cfg.max_inflight_per_stream
                << ", max_inflight_total=" << cfg.max_inflight_total << ")\n";
      return 0;
    }
    run_app(cfg);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[ERR] " << e.what() << "\n";
    return 1;
  }
}

}  // namespace fused_app
