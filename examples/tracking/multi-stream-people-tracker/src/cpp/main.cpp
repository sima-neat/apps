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
#include "examples/tracking/multi-stream-people-tracker/src/cpp/utils/tracker_api.cpp"
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
#include <optional>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using multi_stream_people_tracker::Detection;
using multi_stream_people_tracker::PeopleTracker;
using multi_stream_people_tracker::TrackedDetection;

namespace {

volatile std::sig_atomic_t g_stop_requested = 0;

void request_stop(int) {
  g_stop_requested = 1;
}

struct AppConfig {
  std::string model_path;
  std::vector<std::string> rtsp_urls;
  bool use_h265 = false;
  int latency_ms = 100;
  bool tcp = true;
  int frames = 0;
  int fps = 0;
  int max_inflight_per_stream = 4;
  int max_inflight_total = 16;
  int person_class_id = 0;
  double min_score = 0.55;
  double nms_iou = 0.60;
  int max_detections = 50;
  bool profile = false;
  int warmup_frames = 30;
  float tracker_iou_threshold = 0.3f;
  int tracker_max_missing = 15;
  std::string insight_host = "127.0.0.1";
  int video_port_base = 9000;
  int metadata_port_base = 9100;
  bool video_enabled = true;
  fs::path save_dir;
  int save_every = 0;
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
  double tracker_ms = 0.0;
  double metadata_send_ms = 0.0;

  void add(double detection_pull, double tracker, double metadata_send, int box_count) {
    if (!enabled)
      return;
    if (frames == 0)
      start_ms = sima_examples::time_ms();
    ++frames;
    boxes += box_count;
    detection_pull_ms += detection_pull;
    tracker_ms += tracker;
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
              << " avg_tracker_ms=" << avg(tracker_ms)
              << " avg_metadata_send_ms=" << avg(metadata_send_ms)
              << " avg_tracks=" << static_cast<double>(boxes) / static_cast<double>(frames) << "\n";
    frames = 0;
    boxes = 0;
    start_ms = 0.0;
    detection_pull_ms = 0.0;
    tracker_ms = 0.0;
    metadata_send_ms = 0.0;
  }
};

struct StreamRuntime {
  int index = 0;
  std::string url;
  simaai::neat::nodes::groups::RtspDecodedInputOptions source_options;
  std::unique_ptr<simaai::neat::MetadataSender> metadata_sender;
  PeopleTracker tracker;
  ProfileWindow profile;
  std::optional<cv::Mat> latest_debug_frame;
  int frame_w = 0;
  int frame_h = 0;
  int output_fps = 0;
  int video_port = 0;
  int processed = 0;
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
  sima_examples::require(!cfg.rtsp_urls.empty(), "streams must be set");
  sima_examples::require(cfg.rtsp_urls.size() <= 4, "this phase supports up to four streams");
  sima_examples::require(!cfg.insight_host.empty(), "output.insight.host must be set");
  sima_examples::require(cfg.latency_ms >= 0, "input.latency_ms must be >= 0");
  sima_examples::require(cfg.frames >= 0, "inference.frames must be >= 0");
  sima_examples::require(cfg.fps >= 0, "inference.fps must be >= 0");
  sima_examples::require(cfg.max_inflight_per_stream == -1 || cfg.max_inflight_per_stream > 0,
                         "inference.max_inflight_per_stream must be -1 or > 0");
  sima_examples::require(cfg.max_inflight_total == -1 || cfg.max_inflight_total > 0,
                         "inference.max_inflight_total must be -1 or > 0");
  sima_examples::require(cfg.person_class_id >= 0, "inference.person_class_id must be >= 0");
  sima_examples::require(cfg.min_score >= 0.0 && cfg.min_score <= 1.0,
                         "inference.min_score must be between 0 and 1");
  sima_examples::require(cfg.nms_iou >= 0.0 && cfg.nms_iou <= 1.0,
                         "inference.nms_iou must be between 0 and 1");
  sima_examples::require(cfg.max_detections > 0, "inference.max_detections must be > 0");
  sima_examples::require(cfg.warmup_frames >= 0, "runtime.warmup_frames must be >= 0");
  sima_examples::require(cfg.tracker_iou_threshold >= 0.0f && cfg.tracker_iou_threshold <= 1.0f,
                         "tracking.iou_threshold must be between 0 and 1");
  sima_examples::require(cfg.tracker_max_missing >= 0, "tracking.max_missing_frames must be >= 0");
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
  cfg.use_h265 = parse_use_h265(raw.string_or("input.codec", "h264"));
  cfg.tcp = raw.bool_or("input.tcp", true);
  cfg.latency_ms = raw.int_or("input.latency_ms", 100);
  cfg.frames = raw.int_or("inference.frames", 0);
  cfg.fps = raw.int_or("inference.fps", 0);
  cfg.max_inflight_per_stream = raw.int_or("inference.max_inflight_per_stream", 4);
  cfg.max_inflight_total = raw.int_or("inference.max_inflight_total", 16);
  cfg.person_class_id = raw.int_or("inference.person_class_id", 0);
  cfg.min_score = raw.double_or("inference.min_score", 0.55);
  cfg.nms_iou = raw.double_or("inference.nms_iou", 0.60);
  cfg.max_detections = raw.int_or("inference.max_detections", 50);
  cfg.profile = raw.bool_or("runtime.profile", false);
  cfg.warmup_frames = raw.int_or("runtime.warmup_frames", 30);
  cfg.tracker_iou_threshold = static_cast<float>(raw.double_or("tracking.iou_threshold", 0.3));
  cfg.tracker_max_missing = raw.int_or("tracking.max_missing_frames", 15);
  cfg.insight_host = raw.string_or("output.insight.host", "");
  cfg.video_port_base = raw.int_or("output.insight.video_port_base", 9000);
  cfg.metadata_port_base = raw.int_or("output.insight.metadata_port_base", 9100);
  cfg.video_enabled = raw.bool_or("output.video_enabled", true);
  cfg.save_dir = raw.string_or("output.debug_dir", "");
  cfg.save_every = raw.int_or("output.save_every", 0);
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

std::vector<Detection> filter_people(const std::vector<objdet::Box>& boxes, int person_class_id) {
  std::vector<Detection> people;
  people.reserve(boxes.size());
  for (const auto& box : boxes) {
    if (box.class_id != person_class_id) {
      continue;
    }
    people.push_back(Detection{box.x1, box.y1, box.x2, box.y2, box.score, box.class_id});
  }
  return people;
}

std::vector<sima_examples::MetadataBox>
build_metadata_tracks(const std::vector<TrackedDetection>& tracks, int frame_w, int frame_h) {
  std::vector<sima_examples::MetadataBox> metadata_boxes;
  metadata_boxes.reserve(tracks.size());
  for (const auto& track : tracks) {
    int x1 = std::max(0, static_cast<int>(track.x1));
    int y1 = std::max(0, static_cast<int>(track.y1));
    int w = std::max(0, static_cast<int>(track.x2 - track.x1));
    int h = std::max(0, static_cast<int>(track.y2 - track.y1));
    if (x1 + w > frame_w)
      w = frame_w - x1;
    if (y1 + h > frame_h)
      h = frame_h - y1;

    sima_examples::MetadataBox obj;
    obj.id = std::to_string(track.track_id);
    obj.label = "person";
    obj.confidence = track.score;
    obj.x = static_cast<float>(x1);
    obj.y = static_cast<float>(y1);
    obj.w = static_cast<float>(std::max(0, w));
    obj.h = static_cast<float>(std::max(0, h));
    metadata_boxes.push_back(obj);
  }
  return metadata_boxes;
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
  opt.codec = cfg.use_h265 ? simaai::neat::nodes::groups::RtspCodec::H265
                           : simaai::neat::nodes::groups::RtspCodec::H264;
  if (probe.width > 0 && probe.height > 0) {
    if (cfg.use_h265) {
      opt.dec_width = probe.width;
      opt.dec_height = probe.height;
    } else {
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

simaai::neat::InputOptions encoded_decode_input_options(bool use_h265) {
  simaai::neat::InputOptions opt;
  opt.payload_type = simaai::neat::PayloadType::Encoded;
  if (use_h265) {
    opt.caps_override = "video/x-h265,parsed=(boolean)true,stream-format=(string)byte-stream,"
                        "alignment=(string)au";
  } else {
    opt.format = simaai::neat::FormatTag::H264;
  }
  opt.memory_policy = simaai::neat::InputMemoryPolicy::Ev74;
  return opt;
}

simaai::neat::InputOptions encoded_video_input_options(bool use_h265) {
  simaai::neat::InputOptions opt;
  opt.payload_type = simaai::neat::PayloadType::Encoded;
  if (use_h265) {
    opt.caps_override = "video/x-h265,parsed=(boolean)true,stream-format=(string)byte-stream,"
                        "alignment=(string)au";
  } else {
    opt.format = simaai::neat::FormatTag::H264;
  }
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
  const int dec_w =
      use_h265 ? opt.dec_width : ((opt.h264_width > 0) ? opt.h264_width : opt.fallback_h264_width);
  const int dec_h = use_h265 ? opt.dec_height
                             : ((opt.h264_height > 0) ? opt.h264_height : opt.fallback_h264_height);

  simaai::neat::SimaDecodeOptions dec;
  dec.type = use_h265 ? simaai::neat::SimaDecodeType::H265 : simaai::neat::SimaDecodeType::H264;
  dec.sima_allocator_type = opt.sima_allocator_type;
  dec.out_format = opt.out_format;
  dec.decoder_name = opt.decoder_name;
  dec.raw_output = opt.decoder_raw_output;
  dec.next_element = opt.decoder_next_element;
  dec.dec_width = dec_w;
  dec.dec_height = dec_h;
  dec.dec_fps = opt.source_fps;
  dec.num_buffers = opt.num_buffers;
  dec.input_buffers = opt.decoder_input_buffers;
  dec.decoder_tuning = opt.decoder_tuning;
  dec.memory_opt = opt.decoder_memory_opt;

  decode.connect(simaai::neat::nodes::Input(input_name, encoded_decode_input_options(use_h265)),
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
build_video_sender_graph(const std::string& input_name, bool use_h265,
                         const simaai::neat::nodes::groups::VideoSenderOptions& video_options) {
  simaai::neat::Graph video("video_sender");
  video.connect(simaai::neat::nodes::Input(input_name, encoded_video_input_options(use_h265)),
                simaai::neat::nodes::groups::VideoSender(video_options));
  return video;
}

std::unique_ptr<simaai::neat::Model> build_model(const AppConfig& cfg) {
  simaai::neat::Model::Options model_opt;
  model_opt.preprocess.kind = simaai::neat::InputKind::Image;
  model_opt.preprocess.enable = simaai::neat::AutoFlag::On;
  model_opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::NV12;
  model_opt.preprocess.preset = simaai::neat::NormalizePreset::COCO_YOLO;
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

StreamRuntime build_stream_runtime(const AppConfig& cfg, int stream_index, const std::string& url) {
  StreamRuntime runtime;
  runtime.index = stream_index;
  runtime.url = url;
  runtime.tracker = PeopleTracker(cfg.tracker_iou_threshold, cfg.tracker_max_missing);
  const auto source_options =
      build_source_options(cfg, url, runtime.output_fps, runtime.frame_w, runtime.frame_h);
  sima_examples::require(runtime.frame_w > 0 && runtime.frame_h > 0,
                         "failed to probe RTSP frame dimensions");
  sima_examples::require(runtime.output_fps > 0, "failed to probe RTSP frame rate");
  if (cfg.fps > 0) {
    runtime.output_fps = cfg.fps;
  }

  runtime.profile.enabled = cfg.profile;
  runtime.profile.stream_index = stream_index;
  runtime.source_options = source_options;
  if (cfg.video_enabled) {
    auto video_options =
        cfg.use_h265 ? simaai::neat::nodes::groups::VideoSenderOptions::H265RtpUdpFromEncoded()
                     : simaai::neat::nodes::groups::VideoSenderOptions::H264RtpUdpFromEncoded();
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
  auto decoder = build_decode_graph("decode_h264", stream.source_options);

  if (cfg.video_enabled) {
    auto encoded_branch = simaai::neat::graphs::Branch("encoded", {"decode_h264", "video_h264"});
    app.graph.connect(source, encoded_branch);
    app.graph.connect(encoded_branch, decoder, realtime_link(stream.index, 3));

    auto video_options =
        cfg.use_h265 ? simaai::neat::nodes::groups::VideoSenderOptions::H265RtpUdpFromEncoded()
                     : simaai::neat::nodes::groups::VideoSenderOptions::H264RtpUdpFromEncoded();
    video_options.host = cfg.insight_host;
    video_options.channel = stream.index;
    video_options.video_port_base = cfg.video_port_base;
    video_options.async = true;
    app.graph.connect(encoded_branch,
                      build_video_sender_graph("video_h264", cfg.use_h265, video_options),
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
                   const std::vector<TrackedDetection>& tracks) {
  const auto metadata_tracks = build_metadata_tracks(tracks, stream.frame_w, stream.frame_h);
  const std::string data_json = sima_examples::metadata_boxes_data_json("tracks", metadata_tracks);
  const int64_t timestamp_ms = sample.pts_ns >= 0 ? sample.pts_ns / 1'000'000 : -1;
  const std::string frame_id = sample.frame_id >= 0 ? std::to_string(sample.frame_id) : "";
  std::string err;
  if (!stream.metadata_sender->send_metadata("tracking", data_json, timestamp_ms, frame_id, &err)) {
    std::cerr << "[warn] stream " << stream.index << " metadata send failed: " << err << "\n";
  }
}

void maybe_save_debug_frame(const AppConfig& cfg, const StreamRuntime& stream, const cv::Mat* frame,
                            const std::vector<TrackedDetection>& tracks) {
  if (cfg.save_dir.empty() || cfg.save_every <= 0 || stream.processed % cfg.save_every != 0) {
    return;
  }
  if (frame == nullptr || frame->empty()) {
    return;
  }

  cv::Mat bgr = frame->clone();
  std::vector<objdet::Box> draw_boxes;
  draw_boxes.reserve(tracks.size());
  for (const auto& track : tracks) {
    draw_boxes.push_back(
        objdet::Box{track.x1, track.y1, track.x2, track.y2, track.score, track.track_id});
  }
  objdet::draw_boxes(bgr, draw_boxes, cfg.min_score, cv::Scalar(0, 255, 0), "track ");
  const auto out_path = cfg.save_dir / ("stream_" + std::to_string(stream.index) + "_frame_" +
                                        std::to_string(stream.processed) + ".jpg");
  if (!cv::imwrite(out_path.string(), bgr)) {
    std::cerr << "[warn] failed to write output frame: " << out_path.string() << "\n";
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

void process_output_sample(StreamRuntime& stream, const AppConfig& cfg,
                           const simaai::neat::Sample& sample, double detection_pull_ms) {
  if (cfg.frames > 0 && stream.processed >= cfg.frames) {
    return;
  }

  std::vector<std::uint8_t> payload;
  std::string err;
  if (!extract_bbox_payload(sample, payload, err)) {
    throw std::runtime_error("stream " + std::to_string(stream.index) +
                             " bbox extract failed: " + err);
  }
  const auto boxes = objdet::parse_boxes_strict(payload, stream.frame_w, stream.frame_h,
                                                cfg.max_detections, false);
  const auto people = filter_people(boxes, cfg.person_class_id);
  const double tracker_start = sima_examples::time_ms();
  const auto tracks = stream.tracker.update(people, stream.processed);
  const double tracker_end = sima_examples::time_ms();

  ++stream.processed;
  const bool warming_up = stream.processed <= cfg.warmup_frames;
  if (!warming_up) {
    const double metadata_start = sima_examples::time_ms();
    send_metadata(stream, sample, tracks);
    const double metadata_end = sima_examples::time_ms();
    if (save_frames_enabled(cfg)) {
      maybe_save_debug_frame(
          cfg, stream, stream.latest_debug_frame ? &*stream.latest_debug_frame : nullptr, tracks);
    }
    stream.profile.add(detection_pull_ms, tracker_end - tracker_start,
                       metadata_end - metadata_start, static_cast<int>(tracks.size()));
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
      stream.latest_debug_frame = std::move(bgr);
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
  if (status == simaai::neat::PullStatus::Timeout || status == simaai::neat::PullStatus::Closed) {
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
  auto previous_sigint = std::signal(SIGINT, request_stop);
  if (cfg.profile) {
    setenv("SIMA_GST_ELEMENT_TIMINGS", "1", 0);
    setenv("SIMA_GST_FLOW_DEBUG", "1", 0);
    setenv("SIMA_GST_BOUNDARY_PROBES", "1", 0);
  }
  if (save_frames_enabled(cfg)) {
    fs::create_directories(cfg.save_dir);
  }

  AppRuntime app;
  app.streams.reserve(cfg.rtsp_urls.size());
  auto detector_graph = build_detector_graph(cfg, app.model);
  auto detections_graph = build_detections_graph();

  for (std::size_t index = 0; index < cfg.rtsp_urls.size(); ++index) {
    app.streams.push_back(build_stream_runtime(cfg, static_cast<int>(index), cfg.rtsp_urls[index]));
    connect_stream_graph(app, cfg, app.streams.back(), detector_graph);
  }
  app.graph.connect(detector_graph, detections_graph);

  if (cfg.profile) {
    std::cout << "Backend:\n" << app.graph.describe_backend() << "\n";
  }

  app.run = app.graph.build(build_run_options());
  while (g_stop_requested == 0 && !all_streams_done(app.streams, cfg.frames)) {
    (void)process_run_once(app, cfg, "detections");
  }
  app.run.close();

  for (auto& stream : app.streams) {
    stream.profile.flush();
    std::cout << "[stream " << stream.index << "] processed=" << stream.processed << "\n";
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
