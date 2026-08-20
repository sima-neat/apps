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

#include "pose_logic.h"

#include "neat.h"
#include "neat/node_groups.h"
#include "neat/nodes.h"
#include "support/object_detection/obj_detection_utils.h"
#include "support/runtime/config_utils.h"
#include "support/runtime/example_utils.h"

#include <nodes/groups/VideoSender.h>
#include <nodes/io/MetadataSender.h>

#include <opencv2/core/mat.hpp>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
namespace neat = simaai::neat;

namespace {

using Clock = std::chrono::steady_clock;

volatile std::sig_atomic_t g_stop_requested = 0;

void request_stop(int) {
  g_stop_requested = 1;
}

struct StreamConfig {
  std::string id;
  std::string url;
  neat::nodes::groups::RtspCodec codec = neat::nodes::groups::RtspCodec::H264;
  int insight_channel = -1;
};

struct AppConfig {
  std::string detector_model_path;
  std::string pose_model_path;
  std::vector<StreamConfig> streams;
  bool tcp = true;
  int latency_ms = 100;
  double detector_min_score = 0.30;
  double detector_nms_iou = 0.60;
  int max_detections = 100;
  int max_inflight_per_stream = 4;
  int max_inflight_total = 16;
  int max_people_per_frame = 4;
  double roi_scale = 1.65;
  double pose_presence_threshold = 0.50;
  int pose_job_timeout_ms = 1000;
  int max_pending_jobs = 64;
  int frame_limit = 0;
  std::string insight_host;
  int video_port_base = 9000;
  int metadata_port_base = 9100;
  bool video_enabled = true;
};

struct CliOptions {
  fs::path config_path;
  bool validate_config_only = false;
};

struct FrameIdentity {
  std::string stream_id;
  int64_t frame_id = -1;
  int64_t pts_ns = -1;
  int64_t dts_ns = -1;
  int64_t duration_ns = -1;
  int64_t input_seq = -1;
  int64_t orig_input_seq = -1;
};

struct FrameJob {
  std::uint64_t job_id = 0;
  int stream_index = 0;
  neat::Tensor rgb;
  std::vector<blazepose_app::Box> people;
  FrameIdentity identity;
  Clock::time_point deadline;
};

struct PoseInputContext {
  std::uint64_t job_id = 0;
  int stream_index = 0;
  int roi_index = 0;
  int roi_count = 0;
  blazepose_app::Box box;
  blazepose_app::Affine affine;
  FrameIdentity identity;
};

struct PreparedPoseInput {
  int roi_index = 0;
  blazepose_app::Box box;
  blazepose_app::Affine affine;
  neat::Tensor tensor;
};

struct PoseAggregate {
  int stream_index = 0;
  int expected = 0;
  int completed = 0;
  FrameIdentity identity;
  Clock::time_point deadline;
  std::vector<blazepose_app::Pose> poses;
};

struct StreamRuntime {
  int index = 0;
  StreamConfig config;
  neat::nodes::groups::RtspDecodedInputOptions source_options;
  int width = 0;
  int height = 0;
  int fps = 0;
  std::unique_ptr<neat::MetadataSender> metadata_sender;
  std::mutex metadata_mutex;
  std::atomic<int> metadata_frames{0};
  std::atomic<std::uint64_t> source_frames{0};
  std::atomic<std::uint64_t> detector_frames{0};
  std::atomic<std::uint64_t> selected_rois{0};
  std::atomic<std::uint64_t> completed_rois{0};
  std::atomic<std::uint64_t> detector_mailbox_drops{0};
  std::atomic<std::uint64_t> pose_mailbox_drops{0};
  std::atomic<std::uint64_t> timed_out_jobs{0};
};

struct SharedState {
  std::mutex mutex;
  std::condition_variable cv;
  std::vector<std::optional<FrameJob>> detector_mailboxes;
  std::vector<std::optional<FrameJob>> pose_mailboxes;
  std::deque<FrameJob> pending_detector_outputs;
  std::deque<PoseInputContext> pending_pose_outputs;
  std::unordered_map<std::uint64_t, PoseAggregate> aggregates;
  std::size_t next_detector_stream = 0;
  std::size_t next_pose_stream = 0;
  bool stopping = false;
  std::exception_ptr error;
};

struct AppRuntime {
  neat::Graph source_graph;
  neat::Run source_run;
  neat::Graph detector_graph;
  neat::Run detector_run;
  neat::Graph pose_graph;
  neat::Run pose_run;
  std::unique_ptr<neat::Model> detector_model;
  std::unique_ptr<neat::Model> pose_model;
  std::vector<std::unique_ptr<StreamRuntime>> streams;
  SharedState state;
  std::atomic<std::uint64_t> next_job_id{1};
};

std::string lower_copy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

neat::nodes::groups::RtspCodec parse_codec(const std::string& value) {
  const std::string lowered = lower_copy(value);
  if (lowered == "h264" || lowered == "avc" || lowered == "h.264") {
    return neat::nodes::groups::RtspCodec::H264;
  }
  if (lowered == "h265" || lowered == "hevc" || lowered == "h.265") {
    return neat::nodes::groups::RtspCodec::H265;
  }
  throw std::runtime_error("stream codec must be h264/avc or h265/hevc");
}

std::string codec_name(neat::nodes::groups::RtspCodec codec) {
  return codec == neat::nodes::groups::RtspCodec::H265 ? "h265" : "h264";
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
  if (value.size() >= 2 && ((value.front() == '\'' && value.back() == '\'') ||
                            (value.front() == '"' && value.back() == '"'))) {
    return value.substr(1, value.size() - 2);
  }
  return value;
}

std::pair<std::string, std::string> parse_key_value(const std::string& text,
                                                    const std::string& where) {
  const std::size_t separator = text.find(':');
  if (separator == std::string::npos) {
    throw std::runtime_error(where + " must be 'key: value'");
  }
  const std::string key = sima_examples::trim_copy(text.substr(0, separator));
  const std::string value = unquote(text.substr(separator + 1));
  if (key.empty()) {
    throw std::runtime_error(where + " has an empty key");
  }
  return {key, value};
}

std::vector<StreamConfig> parse_streams(const fs::path& config_path) {
  std::ifstream input(config_path);
  if (!input.is_open()) {
    throw std::runtime_error("failed to open config file: " + config_path.string());
  }

  std::vector<StreamConfig> streams;
  std::optional<StreamConfig> current;
  bool in_streams = false;
  int streams_indent = -1;
  auto commit = [&]() {
    if (current.has_value()) {
      streams.push_back(std::move(*current));
      current.reset();
    }
  };

  std::string raw_line;
  int line_number = 0;
  while (std::getline(input, raw_line)) {
    ++line_number;
    const std::string without_comment = strip_inline_comment(raw_line);
    if (sima_examples::trim_copy(without_comment).empty()) {
      continue;
    }
    int indent = 0;
    while (indent < static_cast<int>(without_comment.size()) &&
           (without_comment[static_cast<std::size_t>(indent)] == ' ' ||
            without_comment[static_cast<std::size_t>(indent)] == '\t')) {
      ++indent;
    }
    const std::string line = sima_examples::trim_copy(without_comment);
    if (!in_streams && line == "streams:") {
      in_streams = true;
      streams_indent = indent;
      continue;
    }
    if (!in_streams) {
      continue;
    }
    if (indent <= streams_indent && line.rfind("- ", 0) != 0) {
      commit();
      break;
    }
    if (line.rfind("- ", 0) == 0) {
      commit();
      current.emplace();
      const auto [key, value] =
          parse_key_value(line.substr(2), "streams line " + std::to_string(line_number));
      if (key != "id") {
        throw std::runtime_error("each streams entry must start with '- id: <unique-id>'");
      }
      current->id = value;
      continue;
    }
    if (!current.has_value()) {
      throw std::runtime_error("streams must contain mapping entries");
    }
    const auto [key, value] = parse_key_value(line, "streams line " + std::to_string(line_number));
    if (key == "url") {
      current->url = value;
    } else if (key == "codec") {
      current->codec = parse_codec(value);
    } else if (key == "insight_channel") {
      try {
        current->insight_channel = std::stoi(value);
      } catch (...) {
        throw std::runtime_error("stream insight_channel must be an integer");
      }
    } else if (key != "id") {
      throw std::runtime_error("unknown stream setting: " + key);
    }
  }
  commit();
  return streams;
}

void validate_config(const AppConfig& cfg) {
  sima_examples::require(!cfg.detector_model_path.empty(), "models.detector_path must be set");
  sima_examples::require(!cfg.pose_model_path.empty(), "models.pose_path must be set");
  sima_examples::require(!cfg.streams.empty(), "streams must be a non-empty list");
  sima_examples::require(!cfg.insight_host.empty(), "output.insight.host must be set");
  sima_examples::require(cfg.latency_ms >= 0, "input.latency_ms must be >= 0");
  sima_examples::require(cfg.detector_min_score >= 0.0 && cfg.detector_min_score <= 1.0,
                         "detector.min_score must be between 0 and 1");
  sima_examples::require(cfg.detector_nms_iou >= 0.0 && cfg.detector_nms_iou <= 1.0,
                         "detector.nms_iou must be between 0 and 1");
  sima_examples::require(cfg.max_detections > 0, "detector.max_detections must be > 0");
  sima_examples::require(cfg.max_inflight_per_stream == -1 || cfg.max_inflight_per_stream > 0,
                         "detector.max_inflight_per_stream must be -1 or > 0");
  sima_examples::require(cfg.max_inflight_total == -1 || cfg.max_inflight_total > 0,
                         "detector.max_inflight_total must be -1 or > 0");
  sima_examples::require(cfg.max_people_per_frame > 0, "pose.max_people_per_frame must be > 0");
  sima_examples::require(cfg.roi_scale > 0.0, "pose.roi_scale must be > 0");
  sima_examples::require(cfg.pose_presence_threshold >= 0.0 && cfg.pose_presence_threshold <= 1.0,
                         "pose.presence_threshold must be between 0 and 1");
  sima_examples::require(cfg.pose_job_timeout_ms > 0, "pose.job_timeout_ms must be > 0");
  sima_examples::require(cfg.max_pending_jobs > 0, "pose.max_pending_jobs must be > 0");
  sima_examples::require(cfg.frame_limit >= 0, "runtime.frames must be >= 0");
  sima_examples::require(cfg.video_port_base > 0, "output.insight.video_port_base must be > 0");
  sima_examples::require(cfg.metadata_port_base > 0,
                         "output.insight.metadata_port_base must be > 0");

  std::set<std::string> ids;
  std::set<int> channels;
  for (const StreamConfig& stream : cfg.streams) {
    sima_examples::require(!stream.id.empty(), "stream id must be set");
    sima_examples::require(!stream.url.empty(), "stream url must be set");
    sima_examples::require(stream.insight_channel >= 0, "stream insight_channel must be >= 0");
    sima_examples::require(ids.insert(stream.id).second, "stream ids must be unique");
    sima_examples::require(channels.insert(stream.insight_channel).second,
                           "stream insight channels must be unique");
  }
}

AppConfig load_app_config(const fs::path& config_path) {
  const auto raw = sima_examples::ScalarConfig::load(config_path);
  AppConfig cfg;
  cfg.detector_model_path = raw.string_or("models.detector_path", "");
  cfg.pose_model_path = raw.string_or("models.pose_path", "");
  cfg.streams = parse_streams(config_path);
  cfg.tcp = raw.bool_or("input.tcp", true);
  cfg.latency_ms = raw.int_or("input.latency_ms", 100);
  cfg.detector_min_score = raw.double_or("detector.min_score", 0.30);
  cfg.detector_nms_iou = raw.double_or("detector.nms_iou", 0.60);
  cfg.max_detections = raw.int_or("detector.max_detections", 100);
  cfg.max_inflight_per_stream = raw.int_or("detector.max_inflight_per_stream", 4);
  cfg.max_inflight_total = raw.int_or("detector.max_inflight_total", 16);
  cfg.max_people_per_frame = raw.int_or("pose.max_people_per_frame", 4);
  cfg.roi_scale = raw.double_or("pose.roi_scale", 1.65);
  cfg.pose_presence_threshold = raw.double_or("pose.presence_threshold", 0.50);
  cfg.pose_job_timeout_ms = raw.int_or("pose.job_timeout_ms", 1000);
  cfg.max_pending_jobs = raw.int_or("pose.max_pending_jobs", 64);
  cfg.frame_limit = raw.int_or("runtime.frames", 0);
  cfg.insight_host = raw.string_or("output.insight.host", "");
  cfg.video_port_base = raw.int_or("output.insight.video_port_base", 9000);
  cfg.metadata_port_base = raw.int_or("output.insight.metadata_port_base", 9100);
  cfg.video_enabled = raw.bool_or("output.video_enabled", true);
  validate_config(cfg);
  return cfg;
}

CliOptions parse_args(int argc, char** argv) {
  CliOptions options;
  options.config_path = sima_examples::default_config_path(SIMANEAT_APPS_EXAMPLE_SOURCE_DIR);
  for (int index = 1; index < argc; ++index) {
    const std::string arg = argv[index];
    if (arg == "--config") {
      if (index + 1 >= argc) {
        throw std::runtime_error("--config requires a path");
      }
      options.config_path = argv[++index];
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

neat::FormatTag encoded_format(neat::nodes::groups::RtspCodec codec) {
  return codec == neat::nodes::groups::RtspCodec::H265 ? neat::FormatTag::H265
                                                       : neat::FormatTag::H264;
}

neat::InputOptions encoded_input_options(neat::nodes::groups::RtspCodec codec,
                                         neat::InputMemoryPolicy memory) {
  neat::InputOptions options;
  options.payload_type = neat::PayloadType::Encoded;
  options.format = encoded_format(codec);
  options.memory_policy = memory;
  return options;
}

neat::nodes::groups::RtspDecodedInputOptions probe_source(const AppConfig& cfg,
                                                          StreamRuntime& runtime) {
  sima_examples::RtspStreamInfo probe;
  sima_examples::RtspProbeOptions probe_options;
  probe_options.payload_type = 96;
  probe_options.latency_ms = cfg.latency_ms;
  probe_options.rtsp_tcp = cfg.tcp;
  if (!sima_examples::probe_rtsp_stream_info(runtime.config.url, probe_options, probe)) {
    throw std::runtime_error("failed to probe RTSP stream: " + runtime.config.url);
  }
  sima_examples::require(probe.width > 0 && probe.height > 0 && probe.fps > 0,
                         "RTSP probe must resolve width, height, and FPS for " + runtime.config.id);
  runtime.width = probe.width;
  runtime.height = probe.height;
  runtime.fps = probe.fps;

  neat::nodes::groups::RtspDecodedInputOptions options;
  options.url = runtime.config.url;
  options.codec = runtime.config.codec;
  options.latency_ms = cfg.latency_ms;
  options.tcp = cfg.tcp;
  options.payload_type = 96;
  options.source_fps = runtime.fps;
  options.insert_queue = true;
  options.out_format = "NV12";
  options.decoder_name = "decoder_" + runtime.config.id;
  options.decoder_raw_output = true;
  options.auto_caps_from_stream = true;
  options.dec_width = runtime.width;
  options.dec_height = runtime.height;
  if (runtime.config.codec == neat::nodes::groups::RtspCodec::H264) {
    options.fallback_h264_width = runtime.width;
    options.fallback_h264_height = runtime.height;
  }
  options.output_caps.enable = true;
  options.output_caps.format = "NV12";
  options.output_caps.width = runtime.width;
  options.output_caps.height = runtime.height;
  options.output_caps.fps = runtime.fps;
  options.output_caps.memory = neat::CapsMemory::Any;
  return options;
}

neat::Graph make_encoded_source(const neat::nodes::groups::RtspDecodedInputOptions& options) {
  neat::nodes::groups::RtspEncodedInputOptions encoded;
  encoded.url = options.url;
  encoded.codec = options.codec;
  encoded.latency_ms = options.latency_ms;
  encoded.tcp = options.tcp;
  encoded.source_fps = options.source_fps;
  encoded.payload_type = options.payload_type;
  encoded.insert_queue = options.insert_queue;
  encoded.auto_caps_from_stream = options.auto_caps_from_stream;
  encoded.fallback_h264_width = options.fallback_h264_width;
  encoded.fallback_h264_height = options.fallback_h264_height;
  return neat::nodes::groups::RtspEncodedInput(encoded);
}

neat::Graph make_decoder(const neat::nodes::groups::RtspDecodedInputOptions& options) {
  neat::SimaDecodeOptions decode;
  decode.type = options.codec == neat::nodes::groups::RtspCodec::H265 ? neat::SimaDecodeType::H265
                                                                      : neat::SimaDecodeType::H264;
  decode.sima_allocator_type = options.sima_allocator_type;
  decode.out_format = "NV12";
  decode.decoder_name = options.decoder_name;
  decode.raw_output = options.decoder_raw_output;
  decode.next_element = options.decoder_next_element;
  decode.dec_width = options.dec_width;
  decode.dec_height = options.dec_height;
  decode.dec_fps = options.source_fps;
  decode.num_buffers = options.num_buffers;
  decode.input_buffers = options.decoder_input_buffers;
  decode.decoder_tuning = options.decoder_tuning;
  decode.memory_opt = options.decoder_memory_opt;

  neat::Graph graph("decoder_" + options.decoder_name);
  graph.connect(neat::nodes::Input(
                    "encoded", encoded_input_options(options.codec, neat::InputMemoryPolicy::Ev74)),
                neat::nodes::SimaDecode(decode));
  graph.add(neat::nodes::CapsRaw("NV12", options.dec_width, options.dec_height, options.source_fps,
                                 neat::CapsMemory::Any));
  graph.add(neat::nodes::Output("analytics_frame"));
  return graph;
}

neat::Graph make_video_sender(const AppConfig& cfg, const StreamRuntime& stream) {
  auto options = neat::nodes::groups::VideoSenderOptions::Passthrough(stream.config.codec);
  options.host = cfg.insight_host;
  options.channel = stream.config.insight_channel;
  options.video_port_base = cfg.video_port_base;
  options.async = false;
  neat::Graph graph("video_" + stream.config.id);
  graph.connect(
      neat::nodes::Input("encoded", encoded_input_options(stream.config.codec,
                                                          neat::InputMemoryPolicy::SystemMemory)),
      neat::nodes::groups::VideoSender(options));
  return graph;
}

std::unique_ptr<neat::Model> make_detector_model(const AppConfig& cfg, int max_width,
                                                 int max_height) {
  neat::Model::Options options;
  options.preprocess.kind = neat::InputKind::Image;
  options.preprocess.enable = neat::AutoFlag::On;
  options.preprocess.color_convert.input_format = neat::PreprocessColorFormat::RGB;
  options.preprocess.input_max_width = max_width;
  options.preprocess.input_max_height = max_height;
  options.preprocess.preset = neat::NormalizePreset::COCO_YOLO;
  options.decode_type = neat::BoxDecodeType::YoloV26;
  options.score_threshold = cfg.detector_min_score;
  options.nms_iou_threshold = cfg.detector_nms_iou;
  options.top_k = cfg.max_detections;
  return std::make_unique<neat::Model>(cfg.detector_model_path, options);
}

std::unique_ptr<neat::Model> make_pose_model(const AppConfig& cfg) {
  neat::Model::Options options;
  options.preprocess.kind = neat::InputKind::Image;
  options.preprocess.enable = neat::AutoFlag::On;
  options.preprocess.color_convert.input_format = neat::PreprocessColorFormat::RGB;
  options.preprocess.resize.enable = neat::AutoFlag::On;
  options.preprocess.resize.width = 256;
  options.preprocess.resize.height = 256;
  options.preprocess.resize.mode = neat::ResizeMode::Stretch;
  options.preprocess.normalize.enable = neat::AutoFlag::On;
  return std::make_unique<neat::Model>(cfg.pose_model_path, options);
}

void validate_detector_contract(const neat::Model& model) {
  const auto inputs = model.input_specs();
  const auto outputs = model.output_specs();
  sima_examples::require(inputs.size() == 1, "YOLO26 must have one input");
  sima_examples::require(inputs[0].shape == std::vector<int64_t>({-1, -1, 3}),
                         "YOLO26 public input must be dynamic HWC RGB");
  sima_examples::require(outputs.size() == 1, "YOLO26 must expose one decoded BBOX output");
}

void validate_pose_contract(const neat::Model& model) {
  const auto inputs = model.input_specs();
  const auto outputs = model.output_specs();
  sima_examples::require(inputs.size() == 1, "BlazePose must have one input");
  sima_examples::require(inputs[0].shape == std::vector<int64_t>({-1, -1, 3}),
                         "BlazePose public input must be dynamic HWC RGB");
  sima_examples::require(outputs.size() == 3, "BlazePose must have three outputs");
  sima_examples::require(outputs[0].shape == std::vector<int64_t>({1, 195}),
                         "BlazePose output 0 must be [1,195]");
  sima_examples::require(outputs[1].shape == std::vector<int64_t>({1, 1}),
                         "BlazePose output 1 must be [1,1]");
  sima_examples::require(outputs[2].shape == std::vector<int64_t>({1, 117}),
                         "BlazePose output 2 must be [1,117]");
}

neat::Sample pose_input_sample(const neat::Tensor& tensor, const PoseInputContext* context) {
  neat::Sample sample = neat::make_tensor_sample("pose_input", tensor);
  sample.payload_type = neat::PayloadType::Tensor;
  sample.media_type = "application/vnd.simaai.tensor";
  if (tensor.semantic.tess.has_value()) {
    sample.format = tensor.semantic.tess->format;
    sample.payload_tag = sample.format;
  }
  if (context != nullptr) {
    sample.stream_id = context->identity.stream_id;
    sample.frame_id = context->identity.frame_id;
    sample.pts_ns = context->identity.pts_ns;
    sample.dts_ns = context->identity.dts_ns;
    sample.duration_ns = context->identity.duration_ns;
    sample.input_seq = context->identity.input_seq;
    sample.orig_input_seq = context->identity.orig_input_seq;
  }
  return sample;
}

void build_pose_run(AppRuntime& app) {
  validate_pose_contract(*app.pose_model);
  cv::Mat seed_image = cv::Mat::zeros(256, 256, CV_8UC3);
  const std::vector<neat::PreprocessRoi> seed_rois = {{0, 0, 0, 256, 256}};
  const neat::TensorList seed_tensors =
      neat::stages::Preproc(std::vector<cv::Mat>{seed_image}, *app.pose_model, seed_rois);
  sima_examples::require(seed_tensors.size() == 1, "BlazePose seed Preproc returned no tensor");
  const neat::Sample seed = pose_input_sample(seed_tensors.front(), nullptr);

  app.pose_graph = neat::Graph("blazepose_runner");
  app.pose_graph.add(neat::nodes::Input("pose_input"));
  app.pose_graph.add(app.pose_model->inference());
  app.pose_graph.add(app.pose_model->postprocess());
  app.pose_graph.add(neat::nodes::Output("pose_output", neat::OutputOptions::EveryFrame(4)));

  neat::RunOptions options;
  options.preset = neat::RunPreset::Reliable;
  options.overflow_policy = neat::OverflowPolicy::Block;
  options.output_memory = neat::OutputMemory::ZeroCopy;
  options.input_timeout_ms = 30000;
  options.startup_preflight = false;
  app.pose_run = app.pose_graph.build(seed, options);
}

neat::GraphLinkOptions realtime_link(const AppConfig& cfg, const StreamRuntime& stream) {
  neat::GraphLinkOptions options;
  options.policy = neat::GraphLinkPolicy::RealtimeLatestByStream;
  options.stream_id = stream.config.id;
  options.max_inflight_per_stream = cfg.max_inflight_per_stream;
  options.max_inflight_total = cfg.max_inflight_total;
  return options;
}

std::string frame_output_name(int stream_index) {
  return "frame_" + std::to_string(stream_index);
}

neat::Graph make_rgb_output(const StreamRuntime& stream) {
  neat::Graph graph("rgb_" + std::to_string(stream.index));
  graph.add(neat::nodes::Input("analytics_frame"));
  graph.add(neat::nodes::VideoConvert());
  graph.add(neat::nodes::CapsRaw("RGB", stream.width, stream.height, stream.fps));
  graph.add(neat::nodes::Output(frame_output_name(stream.index), neat::OutputOptions::Latest()));
  return graph;
}

void initialize_streams(AppRuntime& app, const AppConfig& cfg) {
  int max_width = 0;
  int max_height = 0;
  for (std::size_t index = 0; index < cfg.streams.size(); ++index) {
    auto stream = std::make_unique<StreamRuntime>();
    stream->index = static_cast<int>(index);
    stream->config = cfg.streams[index];
    stream->source_options = probe_source(cfg, *stream);
    max_width = std::max(max_width, stream->width);
    max_height = std::max(max_height, stream->height);

    neat::MetadataSenderOptions metadata_options;
    metadata_options.host = cfg.insight_host;
    metadata_options.channel = stream->config.insight_channel;
    metadata_options.metadata_port_base = cfg.metadata_port_base;
    neat::MetadataSenderSendOptions send_options;
    send_options.nonblocking = true;
    std::string error;
    stream->metadata_sender =
        std::make_unique<neat::MetadataSender>(metadata_options, send_options, &error);
    sima_examples::require(stream->metadata_sender->ok(), error);
    app.streams.push_back(std::move(stream));
  }
  app.detector_model = make_detector_model(cfg, max_width, max_height);
}

void build_source_run(AppRuntime& app, const AppConfig& cfg) {
  app.source_graph = neat::Graph("multistream_blazepose3d_sources");
  for (const auto& stream_ptr : app.streams) {
    StreamRuntime& stream = *stream_ptr;
    const neat::Graph source = make_encoded_source(stream.source_options);
    const neat::Graph decoder = make_decoder(stream.source_options);
    app.source_graph.connect(source, decoder);
    app.source_graph.connect(decoder, make_rgb_output(stream), realtime_link(cfg, stream));
    if (cfg.video_enabled) {
      app.source_graph.connect(source, make_video_sender(cfg, stream));
    }

    std::cout << "[stream " << stream.config.id << "] codec=" << codec_name(stream.config.codec)
              << " source=" << stream.width << "x" << stream.height << "@" << stream.fps
              << " channel=" << stream.config.insight_channel << " video=";
    if (cfg.video_enabled) {
      std::cout << cfg.video_port_base + stream.config.insight_channel;
    } else {
      std::cout << "disabled";
    }
    std::cout << " metadata=" << stream.metadata_sender->metadata_port() << "\n";
  }

  neat::RunOptions options;
  options.preset = neat::RunPreset::Realtime;
  options.output_memory = neat::OutputMemory::ZeroCopy;
  app.source_run = app.source_graph.build(options);
}

neat::Sample image_input_sample(const std::string& name, const neat::Tensor& tensor,
                                const FrameIdentity* identity) {
  neat::Sample sample = neat::make_tensor_sample(name, tensor);
  sample.payload_type = neat::PayloadType::Image;
  sample.media_type = "video/x-raw";
  sample.format = "RGB";
  sample.payload_tag = sample.format;
  if (identity != nullptr) {
    sample.stream_id = identity->stream_id;
    sample.frame_id = identity->frame_id;
    sample.pts_ns = identity->pts_ns;
    sample.dts_ns = identity->dts_ns;
    sample.duration_ns = identity->duration_ns;
    sample.input_seq = identity->input_seq;
    sample.orig_input_seq = identity->orig_input_seq;
  }
  return sample;
}

void build_detector_run(AppRuntime& app) {
  validate_detector_contract(*app.detector_model);
  int seed_width = 0;
  int seed_height = 0;
  for (const auto& stream : app.streams) {
    seed_width = std::max(seed_width, stream->width);
    seed_height = std::max(seed_height, stream->height);
  }
  cv::Mat seed_image = cv::Mat::zeros(seed_height, seed_width, CV_8UC3);
  const neat::Tensor seed_tensor = neat::Tensor::from_cv_mat(
      seed_image, neat::ImageSpec::PixelFormat::RGB, neat::TensorMemory::EV74);
  const neat::Sample seed = image_input_sample("detector_input", seed_tensor, nullptr);

  app.detector_graph = neat::Graph("yolo26_runner");
  auto input_options = app.detector_model->input_appsrc_options(false);
  input_options.block = true;
  neat::Graph input_graph;
  input_graph.add(neat::nodes::Input("detector_input", input_options));
  const neat::Graph model_graph = app.detector_model->graph();
  neat::Graph output_graph;
  output_graph.add(neat::nodes::Output("detector_output", neat::OutputOptions::EveryFrame(4)));
  app.detector_graph.connect(input_graph, model_graph);
  app.detector_graph.connect(model_graph, output_graph);

  neat::RunOptions options;
  options.preset = neat::RunPreset::Reliable;
  options.overflow_policy = neat::OverflowPolicy::Block;
  options.output_memory = neat::OutputMemory::ZeroCopy;
  options.input_timeout_ms = 30000;
  options.startup_preflight = false;
  app.detector_run = app.detector_graph.build(seed, options);
}

bool extract_bbox_payload(const neat::Sample& sample, std::vector<std::uint8_t>& payload,
                          std::string& error) {
  if (sample.kind == neat::SampleKind::Bundle) {
    for (const neat::Sample& field : sample.fields) {
      if (extract_bbox_payload(field, payload, error)) {
        return true;
      }
    }
    error = "bundle missing BBOX field";
    return false;
  }
  if (sample.kind == neat::SampleKind::TensorSet && !sample.tensors.empty()) {
    neat::Sample tensor_sample = sample;
    tensor_sample.kind = neat::SampleKind::Tensor;
    tensor_sample.tensor = sample.tensors.front();
    tensor_sample.tensors.clear();
    return objdet::extract_bbox_payload(tensor_sample, payload, error);
  }
  return objdet::extract_bbox_payload(sample, payload, error);
}

neat::Tensor require_rgb_tensor(const neat::Sample& sample) {
  const neat::TensorList tensors = neat::tensors_from_sample(sample, false);
  if (tensors.size() != 1) {
    throw std::runtime_error("RGB frame output must contain one tensor");
  }
  const auto format = tensors.front().image_format();
  if (!format.has_value() || *format != neat::ImageSpec::PixelFormat::RGB) {
    throw std::runtime_error("VideoConvert output is not RGB");
  }
  return tensors.front();
}

std::vector<blazepose_app::Box> select_people(const neat::Sample& detections, int width, int height,
                                              const AppConfig& cfg) {
  std::vector<std::uint8_t> payload;
  std::string error;
  if (!extract_bbox_payload(detections, payload, error)) {
    throw std::runtime_error("failed to read detector BBOX output: " + error);
  }
  const std::vector<objdet::Box> boxes =
      objdet::parse_boxes_strict(payload, width, height, cfg.max_detections, false);
  std::vector<blazepose_app::Box> people;
  for (const objdet::Box& box : boxes) {
    if (box.class_id == 0) {
      people.push_back({box.x1, box.y1, box.x2, box.y2, box.score, box.class_id});
    }
  }
  std::sort(people.begin(), people.end(),
            [](const auto& left, const auto& right) { return left.score > right.score; });
  if (people.size() > static_cast<std::size_t>(cfg.max_people_per_frame)) {
    people.resize(static_cast<std::size_t>(cfg.max_people_per_frame));
  }
  return people;
}

FrameIdentity identity_from_sample(const neat::Sample& sample) {
  return {sample.stream_id,   sample.frame_id,  sample.pts_ns,        sample.dts_ns,
          sample.duration_ns, sample.input_seq, sample.orig_input_seq};
}

void publish_metadata(StreamRuntime& stream, const FrameIdentity& identity,
                      std::vector<blazepose_app::Pose> poses) {
  const std::string data = blazepose_app::poses_data_json(std::move(poses)).dump();
  const int64_t timestamp_ms = identity.pts_ns >= 0 ? identity.pts_ns / 1'000'000 : -1;
  const std::string frame_id = identity.frame_id >= 0 ? std::to_string(identity.frame_id) : "";
  std::string error;
  std::lock_guard<std::mutex> lock(stream.metadata_mutex);
  if (!stream.metadata_sender->send_metadata("pose-estimation", data, timestamp_ms, frame_id,
                                             &error)) {
    std::cerr << "[warn] stream " << stream.config.id << " metadata send failed: " << error << "\n";
  }
  ++stream.metadata_frames;
}

std::vector<float> tensor_floats(const neat::Tensor& tensor, std::size_t expected) {
  const std::vector<std::uint8_t> bytes = tensor.copy_payload_bytes();
  if (bytes.size() != expected * sizeof(float)) {
    throw std::runtime_error("unexpected BlazePose output byte count");
  }
  std::vector<float> values(expected);
  std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
}

std::optional<blazepose_app::Pose> parse_pose_output(const neat::Sample& sample,
                                                     const PoseInputContext& context,
                                                     const AppConfig& cfg) {
  const neat::TensorList tensors = neat::tensors_from_sample(sample, false);
  if (tensors.size() != 3) {
    throw std::runtime_error("BlazePose output must contain three tensors");
  }
  const std::vector<float> presence = tensor_floats(tensors[1], 1);
  if (presence[0] < cfg.pose_presence_threshold) {
    return std::nullopt;
  }
  const std::vector<float> landmarks = tensor_floats(tensors[0], 195);
  return blazepose_app::decode_pose(landmarks, context.affine, context.box, context.roi_index);
}

void record_error(AppRuntime& app) {
  std::lock_guard<std::mutex> lock(app.state.mutex);
  if (!app.state.error) {
    app.state.error = std::current_exception();
  }
  app.state.stopping = true;
  app.state.cv.notify_all();
}

std::optional<FrameJob> take_next_job(AppRuntime& app,
                                      std::vector<std::optional<FrameJob>>& mailboxes,
                                      std::size_t& next_stream) {
  std::unique_lock<std::mutex> lock(app.state.mutex);
  app.state.cv.wait(lock, [&]() {
    return app.state.stopping || std::any_of(mailboxes.begin(), mailboxes.end(),
                                             [](const auto& item) { return item.has_value(); });
  });
  if (app.state.stopping) {
    return std::nullopt;
  }
  for (std::size_t offset = 0; offset < mailboxes.size(); ++offset) {
    const std::size_t index = (next_stream + offset) % mailboxes.size();
    if (mailboxes[index].has_value()) {
      FrameJob job = std::move(*mailboxes[index]);
      mailboxes[index].reset();
      next_stream = (index + 1) % mailboxes.size();
      return job;
    }
  }
  return std::nullopt;
}

blazepose_app::Affine affine_from_tensor(const neat::Tensor& tensor) {
  if (!tensor.semantic.preprocess.has_value()) {
    throw std::runtime_error("BlazePose Preproc output is missing affine metadata");
  }
  const auto& meta = *tensor.semantic.preprocess;
  return {meta.affine_m00, meta.affine_m01, meta.affine_m02,
          meta.affine_m10, meta.affine_m11, meta.affine_m12};
}

void pull_source_frames(AppRuntime& app, const AppConfig& cfg, int stream_index) {
  try {
    StreamRuntime& stream = *app.streams[static_cast<std::size_t>(stream_index)];
    const std::string output = frame_output_name(stream_index);
    while (true) {
      neat::Sample sample;
      neat::PullError error;
      const auto status = app.source_run.pull(output, 50, sample, &error);
      if (status == neat::PullStatus::Timeout) {
        std::lock_guard<std::mutex> lock(app.state.mutex);
        if (app.state.stopping) {
          return;
        }
        continue;
      }
      if (status == neat::PullStatus::Closed) {
        return;
      }
      if (status != neat::PullStatus::Ok) {
        throw std::runtime_error("failed to pull RGB source frame: " + error.message);
      }
      FrameJob job;
      job.job_id = app.next_job_id.fetch_add(1);
      job.stream_index = stream_index;
      job.rgb = require_rgb_tensor(sample);
      job.identity = identity_from_sample(sample);
      job.identity.stream_id = stream.config.id;
      const std::uint64_t pull_sequence = stream.source_frames.fetch_add(1) + 1;
      job.identity.frame_id =
          blazepose_app::select_frame_id(job.identity.frame_id, job.identity.orig_input_seq,
                                         job.identity.input_seq, pull_sequence);
      job.deadline = Clock::now() + std::chrono::milliseconds(cfg.pose_job_timeout_ms);

      std::lock_guard<std::mutex> lock(app.state.mutex);
      if (app.state.stopping) {
        return;
      }
      auto& mailbox = app.state.detector_mailboxes[static_cast<std::size_t>(stream_index)];
      if (mailbox.has_value()) {
        ++stream.detector_mailbox_drops;
      }
      mailbox = std::move(job);
      app.state.cv.notify_all();
    }
  } catch (...) {
    record_error(app);
  }
}

void dispatch_detector_jobs(AppRuntime& app, const AppConfig& cfg) {
  try {
    while (true) {
      std::optional<FrameJob> maybe_job =
          take_next_job(app, app.state.detector_mailboxes, app.state.next_detector_stream);
      if (!maybe_job.has_value()) {
        return;
      }
      FrameJob job = std::move(*maybe_job);
      StreamRuntime& stream = *app.streams[static_cast<std::size_t>(job.stream_index)];
      if (Clock::now() >= job.deadline) {
        ++stream.timed_out_jobs;
        publish_metadata(stream, job.identity, {});
        continue;
      }
      const neat::Tensor detector_frame = job.rgb.cvu();
      const neat::Sample input =
          image_input_sample("detector_input", detector_frame, &job.identity);
      {
        std::unique_lock<std::mutex> lock(app.state.mutex);
        app.state.cv.wait(lock, [&]() {
          return app.state.stopping || app.state.pending_detector_outputs.size() <
                                           static_cast<std::size_t>(cfg.max_pending_jobs);
        });
        if (app.state.stopping) {
          return;
        }
        app.state.pending_detector_outputs.push_back(job);
      }
      if (!app.detector_run.push("detector_input", input)) {
        std::lock_guard<std::mutex> lock(app.state.mutex);
        if (app.state.stopping) {
          return;
        }
        throw std::runtime_error("YOLO26 Run rejected a frame input");
      }
    }
  } catch (...) {
    record_error(app);
  }
}

void pull_detector_outputs(AppRuntime& app, const AppConfig& cfg) {
  try {
    while (true) {
      neat::Sample sample;
      neat::PullError error;
      const auto status = app.detector_run.pull("detector_output", 20, sample, &error);
      if (status == neat::PullStatus::Timeout) {
        std::lock_guard<std::mutex> lock(app.state.mutex);
        if (app.state.stopping) {
          return;
        }
        continue;
      }
      if (status == neat::PullStatus::Closed) {
        return;
      }
      if (status != neat::PullStatus::Ok) {
        throw std::runtime_error("failed to pull YOLO26 output: " + error.message);
      }

      FrameJob job;
      {
        std::lock_guard<std::mutex> lock(app.state.mutex);
        if (app.state.pending_detector_outputs.empty()) {
          if (app.state.stopping) {
            return;
          }
          throw std::runtime_error("YOLO26 output arrived without pending frame context");
        }
        job = std::move(app.state.pending_detector_outputs.front());
        app.state.pending_detector_outputs.pop_front();
        app.state.cv.notify_all();
      }
      StreamRuntime& stream = *app.streams[static_cast<std::size_t>(job.stream_index)];
      ++stream.detector_frames;
      if (Clock::now() >= job.deadline) {
        ++stream.timed_out_jobs;
        publish_metadata(stream, job.identity, {});
        continue;
      }
      job.people = select_people(sample, stream.width, stream.height, cfg);
      stream.selected_rois.fetch_add(job.people.size());
      if (job.people.empty()) {
        publish_metadata(stream, job.identity, {});
        continue;
      }

      std::lock_guard<std::mutex> lock(app.state.mutex);
      if (app.state.stopping) {
        return;
      }
      auto& mailbox = app.state.pose_mailboxes[static_cast<std::size_t>(job.stream_index)];
      if (mailbox.has_value()) {
        ++stream.pose_mailbox_drops;
      }
      mailbox = std::move(job);
      app.state.cv.notify_all();
    }
  } catch (...) {
    record_error(app);
  }
}

void dispatch_pose_jobs(AppRuntime& app, const AppConfig& cfg) {
  try {
    while (true) {
      std::optional<FrameJob> maybe_job =
          take_next_job(app, app.state.pose_mailboxes, app.state.next_pose_stream);
      if (!maybe_job.has_value()) {
        return;
      }
      FrameJob job = std::move(*maybe_job);
      StreamRuntime& stream = *app.streams[static_cast<std::size_t>(job.stream_index)];
      if (Clock::now() >= job.deadline) {
        ++stream.timed_out_jobs;
        publish_metadata(stream, job.identity, {});
        continue;
      }
      if (job.people.empty()) {
        publish_metadata(stream, job.identity, {});
        continue;
      }

      auto rgb_view = job.rgb.map_cv_mat_view(neat::ImageSpec::PixelFormat::RGB);
      if (!rgb_view.has_value()) {
        throw std::runtime_error("failed to map packed RGB frame without copying");
      }
      std::vector<PreparedPoseInput> prepared_inputs;
      prepared_inputs.reserve(job.people.size());
      std::vector<blazepose_app::Roi> requested_rois;
      requested_rois.reserve(job.people.size());
      for (const blazepose_app::Box& person : job.people) {
        requested_rois.push_back(blazepose_app::square_roi(person, cfg.roi_scale));
      }
      const auto plan =
          blazepose_app::batch_crop_plan(requested_rois, rgb_view->mat.cols, rgb_view->mat.rows);
      if (!plan.has_value()) {
        publish_metadata(stream, job.identity, {});
        continue;
      }
      const cv::Rect image_rect(plan->image.x, plan->image.y, plan->image.width,
                                plan->image.height);
      const cv::Mat crop_view = rgb_view->mat(image_rect);
      std::vector<neat::PreprocessRoi> crop_rois;
      crop_rois.reserve(plan->rois.size());
      for (const blazepose_app::Roi& roi : plan->rois) {
        crop_rois.push_back({0, roi.x, roi.y, roi.width, roi.height});
      }
      const neat::TensorList output =
          neat::stages::Preproc(std::vector<cv::Mat>{crop_view}, *app.pose_model, crop_rois);
      if (output.size() != plan->indices.size()) {
        throw std::runtime_error("BlazePose Preproc output count does not match ROI count");
      }
      for (std::size_t index = 0; index < output.size(); ++index) {
        const std::size_t person_index = plan->indices[index];
        const blazepose_app::Affine affine = blazepose_app::offset_affine(
            affine_from_tensor(output[index]), plan->image.x, plan->image.y);
        // Detached asynchronous Runs may retain their input after push(). Give
        // each ROI independent EV74 storage so Preproc can recycle its pool.
        prepared_inputs.push_back({static_cast<int>(person_index), job.people[person_index], affine,
                                   output[index].clone().cvu()});
      }

      {
        std::unique_lock<std::mutex> lock(app.state.mutex);
        app.state.cv.wait(lock, [&]() {
          return app.state.stopping ||
                 app.state.aggregates.size() < static_cast<std::size_t>(cfg.max_pending_jobs);
        });
        if (app.state.stopping) {
          return;
        }
        PoseAggregate aggregate;
        aggregate.stream_index = job.stream_index;
        aggregate.expected = static_cast<int>(prepared_inputs.size());
        aggregate.identity = job.identity;
        aggregate.deadline = job.deadline;
        app.state.aggregates.emplace(job.job_id, std::move(aggregate));
      }

      for (const PreparedPoseInput& prepared : prepared_inputs) {
        PoseInputContext context;
        context.job_id = job.job_id;
        context.stream_index = job.stream_index;
        context.roi_index = prepared.roi_index;
        context.roi_count = static_cast<int>(prepared_inputs.size());
        context.box = prepared.box;
        context.affine = prepared.affine;
        context.identity = job.identity;
        const neat::Sample input = pose_input_sample(prepared.tensor, &context);

        {
          std::lock_guard<std::mutex> lock(app.state.mutex);
          if (app.state.stopping) {
            return;
          }
          app.state.pending_pose_outputs.push_back(context);
        }
        if (!app.pose_run.push("pose_input", input)) {
          std::lock_guard<std::mutex> lock(app.state.mutex);
          if (app.state.stopping) {
            return;
          }
          throw std::runtime_error("BlazePose Run rejected an ROI input");
        }
      }
    }
  } catch (...) {
    record_error(app);
  }
}

void publish_completed_aggregate(AppRuntime& app, std::uint64_t job_id) {
  PoseAggregate aggregate;
  {
    std::lock_guard<std::mutex> lock(app.state.mutex);
    const auto found = app.state.aggregates.find(job_id);
    if (found == app.state.aggregates.end()) {
      return;
    }
    aggregate = std::move(found->second);
    app.state.aggregates.erase(found);
    app.state.cv.notify_all();
  }
  publish_metadata(*app.streams[static_cast<std::size_t>(aggregate.stream_index)],
                   aggregate.identity, std::move(aggregate.poses));
}

void expire_pose_jobs(AppRuntime& app) {
  std::vector<std::uint64_t> expired;
  const auto now = Clock::now();
  {
    std::lock_guard<std::mutex> lock(app.state.mutex);
    for (const auto& [job_id, aggregate] : app.state.aggregates) {
      if (now >= aggregate.deadline) {
        expired.push_back(job_id);
      }
    }
  }
  for (const std::uint64_t job_id : expired) {
    int stream_index = -1;
    {
      std::lock_guard<std::mutex> lock(app.state.mutex);
      const auto found = app.state.aggregates.find(job_id);
      if (found != app.state.aggregates.end()) {
        stream_index = found->second.stream_index;
      }
    }
    if (stream_index >= 0) {
      ++app.streams[static_cast<std::size_t>(stream_index)]->timed_out_jobs;
      publish_completed_aggregate(app, job_id);
    }
  }
}

void pull_pose_outputs(AppRuntime& app, const AppConfig& cfg) {
  try {
    while (true) {
      {
        std::lock_guard<std::mutex> lock(app.state.mutex);
        if (app.state.stopping && app.state.pending_pose_outputs.empty()) {
          return;
        }
      }

      neat::Sample sample;
      neat::PullError error;
      const auto status = app.pose_run.pull("pose_output", 20, sample, &error);
      if (status == neat::PullStatus::Timeout) {
        expire_pose_jobs(app);
        continue;
      }
      if (status == neat::PullStatus::Closed) {
        return;
      }
      if (status != neat::PullStatus::Ok) {
        throw std::runtime_error("failed to pull BlazePose output: " + error.message);
      }

      PoseInputContext context;
      {
        std::lock_guard<std::mutex> lock(app.state.mutex);
        if (app.state.pending_pose_outputs.empty()) {
          if (app.state.stopping) {
            return;
          }
          throw std::runtime_error("BlazePose output arrived without pending ROI context");
        }
        context = std::move(app.state.pending_pose_outputs.front());
        app.state.pending_pose_outputs.pop_front();
      }
      const auto pose = parse_pose_output(sample, context, cfg);
      bool complete = false;
      {
        std::lock_guard<std::mutex> lock(app.state.mutex);
        const auto found = app.state.aggregates.find(context.job_id);
        if (found != app.state.aggregates.end()) {
          ++found->second.completed;
          if (pose.has_value()) {
            found->second.poses.push_back(*pose);
          }
          complete = found->second.completed == found->second.expected;
        }
      }
      ++app.streams[static_cast<std::size_t>(context.stream_index)]->completed_rois;
      if (complete) {
        publish_completed_aggregate(app, context.job_id);
      }
      expire_pose_jobs(app);
    }
  } catch (...) {
    record_error(app);
  }
}

bool all_streams_done(const AppRuntime& app, int frame_limit) {
  if (frame_limit <= 0) {
    return false;
  }
  return std::all_of(app.streams.begin(), app.streams.end(), [frame_limit](const auto& stream) {
    return stream->metadata_frames.load() >= frame_limit;
  });
}

void stop_runtime(AppRuntime& app) {
  {
    std::lock_guard<std::mutex> lock(app.state.mutex);
    app.state.stopping = true;
    for (auto& mailbox : app.state.detector_mailboxes) {
      mailbox.reset();
    }
    for (auto& mailbox : app.state.pose_mailboxes) {
      mailbox.reset();
    }
    app.state.pending_detector_outputs.clear();
    app.state.pending_pose_outputs.clear();
    app.state.aggregates.clear();
    app.state.cv.notify_all();
  }
  app.source_run.close();
  app.detector_run.close();
  app.pose_run.close();
}

void print_summary(const AppRuntime& app, double elapsed_seconds) {
  std::uint64_t total_frames = 0;
  std::uint64_t total_rois = 0;
  for (const auto& stream : app.streams) {
    total_frames += static_cast<std::uint64_t>(stream->metadata_frames.load());
    total_rois += stream->completed_rois.load();
    std::cout << "[summary stream=" << stream->config.id
              << "] source_frames=" << stream->source_frames.load()
              << " detector_frames=" << stream->detector_frames.load()
              << " metadata_frames=" << stream->metadata_frames.load()
              << " selected_rois=" << stream->selected_rois.load()
              << " completed_rois=" << stream->completed_rois.load()
              << " detector_mailbox_drops=" << stream->detector_mailbox_drops.load()
              << " pose_mailbox_drops=" << stream->pose_mailbox_drops.load()
              << " timed_out_jobs=" << stream->timed_out_jobs.load() << "\n";
  }
  const double frame_fps = elapsed_seconds > 0.0 ? total_frames / elapsed_seconds : 0.0;
  const double pose_fps = elapsed_seconds > 0.0 ? total_rois / elapsed_seconds : 0.0;
  std::cout << "[summary aggregate] elapsed_s=" << elapsed_seconds << " metadata_fps=" << frame_fps
            << " pose_fps=" << pose_fps << "\n";
}

void run_app(const AppConfig& cfg) {
  if (!fs::exists(cfg.detector_model_path)) {
    throw std::runtime_error("detector model not found: " + cfg.detector_model_path);
  }
  if (!fs::exists(cfg.pose_model_path)) {
    throw std::runtime_error("pose model not found: " + cfg.pose_model_path);
  }

  AppRuntime app;
  app.streams.reserve(cfg.streams.size());
  app.state.detector_mailboxes.resize(cfg.streams.size());
  app.state.pose_mailboxes.resize(cfg.streams.size());
  initialize_streams(app, cfg);
  build_source_run(app, cfg);
  build_detector_run(app);
  app.pose_model = make_pose_model(cfg);
  build_pose_run(app);

  g_stop_requested = 0;
  auto previous_signal = std::signal(SIGINT, request_stop);
  const auto start = Clock::now();
  std::vector<std::thread> source_pullers;
  source_pullers.reserve(app.streams.size());
  for (const auto& stream : app.streams) {
    const int stream_index = stream->index;
    source_pullers.emplace_back(
        [&, stream_index]() { pull_source_frames(app, cfg, stream_index); });
  }
  std::thread detector_dispatcher([&]() { dispatch_detector_jobs(app, cfg); });
  std::thread detector_puller([&]() { pull_detector_outputs(app, cfg); });
  std::thread pose_dispatcher([&]() { dispatch_pose_jobs(app, cfg); });
  std::thread pose_puller([&]() { pull_pose_outputs(app, cfg); });

  try {
    while (g_stop_requested == 0 && !all_streams_done(app, cfg.frame_limit)) {
      std::unique_lock<std::mutex> lock(app.state.mutex);
      app.state.cv.wait_for(lock, std::chrono::milliseconds(50), [&]() {
        return app.state.error || g_stop_requested != 0 || all_streams_done(app, cfg.frame_limit);
      });
      if (app.state.error) {
        std::rethrow_exception(app.state.error);
      }
    }
  } catch (...) {
    record_error(app);
  }

  stop_runtime(app);
  for (std::thread& puller : source_pullers) {
    puller.join();
  }
  detector_dispatcher.join();
  detector_puller.join();
  pose_dispatcher.join();
  pose_puller.join();
  std::signal(SIGINT, previous_signal);
  const double elapsed = std::chrono::duration<double>(Clock::now() - start).count();
  print_summary(app, elapsed);
  if (app.state.error) {
    std::rethrow_exception(app.state.error);
  }
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
      std::cout << "Config validated: " << cli.config_path << " (streams=" << cfg.streams.size()
                << ", max_people_per_frame=" << cfg.max_people_per_frame << ")\n";
      return 0;
    }
    run_app(cfg);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[ERR] " << error.what() << "\n";
    return 1;
  }
}
