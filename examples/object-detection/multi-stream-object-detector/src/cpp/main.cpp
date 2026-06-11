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
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct AppConfig {
  std::string model_path;
  fs::path labels_path;
  std::vector<std::string> rtsp_urls;
  int latency_ms = 100;
  bool tcp = true;
  int frames = 0;
  int fps = 0;
  double min_score = 0.55;
  double nms_iou = 0.60;
  int max_detections = 50;
  bool profile = false;
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
  std::unique_ptr<simaai::neat::Model> model;
  simaai::neat::Graph graph;
  simaai::neat::Run run;
  std::unique_ptr<simaai::neat::MetadataSender> metadata_sender;
  std::vector<std::string> labels;
  ProfileWindow profile;
  int frame_w = 0;
  int frame_h = 0;
  int output_fps = 0;
  int video_port = 0;
  int processed = 0;
  bool closed = false;
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
  sima_examples::require(cfg.rtsp_urls.size() <= 4, "this phase supports up to four streams");
  sima_examples::require(!cfg.insight_host.empty(), "output.insight.host must be set");
  sima_examples::require(cfg.latency_ms >= 0, "input.latency_ms must be >= 0");
  sima_examples::require(cfg.frames >= 0, "inference.frames must be >= 0");
  sima_examples::require(cfg.fps >= 0, "inference.fps must be >= 0");
  sima_examples::require(cfg.min_score >= 0.0 && cfg.min_score <= 1.0,
                         "inference.min_score must be between 0 and 1");
  sima_examples::require(cfg.nms_iou >= 0.0 && cfg.nms_iou <= 1.0,
                         "inference.nms_iou must be between 0 and 1");
  sima_examples::require(cfg.max_detections > 0, "inference.max_detections must be > 0");
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
  cfg.frames = raw.int_or("inference.frames", 0);
  cfg.fps = raw.int_or("inference.fps", 0);
  cfg.min_score = raw.double_or("inference.min_score", 0.55);
  cfg.nms_iou = raw.double_or("inference.nms_iou", 0.60);
  cfg.max_detections = raw.int_or("inference.max_detections", 50);
  cfg.profile = raw.bool_or("runtime.profile", false);
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
  opt.auto_caps_from_stream = true;
  if (probe.width > 0 && probe.height > 0) {
    opt.fallback_h264_width = probe.width;
    opt.fallback_h264_height = probe.height;
    width_out = probe.width;
    height_out = probe.height;
  }
  if (probe.fps > 0) {
    opt.fallback_h264_fps = probe.fps;
    fps_out = probe.fps;
  }
  return opt;
}

std::unique_ptr<simaai::neat::Model> make_model(const AppConfig& cfg) {
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

StreamRuntime build_stream(const AppConfig& cfg, int stream_index, const std::string& url,
                           const std::vector<std::string>& labels) {
  StreamRuntime runtime;
  runtime.index = stream_index;
  runtime.url = url;
  const auto source_options =
      make_source_options(cfg, url, runtime.output_fps, runtime.frame_w, runtime.frame_h);
  sima_examples::require(runtime.frame_w > 0 && runtime.frame_h > 0,
                         "failed to probe RTSP frame dimensions");
  sima_examples::require(runtime.output_fps > 0, "failed to probe RTSP frame rate");
  if (cfg.fps > 0) {
    runtime.output_fps = cfg.fps;
  }

  runtime.model = make_model(cfg);
  runtime.labels = labels;
  runtime.profile.enabled = cfg.profile;
  runtime.profile.stream_index = stream_index;

  auto source = simaai::neat::nodes::groups::RtspDecodedInput(source_options);
  const bool save_debug_frames = !cfg.save_dir.empty() && cfg.save_every > 0;
  std::vector<std::string> outputs = {"model"};
  if (cfg.video_enabled) {
    outputs.push_back("video");
  }
  if (save_debug_frames) {
    outputs.push_back("debug_frame");
  }
  auto branch = simaai::neat::graphs::Branch("source", outputs);

  runtime.graph.connect(source, branch);
  if (cfg.video_enabled) {
    auto video_options = simaai::neat::nodes::groups::VideoSenderOptions::H264RtpUdpFromRaw(
        runtime.frame_w, runtime.frame_h, runtime.output_fps);
    video_options.host = cfg.insight_host;
    video_options.channel = stream_index;
    video_options.video_port_base = cfg.video_port_base;
    video_options.encoder.bitrate_kbps = 1000;
    runtime.video_port = video_options.video_port();
    simaai::neat::Graph video_graph("video");
    video_graph.connect(simaai::neat::nodes::Input("video"),
                        simaai::neat::nodes::groups::VideoSender(video_options));
    runtime.graph.connect(branch, video_graph);
  }

  simaai::neat::Graph model_graph("model");
  model_graph.connect(simaai::neat::nodes::Input("model"), *runtime.model);
  simaai::neat::Graph detections_graph("detections");
  detections_graph.add(
      simaai::neat::nodes::Output("detections", simaai::neat::OutputOptions::EveryFrame(4)));
  runtime.graph.connect(branch, model_graph);
  runtime.graph.connect(model_graph, detections_graph);

  if (save_debug_frames) {
    simaai::neat::Graph frames("debug_frame");
    frames.add(
        simaai::neat::nodes::Output("debug_frame", simaai::neat::OutputOptions::EveryFrame(4)));
    auto debug_join = simaai::neat::graphs::Combine({"debug_frame", "detections"}, "debug_output",
                                                    simaai::neat::CombinePolicy::ByFrame);
    runtime.graph.connect(branch, frames);
    runtime.graph.connect(frames, debug_join);
    runtime.graph.connect(detections_graph, debug_join);
  }
  if (cfg.profile) {
    std::cout << "Backend stream=" << stream_index << ":\n"
              << runtime.graph.describe_backend() << "\n";
  }

  simaai::neat::RunOptions run_options;
  run_options.preset = simaai::neat::RunPreset::Realtime;
  run_options.queue_depth = 3;
  run_options.overflow_policy = simaai::neat::OverflowPolicy::KeepLatest;
  run_options.output_memory = simaai::neat::OutputMemory::ZeroCopy;
  runtime.run = runtime.graph.build(run_options);

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

void send_metadata(StreamRuntime& stream, const simaai::neat::Sample& sample,
                   const std::vector<objdet::Box>& boxes) {
  const auto metadata_boxes =
      build_metadata_boxes(boxes, stream.labels, stream.frame_w, stream.frame_h);
  const std::string data_json = sima_examples::metadata_boxes_data_json("objects", metadata_boxes);
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  const int64_t ts_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
  const int64_t frame_id = sample.frame_id >= 0 ? sample.frame_id : 0;
  std::string err;
  if (!stream.metadata_sender->send_metadata("object-detection", data_json, ts_ms,
                                             std::to_string(frame_id), &err)) {
    std::cerr << "[warn] stream " << stream.index << " metadata send failed: " << err << "\n";
  }
}

void maybe_save_debug_frame(const AppConfig& cfg, const StreamRuntime& stream,
                            const simaai::neat::Sample& sample,
                            const std::vector<objdet::Box>& boxes) {
  if (cfg.save_dir.empty() || cfg.save_every <= 0 || stream.processed % cfg.save_every != 0) {
    return;
  }

  const auto tensors = simaai::neat::tensors_from_sample(sample, false);
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
  const auto out_path = cfg.save_dir / ("stream_" + std::to_string(stream.index) + "_frame_" +
                                        std::to_string(stream.processed) + ".jpg");
  if (!cv::imwrite(out_path.string(), bgr)) {
    std::cerr << "[warn] failed to write output frame: " << out_path.string() << "\n";
  }
}

bool process_stream_once(StreamRuntime& stream, const AppConfig& cfg,
                         const std::string& output_name) {
  simaai::neat::Sample sample;
  simaai::neat::PullError pull_error;
  const double pull_start = sima_examples::time_ms();
  const auto status = stream.run.pull(output_name, 2000, sample, &pull_error);
  const double pull_end = sima_examples::time_ms();
  if (status == simaai::neat::PullStatus::Timeout) {
    return false;
  }
  if (status == simaai::neat::PullStatus::Closed) {
    stream.closed = true;
    return false;
  }
  if (status != simaai::neat::PullStatus::Ok) {
    throw std::runtime_error("stream " + std::to_string(stream.index) +
                             " failed to pull detections: " + pull_error.message);
  }

  std::vector<std::uint8_t> payload;
  std::string err;
  if (!extract_bbox_payload(sample, payload, err)) {
    throw std::runtime_error("stream " + std::to_string(stream.index) +
                             " bbox extract failed: " + err);
  }
  const auto boxes = objdet::parse_boxes_strict(payload, stream.frame_w, stream.frame_h,
                                                cfg.max_detections, false);

  const double metadata_start = sima_examples::time_ms();
  send_metadata(stream, sample, boxes);
  const double metadata_end = sima_examples::time_ms();

  ++stream.processed;
  maybe_save_debug_frame(cfg, stream, sample, boxes);
  stream.profile.add(pull_end - pull_start, metadata_end - metadata_start,
                     static_cast<int>(boxes.size()));
  return true;
}

void run_app(const AppConfig& cfg) {
  if (cfg.profile) {
    setenv("SIMA_GST_ELEMENT_TIMINGS", "1", 0);
    setenv("SIMA_GST_FLOW_DEBUG", "1", 0);
    setenv("SIMA_GST_BOUNDARY_PROBES", "1", 0);
  }
  if (!cfg.save_dir.empty()) {
    fs::create_directories(cfg.save_dir);
  }

  const auto labels = load_labels(cfg.labels_path);
  std::vector<StreamRuntime> streams;
  streams.reserve(cfg.rtsp_urls.size());
  for (std::size_t index = 0; index < cfg.rtsp_urls.size(); ++index) {
    streams.push_back(build_stream(cfg, static_cast<int>(index), cfg.rtsp_urls[index], labels));
  }

  const std::string output_name =
      (!cfg.save_dir.empty() && cfg.save_every > 0) ? "debug_output" : "detections";
  bool running = true;
  while (running) {
    running = false;
    for (auto& stream : streams) {
      if (stream.closed || (cfg.frames > 0 && stream.processed >= cfg.frames)) {
        continue;
      }
      running = true;
      (void)process_stream_once(stream, cfg, output_name);
    }
  }

  for (auto& stream : streams) {
    stream.profile.flush();
    stream.run.close();
    std::cout << "[stream " << stream.index << "] processed=" << stream.processed << "\n";
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
      std::cout << "Config validated: " << cli.config_path << " (streams=" << cfg.rtsp_urls.size()
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
