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
#include "neat/nodes.h"
#include "neat/node_groups.h"
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
  std::string rtsp_url;
  int latency_ms = 200;
  bool tcp = true;
  int frames = 0;
  double min_score = 0.55;
  double nms_iou = 0.60;
  int max_detections = 50;
  bool profile = false;
  int profile_interval = 100;
  std::string insight_host = "127.0.0.1";
  int video_port = 9000;
  int metadata_port = 9100;
  fs::path save_dir;
  int save_every = 0;
};

struct CliOptions {
  fs::path config_path;
  bool validate_config_only = false;
};

struct ProfileWindow {
  bool enabled = false;
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
    const auto avg = [this](double value) { return value / static_cast<double>(frames); };
    const double output_fps = elapsed > 0.0 ? static_cast<double>(frames) * 1000.0 / elapsed : 0.0;
    std::cout << "[profile] frames=" << frames << " output_fps=" << output_fps
              << " avg_detection_pull_ms=" << avg(detection_pull_ms)
              << " avg_metadata_send_ms=" << avg(metadata_send_ms)
              << " avg_boxes=" << static_cast<double>(boxes) / static_cast<double>(frames) << "\n";
    frames = 0;
    boxes = 0;
    start_ms = 0.0;
    detection_pull_ms = 0.0;
    metadata_send_ms = 0.0;
  }
};

struct PipelineRuntime {
  std::unique_ptr<simaai::neat::Model> model;
  simaai::neat::Graph graph;
  simaai::neat::Run run;
  std::unique_ptr<simaai::neat::MetadataSender> metadata_sender;
  std::vector<std::string> labels;
  int frame_w = 0;
  int frame_h = 0;
  int output_fps = 0;
  int video_port = 0;
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

void validate_config(const AppConfig& cfg) {
  sima_examples::require(!cfg.rtsp_url.empty(), "source.rtsp_url must be set");
  sima_examples::require(!cfg.model_path.empty(), "model.path must be set");
  sima_examples::require(!cfg.labels_path.empty(), "model.labels must be set");
  sima_examples::require(!cfg.insight_host.empty(), "output.insight.host must be set");
  sima_examples::require(cfg.latency_ms >= 0, "source.latency_ms must be >= 0");
  sima_examples::require(cfg.frames >= 0, "inference.frames must be >= 0");
  sima_examples::require(cfg.min_score >= 0.0 && cfg.min_score <= 1.0,
                         "inference.min_score must be between 0 and 1");
  sima_examples::require(cfg.nms_iou >= 0.0 && cfg.nms_iou <= 1.0,
                         "inference.nms_iou must be between 0 and 1");
  sima_examples::require(cfg.max_detections > 0, "inference.max_detections must be > 0");
  sima_examples::require(cfg.profile_interval > 0, "runtime.profile_interval must be > 0");
  sima_examples::require(cfg.video_port > 0, "output.insight.video_port must be > 0");
  sima_examples::require(cfg.metadata_port > 0, "output.insight.metadata_port must be > 0");
  sima_examples::require(cfg.save_every >= 0, "output.save_every must be >= 0");
}

AppConfig load_app_config(const fs::path& config_path) {
  const auto raw = sima_examples::ScalarConfig::load(config_path);
  AppConfig cfg;
  const auto default_labels =
      fs::path(SIMANEAT_APPS_EXAMPLE_SOURCE_DIR).parent_path() / "common" / "coco_label.txt";
  cfg.model_path = raw.string_or("model.path", "");
  cfg.labels_path = raw.string_or("model.labels", default_labels.string());
  cfg.rtsp_url = raw.string_or("source.rtsp_url", "");
  cfg.latency_ms = raw.int_or("source.latency_ms", 200);
  cfg.tcp = raw.bool_or("source.tcp", true);
  cfg.frames = raw.int_or("inference.frames", 0);
  cfg.min_score = raw.double_or("inference.min_score", 0.55);
  cfg.nms_iou = raw.double_or("inference.nms_iou", 0.60);
  cfg.max_detections = raw.int_or("inference.max_detections", 50);
  cfg.profile = raw.bool_or("runtime.profile", false);
  cfg.profile_interval = raw.int_or("runtime.profile_interval", 100);
  cfg.insight_host = raw.string_or("output.insight.host", "");
  cfg.video_port = raw.int_or("output.insight.video_port", 9000);
  cfg.metadata_port = raw.int_or("output.insight.metadata_port", 9100);
  cfg.save_dir = raw.string_or("output.save_dir", "");
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
make_source_options(const AppConfig& cfg, int& fps_out, int& width_out, int& height_out) {
  sima_examples::RtspStreamInfo probe;
  sima_examples::RtspProbeOptions probe_options;
  probe_options.payload_type = 96;
  probe_options.latency_ms = cfg.latency_ms;
  probe_options.rtsp_tcp = cfg.tcp;
  probe_options.debug = cfg.profile;
  (void)sima_examples::probe_rtsp_stream_info(cfg.rtsp_url, probe_options, probe);

  // RTSP options
  simaai::neat::nodes::groups::RtspDecodedInputOptions opt;
  opt.url = cfg.rtsp_url;
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

simaai::neat::Graph
make_encoded_rtsp_source(const simaai::neat::nodes::groups::RtspDecodedInputOptions& opt) {
  simaai::neat::Graph source("rtsp_encoded_source");
  const bool use_auto_caps = opt.auto_caps_from_stream &&
                             (opt.h264_fps <= 0 || opt.h264_width <= 0 || opt.h264_height <= 0);
  const bool insert_queue = opt.insert_queue && !opt.sync_mode;
  source.add(simaai::neat::nodes::RTSPInput(opt.url, opt.latency_ms, opt.tcp));
  if (insert_queue) {
    source.add(simaai::neat::nodes::Queue());
  }
  source.add(simaai::neat::nodes::H264Depacketize(opt.payload_type, opt.h264_parse_config_interval,
                                                  opt.h264_fps, opt.h264_width, opt.h264_height,
                                                  /*enforce_h264_caps=*/!use_auto_caps));
  if (insert_queue) {
    source.add(simaai::neat::nodes::Queue());
  }
  if (use_auto_caps) {
    source.add(simaai::neat::nodes::H264CapsFixup(opt.fallback_h264_fps, opt.fallback_h264_width,
                                                  opt.fallback_h264_height));
  }
  return source;
}

simaai::neat::Graph
make_h264_decode_graph(const std::string& input_name, const std::string& output_name,
                       const simaai::neat::nodes::groups::RtspDecodedInputOptions& opt) {
  simaai::neat::Graph decode("rtsp_h264_decode");
  const int dec_w = (opt.h264_width > 0) ? opt.h264_width : opt.fallback_h264_width;
  const int dec_h = (opt.h264_height > 0) ? opt.h264_height : opt.fallback_h264_height;
  const int dec_fps = (opt.h264_fps > 0) ? opt.h264_fps : opt.fallback_h264_fps;

  decode.add(simaai::neat::nodes::Input(input_name));
  decode.add(simaai::neat::nodes::H264Decode(opt.sima_allocator_type, opt.out_format,
                                             opt.decoder_name, opt.decoder_raw_output,
                                             opt.decoder_next_element, dec_w, dec_h, dec_fps));
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
  decode.add(simaai::neat::nodes::Output(output_name));
  return decode;
}

std::unique_ptr<simaai::neat::Model> make_model(const AppConfig& cfg) {
  // Model options
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

PipelineRuntime build_pipeline(const AppConfig& cfg) {
  PipelineRuntime runtime;
  const auto source_options =
      make_source_options(cfg, runtime.output_fps, runtime.frame_w, runtime.frame_h);
  sima_examples::require(runtime.frame_w > 0 && runtime.frame_h > 0,
                         "failed to probe RTSP frame dimensions");
  sima_examples::require(runtime.output_fps > 0, "failed to probe RTSP frame rate");

  runtime.model = make_model(cfg);
  runtime.labels = load_labels(cfg.labels_path);

  const bool save_debug_frames = !cfg.save_dir.empty() && cfg.save_every > 0;
  auto source = make_encoded_rtsp_source(source_options);
  auto encoded_branch = simaai::neat::graphs::Branch("encoded_source", {"video", "decode"});
  auto decode_graph = make_h264_decode_graph(
      "decode", save_debug_frames ? "decoded_source" : "model", source_options);

  auto video_options = simaai::neat::nodes::groups::VideoSenderOptions::H264RtpUdpFromEncoded();
  video_options.host = cfg.insight_host;
  video_options.channel = 0;
  video_options.video_port_base = cfg.video_port;
  runtime.video_port = video_options.video_port();
  simaai::neat::Graph video_graph("video");
  video_graph.connect(simaai::neat::nodes::Input("video"),
                      simaai::neat::nodes::groups::VideoSender(video_options));

  simaai::neat::Graph model_graph("model");
  model_graph.connect(simaai::neat::nodes::Input("model"), *runtime.model);
  simaai::neat::Graph detections_graph("detections");
  detections_graph.add(
      simaai::neat::nodes::Output("detections", simaai::neat::OutputOptions::EveryFrame(4)));

  simaai::neat::GraphLinkOptions live_link_options;
  live_link_options.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
  runtime.graph.connect(source, encoded_branch);
  runtime.graph.connect(encoded_branch, video_graph, live_link_options);
  if (save_debug_frames) {
    auto decoded_branch = simaai::neat::graphs::Branch("decoded_source", {"model", "debug_frame"});
    runtime.graph.connect(encoded_branch, decode_graph);
    runtime.graph.connect(decode_graph, decoded_branch);
    runtime.graph.connect(decoded_branch, model_graph);
    simaai::neat::Graph frames("debug_frame");
    frames.add(
        simaai::neat::nodes::Output("debug_frame", simaai::neat::OutputOptions::EveryFrame(4)));
    auto debug_join = simaai::neat::graphs::Combine({"debug_frame", "detections"}, "debug_output",
                                                    simaai::neat::CombinePolicy::ByPts);
    runtime.graph.connect(decoded_branch, frames);
    runtime.graph.connect(frames, debug_join);
    runtime.graph.connect(detections_graph, debug_join);
  } else {
    runtime.graph.connect(encoded_branch, decode_graph, live_link_options);
    runtime.graph.connect(decode_graph, model_graph);
  }
  runtime.graph.connect(model_graph, detections_graph);
  if (cfg.profile) {
    std::cout << "Backend:\n" << runtime.graph.describe_backend() << "\n";
  }

  // Runtime options
  simaai::neat::RunOptions run_options;
  run_options.preset = simaai::neat::RunPreset::Realtime;
  run_options.queue_depth = 3;
  run_options.overflow_policy = simaai::neat::OverflowPolicy::KeepLatest;
  run_options.output_memory = simaai::neat::OutputMemory::ZeroCopy;
  runtime.run = runtime.graph.build(run_options);

  simaai::neat::MetadataSenderOptions metadata_options;
  metadata_options.host = cfg.insight_host;
  metadata_options.channel = 0;
  metadata_options.metadata_port_base = cfg.metadata_port;
  std::string metadata_err;
  runtime.metadata_sender =
      std::make_unique<simaai::neat::MetadataSender>(metadata_options, &metadata_err);
  sima_examples::require(runtime.metadata_sender->ok(), metadata_err);

  std::cout << "rtsp=" << cfg.rtsp_url << " stream=" << runtime.frame_w << "x" << runtime.frame_h
            << "@" << runtime.output_fps << " insight=" << cfg.insight_host
            << " video=" << runtime.video_port
            << " metadata=" << runtime.metadata_sender->metadata_port() << " channel=0\n";
  return runtime;
}

void send_metadata(PipelineRuntime& runtime, const simaai::neat::Sample& sample,
                   const std::vector<objdet::Box>& boxes) {
  const auto metadata_boxes =
      build_metadata_boxes(boxes, runtime.labels, runtime.frame_w, runtime.frame_h);
  const std::string data_json = sima_examples::metadata_boxes_data_json("objects", metadata_boxes);
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  const int64_t ts_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
  const int64_t frame_id = sample.frame_id >= 0 ? sample.frame_id : 0;
  std::string err;
  if (!runtime.metadata_sender->send_metadata("object-detection", data_json, ts_ms,
                                              std::to_string(frame_id), &err)) {
    std::cerr << "[warn] insight metadata send failed: " << err << "\n";
  }
}

void maybe_save_debug_frame(const AppConfig& cfg, int processed, const simaai::neat::Sample& sample,
                            const std::vector<objdet::Box>& boxes) {
  if (cfg.save_dir.empty() || cfg.save_every <= 0 || processed % cfg.save_every != 0) {
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
  const auto out_path = cfg.save_dir / ("frame_" + std::to_string(processed) + ".jpg");
  if (!cv::imwrite(out_path.string(), bgr)) {
    std::cerr << "[warn] failed to write output frame: " << out_path.string() << "\n";
  }
}

void run_pipeline(PipelineRuntime& runtime, const AppConfig& cfg) {
  ProfileWindow profile;
  profile.enabled = cfg.profile;
  profile.interval = cfg.profile_interval;

  const std::string output_name =
      (!cfg.save_dir.empty() && cfg.save_every > 0) ? "debug_output" : "detections";
  int processed = 0;
  while (cfg.frames <= 0 || processed < cfg.frames) {
    simaai::neat::Sample detection_sample;
    simaai::neat::PullError pull_error;
    const double pull_start = sima_examples::time_ms();
    const auto status = runtime.run.pull(output_name, 20000, detection_sample, &pull_error);
    const double pull_end = sima_examples::time_ms();
    if (status == simaai::neat::PullStatus::Timeout) {
      std::cerr << "[warn] timed out waiting for detections\n";
      continue;
    }
    if (status == simaai::neat::PullStatus::Closed) {
      break;
    }
    if (status != simaai::neat::PullStatus::Ok) {
      throw std::runtime_error("failed to pull detections: " + pull_error.message);
    }

    std::vector<std::uint8_t> payload;
    std::string err;
    if (!extract_bbox_payload(detection_sample, payload, err)) {
      throw std::runtime_error("bbox extract failed: " + err);
    }
    const auto boxes = objdet::parse_boxes_strict(payload, runtime.frame_w, runtime.frame_h,
                                                  cfg.max_detections, false);

    const double metadata_start = sima_examples::time_ms();
    send_metadata(runtime, detection_sample, boxes);
    const double metadata_end = sima_examples::time_ms();

    ++processed;
    maybe_save_debug_frame(cfg, processed, detection_sample, boxes);
    profile.add(pull_end - pull_start, metadata_end - metadata_start,
                static_cast<int>(boxes.size()));
  }

  profile.flush();
  std::cout << "processed=" << processed << " video_sender=" << cfg.insight_host << ":"
            << runtime.video_port << "\n";
}

} // namespace

int main(int argc, char** argv) {
  try {
    const CliOptions cli = parse_args(argc, argv);
    const AppConfig cfg = load_app_config(cli.config_path);
    if (cli.validate_config_only) {
      std::cout << "Config validated: " << cli.config_path << "\n";
      return 0;
    }
    if (!cfg.save_dir.empty()) {
      fs::create_directories(cfg.save_dir);
    }

    if (cfg.profile) {
      setenv("SIMA_GST_ELEMENT_TIMINGS", "1", 0);
      setenv("SIMA_GST_FLOW_DEBUG", "1", 0);
      setenv("SIMA_GST_BOUNDARY_PROBES", "1", 0);
    }
    PipelineRuntime runtime = build_pipeline(cfg);
    run_pipeline(runtime, cfg);
    runtime.run.close();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[ERR] " << e.what() << "\n";
    return 1;
  }
}
