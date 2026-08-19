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
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

enum class SourceType { Rtsp, Http };
enum class SourceCodec { H264, H265, Mjpeg };

struct AppConfig {
  std::string model_path;
  fs::path labels_path;
  std::string source_url;
  SourceType source_type = SourceType::Rtsp;
  SourceCodec source_codec = SourceCodec::H264;
  int latency_ms = 200;
  bool tcp = true;
  int source_fps = 0;
  bool ssl_strict = true;
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
  /// Run output the loop pulls: the detections alone, or the frame-joined bundle when saving.
  std::string output_name;
  int frame_w = 0;
  int frame_h = 0;
  int output_fps = 0;
  int video_port = 0;
};

std::string lower_copy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

SourceType parse_source_type(const std::string& value) {
  const std::string lowered = lower_copy(value);
  if (lowered == "rtsp") {
    return SourceType::Rtsp;
  }
  if (lowered == "http" || lowered == "https") {
    return SourceType::Http;
  }
  throw std::runtime_error("source.type must be rtsp or http");
}

SourceCodec parse_source_codec(const std::string& value) {
  const std::string lowered = lower_copy(value);
  if (lowered == "h264" || lowered == "avc" || lowered == "h.264") {
    return SourceCodec::H264;
  }
  if (lowered == "h265" || lowered == "hevc" || lowered == "h.265") {
    return SourceCodec::H265;
  }
  if (lowered == "mjpeg" || lowered == "jpeg") {
    return SourceCodec::Mjpeg;
  }
  throw std::runtime_error("source.codec must be h264/avc, h265/hevc, or mjpeg");
}

const char* source_type_name(SourceType value) {
  return value == SourceType::Rtsp ? "rtsp" : "http";
}

const char* source_codec_name(SourceCodec value) {
  if (value == SourceCodec::H264)
    return "h264";
  return value == SourceCodec::H265 ? "h265" : "mjpeg";
}

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
  sima_examples::require(!cfg.source_url.empty(), "source.url or source.rtsp_url must be set");
  sima_examples::require(!cfg.model_path.empty(), "model.path must be set");
  sima_examples::require(!cfg.labels_path.empty(), "model.labels must be set");
  sima_examples::require(!cfg.insight_host.empty(), "output.insight.host must be set");
  sima_examples::require(cfg.latency_ms >= 0, "source.latency_ms must be >= 0");
  sima_examples::require(cfg.source_fps >= 0, "source.fps must be >= 0");
  if (cfg.source_type == SourceType::Http) {
    sima_examples::require(cfg.source_codec == SourceCodec::Mjpeg,
                           "source.codec must be mjpeg for source.type=http");
  }
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
  const std::string legacy_rtsp_url = raw.string_or("source.rtsp_url", "");
  cfg.source_url = raw.string_or("source.url", legacy_rtsp_url);
  cfg.source_type = parse_source_type(raw.string_or("source.type", "rtsp"));
  cfg.source_codec = parse_source_codec(raw.string_or("source.codec", "h264"));
  cfg.latency_ms = raw.int_or("source.latency_ms", 200);
  cfg.tcp = raw.bool_or("source.tcp", true);
  cfg.source_fps = raw.int_or("source.fps", 0);
  cfg.ssl_strict = raw.bool_or("source.ssl_strict", true);
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

const simaai::neat::Sample* find_field(const simaai::neat::Sample& sample,
                                       const std::string& label) {
  if (sample.stream_label == label) {
    return &sample;
  }
  for (const auto& field : sample.fields) {
    if (const auto* found = find_field(field, label)) {
      return found;
    }
  }
  return nullptr;
}

const simaai::neat::Sample& joined_field(const simaai::neat::Sample& sample,
                                         const std::string& label, size_t bundle_index) {
  if (const auto* field = find_field(sample, label)) {
    return *field;
  }
  if (sample.kind == simaai::neat::SampleKind::Bundle && sample.fields.size() > bundle_index) {
    return sample.fields[bundle_index];
  }
  throw std::runtime_error("joined output missing " + label + " field");
}

simaai::neat::Tensor frame_tensor_from_sample(const simaai::neat::Sample& sample) {
  const auto tensors = simaai::neat::tensors_from_sample(joined_field(sample, "frame", 0U), true);
  return tensors.front();
}

cv::Mat tensor_bgr_from_decoded(const simaai::neat::Tensor& tensor) {
  cv::Mat bgr;
  std::string err;
  if (sima_examples::nv12_to_bgr(tensor, bgr, err)) {
    return bgr;
  }
  return tensor.to_cv_mat_copy(simaai::neat::ImageSpec::PixelFormat::BGR);
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

struct SourceGeometry {
  int width = 0;
  int height = 0;
  int fps = 0;
};

int fps_from_rate(const std::string& value) {
  if (value.empty() || value == "0/0" || value == "0/1")
    return 0;
  try {
    const auto slash = value.find('/');
    double fps = 0.0;
    if (slash == std::string::npos) {
      fps = std::stod(value);
    } else {
      const double den = std::stod(value.substr(slash + 1));
      if (den <= 0.0)
        return 0;
      fps = std::stod(value.substr(0, slash)) / den;
    }
    return fps > 0.0 ? static_cast<int>(std::lround(fps)) : 0;
  } catch (...) {
    return 0;
  }
}

std::string shell_quote(const std::string& value) {
  std::string out = "'";
  for (const char c : value) {
    out += c == '\'' ? "'\\''" : std::string(1, c);
  }
  out += "'";
  return out;
}

SourceGeometry probe_ffprobe_geometry(const AppConfig& cfg) {
  SourceGeometry geometry;
  std::string command =
      "ffprobe -v error -rw_timeout 5000000 -select_streams v:0 "
      "-show_entries stream=width,height,r_frame_rate,avg_frame_rate -of default=nw=1 ";
  if (!cfg.ssl_strict) {
    command += "-tls_verify 0 ";
  }
  command += shell_quote(cfg.source_url) + " 2>/dev/null";

  FILE* pipe = popen(command.c_str(), "r");
  if (!pipe) {
    return geometry;
  }

  int avg_fps = 0;
  int r_fps = 0;
  std::array<char, 256> buffer{};
  while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe)) {
    std::string line(buffer.data());
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
      line.pop_back();
    }
    const auto eq = line.find('=');
    if (eq == std::string::npos) {
      continue;
    }
    const std::string key = line.substr(0, eq);
    const std::string value = line.substr(eq + 1);
    if (key == "width") {
      geometry.width = std::atoi(value.c_str());
    } else if (key == "height") {
      geometry.height = std::atoi(value.c_str());
    } else if (key == "avg_frame_rate") {
      avg_fps = fps_from_rate(value);
    } else if (key == "r_frame_rate") {
      r_fps = fps_from_rate(value);
    }
  }
  pclose(pipe);
  geometry.fps = avg_fps > 0 ? avg_fps : r_fps;
  return geometry;
}

void fill_missing_geometry(SourceGeometry& dst, const SourceGeometry& src) {
  if (dst.width <= 0)
    dst.width = src.width;
  if (dst.height <= 0)
    dst.height = src.height;
  if (dst.fps <= 0)
    dst.fps = src.fps;
}

void require_mjpeg_fps(const AppConfig& cfg, const SourceGeometry& geometry) {
  if (cfg.source_codec == SourceCodec::Mjpeg && geometry.fps <= 0) {
    throw std::runtime_error(
        "MJPEG source did not provide a valid frame rate; set source.fps or use a source with "
        "probeable FPS metadata");
  }
}

simaai::neat::nodes::groups::RtspDecodedInputOptions
make_rtsp_source_options(const AppConfig& cfg, const SourceGeometry& geometry) {
  simaai::neat::nodes::groups::RtspDecodedInputOptions opt;
  opt.url = cfg.source_url;
  opt.latency_ms = cfg.latency_ms;
  opt.tcp = cfg.tcp;
  opt.insert_queue = true;
  opt.out_format = "NV12";
  opt.decoder_name = "decoder";
  opt.decoder_raw_output = true;
  opt.auto_caps_from_stream = true;
  opt.source_fps = geometry.fps;
  opt.codec = cfg.source_codec == SourceCodec::H264 ? simaai::neat::nodes::groups::RtspCodec::H264
              : cfg.source_codec == SourceCodec::H265
                  ? simaai::neat::nodes::groups::RtspCodec::H265
                  : simaai::neat::nodes::groups::RtspCodec::MJPEG;
  if (cfg.source_codec == SourceCodec::H264) {
    opt.payload_type = 96;
    opt.fallback_h264_width = geometry.width;
    opt.fallback_h264_height = geometry.height;
  } else if (cfg.source_codec == SourceCodec::H265) {
    opt.payload_type = 96;
    opt.dec_width = geometry.width;
    opt.dec_height = geometry.height;
  } else {
    opt.mjpeg_payload_type = 26;
    opt.dec_width = geometry.width;
    opt.dec_height = geometry.height;
  }
  if (geometry.width > 0 && geometry.height > 0 && geometry.fps > 0) {
    opt.output_caps.enable = true;
    opt.output_caps.format = "NV12";
    opt.output_caps.width = geometry.width;
    opt.output_caps.height = geometry.height;
    opt.output_caps.fps = geometry.fps;
    opt.output_caps.memory = simaai::neat::CapsMemory::Any;
  }
  return opt;
}

simaai::neat::nodes::groups::HttpMjpegDecodedInputOptions
make_http_mjpeg_source_options(const AppConfig& cfg, const SourceGeometry& geometry) {
  simaai::neat::nodes::groups::HttpMjpegDecodedInputOptions opt;
  opt.url = cfg.source_url;
  opt.decoder_name = "decoder";
  opt.decoder_raw_output = true;
  opt.source_fps = geometry.fps;
  opt.ssl_strict = cfg.ssl_strict;
  if (geometry.width > 0 && geometry.height > 0 && geometry.fps > 0) {
    opt.output_caps.enable = true;
    opt.output_caps.format = "NV12";
    opt.output_caps.width = geometry.width;
    opt.output_caps.height = geometry.height;
    opt.output_caps.fps = geometry.fps;
    opt.output_caps.memory = simaai::neat::CapsMemory::Any;
  }
  return opt;
}

simaai::neat::Graph make_source_graph(const AppConfig& cfg, const SourceGeometry& geometry) {
  if (cfg.source_type == SourceType::Rtsp) {
    return simaai::neat::nodes::groups::RtspDecodedInput(make_rtsp_source_options(cfg, geometry));
  }
  return simaai::neat::nodes::groups::HttpMjpegDecodedInput(
      make_http_mjpeg_source_options(cfg, geometry));
}

SourceGeometry probe_rtsp_h264_geometry(const AppConfig& cfg) {
  sima_examples::RtspStreamInfo probe;
  sima_examples::RtspProbeOptions probe_options;
  probe_options.payload_type = 96;
  probe_options.latency_ms = cfg.latency_ms;
  probe_options.rtsp_tcp = cfg.tcp;
  probe_options.debug = cfg.profile;
  (void)sima_examples::probe_rtsp_stream_info(cfg.source_url, probe_options, probe);

  SourceGeometry geometry;
  geometry.width = probe.width;
  geometry.height = probe.height;
  geometry.fps = probe.fps;
  return geometry;
}

SourceGeometry probe_rtsp_geometry(const AppConfig& cfg) {
  SourceGeometry geometry = probe_ffprobe_geometry(cfg);
  if (cfg.source_fps > 0) {
    geometry.fps = cfg.source_fps;
  }

  if (cfg.source_codec == SourceCodec::H264) {
    SourceGeometry rtsp_geometry = probe_rtsp_h264_geometry(cfg);
    if (cfg.source_fps > 0) {
      rtsp_geometry.fps = cfg.source_fps;
    }
    fill_missing_geometry(geometry, rtsp_geometry);
    return geometry;
  }

  if (geometry.width <= 0 || geometry.height <= 0 || geometry.fps <= 0) {
    cv::VideoCapture cap(cfg.source_url);
    if (!cap.isOpened()) {
      throw std::runtime_error("failed to open RTSP source for probing: " + cfg.source_url);
    }
    SourceGeometry cv_geometry;
    cv_geometry.width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    cv_geometry.height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    cv_geometry.fps = static_cast<int>(std::lround(cap.get(cv::CAP_PROP_FPS)));
    cap.release();
    fill_missing_geometry(geometry, cv_geometry);
  }
  if (cfg.source_fps > 0) {
    geometry.fps = cfg.source_fps;
  }
  require_mjpeg_fps(cfg, geometry);
  return geometry;
}

SourceGeometry probe_decoded_source_geometry(const AppConfig& cfg, int fps) {
  SourceGeometry geometry;
  geometry.fps = fps;

  simaai::neat::Graph probe_graph("source_probe");
  probe_graph.add(make_source_graph(cfg, geometry));
  probe_graph.add(simaai::neat::nodes::Output("frame", simaai::neat::OutputOptions::EveryFrame(1)));

  simaai::neat::RunOptions run_options;
  run_options.preset = simaai::neat::RunPreset::Realtime;
  run_options.queue_depth = 3;
  run_options.overflow_policy = simaai::neat::OverflowPolicy::KeepLatest;
  run_options.output_memory = simaai::neat::OutputMemory::ZeroCopy;
  simaai::neat::Run run = probe_graph.build(run_options);

  simaai::neat::Sample sample;
  simaai::neat::PullError pull_error;
  const auto status = run.pull("frame", 20000, sample, &pull_error);
  run.close();
  if (status != simaai::neat::PullStatus::Ok) {
    throw std::runtime_error("failed to probe decoded source frame: " + pull_error.message);
  }

  const auto tensors = simaai::neat::tensors_from_sample(sample, false);
  if (!tensors.empty()) {
    (void)sima_examples::infer_dims(tensors.front(), geometry.width, geometry.height);
  }
  return geometry;
}

SourceGeometry resolve_source_geometry(const AppConfig& cfg) {
  if (cfg.source_type == SourceType::Rtsp) {
    return probe_rtsp_geometry(cfg);
  }
  SourceGeometry geometry = probe_ffprobe_geometry(cfg);
  if (cfg.source_fps > 0) {
    geometry.fps = cfg.source_fps;
  }
  require_mjpeg_fps(cfg, geometry);
  if (geometry.width <= 0 || geometry.height <= 0) {
    fill_missing_geometry(geometry, probe_decoded_source_geometry(cfg, geometry.fps));
  }
  return geometry;
}

std::unique_ptr<simaai::neat::Model> make_model(const AppConfig& cfg,
                                                const SourceGeometry& geometry) {
  // Model options
  simaai::neat::Model::Options model_opt;
  model_opt.preprocess.kind = simaai::neat::InputKind::Image;
  model_opt.preprocess.enable = simaai::neat::AutoFlag::On;
  model_opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::NV12;
  if (geometry.width > 0 && geometry.height > 0) {
    model_opt.preprocess.input_max_width = geometry.width;
    model_opt.preprocess.input_max_height = geometry.height;
  }
  model_opt.preprocess.preset = simaai::neat::NormalizePreset::COCO_YOLO;
  model_opt.decode_type = simaai::neat::BoxDecodeType::YoloV26;
  model_opt.score_threshold = cfg.min_score;
  model_opt.nms_iou_threshold = cfg.nms_iou;
  model_opt.top_k = cfg.max_detections;
  return std::make_unique<simaai::neat::Model>(cfg.model_path, model_opt);
}

PipelineRuntime build_pipeline(const AppConfig& cfg) {
  PipelineRuntime runtime;
  const SourceGeometry geometry = resolve_source_geometry(cfg);
  runtime.frame_w = geometry.width;
  runtime.frame_h = geometry.height;
  runtime.output_fps = geometry.fps;
  sima_examples::require(runtime.frame_w > 0 && runtime.frame_h > 0,
                         "failed to probe source frame dimensions");
  sima_examples::require(runtime.output_fps > 0, "failed to resolve source frame rate");

  runtime.model = make_model(cfg, geometry);
  runtime.labels = load_labels(cfg.labels_path);

  auto video_options = simaai::neat::nodes::groups::VideoSenderOptions::H264RtpUdpFromRaw(
      runtime.frame_w, runtime.frame_h, runtime.output_fps);
  video_options.host = cfg.insight_host;
  video_options.channel = 0;
  video_options.video_port_base = cfg.video_port;
  video_options.encoder.bitrate_kbps = 1000;
  runtime.video_port = video_options.video_port();

  // Insight correlates the RTP timestamp with the metadata timestamp, so the encoder and the
  // detections must stay in one Run and therefore on one GStreamer timeline.
  const bool save_frames = !cfg.save_dir.empty();
  auto source = make_source_graph(cfg, geometry);
  auto branch = save_frames ? simaai::neat::graphs::Branch("source", {"video", "model", "frame"})
                            : simaai::neat::graphs::Branch("source", {"video", "model"});

  simaai::neat::Graph video_graph("video");
  video_graph.connect(simaai::neat::nodes::Input("video"),
                      simaai::neat::nodes::groups::VideoSender(video_options));

  simaai::neat::Graph model_graph("model");
  model_graph.connect(simaai::neat::nodes::Input("model"), *runtime.model);
  simaai::neat::Graph detections_graph("detections");
  detections_graph.add(
      simaai::neat::nodes::Output("detections", simaai::neat::OutputOptions::EveryFrame(4)));

  runtime.graph.connect(source, branch);
  runtime.graph.connect(branch, video_graph);
  runtime.graph.connect(branch, model_graph);
  runtime.graph.connect(model_graph, detections_graph);
  if (save_frames) {
    simaai::neat::Graph frame_graph("frame");
    frame_graph.add(
        simaai::neat::nodes::Output("frame", simaai::neat::OutputOptions::EveryFrame(4)));
    auto joined = simaai::neat::graphs::Combine({"frame", "detections"}, "detector_output",
                                                simaai::neat::CombinePolicy::ByFrame);
    runtime.graph.connect(branch, frame_graph);
    runtime.graph.connect(frame_graph, joined);
    runtime.graph.connect(detections_graph, joined);
  }
  runtime.output_name = save_frames ? "detector_output" : "detections";
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

  std::cout << "source=" << cfg.source_url << " type=" << source_type_name(cfg.source_type)
            << " codec=" << source_codec_name(cfg.source_codec) << " stream=" << runtime.frame_w
            << "x" << runtime.frame_h << "@" << runtime.output_fps
            << " insight=" << cfg.insight_host << " video=" << runtime.video_port
            << " metadata=" << runtime.metadata_sender->metadata_port() << " channel=0\n";
  return runtime;
}

void send_metadata(PipelineRuntime& runtime, const simaai::neat::Sample& sample,
                   const std::vector<objdet::Box>& boxes) {
  const auto metadata_boxes =
      build_metadata_boxes(boxes, runtime.labels, runtime.frame_w, runtime.frame_h);
  const std::string data_json = sima_examples::metadata_boxes_data_json("objects", metadata_boxes);
  const int64_t ts_ms = sample.pts_ns >= 0 ? sample.pts_ns / 1'000'000 : -1;
  const std::string frame_id = sample.frame_id >= 0 ? std::to_string(sample.frame_id) : "";
  std::string err;
  if (!runtime.metadata_sender->send_metadata("object-detection", data_json, ts_ms, frame_id,
                                              &err)) {
    std::cerr << "[warn] insight metadata send failed: " << err << "\n";
  }
}

void maybe_save_debug_frame(const AppConfig& cfg, int processed, const simaai::neat::Sample& sample,
                            const std::vector<objdet::Box>& boxes) {
  if (cfg.save_dir.empty() || cfg.save_every <= 0 || processed % cfg.save_every != 0) {
    return;
  }

  cv::Mat bgr = tensor_bgr_from_decoded(frame_tensor_from_sample(sample));
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

  int processed = 0;
  while (cfg.frames <= 0 || processed < cfg.frames) {
    simaai::neat::Sample detection_sample;
    simaai::neat::PullError pull_error;
    const double pull_start = sima_examples::time_ms();
    const auto status = runtime.run.pull(runtime.output_name, 20000, detection_sample, &pull_error);
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
