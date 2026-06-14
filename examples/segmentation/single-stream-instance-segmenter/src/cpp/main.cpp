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
#include "support/runtime/config_utils.h"
#include "support/runtime/example_utils.h"
#include <nodes/groups/VideoSender.h>
#include <nodes/io/MetadataSender.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace fs = std::filesystem;

using sima_examples::time_ms;

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
  double mask_alpha = 0.55;
  double mask_threshold = 0.50;
  bool draw_boxes = true;
};

struct CliOptions {
  fs::path config_path;
  bool validate_config_only = false;
};

struct SegmentationDetection {
  float x1 = 0.0f;
  float y1 = 0.0f;
  float x2 = 0.0f;
  float y2 = 0.0f;
  float score = 0.0f;
  int class_id = -1;
  cv::Mat mask;
};

struct ProfileWindow {
  bool enabled = false;
  int interval = 100;
  int frames = 0;
  int boxes = 0;
  double start_ms = 0.0;
  double pull_ms = 0.0;
  double decode_ms = 0.0;
  double overlay_ms = 0.0;
  double video_push_ms = 0.0;
  double metadata_ms = 0.0;

  void add(double pull, double decode, double overlay, double video_push, double metadata,
           int box_count) {
    if (!enabled) {
      return;
    }
    if (frames == 0) {
      start_ms = time_ms();
    }
    frames += 1;
    boxes += box_count;
    pull_ms += pull;
    decode_ms += decode;
    overlay_ms += overlay;
    video_push_ms += video_push;
    metadata_ms += metadata;
    if (frames >= interval) {
      flush();
    }
  }

  void flush() {
    if (!enabled || frames <= 0) {
      return;
    }
    const double elapsed_ms = std::max(time_ms() - start_ms, 1e-6);
    const double n = static_cast<double>(frames);
    const double fps = static_cast<double>(frames) * 1000.0 / elapsed_ms;
    std::cout << "[profile] frames=" << frames << " output_fps=" << fps
              << " avg_pull_ms=" << pull_ms / n << " avg_decode_ms=" << decode_ms / n
              << " avg_overlay_ms=" << overlay_ms / n << " avg_video_push_ms=" << video_push_ms / n
              << " avg_metadata_ms=" << metadata_ms / n
              << " avg_instances=" << static_cast<double>(boxes) / n << "\n";
    reset();
  }

  void reset() {
    frames = 0;
    boxes = 0;
    start_ms = 0.0;
    pull_ms = 0.0;
    decode_ms = 0.0;
    overlay_ms = 0.0;
    video_push_ms = 0.0;
    metadata_ms = 0.0;
  }
};

struct PipelineRuntime {
  std::unique_ptr<simaai::neat::Model> model;
  simaai::neat::Graph graph;
  simaai::neat::Run run;
  simaai::neat::Graph video_graph;
  simaai::neat::Run video_run;
  std::unique_ptr<simaai::neat::MetadataSender> metadata_sender;
  std::vector<std::string> labels;
  int frame_w = 0;
  int frame_h = 0;
  int output_fps = 30;
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
  sima_examples::require(cfg.mask_alpha >= 0.0 && cfg.mask_alpha <= 1.0,
                         "output.mask_alpha must be between 0 and 1");
  sima_examples::require(cfg.mask_threshold >= 0.0 && cfg.mask_threshold <= 1.0,
                         "output.mask_threshold must be between 0 and 1");
}

AppConfig load_app_config(const fs::path& config_path) {
  const auto raw = sima_examples::ScalarConfig::load(config_path);
  const fs::path default_labels =
      sima_examples::default_config_path(SIMANEAT_APPS_EXAMPLE_SOURCE_DIR).parent_path() /
      "coco_label.txt";

  AppConfig cfg;
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
  cfg.mask_alpha = raw.double_or("output.mask_alpha", 0.55);
  cfg.mask_threshold = raw.double_or("output.mask_threshold", 0.50);
  cfg.draw_boxes = raw.bool_or("output.draw_boxes", true);
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

std::vector<float> tensor_to_floats(const simaai::neat::Tensor& tensor) {
  if (tensor.dtype != simaai::neat::TensorDType::Float32) {
    throw std::runtime_error("expected Float32 tensor");
  }
  const auto bytes = tensor.copy_dense_bytes_tight();
  if (bytes.size() % sizeof(float) != 0) {
    throw std::runtime_error("float tensor byte size is not aligned");
  }
  std::vector<float> values(bytes.size() / sizeof(float));
  if (!values.empty()) {
    std::memcpy(values.data(), bytes.data(), bytes.size());
  }
  return values;
}

std::vector<std::uint8_t> tensor_to_u8(const simaai::neat::Tensor& tensor) {
  if (tensor.dtype != simaai::neat::TensorDType::UInt8) {
    throw std::runtime_error("expected UInt8 tensor");
  }
  return tensor.copy_dense_bytes_tight();
}

std::vector<SegmentationDetection>
decode_segmentation_output(const simaai::neat::TensorList& tensors, int frame_w, int frame_h,
                           int max_detections) {
  if (tensors.empty()) {
    throw std::runtime_error("model returned no segmentation tensors");
  }

  const auto decoded =
      simaai::neat::decode_segmentation(tensors, frame_w, frame_h, max_detections, false);
  std::vector<SegmentationDetection> detections;
  const size_t mask_bytes = 160U * 160U;
  for (const auto& item : decoded) {
    const auto boxes = tensor_to_floats(item.boxes);
    const auto masks = tensor_to_u8(item.masks);
    const int count = static_cast<int>(boxes.size() / 6U);
    for (int i = 0; i < count; ++i) {
      const float* row = boxes.data() + static_cast<size_t>(i) * 6U;
      if (row[2] <= row[0] || row[3] <= row[1]) {
        continue;
      }
      SegmentationDetection det;
      det.x1 = row[0];
      det.y1 = row[1];
      det.x2 = row[2];
      det.y2 = row[3];
      det.score = row[4];
      det.class_id = static_cast<int>(row[5]);
      if (masks.size() >= (static_cast<size_t>(i) + 1U) * mask_bytes) {
        cv::Mat mask(160, 160, CV_8UC1,
                     const_cast<std::uint8_t*>(masks.data() + static_cast<size_t>(i) * mask_bytes));
        det.mask = mask.clone();
      }
      detections.push_back(std::move(det));
      if (static_cast<int>(detections.size()) >= max_detections) {
        return detections;
      }
    }
  }
  return detections;
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

std::unique_ptr<simaai::neat::Model> make_model(const AppConfig& cfg) {
  simaai::neat::Model::Options opt;
  opt.preprocess.kind = simaai::neat::InputKind::Image;
  opt.preprocess.enable = simaai::neat::AutoFlag::On;
  opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::NV12;
  opt.preprocess.preset = simaai::neat::NormalizePreset::COCO_YOLO;
  opt.decode_type = simaai::neat::BoxDecodeType::YoloV26Seg;
  opt.score_threshold = cfg.min_score;
  opt.nms_iou_threshold = cfg.nms_iou;
  opt.top_k = cfg.max_detections;
  return std::make_unique<simaai::neat::Model>(cfg.model_path, opt);
}

cv::Mat tensor_bgr_from_decoded(const simaai::neat::Tensor& tensor) {
  cv::Mat bgr;
  std::string err;
  if (sima_examples::nv12_to_bgr(tensor, bgr, err)) {
    return bgr;
  }
  return tensor.to_cv_mat_copy(simaai::neat::ImageSpec::PixelFormat::BGR);
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

simaai::neat::TensorList segment_tensors_from_sample(const simaai::neat::Sample& sample) {
  return simaai::neat::tensors_from_sample(joined_field(sample, "segments", 1U), true);
}

std::string class_name(const std::vector<std::string>& labels, int class_id) {
  return class_id >= 0 && class_id < static_cast<int>(labels.size()) ? labels[class_id] : "unknown";
}

cv::Scalar class_color(int class_id) {
  static const std::vector<cv::Scalar> palette = {
      cv::Scalar(56, 56, 255),  cv::Scalar(151, 157, 255), cv::Scalar(31, 112, 255),
      cv::Scalar(29, 178, 255), cv::Scalar(49, 210, 207),  cv::Scalar(10, 249, 72),
      cv::Scalar(23, 204, 146), cv::Scalar(134, 219, 61),  cv::Scalar(52, 147, 26),
      cv::Scalar(187, 212, 0),  cv::Scalar(255, 194, 0),   cv::Scalar(168, 153, 44),
  };
  return palette[static_cast<size_t>(std::max(class_id, 0)) % palette.size()];
}

cv::Rect frame_rect_for_detection(const SegmentationDetection& det, const cv::Size& frame_size) {
  const int x0 = std::clamp(static_cast<int>(std::floor(det.x1)), 0, frame_size.width - 1);
  const int y0 = std::clamp(static_cast<int>(std::floor(det.y1)), 0, frame_size.height - 1);
  const int x1 = std::clamp(static_cast<int>(std::ceil(det.x2)), x0 + 1, frame_size.width);
  const int y1 = std::clamp(static_cast<int>(std::ceil(det.y2)), y0 + 1, frame_size.height);
  return cv::Rect(x0, y0, x1 - x0, y1 - y0);
}

cv::Rect mask_rect_for_frame_rect(const cv::Rect& frame_rect, const cv::Size& frame_size,
                                  const cv::Size& mask_size) {
  const int model_w = mask_size.width * 4;
  const int model_h = mask_size.height * 4;
  const double scale =
      std::min(static_cast<double>(model_w) / static_cast<double>(frame_size.width),
               static_cast<double>(model_h) / static_cast<double>(frame_size.height));
  const double pad_x =
      (static_cast<double>(model_w) - static_cast<double>(frame_size.width) * scale) * 0.5;
  const double pad_y =
      (static_cast<double>(model_h) - static_cast<double>(frame_size.height) * scale) * 0.5;
  const auto to_mask_x = [&](double frame_x) {
    return (frame_x * scale + pad_x) * static_cast<double>(mask_size.width) /
           static_cast<double>(model_w);
  };
  const auto to_mask_y = [&](double frame_y) {
    return (frame_y * scale + pad_y) * static_cast<double>(mask_size.height) /
           static_cast<double>(model_h);
  };

  const int x0 = std::clamp(static_cast<int>(std::floor(to_mask_x(frame_rect.x))), 0,
                            std::max(0, mask_size.width - 1));
  const int y0 = std::clamp(static_cast<int>(std::floor(to_mask_y(frame_rect.y))), 0,
                            std::max(0, mask_size.height - 1));
  const int x1 = std::clamp(static_cast<int>(std::ceil(to_mask_x(frame_rect.x + frame_rect.width))),
                            x0 + 1, mask_size.width);
  const int y1 =
      std::clamp(static_cast<int>(std::ceil(to_mask_y(frame_rect.y + frame_rect.height))), y0 + 1,
                 mask_size.height);
  return cv::Rect(x0, y0, x1 - x0, y1 - y0);
}

cv::Mat project_letterbox_mask_roi(const cv::Mat& mask, const cv::Rect& frame_rect,
                                   const cv::Size& frame_size) {
  const cv::Rect mask_rect =
      mask_rect_for_frame_rect(frame_rect, frame_size, cv::Size(mask.cols, mask.rows));
  cv::Mat projected;
  cv::resize(mask(mask_rect), projected, frame_rect.size(), 0, 0, cv::INTER_LINEAR);
  return projected;
}

void draw_box(cv::Mat& frame, const SegmentationDetection& det,
              const std::vector<std::string>& labels) {
  const cv::Rect rect = frame_rect_for_detection(det, frame.size());
  const cv::Scalar color = class_color(det.class_id);
  cv::rectangle(frame, rect, color, 2);
  cv::putText(frame,
              class_name(labels, det.class_id) + " " + std::to_string(det.score).substr(0, 4),
              cv::Point(rect.x, std::max(0, rect.y - 4)), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
              cv::LINE_AA);
}

cv::Mat overlay_segmentation(const cv::Mat& frame,
                             const std::vector<SegmentationDetection>& detections,
                             const std::vector<std::string>& labels, const AppConfig& cfg) {
  cv::Mat annotated = frame.clone();
  for (const auto& det : detections) {
    if (det.score < cfg.min_score || det.mask.empty()) {
      continue;
    }
    const cv::Rect frame_rect = frame_rect_for_detection(det, annotated.size());
    cv::Mat resized_mask = project_letterbox_mask_roi(det.mask, frame_rect, annotated.size());
    cv::Mat binary_mask;
    cv::threshold(resized_mask, binary_mask, cfg.mask_threshold * 255.0, 255, cv::THRESH_BINARY);
    if (cv::countNonZero(binary_mask) > 0) {
      cv::Mat annotated_roi = annotated(frame_rect);
      cv::Mat mask_color(frame_rect.size(), annotated.type(), class_color(det.class_id));
      cv::Mat blended;
      cv::addWeighted(annotated_roi, 1.0 - cfg.mask_alpha, mask_color, cfg.mask_alpha, 0.0,
                      blended);
      blended.copyTo(annotated_roi, binary_mask);

      std::vector<std::vector<cv::Point>> contours;
      cv::findContours(binary_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
      cv::drawContours(annotated_roi, contours, -1, class_color(det.class_id), 2);
    }
    if (cfg.draw_boxes) {
      draw_box(annotated, det, labels);
    }
  }
  return annotated;
}

std::vector<sima_examples::MetadataBox>
build_metadata_boxes(const std::vector<SegmentationDetection>& detections,
                     const std::vector<std::string>& labels, int frame_w, int frame_h) {
  std::vector<sima_examples::MetadataBox> boxes;
  boxes.reserve(detections.size());
  int object_index = 1;
  for (const auto& det : detections) {
    const cv::Rect rect = frame_rect_for_detection(det, cv::Size(frame_w, frame_h));
    sima_examples::MetadataBox obj;
    obj.id = "obj_" + std::to_string(object_index++);
    obj.label = class_name(labels, det.class_id);
    obj.confidence = det.score;
    obj.x = static_cast<float>(rect.x);
    obj.y = static_cast<float>(rect.y);
    obj.w = static_cast<float>(rect.width);
    obj.h = static_cast<float>(rect.height);
    boxes.push_back(obj);
  }
  return boxes;
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

  auto source = simaai::neat::nodes::groups::RtspDecodedInput(source_options);
  auto branch = simaai::neat::graphs::Branch("source", {"frame", "model"});

  simaai::neat::Graph frame_graph("frame");
  frame_graph.add(simaai::neat::nodes::Output("frame", simaai::neat::OutputOptions::EveryFrame(4)));

  simaai::neat::Graph model_graph("model");
  model_graph.connect(simaai::neat::nodes::Input("model"), *runtime.model);

  simaai::neat::Graph segments_graph("segments");
  segments_graph.add(
      simaai::neat::nodes::Output("segments", simaai::neat::OutputOptions::EveryFrame(4)));

  auto joined = simaai::neat::graphs::Combine({"frame", "segments"}, "segmentation_output",
                                              simaai::neat::CombinePolicy::ByFrame);

  simaai::neat::GraphLinkOptions live_link_options;
  live_link_options.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
  runtime.graph.connect(source, branch);
  runtime.graph.connect(branch, frame_graph, live_link_options);
  runtime.graph.connect(branch, model_graph, live_link_options);
  runtime.graph.connect(model_graph, segments_graph);
  runtime.graph.connect(frame_graph, joined);
  runtime.graph.connect(segments_graph, joined);
  if (cfg.profile) {
    std::cout << "Backend:\n" << runtime.graph.describe_backend() << "\n";
  }

  simaai::neat::RunOptions run_options;
  run_options.preset = simaai::neat::RunPreset::Realtime;
  run_options.queue_depth = 3;
  run_options.overflow_policy = simaai::neat::OverflowPolicy::KeepLatest;
  run_options.output_memory = simaai::neat::OutputMemory::ZeroCopy;
  runtime.run = runtime.graph.build(run_options);

  simaai::neat::InputOptions video_input;
  video_input.payload_type = simaai::neat::PayloadType::Image;
  video_input.format = "RGB";
  video_input.width = runtime.frame_w;
  video_input.height = runtime.frame_h;
  video_input.depth = 3;
  video_input.use_simaai_pool = false;

  auto video_options = simaai::neat::nodes::groups::VideoSenderOptions::H264RtpUdpFromRaw(
      runtime.frame_w, runtime.frame_h, runtime.output_fps);
  video_options.host = cfg.insight_host;
  video_options.channel = 0;
  video_options.video_port_base = cfg.video_port;
  video_options.encoder.bitrate_kbps = 1000;
  runtime.video_port = video_options.video_port();

  runtime.video_graph.add(simaai::neat::nodes::Input(video_input));
  runtime.video_graph.add(simaai::neat::nodes::groups::VideoSender(video_options));
  cv::Mat seed(runtime.frame_h, runtime.frame_w, CV_8UC3, cv::Scalar(0, 0, 0));
  runtime.video_run = runtime.video_graph.build(std::vector<cv::Mat>{seed});

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
                   const std::vector<SegmentationDetection>& detections) {
  const auto boxes =
      build_metadata_boxes(detections, runtime.labels, runtime.frame_w, runtime.frame_h);
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  const int64_t ts_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
  const int64_t frame_id = sample.frame_id >= 0 ? sample.frame_id : 0;
  std::string err;
  if (!runtime.metadata_sender->send_metadata(
          "instance-segmentation", sima_examples::metadata_boxes_data_json("objects", boxes), ts_ms,
          std::to_string(frame_id), &err)) {
    std::cerr << "[warn] insight metadata send failed: " << err << "\n";
  }
}

void push_annotated_video(PipelineRuntime& runtime, const cv::Mat& annotated_bgr) {
  cv::Mat rgb;
  cv::cvtColor(annotated_bgr, rgb, cv::COLOR_BGR2RGB);
  const auto tensor = simaai::neat::Tensor::from_cv_mat(
      rgb, simaai::neat::ImageSpec::PixelFormat::RGB, simaai::neat::TensorMemory::EV74);
  if (!runtime.video_run.push(simaai::neat::TensorList{tensor})) {
    throw std::runtime_error("Insight video push failed");
  }
}

void maybe_save_frame(const AppConfig& cfg, int processed, const cv::Mat& annotated_bgr) {
  if (cfg.save_dir.empty() || cfg.save_every <= 0 || processed % cfg.save_every != 0) {
    return;
  }
  const auto out_path = cfg.save_dir / ("frame_" + std::to_string(processed) + ".jpg");
  if (!cv::imwrite(out_path.string(), annotated_bgr)) {
    std::cerr << "[warn] failed to write output frame: " << out_path.string() << "\n";
  }
}

void run_pipeline(PipelineRuntime& runtime, const AppConfig& cfg) {
  ProfileWindow profile;
  profile.enabled = cfg.profile;
  profile.interval = cfg.profile_interval;

  int processed = 0;
  while (cfg.frames <= 0 || processed < cfg.frames) {
    simaai::neat::Sample sample;
    simaai::neat::PullError pull_error;
    const double pull_start = time_ms();
    const auto status = runtime.run.pull("segmentation_output", 20000, sample, &pull_error);
    const double pull_end = time_ms();
    if (status == simaai::neat::PullStatus::Timeout) {
      std::cerr << "[warn] timed out waiting for segmentation output\n";
      continue;
    }
    if (status == simaai::neat::PullStatus::Closed) {
      break;
    }
    if (status != simaai::neat::PullStatus::Ok) {
      throw std::runtime_error("failed to pull segmentation output: " + pull_error.message);
    }

    const double decode_start = time_ms();
    const cv::Mat frame = tensor_bgr_from_decoded(frame_tensor_from_sample(sample));
    const auto detections = decode_segmentation_output(
        segment_tensors_from_sample(sample), runtime.frame_w, runtime.frame_h, cfg.max_detections);
    const double decode_end = time_ms();

    const double overlay_start = time_ms();
    const cv::Mat annotated = overlay_segmentation(frame, detections, runtime.labels, cfg);
    const double overlay_end = time_ms();

    const double video_start = time_ms();
    push_annotated_video(runtime, annotated);
    const double video_end = time_ms();

    const double metadata_start = time_ms();
    send_metadata(runtime, sample, detections);
    const double metadata_end = time_ms();

    ++processed;
    maybe_save_frame(cfg, processed, annotated);
    profile.add(pull_end - pull_start, decode_end - decode_start, overlay_end - overlay_start,
                video_end - video_start, metadata_end - metadata_start,
                static_cast<int>(detections.size()));
  }

  profile.flush();
  std::cout << "processed=" << processed << " video_sender=" << cfg.insight_host << ":"
            << runtime.video_port << "\n";
}

} // namespace

int main(int argc, char** argv) {
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);

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
    runtime.video_run.close();
    runtime.run.close();
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "Error: " << ex.what() << "\n";
    return 2;
  }
}
