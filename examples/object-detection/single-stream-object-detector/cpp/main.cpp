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

#include "support/runtime/example_utils.h"
#include "support/runtime/config_utils.h"
#include "support/object_detection/obj_detection_utils.h"
#include "neat.h"
#include "neat/models.h"
#include "neat/nodes.h"
#include "neat/node_groups.h"
#include <nodes/groups/VideoSender.h>
#include <nodes/io/MetadataSender.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

using sima_examples::time_ms;

namespace {

// single-stream-object-detector is a reference pipeline for the common deployment:
// one RTSP source, one YOLO detector, and Insight video/metadata output.
//
// The code keeps ingest, inference, and output transport separate so each
// stage can be reasoned about and debugged independently.

struct ModelConfig {
  std::string path;
};

struct SourceConfig {
  std::string rtsp_url;
  int latency_ms = 200;
  bool tcp = true;
};

struct InferenceConfig {
  int frames = 0;
  double min_score = 0.55;
  double nms_iou = 0.50;
  int max_detections = 100;
};

struct RuntimeConfig {
  bool profile = false;
  int profile_interval = 100;
};

struct InsightConfig {
  std::string host = "127.0.0.1";
  int video_port = 9000;
  int metadata_port = 9100;
};

struct AppConfig {
  ModelConfig model;
  SourceConfig source;
  InferenceConfig inference;
  RuntimeConfig runtime;
  InsightConfig insight;
};

struct CliOptions {
  fs::path config_path;
  bool validate_config_only = false;
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
  sima_examples::require(!cfg.source.rtsp_url.empty(), "source.rtsp_url must be set");
  sima_examples::require(!cfg.model.path.empty(), "model.path must be set");
  sima_examples::require(!cfg.insight.host.empty(), "output.insight.host must be set");
  sima_examples::require(cfg.source.latency_ms >= 0, "source.latency_ms must be >= 0");
  sima_examples::require(cfg.inference.frames >= 0, "inference.frames must be >= 0");
  sima_examples::require(cfg.inference.min_score >= 0.0 && cfg.inference.min_score <= 1.0,
                         "inference.min_score must be between 0 and 1");
  sima_examples::require(cfg.inference.nms_iou >= 0.0 && cfg.inference.nms_iou <= 1.0,
                         "inference.nms_iou must be between 0 and 1");
  sima_examples::require(cfg.inference.max_detections > 0, "inference.max_detections must be > 0");
  sima_examples::require(cfg.runtime.profile_interval > 0, "runtime.profile_interval must be > 0");
  sima_examples::require(cfg.insight.video_port > 0, "output.insight.video_port must be > 0");
  sima_examples::require(cfg.insight.metadata_port > 0, "output.insight.metadata_port must be > 0");
}

AppConfig load_app_config(const fs::path& config_path) {
  const auto raw = sima_examples::ScalarConfig::load(config_path);
  AppConfig cfg;
  cfg.model.path = raw.string_or("model.path", "");
  cfg.source.rtsp_url = raw.string_or("source.rtsp_url", "");
  cfg.source.latency_ms = raw.int_or("source.latency_ms", 200);
  cfg.source.tcp = raw.bool_or("source.tcp", true);
  cfg.inference.frames = raw.int_or("inference.frames", 0);
  cfg.inference.min_score = raw.double_or("inference.min_score", 0.55);
  cfg.inference.nms_iou = raw.double_or("inference.nms_iou", 0.50);
  cfg.inference.max_detections = raw.int_or("inference.max_detections", 100);
  cfg.runtime.profile = raw.bool_or("runtime.profile", false);
  cfg.runtime.profile_interval = raw.int_or("runtime.profile_interval", 100);
  cfg.insight.host = raw.string_or("output.insight.host", "");
  cfg.insight.video_port = raw.int_or("output.insight.video_port", 9000);
  cfg.insight.metadata_port = raw.int_or("output.insight.metadata_port", 9100);
  validate_config(cfg);
  return cfg;
}

using sima_examples::infer_dims;
using sima_examples::make_blank_nv12_tensor;
using sima_examples::nv12_copy_to_cpu_tensor;

// This sample uses the standard 80-class COCO label order expected by the
// bundled YOLOv8 model. Keeping the mapping local to the sample avoids baking
// model-specific semantics into generic Insight helpers used by other models.
std::vector<std::string> yolo_coco_labels() {
  return {
      "person",        "bicycle",      "car",
      "motorcycle",    "airplane",     "bus",
      "train",         "truck",        "boat",
      "traffic light", "fire hydrant", "stop sign",
      "parking meter", "bench",        "bird",
      "cat",           "dog",          "horse",
      "sheep",         "cow",          "elephant",
      "bear",          "zebra",        "giraffe",
      "backpack",      "umbrella",     "handbag",
      "tie",           "suitcase",     "frisbee",
      "skis",          "snowboard",    "sports ball",
      "kite",          "baseball bat", "baseball glove",
      "skateboard",    "surfboard",    "tennis racket",
      "bottle",        "wine glass",   "cup",
      "fork",          "knife",        "spoon",
      "bowl",          "banana",       "apple",
      "sandwich",      "orange",       "broccoli",
      "carrot",        "hot dog",      "pizza",
      "donut",         "cake",         "chair",
      "couch",         "potted plant", "bed",
      "dining table",  "toilet",       "tv",
      "laptop",        "mouse",        "remote",
      "keyboard",      "cell phone",   "microwave",
      "oven",          "toaster",      "sink",
      "refrigerator",  "book",         "clock",
      "vase",          "scissors",     "teddy bear",
      "hair drier",    "toothbrush",
  };
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

void enable_insight_diagnostics(bool enabled) {
  if (!enabled)
    return;
  setenv("SIMA_GST_ELEMENT_TIMINGS", "1", 0);
  setenv("SIMA_GST_FLOW_DEBUG", "1", 0);
  setenv("SIMA_GST_BOUNDARY_PROBES", "1", 0);
}

void print_pipeline_report(const char* label, const simaai::neat::Run& run, bool enabled) {
  if (!enabled)
    return;
  simaai::neat::RunReportOptions opt;
  opt.include_pipeline = false;
  opt.include_stage_timings = false;
  opt.include_element_timings = true;
  opt.include_boundaries = false;
  opt.include_flow_stats = true;
  opt.include_node_reports = true;
  opt.include_next_cpu = false;
  opt.include_queue_depth = false;
  opt.include_num_buffers = false;
  opt.include_run_stats = true;
  opt.include_input_stats = true;
  opt.include_system_info = false;
  std::cout << "[TIMING] " << label << "\n" << run.report(opt);
}

void print_stream_summary(const char* label, const simaai::neat::Run& run, bool enabled) {
  if (!enabled)
    return;
  const auto stats = run.input_stats();
  std::cout << "[STREAM] " << label << " avg_push_us=" << stats.avg_push_us
            << " avg_pull_wait_us=" << stats.avg_pull_wait_us
            << " avg_decode_us=" << stats.avg_decode_us << " avg_copy_us=" << stats.avg_copy_us
            << " push_failures=" << stats.push_failures << "\n";
}

double fps_from_count(int count, double elapsed_ms) {
  if (elapsed_ms <= 0.0)
    return 0.0;
  return static_cast<double>(count) * 1000.0 / elapsed_ms;
}

void print_throughput_summary(int produced, int det_outputs, int published,
                              double producer_start_ms, double producer_end_ms,
                              double consumer_start_ms, double consumer_end_ms) {
  const double producer_elapsed_ms =
      (producer_end_ms > producer_start_ms) ? (producer_end_ms - producer_start_ms) : 0.0;
  const double consumer_elapsed_ms =
      (consumer_end_ms > consumer_start_ms) ? (consumer_end_ms - consumer_start_ms) : 0.0;
  std::cout << "[THROUGHPUT] produced=" << produced
            << " fps=" << fps_from_count(produced, producer_elapsed_ms)
            << " yolo_out=" << det_outputs
            << " fps=" << fps_from_count(det_outputs, consumer_elapsed_ms)
            << " published=" << published
            << " fps=" << fps_from_count(published, consumer_elapsed_ms)
            << " producer_ms=" << producer_elapsed_ms << " consumer_ms=" << consumer_elapsed_ms
            << "\n";
}

struct FrameItem {
  int index = 0;
  simaai::neat::Tensor frame;
  double pull_ts_ms = 0.0;
  double rtsp_pull_ms = 0.0;
};

struct PendingFrame {
  int index = 0;
  double pull_ts_ms = 0.0;
  simaai::neat::Tensor frame;
};

struct FrameQueue {
  explicit FrameQueue(size_t max_size_in) : max_size(max_size_in) {}

  bool push(FrameItem item) {
    std::unique_lock<std::mutex> lock(mu);
    cond.wait(lock, [&]() { return closed || items.size() < max_size; });
    if (closed)
      return false;
    items.push_back(std::move(item));
    lock.unlock();
    cond.notify_all();
    return true;
  }

  bool pop(FrameItem& out) {
    std::unique_lock<std::mutex> lock(mu);
    cond.wait(lock, [&]() { return closed || !items.empty(); });
    if (items.empty())
      return false;
    out = std::move(items.front());
    items.pop_front();
    lock.unlock();
    cond.notify_all();
    return true;
  }

  void close() {
    std::lock_guard<std::mutex> lock(mu);
    closed = true;
    cond.notify_all();
  }

private:
  size_t max_size = 0;
  std::mutex mu;
  std::condition_variable cond;
  std::deque<FrameItem> items;
  bool closed = false;
};

struct ProducerStats {
  int count = 0;
};

struct TimingAccumulator {
  int count = 0;
  double sum_ms = 0.0;
  double max_ms = 0.0;

  void add(double ms) {
    sum_ms += ms;
    if (ms > max_ms)
      max_ms = ms;
    count += 1;
  }

  double avg() const {
    return count > 0 ? sum_ms / static_cast<double>(count) : 0.0;
  }

  void reset() {
    count = 0;
    sum_ms = 0.0;
    max_ms = 0.0;
  }
};

struct FrameProfile {
  double rtsp_pull_ms = 0.0;
  double queue_pop_ms = 0.0;
  double yolo_push_ms = 0.0;
  double yolo_pull_ms = 0.0;
  double bbox_extract_ms = 0.0;
  double bbox_parse_ms = 0.0;
  double video_input_ms = 0.0;
  double video_push_ms = 0.0;
  double e2e_ms = 0.0;
  int boxes = 0;
};

struct ProfileWindow {
  bool enabled = false;
  int interval = 100;
  int frames = 0;
  int boxes = 0;
  double start_ms = 0.0;
  TimingAccumulator rtsp_pull;
  TimingAccumulator queue_pop;
  TimingAccumulator yolo_push;
  TimingAccumulator yolo_pull;
  TimingAccumulator bbox_extract;
  TimingAccumulator bbox_parse;
  TimingAccumulator video_input;
  TimingAccumulator video_push;
  TimingAccumulator e2e;

  void add(const FrameProfile& profile, int published_total) {
    if (!enabled)
      return;
    if (frames == 0)
      start_ms = time_ms();
    frames += 1;
    boxes += profile.boxes;
    rtsp_pull.add(profile.rtsp_pull_ms);
    queue_pop.add(profile.queue_pop_ms);
    yolo_push.add(profile.yolo_push_ms);
    yolo_pull.add(profile.yolo_pull_ms);
    bbox_extract.add(profile.bbox_extract_ms);
    bbox_parse.add(profile.bbox_parse_ms);
    video_input.add(profile.video_input_ms);
    video_push.add(profile.video_push_ms);
    e2e.add(profile.e2e_ms);
    if (frames >= interval)
      flush(published_total);
  }

  void flush(int published_total) {
    if (!enabled || frames <= 0)
      return;
    const double elapsed_ms = time_ms() - start_ms;
    std::cout << "[profile] frames=" << frames << " published=" << published_total
              << " fps=" << fps_from_count(frames, elapsed_ms)
              << " avg_rtsp_pull_ms=" << rtsp_pull.avg() << " avg_queue_pop_ms=" << queue_pop.avg()
              << " avg_yolo_ms=" << (yolo_push.avg() + yolo_pull.avg())
              << " avg_bbox_ms=" << (bbox_extract.avg() + bbox_parse.avg())
              << " avg_video_input_ms=" << video_input.avg()
              << " avg_video_push_ms=" << video_push.avg() << " avg_e2e_ms=" << e2e.avg()
              << " avg_boxes=" << (static_cast<double>(boxes) / static_cast<double>(frames))
              << " max_e2e_ms=" << e2e.max_ms << "\n";
    reset();
  }

  void reset() {
    frames = 0;
    boxes = 0;
    start_ms = 0.0;
    rtsp_pull.reset();
    queue_pop.reset();
    yolo_push.reset();
    yolo_pull.reset();
    bbox_extract.reset();
    bbox_parse.reset();
    video_input.reset();
    video_push.reset();
    e2e.reset();
  }
};

struct RtspRuntime {
  simaai::neat::Graph source_graph;
  simaai::neat::Run source_run;
  simaai::neat::Tensor first_frame;
  double first_pull_ms = 0.0;
  double first_pull_ts = 0.0;
  int frame_w = 0;
  int frame_h = 0;
  int output_fps = 30;
};

struct InsightRuntime {
  std::string host;
  int video_port = 0;
  simaai::neat::Graph video_graph;
  simaai::neat::Run video_run;
  std::unique_ptr<simaai::neat::MetadataSender> metadata_sender;
  std::vector<std::string> labels;
};

struct DetectorRuntime {
  std::unique_ptr<simaai::neat::Model> model;
  simaai::neat::Graph detector_graph;
  simaai::neat::Run detector_run;
};

struct WorkerSharedState {
  std::optional<int> frame_limit;
  FrameQueue& queue;
  ProducerStats& producer_stats;
  ProfileWindow& profile_window;
  std::atomic<bool>& stop;
  std::atomic<int>& published;
  std::atomic<int>& det_outputs;
  double& producer_start_ms;
  double& producer_end_ms;
  double& consumer_start_ms;
  double& consumer_end_ms;
};

RtspRuntime build_rtsp_runtime(const AppConfig& cfg) {
  RtspRuntime runtime;

  sima_examples::RtspStreamInfo rtsp_probe;
  sima_examples::RtspProbeOptions rtsp_probe_opt;
  rtsp_probe_opt.payload_type = 96;
  rtsp_probe_opt.latency_ms = cfg.source.latency_ms;
  rtsp_probe_opt.rtsp_tcp = cfg.source.tcp;
  rtsp_probe_opt.debug = cfg.runtime.profile;
  (void)sima_examples::probe_rtsp_stream_info(cfg.source.rtsp_url, rtsp_probe_opt, rtsp_probe);

  simaai::neat::nodes::groups::RtspDecodedInputOptions source_options;
  source_options.url = cfg.source.rtsp_url;
  source_options.latency_ms = cfg.source.latency_ms;
  source_options.tcp = cfg.source.tcp;
  source_options.payload_type = 96;
  source_options.insert_queue = true;
  source_options.out_format = "NV12";
  source_options.decoder_name = "decoder";
  source_options.decoder_raw_output = true;
  source_options.auto_caps_from_stream = true;
  if (rtsp_probe.width > 0 && rtsp_probe.height > 0) {
    source_options.fallback_h264_width = rtsp_probe.width;
    source_options.fallback_h264_height = rtsp_probe.height;
    std::cout << "[init] probed RTSP decode dims " << rtsp_probe.width << "x" << rtsp_probe.height;
    if (rtsp_probe.fps > 0)
      std::cout << " @" << rtsp_probe.fps << " fps";
    std::cout << "\n";
  }
  if (rtsp_probe.fps > 0)
    source_options.fallback_h264_fps = rtsp_probe.fps;
  runtime.output_fps = (rtsp_probe.fps > 0) ? rtsp_probe.fps : 30;

  runtime.source_graph.add(simaai::neat::nodes::groups::RtspDecodedInput(source_options));
  runtime.source_graph.add(simaai::neat::nodes::Output());
  simaai::neat::RunOptions source_run_options;
  source_run_options.enable_metrics = cfg.runtime.profile;
  source_run_options.queue_depth = 4;
  source_run_options.overflow_policy = simaai::neat::OverflowPolicy::KeepLatest;
  runtime.source_run = runtime.source_graph.build(source_run_options);

  const double first_pull_start = time_ms();
  simaai::neat::Sample first_sample;
  simaai::neat::PullError first_pull_error;
  const auto first_pull_status = runtime.source_run.pull(5000, first_sample, &first_pull_error);
  if (first_pull_status != simaai::neat::PullStatus::Ok) {
    if (first_pull_status == simaai::neat::PullStatus::Timeout) {
      throw std::runtime_error(
          "Timed out waiting for first RTSP frame. This is usually upstream connectivity or stream "
          "delivery, not framerate derivation. If diagnostics show zero buffers at rtspsrc/depay/"
          "decoder, the device is not receiving RTP from the source.");
    }
    throw std::runtime_error("Failed waiting for first RTSP frame: " + first_pull_error.message);
  }
  auto first_tensors = simaai::neat::tensors_from_sample(first_sample);
  runtime.first_frame = std::move(first_tensors.front());
  const double first_pull_end = time_ms();
  runtime.first_pull_ms = first_pull_end - first_pull_start;
  runtime.first_pull_ts = first_pull_end;
  sima_examples::require(infer_dims(runtime.first_frame, runtime.frame_w, runtime.frame_h),
                         "first frame missing dimensions");
  if (runtime.frame_w == 1280 && runtime.frame_h == 720 && source_options.h264_width <= 0 &&
      source_options.h264_height <= 0 && source_options.fallback_h264_width <= 0 &&
      source_options.fallback_h264_height <= 0) {
    std::fprintf(stderr, "[WARN] deriving width=1280 and height=720 from SDP or timestamp\n");
  }
  return runtime;
}

InsightRuntime build_insight_runtime(const AppConfig& cfg, int frame_w, int frame_h,
                                     int output_fps) {
  InsightRuntime runtime;
  runtime.host = cfg.insight.host;
  runtime.video_port = cfg.insight.video_port;

  simaai::neat::InputOptions video_input_options;
  video_input_options.format = "NV12";
  video_input_options.width = frame_w;
  video_input_options.height = frame_h;
  video_input_options.caps_override = "video/x-raw,format=NV12,width=" + std::to_string(frame_w) +
                                      ",height=" + std::to_string(frame_h) +
                                      ",framerate=" + std::to_string(output_fps) + "/1";
  video_input_options.use_simaai_pool = false;
  runtime.video_graph.add(simaai::neat::nodes::Input(video_input_options));
  auto video_sender_options = simaai::neat::nodes::groups::VideoSenderOptions::H264RtpUdpFromRaw(
      frame_w, frame_h, output_fps);
  video_sender_options.host = runtime.host;
  video_sender_options.channel = 0;
  video_sender_options.video_port_base = cfg.insight.video_port;
  runtime.video_port = video_sender_options.video_port();
  video_sender_options.encoder.bitrate_kbps = 4000;
  runtime.video_graph.add(simaai::neat::nodes::groups::VideoSender(video_sender_options));

  simaai::neat::Tensor video_seed;
  std::string video_seed_err;
  sima_examples::require(make_blank_nv12_tensor(frame_w, frame_h, video_seed, video_seed_err),
                         video_seed_err);
  simaai::neat::RunOptions video_run_options;
  video_run_options.enable_metrics = cfg.runtime.profile;
  runtime.video_run = runtime.video_graph.build(simaai::neat::TensorList{video_seed},
                                                simaai::neat::RunMode::Async, video_run_options);
  std::cout << "video_sender=" << runtime.host << ":" << runtime.video_port << "\n";

  simaai::neat::MetadataSenderOptions metadata_options;
  metadata_options.host = cfg.insight.host;
  metadata_options.channel = 0;
  metadata_options.metadata_port_base = cfg.insight.metadata_port;
  std::string metadata_err;
  runtime.metadata_sender =
      std::make_unique<simaai::neat::MetadataSender>(metadata_options, &metadata_err);
  sima_examples::require(runtime.metadata_sender->ok(), metadata_err);
  runtime.labels = yolo_coco_labels();
  std::cout << "insight host=" << runtime.metadata_sender->host()
            << " video_port=" << runtime.video_port
            << " metadata_port=" << runtime.metadata_sender->metadata_port() << " channel=0\n";
  return runtime;
}

DetectorRuntime build_detector_runtime(const AppConfig& cfg, int frame_w, int frame_h) {
  // NEAT boundary: build detection Graph/Run pipeline.
  DetectorRuntime runtime;

  // NEAT boundary: build model + async inference runtime.
  simaai::neat::Model::Options model_opt;
  model_opt.preprocess.kind = simaai::neat::InputKind::Image;
  model_opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::BGR;
  model_opt.preprocess.input_max_width = frame_w;
  model_opt.preprocess.input_max_height = frame_h;
  model_opt.preprocess.input_max_depth = 3;
  model_opt.decode_type = simaai::neat::BoxDecodeType::YoloV8;
  model_opt.score_threshold = cfg.inference.min_score;
  model_opt.nms_iou_threshold = cfg.inference.nms_iou;
  model_opt.top_k = cfg.inference.max_detections;
  model_opt.boxdecode_original_width = frame_w;
  model_opt.boxdecode_original_height = frame_h;
  runtime.model = std::make_unique<simaai::neat::Model>(cfg.model.path, model_opt);
  std::cout << "[init] model configured for " << frame_w << "x" << frame_h << " BGR\n";

  simaai::neat::InputOptions appsrc_options = runtime.model->input_appsrc_options(false);
  appsrc_options.payload_type = simaai::neat::PayloadType::Image;
  appsrc_options.format = "BGR";
  appsrc_options.width = frame_w;
  appsrc_options.height = frame_h;
  appsrc_options.depth = 3;

  runtime.detector_graph.add(simaai::neat::nodes::Input(appsrc_options));
  runtime.detector_graph.add(simaai::neat::nodes::groups::Preprocess(*runtime.model));
  runtime.detector_graph.add(simaai::neat::nodes::groups::Infer(*runtime.model));
  runtime.detector_graph.add(simaai::neat::nodes::SimaBoxDecode(
      *runtime.model, simaai::neat::BoxDecodeType::YoloV8, cfg.inference.min_score,
      cfg.inference.nms_iou, cfg.inference.max_detections, "", std::nullopt, std::nullopt, frame_w,
      frame_h));
  runtime.detector_graph.add(simaai::neat::nodes::Output());

  simaai::neat::RunOptions detector_run_options;
  detector_run_options.preset = simaai::neat::RunPreset::Reliable;
  detector_run_options.enable_metrics = cfg.runtime.profile;
  detector_run_options.queue_depth = 4;
  detector_run_options.overflow_policy = simaai::neat::OverflowPolicy::KeepLatest;
  detector_run_options.output_memory = simaai::neat::OutputMemory::Owned;
  cv::Mat detector_seed(frame_h, frame_w, CV_8UC3, cv::Scalar(0, 0, 0));
  std::cout << "[init] building YOLO pipeline\n";
  runtime.detector_run = runtime.detector_graph.build(
      std::vector<cv::Mat>{detector_seed}, simaai::neat::RunMode::Async, detector_run_options);
  std::cout << "[init] YOLO pipeline ready\n";
  return runtime;
}

std::optional<int> frame_limit_from_config(const AppConfig& cfg) {
  if (cfg.inference.frames > 0) {
    return cfg.inference.frames;
  }
  return std::nullopt;
}

std::vector<sima_examples::MetadataBox>
build_detection_metadata_boxes(const std::vector<objdet::Box>& boxes,
                               const std::vector<std::string>& labels, int frame_w, int frame_h) {
  std::vector<sima_examples::MetadataBox> metadata_boxes;
  metadata_boxes.reserve(boxes.size());
  int object_index = 1;
  for (const auto& box : boxes) {
    int x1 = static_cast<int>(box.x1);
    int y1 = static_cast<int>(box.y1);
    int w = static_cast<int>(box.x2 - box.x1);
    int h = static_cast<int>(box.y2 - box.y1);
    if (x1 < 0)
      x1 = 0;
    if (y1 < 0)
      y1 = 0;
    if (w < 0)
      w = 0;
    if (h < 0)
      h = 0;
    if (x1 + w > frame_w)
      w = frame_w - x1;
    if (y1 + h > frame_h)
      h = frame_h - y1;
    if (w < 0)
      w = 0;
    if (h < 0)
      h = 0;
    sima_examples::MetadataBox obj;
    obj.id = "obj_" + std::to_string(object_index++);
    obj.label = (box.class_id >= 0 && box.class_id < static_cast<int>(labels.size()))
                    ? labels[box.class_id]
                    : "unknown";
    obj.confidence = box.score;
    obj.x = static_cast<float>(x1);
    obj.y = static_cast<float>(y1);
    obj.w = static_cast<float>(w);
    obj.h = static_cast<float>(h);
    metadata_boxes.push_back(obj);
  }
  return metadata_boxes;
}

void producer_worker(simaai::neat::Run& source_run, simaai::neat::Tensor first_frame,
                     double first_pull_ms, double first_pull_ts, WorkerSharedState& state) {
  state.producer_start_ms = time_ms();
  int produced = 0;
  bool use_first = true;
  while (!state.stop.load() && (!state.frame_limit || produced < *state.frame_limit)) {
    simaai::neat::Tensor frame;
    double pull_ms = 0.0;
    double pull_ts = 0.0;
    if (use_first) {
      frame = std::move(first_frame);
      use_first = false;
      pull_ms = first_pull_ms;
      pull_ts = first_pull_ts;
    } else {
      const double t0 = time_ms();
      auto frame_opt = source_run.pull();
      if (!frame_opt.has_value())
        continue;
      const double t1 = time_ms();
      auto tensors = simaai::neat::tensors_from_sample(*frame_opt);
      frame = std::move(tensors.front());
      pull_ms = t1 - t0;
      pull_ts = t1;
    }
    FrameItem item;
    item.index = produced;
    item.frame = std::move(frame);
    item.pull_ts_ms = pull_ts;
    item.rtsp_pull_ms = pull_ms;

    if (!state.queue.push(std::move(item)))
      break;

    produced += 1;
    state.producer_stats.count = produced;
  }
  state.queue.close();
  state.producer_end_ms = time_ms();
}

void consumer_worker(simaai::neat::Run& detector_run, simaai::neat::Run& video_run,
                     simaai::neat::MetadataSender& metadata_sender,
                     const std::vector<std::string>& insight_labels, int frame_w, int frame_h,
                     int max_detections, WorkerSharedState& state) {
  state.consumer_start_ms = time_ms();
  int out_pulls = 0;
  while (!state.stop.load() &&
         (!state.frame_limit || state.published.load() < *state.frame_limit)) {
    FrameProfile frame_profile;
    FrameItem item;
    const double q0 = time_ms();
    if (!state.queue.pop(item))
      break;
    const double q1 = time_ms();
    frame_profile.rtsp_pull_ms = item.rtsp_pull_ms;
    frame_profile.queue_pop_ms = q1 - q0;

    PendingFrame decoded_frame;
    decoded_frame.index = item.index;
    decoded_frame.pull_ts_ms = item.pull_ts_ms;
    decoded_frame.frame = std::move(item.frame);

    // NEAT boundary: push current frame into async YOLO run, then pull next output.
    const double t_push0 = time_ms();
    const cv::Mat detector_input =
        decoded_frame.frame.to_cv_mat_copy(simaai::neat::ImageSpec::PixelFormat::BGR);
    const bool pushed = detector_run.push(std::vector<cv::Mat>{detector_input});
    const double t_push1 = time_ms();
    frame_profile.yolo_push_ms = t_push1 - t_push0;
    if (!pushed) {
      std::cerr << "[warn] push failed\n";
      continue;
    }
    const double t_pull0 = time_ms();
    auto detection_sample = detector_run.pull();
    const double t_pull1 = time_ms();
    frame_profile.yolo_pull_ms = t_pull1 - t_pull0;
    if (!detection_sample.has_value())
      continue;
    out_pulls += 1;
    state.det_outputs.store(out_pulls);

    PendingFrame pending = std::move(decoded_frame);

    const double t_extract0 = time_ms();
    std::vector<uint8_t> bbox_payload;
    std::string err;
    if (!extract_bbox_payload(*detection_sample, bbox_payload, err)) {
      std::cerr << "[warn] bbox extract failed: " << err << "\n";
      continue;
    }
    const double t_extract1 = time_ms();
    frame_profile.bbox_extract_ms = t_extract1 - t_extract0;

    const double t_parse0 = time_ms();
    std::vector<objdet::Box> detections;
    try {
      detections =
          objdet::parse_boxes_strict(bbox_payload, frame_w, frame_h, max_detections, false);
    } catch (const std::exception& ex) {
      std::cerr << "[warn] bbox parse failed: " << ex.what() << "\n";
      continue;
    }
    const double t_parse1 = time_ms();
    frame_profile.bbox_parse_ms = t_parse1 - t_parse0;
    frame_profile.boxes = static_cast<int>(detections.size());

    std::vector<sima_examples::MetadataBox> metadata_boxes =
        build_detection_metadata_boxes(detections, insight_labels, frame_w, frame_h);

    // Contract: publish video first, then publish the matching metadata side-channel payload.
    double output_ts = 0.0;
    const double t_video_input0 = time_ms();
    simaai::neat::Tensor nv12_frame;
    std::string nv12_err;
    const bool converted = nv12_copy_to_cpu_tensor(pending.frame, nv12_frame, nv12_err);
    if (!converted) {
      std::cerr << "[warn] VideoSender input copy failed: " << nv12_err << "\n";
      continue;
    }
    const double t_video_input1 = time_ms();
    frame_profile.video_input_ms = t_video_input1 - t_video_input0;

    // NEAT boundary: push the raw frame into VideoSender; the nodegroup owns conversion, H.264
    // encoding, RTP packetization, and UDP output.
    const double t_video_sender_push0 = time_ms();
    if (!video_run.push(simaai::neat::TensorList{nv12_frame})) {
      std::cerr << "[warn] VideoSender push failed\n";
      continue;
    }
    const double t_video_sender_push1 = time_ms();
    frame_profile.video_push_ms = t_video_sender_push1 - t_video_sender_push0;
    output_ts = t_video_sender_push1;
    const int64_t fid = detection_sample->frame_id >= 0 ? detection_sample->frame_id
                                                        : static_cast<int64_t>(pending.index);
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    const int64_t ts_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
    const std::string data_json =
        sima_examples::metadata_boxes_data_json("objects", metadata_boxes);
    std::string metadata_err;
    if (!metadata_sender.send_metadata("object-detection", data_json, ts_ms, std::to_string(fid),
                                       &metadata_err)) {
      std::cerr << "[warn] insight metadata send failed: " << metadata_err << "\n";
    }

    frame_profile.e2e_ms = output_ts - pending.pull_ts_ms;

    const int published_now = state.published.fetch_add(1) + 1;
    state.profile_window.add(frame_profile, published_now);
  }
  state.stop.store(true);
  state.queue.close();
  state.consumer_end_ms = time_ms();
}

} // namespace

int main(int argc, char** argv) {
  try {
    // Lifecycle: setup -> start workers -> join -> summary -> teardown.
    const CliOptions cli = parse_args(argc, argv);
    const AppConfig cfg = load_app_config(cli.config_path);
    if (cli.validate_config_only) {
      std::cout << "Config validated: " << cli.config_path << "\n";
      return 0;
    }

    enable_insight_diagnostics(cfg.runtime.profile);

    RtspRuntime rtsp_runtime = build_rtsp_runtime(cfg);
    InsightRuntime insight_runtime = build_insight_runtime(
        cfg, rtsp_runtime.frame_w, rtsp_runtime.frame_h, rtsp_runtime.output_fps);
    DetectorRuntime detector_runtime =
        build_detector_runtime(cfg, rtsp_runtime.frame_w, rtsp_runtime.frame_h);

    std::optional<int> frame_limit = frame_limit_from_config(cfg);
    std::cout << "mode=insight"
              << " frame_limit=" << (frame_limit ? std::to_string(*frame_limit) : "inf")
              << " profile=" << (cfg.runtime.profile ? "1" : "0") << "\n";

    // Contract: bounded queue preserves backpressure and producer closes it on exit.
    FrameQueue queue(300);
    ProducerStats producer_stats;
    ProfileWindow profile_window;
    profile_window.enabled = cfg.runtime.profile;
    profile_window.interval = cfg.runtime.profile_interval;
    std::atomic<bool> stop{false};
    std::atomic<int> published{0};
    std::atomic<int> det_outputs{0};
    double producer_start_ms = 0.0;
    double producer_end_ms = 0.0;
    double consumer_start_ms = 0.0;
    double consumer_end_ms = 0.0;

    WorkerSharedState worker_state{
        frame_limit,     queue,       producer_stats,    profile_window,  stop,
        published,       det_outputs, producer_start_ms, producer_end_ms, consumer_start_ms,
        consumer_end_ms,
    };

    // Contract: start producer first, then consumer; both terminate when queue closes or stop is
    // set.
    std::thread producer_thread(producer_worker, std::ref(rtsp_runtime.source_run),
                                std::move(rtsp_runtime.first_frame), rtsp_runtime.first_pull_ms,
                                rtsp_runtime.first_pull_ts, std::ref(worker_state));
    std::thread consumer_thread(
        consumer_worker, std::ref(detector_runtime.detector_run),
        std::ref(insight_runtime.video_run), std::ref(*insight_runtime.metadata_sender),
        std::cref(insight_runtime.labels), rtsp_runtime.frame_w, rtsp_runtime.frame_h,
        cfg.inference.max_detections, std::ref(worker_state));

    if (producer_thread.joinable())
      producer_thread.join();
    if (consumer_thread.joinable())
      consumer_thread.join();

    std::cout << "published=" << published.load() << " video_sender=" << insight_runtime.host << ":"
              << insight_runtime.video_port << "\n";
    print_throughput_summary(producer_stats.count, det_outputs.load(), published.load(),
                             producer_start_ms, producer_end_ms, consumer_start_ms,
                             consumer_end_ms);
    profile_window.flush(published.load());
    print_stream_summary("rtsp", rtsp_runtime.source_run, cfg.runtime.profile);
    print_stream_summary("yolo", detector_runtime.detector_run, cfg.runtime.profile);
    print_pipeline_report("yolo", detector_runtime.detector_run, cfg.runtime.profile);
    print_stream_summary("video_sender", insight_runtime.video_run, cfg.runtime.profile);
    print_pipeline_report("video_sender", insight_runtime.video_run, cfg.runtime.profile);

    // Contract: worker threads are joined before pipeline teardown. Explicit close avoids handing
    // live appsrc/encoder teardown to Run move-assignment while bounded examples are exiting.
    detector_runtime.detector_run.close();
    insight_runtime.video_run.close();
    rtsp_runtime.source_run.close();
    return 0;

  } catch (const std::exception& e) {
    std::cerr << "[ERR] " << e.what() << "\n";
    return 1;
  }
}
