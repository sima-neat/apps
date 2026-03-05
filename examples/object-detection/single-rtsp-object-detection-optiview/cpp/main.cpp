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
#include "support/object_detection/obj_detection_utils.h"
#include "neat/session.h"
#include "neat/models.h"
#include "neat/nodes.h"
#include "neat/node_groups.h"

#include <gst/gst.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h> // getpid()

namespace fs = std::filesystem;

using sima_examples::parse_int_arg;
using sima_examples::time_ms;

namespace {

// single-rtsp-object-detection-optiview is a reference pipeline for the common deployment:
// one RTSP source, one YOLO detector, and OptiView video/JSON output.
//
// The code keeps ingest, inference, and output transport separate so each
// stage can be reasoned about and debugged independently.

struct Config {
  std::string url;
  std::string mpk;
  int frames = 300;
  bool frames_set = false;
  bool debug = false;
  std::string optiview_host = "127.0.0.1";
  int optiview_video_port = 9000;
  int optiview_json_port = 9100;
};

// Keep runtime configuration intentionally small. The remaining options cover
// only source selection, bounded runs, debug timings, and OptiView address/ports.
Config parse_config(int argc, char** argv) {
  Config cfg;
  sima_examples::get_arg(argc, argv, "--rtsp", cfg.url);
  sima_examples::get_arg(argc, argv, "--mpk", cfg.mpk);
  cfg.frames_set = parse_int_arg(argc, argv, "--frames", cfg.frames);
  cfg.debug = sima_examples::has_flag(argc, argv, "--debug");
  sima_examples::get_arg(argc, argv, "--optiview-host", cfg.optiview_host);
  if (cfg.optiview_host.empty())
    cfg.optiview_host = "127.0.0.1";
  parse_int_arg(argc, argv, "--optiview-video-port", cfg.optiview_video_port);
  parse_int_arg(argc, argv, "--optiview-json-port", cfg.optiview_json_port);
  return cfg;
}

using sima_examples::infer_dims;
using sima_examples::make_blank_nv12_tensor;
using sima_examples::nv12_copy_to_cpu_tensor;

void print_time(const char* label, double ms, bool enabled) {
  if (!enabled)
    return;
  std::cout << label << " " << ms << "\n";
}

// This sample uses the standard 80-class COCO label order expected by the
// bundled YOLOv8 model. Keeping the mapping local to the sample avoids baking
// model-specific semantics into generic OptiView helpers used by other models.
std::vector<std::string> yolo_coco_labels() {
  return {
      "person",         "bicycle",      "car",           "motorcycle",    "airplane",
      "bus",            "train",        "truck",         "boat",          "traffic light",
      "fire hydrant",   "stop sign",    "parking meter", "bench",         "bird",
      "cat",            "dog",          "horse",         "sheep",         "cow",
      "elephant",       "bear",         "zebra",         "giraffe",       "backpack",
      "umbrella",       "handbag",      "tie",           "suitcase",      "frisbee",
      "skis",           "snowboard",    "sports ball",   "kite",          "baseball bat",
      "baseball glove", "skateboard",   "surfboard",     "tennis racket", "bottle",
      "wine glass",     "cup",          "fork",          "knife",         "spoon",
      "bowl",           "banana",       "apple",         "sandwich",      "orange",
      "broccoli",       "carrot",       "hot dog",       "pizza",         "donut",
      "cake",           "chair",        "couch",         "potted plant",  "bed",
      "dining table",   "toilet",       "tv",            "laptop",        "mouse",
      "remote",         "keyboard",     "cell phone",    "microwave",     "oven",
      "toaster",        "sink",         "refrigerator",  "book",          "clock",
      "vase",           "scissors",     "teddy bear",    "hair drier",    "toothbrush",
  };
}

void enable_optiview_diagnostics(bool enabled) {
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

void print_throughput_summary(int produced, int det_outputs, int published, double producer_start_ms,
                              double producer_end_ms, double consumer_start_ms,
                              double consumer_end_ms) {
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

struct ProducerTiming {
  int count = 0;
  double rtsp_pull_sum = 0.0;
  double rtsp_pull_max = 0.0;
  double queue_push_sum = 0.0;
  double queue_push_max = 0.0;

  void add_rtsp_pull(double ms) {
    rtsp_pull_sum += ms;
    if (ms > rtsp_pull_max)
      rtsp_pull_max = ms;
  }

  void add_queue_push(double ms) {
    queue_push_sum += ms;
    if (ms > queue_push_max)
      queue_push_max = ms;
  }

  void print() const {
    if (count <= 0)
      return;
    std::cout << "producer_avg_rtsp_pull_ms " << (rtsp_pull_sum / count)
              << " producer_max_rtsp_pull_ms " << rtsp_pull_max << " producer_avg_queue_push_ms "
              << (queue_push_sum / count) << " producer_max_queue_push_ms " << queue_push_max
              << "\n";
  }
};

struct ConsumerTiming {
  int count = 0;
  double queue_pop_sum = 0.0;
  double queue_pop_max = 0.0;
  double convert_sum = 0.0;
  double convert_max = 0.0;
  double udp_convert_sum = 0.0;
  double udp_convert_max = 0.0;
  double yolo_push_sum = 0.0;
  double yolo_push_max = 0.0;
  double yolo_pull_sum = 0.0;
  double yolo_pull_max = 0.0;
  double udp_push_sum = 0.0;
  double udp_push_max = 0.0;
  double bbox_extract_sum = 0.0;
  double bbox_extract_max = 0.0;
  double bbox_parse_sum = 0.0;
  double bbox_parse_max = 0.0;
  double overlay_sum = 0.0;
  double overlay_max = 0.0;
  double write_sum = 0.0;
  double write_max = 0.0;
  double e2e_sum = 0.0;
  double e2e_max = 0.0;

  void add_queue_pop(double ms) {
    queue_pop_sum += ms;
    if (ms > queue_pop_max)
      queue_pop_max = ms;
  }

  void add_convert(double ms) {
    convert_sum += ms;
    if (ms > convert_max)
      convert_max = ms;
  }

  void add_udp_convert(double ms) {
    udp_convert_sum += ms;
    if (ms > udp_convert_max)
      udp_convert_max = ms;
  }

  void add_yolo_push(double ms) {
    yolo_push_sum += ms;
    if (ms > yolo_push_max)
      yolo_push_max = ms;
  }

  void add_yolo_pull(double ms) {
    yolo_pull_sum += ms;
    if (ms > yolo_pull_max)
      yolo_pull_max = ms;
  }

  void add_udp_push(double ms) {
    udp_push_sum += ms;
    if (ms > udp_push_max)
      udp_push_max = ms;
  }

  void add_bbox_extract(double ms) {
    bbox_extract_sum += ms;
    if (ms > bbox_extract_max)
      bbox_extract_max = ms;
  }

  void add_bbox_parse(double ms) {
    bbox_parse_sum += ms;
    if (ms > bbox_parse_max)
      bbox_parse_max = ms;
  }

  void add_overlay(double ms) {
    overlay_sum += ms;
    if (ms > overlay_max)
      overlay_max = ms;
  }

  void add_write(double ms) {
    write_sum += ms;
    if (ms > write_max)
      write_max = ms;
  }

  void add_e2e(double ms) {
    e2e_sum += ms;
    if (ms > e2e_max)
      e2e_max = ms;
  }

  void print() const {
    if (count <= 0)
      return;
    std::cout << "consumer_avg_queue_pop_ms " << (queue_pop_sum / count)
              << " consumer_max_queue_pop_ms " << queue_pop_max << " consumer_avg_convert_ms "
              << (convert_sum / count) << " consumer_max_convert_ms " << convert_max
              << " consumer_avg_udp_convert_ms " << (udp_convert_sum / count)
              << " consumer_max_udp_convert_ms " << udp_convert_max << " consumer_avg_yolo_push_ms "
              << (yolo_push_sum / count) << " consumer_max_yolo_push_ms " << yolo_push_max
              << " consumer_avg_yolo_pull_ms " << (yolo_pull_sum / count)
              << " consumer_max_yolo_pull_ms " << yolo_pull_max << " consumer_avg_udp_push_ms "
              << (udp_push_sum / count) << " consumer_max_udp_push_ms " << udp_push_max
              << " consumer_avg_bbox_extract_ms " << (bbox_extract_sum / count)
              << " consumer_max_bbox_extract_ms " << bbox_extract_max
              << " consumer_avg_bbox_parse_ms " << (bbox_parse_sum / count)
              << " consumer_max_bbox_parse_ms " << bbox_parse_max << " consumer_avg_overlay_ms "
              << (overlay_sum / count) << " consumer_max_overlay_ms " << overlay_max
              << " consumer_avg_write_ms " << (write_sum / count) << " consumer_max_write_ms "
              << write_max << " consumer_avg_e2e_ms " << (e2e_sum / count)
              << " consumer_max_e2e_ms " << e2e_max << "\n";
  }
};

struct RtspRuntime {
  simaai::neat::Session session;
  simaai::neat::Run run;
  simaai::neat::Tensor first_frame;
  double first_pull_ms = 0.0;
  double first_pull_ts = 0.0;
  int frame_w = 0;
  int frame_h = 0;
  int output_fps = 30;
};

struct OptiViewRuntime {
  std::string host;
  int video_port = 0;
  simaai::neat::Session session;
  simaai::neat::Run video_run;
  std::unique_ptr<sima_examples::OptiViewSender> sender;
  std::vector<std::string> labels;
};

struct YoloRuntime {
  float min_score = 0.52f;
  int topk = 100;
  std::unique_ptr<simaai::neat::Model> model;
  simaai::neat::Session session;
  simaai::neat::Run run;
};

struct WorkerSharedState {
  Config& cfg;
  std::optional<int> frame_limit;
  FrameQueue& queue;
  ProducerTiming& producer_stats;
  ConsumerTiming& consumer_stats;
  std::atomic<bool>& stop;
  std::atomic<int>& published;
  std::atomic<int>& det_outputs;
  double& producer_start_ms;
  double& producer_end_ms;
  double& consumer_start_ms;
  double& consumer_end_ms;
};

RtspRuntime build_rtsp_runtime(const Config& cfg) {
  RtspRuntime runtime;

  sima_examples::RtspStreamInfo rtsp_probe;
  sima_examples::RtspProbeOptions rtsp_probe_opt;
  rtsp_probe_opt.payload_type = 96;
  rtsp_probe_opt.latency_ms = 200;
  rtsp_probe_opt.rtsp_tcp = true;
  rtsp_probe_opt.debug = cfg.debug;
  (void)sima_examples::probe_rtsp_stream_info(cfg.url, rtsp_probe_opt, rtsp_probe);

  simaai::neat::nodes::groups::RtspDecodedInputOptions cam_opt;
  cam_opt.url = cfg.url;
  cam_opt.latency_ms = 200;
  cam_opt.tcp = true;
  cam_opt.payload_type = 96;
  cam_opt.insert_queue = true;
  cam_opt.out_format = "NV12";
  cam_opt.decoder_name = "decoder";
  cam_opt.decoder_raw_output = true;
  cam_opt.auto_caps_from_stream = true;
  if (rtsp_probe.width > 0 && rtsp_probe.height > 0) {
    cam_opt.fallback_h264_width = rtsp_probe.width;
    cam_opt.fallback_h264_height = rtsp_probe.height;
    std::cout << "[init] probed RTSP decode dims " << rtsp_probe.width << "x" << rtsp_probe.height;
    if (rtsp_probe.fps > 0)
      std::cout << " @" << rtsp_probe.fps << " fps";
    std::cout << "\n";
  }
  if (rtsp_probe.fps > 0)
    cam_opt.fallback_h264_fps = rtsp_probe.fps;
  runtime.output_fps = (rtsp_probe.fps > 0) ? rtsp_probe.fps : 30;

  runtime.session.add(simaai::neat::nodes::groups::RtspDecodedInput(cam_opt));
  runtime.session.add(simaai::neat::nodes::Output());
  simaai::neat::RunOptions cam_run_opt;
  cam_run_opt.enable_metrics = true;
  cam_run_opt.queue_depth = 4;
  cam_run_opt.overflow_policy = simaai::neat::OverflowPolicy::KeepLatest;
  runtime.run = runtime.session.build(cam_run_opt);

  const double first_pull_start = time_ms();
  try {
    runtime.first_frame = runtime.run.pull_tensor_or_throw(5000);
  } catch (const std::exception& e) {
    const std::string msg = e.what();
    if (msg.find("timeout") != std::string::npos) {
      throw std::runtime_error(
          "Timed out waiting for first RTSP frame. This is usually upstream connectivity or stream "
          "delivery, not framerate derivation. If diagnostics show zero buffers at rtspsrc/depay/"
          "decoder, the device is not receiving RTP from the source.");
    }
    throw;
  }
  const double first_pull_end = time_ms();
  runtime.first_pull_ms = first_pull_end - first_pull_start;
  runtime.first_pull_ts = first_pull_end;
  sima_examples::require(infer_dims(runtime.first_frame, runtime.frame_w, runtime.frame_h),
                         "first frame missing dimensions");
  if (runtime.frame_w == 1280 && runtime.frame_h == 720 && cam_opt.h264_width <= 0 &&
      cam_opt.h264_height <= 0 && cam_opt.fallback_h264_width <= 0 &&
      cam_opt.fallback_h264_height <= 0) {
    std::fprintf(stderr, "[WARN] deriving width=1280 and height=720 from SDP or timestamp\n");
  }
  return runtime;
}

OptiViewRuntime build_optiview_runtime(const Config& cfg, int frame_w, int frame_h, int output_fps) {
  OptiViewRuntime runtime;
  runtime.host = cfg.optiview_host;
  runtime.video_port = cfg.optiview_video_port;

  simaai::neat::InputOptions udp_src;
  udp_src.format = "NV12";
  udp_src.width = frame_w;
  udp_src.height = frame_h;
  udp_src.caps_override = "video/x-raw,format=NV12,width=" + std::to_string(frame_w) +
                          ",height=" + std::to_string(frame_h) + ",framerate=" +
                          std::to_string(output_fps) + "/1";
  udp_src.use_simaai_pool = false;
  runtime.session.add(simaai::neat::nodes::Input(udp_src));
  runtime.session.add(simaai::neat::nodes::H264EncodeSima(frame_w, frame_h, output_fps, 4000));
  runtime.session.add(simaai::neat::nodes::H264Parse());
  runtime.session.add(simaai::neat::nodes::H264Packetize(96, 1));
  simaai::neat::UdpOutputOptions udp_opt;
  udp_opt.host = runtime.host;
  udp_opt.port = runtime.video_port;
  runtime.session.add(simaai::neat::nodes::UdpOutput(udp_opt));

  simaai::neat::Tensor udp_dummy;
  std::string udp_err;
  sima_examples::require(make_blank_nv12_tensor(frame_w, frame_h, udp_dummy, udp_err), udp_err);
  simaai::neat::RunOptions udp_run_opt;
  udp_run_opt.enable_metrics = true;
  runtime.video_run =
      runtime.session.build(udp_dummy, simaai::neat::RunMode::Async, udp_run_opt);
  std::cout << "udp=" << runtime.host << ":" << runtime.video_port << "\n";

  sima_examples::OptiViewOptions opt;
  opt.host = cfg.optiview_host;
  opt.channel = 0;
  opt.video_port_base = cfg.optiview_video_port;
  opt.json_port_base = cfg.optiview_json_port;
  std::string opt_err;
  runtime.sender = std::make_unique<sima_examples::OptiViewSender>(opt, &opt_err);
  sima_examples::require(runtime.sender->ok(), opt_err);
  runtime.labels = yolo_coco_labels();
  std::cout << "optiview host=" << runtime.sender->host()
            << " video_port=" << runtime.sender->video_port()
            << " json_port=" << runtime.sender->json_port() << " channel=0\n";
  return runtime;
}

YoloRuntime build_yolo_runtime(const Config& cfg, int frame_w, int frame_h) {
  // NEAT boundary: build detection Session/Run graph.
  YoloRuntime runtime;

  // NEAT boundary: build model + async inference runtime.
  simaai::neat::Model::Options model_opt;
  model_opt.media_type = "video/x-raw";
  model_opt.format = "NV12";
  model_opt.preproc.input_width = frame_w;
  model_opt.preproc.input_height = frame_h;
  model_opt.input_max_width = frame_w;
  model_opt.input_max_height = frame_h;
  model_opt.input_max_depth = 1;
  runtime.model = std::make_unique<simaai::neat::Model>(cfg.mpk, model_opt);
  std::cout << "[init] model configured for " << frame_w << "x" << frame_h << " NV12\n";

  simaai::neat::InputOptions ysrc = runtime.model->input_appsrc_options(false);
  ysrc.media_type = "video/x-raw";
  ysrc.format = "NV12";
  ysrc.width = frame_w;
  ysrc.height = frame_h;
  ysrc.depth = 1;

  runtime.session.add(simaai::neat::nodes::Input(ysrc));
  runtime.session.add(simaai::neat::nodes::groups::Preprocess(*runtime.model));
  runtime.session.add(simaai::neat::nodes::groups::Infer(*runtime.model));
  runtime.session.add(simaai::neat::nodes::SimaBoxDecode(*runtime.model, "yolov8", frame_w, frame_h,
                                                         runtime.min_score, 0.5f, runtime.topk));
  runtime.session.add(simaai::neat::nodes::Output());

  simaai::neat::RunOptions det_run_opt;
  det_run_opt.preset = simaai::neat::RunPreset::Reliable;
  det_run_opt.enable_metrics = true;
  det_run_opt.queue_depth = 4;
  det_run_opt.overflow_policy = simaai::neat::OverflowPolicy::KeepLatest;
  det_run_opt.output_memory = simaai::neat::OutputMemory::Owned;
  simaai::neat::Tensor det_dummy;
  std::string det_err;
  sima_examples::require(make_blank_nv12_tensor(frame_w, frame_h, det_dummy, det_err), det_err);
  std::cout << "[init] building YOLO pipeline\n";
  runtime.run = runtime.session.build(det_dummy, simaai::neat::RunMode::Async, det_run_opt);
  std::cout << "[init] YOLO pipeline ready\n";
  return runtime;
}

std::optional<int> resolve_frame_limit(const Config& cfg) {
  if (cfg.frames_set) {
    sima_examples::require(cfg.frames > 0, "--frames must be > 0");
    return cfg.frames;
  }
  return std::nullopt;
}

std::vector<sima_examples::OptiViewObject> build_optiview_objects(const std::vector<objdet::Box>& boxes,
                                                                   int frame_w, int frame_h) {
  std::vector<sima_examples::OptiViewObject> optiview_objects;
  optiview_objects.reserve(boxes.size());
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
    sima_examples::OptiViewObject obj;
    obj.x = x1;
    obj.y = y1;
    obj.w = w;
    obj.h = h;
    obj.score = box.score;
    obj.class_id = box.class_id;
    optiview_objects.push_back(obj);
  }
  return optiview_objects;
}

void producer_worker(simaai::neat::Run& cam, simaai::neat::Tensor first_frame, double first_pull_ms,
                     double first_pull_ts, WorkerSharedState& state) {
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
      auto frame_opt = cam.pull_tensor();
      if (!frame_opt.has_value())
        continue;
      const double t1 = time_ms();
      frame = std::move(*frame_opt);
      pull_ms = t1 - t0;
      pull_ts = t1;
    }
    print_time("rtsp_pull_ms", pull_ms, state.cfg.debug);
    state.producer_stats.add_rtsp_pull(pull_ms);

    FrameItem item;
    item.index = produced;
    item.frame = std::move(frame);
    item.pull_ts_ms = pull_ts;

    const double q0 = time_ms();
    if (!state.queue.push(std::move(item)))
      break;
    const double q1 = time_ms();
    const double queue_ms = q1 - q0;
    print_time("queue_push_ms", queue_ms, state.cfg.debug);
    state.producer_stats.add_queue_push(queue_ms);

    produced += 1;
    state.producer_stats.count = produced;
  }
  state.queue.close();
  state.producer_end_ms = time_ms();
}

void consumer_worker(simaai::neat::Run& det, simaai::neat::Run& udp_run,
                     sima_examples::OptiViewSender& optiview_sender,
                     const std::vector<std::string>& optiview_labels, int frame_w, int frame_h,
                     int topk, WorkerSharedState& state) {
  state.consumer_start_ms = time_ms();
  int out_pulls = 0;
  while (!state.stop.load() && (!state.frame_limit || state.published.load() < *state.frame_limit)) {
    FrameItem item;
    const double q0 = time_ms();
    if (!state.queue.pop(item))
      break;
    const double q1 = time_ms();
    const double queue_ms = q1 - q0;
    print_time("queue_pop_ms", queue_ms, state.cfg.debug);
    state.consumer_stats.add_queue_pop(queue_ms);

    PendingFrame pending_current;
    pending_current.index = item.index;
    pending_current.pull_ts_ms = item.pull_ts_ms;
    pending_current.frame = std::move(item.frame);

    // NEAT boundary: push current frame into async YOLO run, then pull next output.
    const double t_push0 = time_ms();
    const bool pushed = det.push(pending_current.frame);
    const double t_push1 = time_ms();
    const double push_ms = t_push1 - t_push0;
    print_time("yolo_push_ms", push_ms, state.cfg.debug);
    state.consumer_stats.add_yolo_push(push_ms);
    if (!pushed) {
      std::cerr << "[warn] push failed\n";
      continue;
    }
    const double t_pull0 = time_ms();
    auto out_opt = det.pull();
    const double t_pull1 = time_ms();
    const double pull_ms = t_pull1 - t_pull0;
    print_time("yolo_pull_ms", pull_ms, state.cfg.debug);
    state.consumer_stats.add_yolo_pull(pull_ms);
    if (!out_opt.has_value())
      continue;
    out_pulls += 1;
    state.det_outputs.store(out_pulls);
    if (state.cfg.debug) {
      std::cout << "[dbg] det pull=" << out_pulls << " kind=" << static_cast<int>(out_opt->kind)
                << " tag=" << out_opt->payload_tag << " format=" << out_opt->format
                << " frame_id=" << out_opt->frame_id << " input_seq=" << out_opt->input_seq
                << "\n";
    }

    PendingFrame pending = std::move(pending_current);

    const double t_extract0 = time_ms();
    std::vector<uint8_t> payload;
    std::string err;
    if (!sima_examples::extract_bbox_payload(*out_opt, payload, err)) {
      std::cerr << "[warn] bbox extract failed: " << err << "\n";
      continue;
    }
    const double t_extract1 = time_ms();
    const double extract_ms = t_extract1 - t_extract0;
    print_time("bbox_extract_ms", extract_ms, state.cfg.debug);
    state.consumer_stats.add_bbox_extract(extract_ms);

    const double t_parse0 = time_ms();
    std::vector<objdet::Box> boxes;
    try {
      boxes = objdet::parse_boxes_strict(payload, frame_w, frame_h, topk, state.cfg.debug);
    } catch (const std::exception& ex) {
      std::cerr << "[warn] bbox parse failed: " << ex.what() << "\n";
      continue;
    }
    const double t_parse1 = time_ms();
    const double parse_ms = t_parse1 - t_parse0;
    print_time("bbox_parse_ms", parse_ms, state.cfg.debug);
    state.consumer_stats.add_bbox_parse(parse_ms);

    std::vector<sima_examples::OptiViewObject> optiview_objects =
        build_optiview_objects(boxes, frame_w, frame_h);
    if (state.cfg.debug) {
      std::cout << "boxes=" << boxes.size() << "\n";
    }

    // Contract: publish video first, then publish the matching JSON side-channel payload.
    double output_ts = 0.0;
    const double t_udp_conv0 = time_ms();
    simaai::neat::Tensor nv12_frame;
    std::string nv12_err;
    const bool converted = nv12_copy_to_cpu_tensor(pending.frame, nv12_frame, nv12_err);
    if (!converted) {
      std::cerr << "[warn] udp convert failed: " << nv12_err << "\n";
      continue;
    }
    const double t_udp_conv1 = time_ms();
    const double udp_conv_ms = t_udp_conv1 - t_udp_conv0;
    print_time("nv12_copy_ms", udp_conv_ms, state.cfg.debug);
    state.consumer_stats.add_udp_convert(udp_conv_ms);

    // NEAT boundary: push frame to OptiView video transport run.
    const double t_udp_push0 = time_ms();
    if (!udp_run.push(nv12_frame)) {
      std::cerr << "[warn] udp push failed\n";
      continue;
    }
    const double t_udp_push1 = time_ms();
    const double udp_push_ms = t_udp_push1 - t_udp_push0;
    print_time("udp_push_ms", udp_push_ms, state.cfg.debug);
    state.consumer_stats.add_udp_push(udp_push_ms);
    output_ts = t_udp_push1;
    const int64_t fid = out_opt->frame_id >= 0 ? out_opt->frame_id : static_cast<int64_t>(pending.index);
    const int64_t ts_ms = static_cast<int64_t>(output_ts);
    std::string json_payload = sima_examples::optiview_make_json(
        ts_ms, std::to_string(fid), optiview_objects, optiview_labels);
    std::string json_err;
    if (!optiview_sender.send_json(json_payload, &json_err)) {
      std::cerr << "[warn] optiview json send failed: " << json_err << "\n";
    }

    const double e2e_ms = output_ts - pending.pull_ts_ms;
    print_time("e2e_ms", e2e_ms, state.cfg.debug);
    state.consumer_stats.add_e2e(e2e_ms);

    const int published_now = state.published.fetch_add(1) + 1;
    state.consumer_stats.count = published_now;
  }
  state.stop.store(true);
  state.queue.close();
  state.consumer_end_ms = time_ms();
}

} // namespace

int main(int argc, char** argv) {
  try {
    // Lifecycle: setup -> start workers -> join -> summary -> teardown.
    Config cfg = parse_config(argc, argv);
    sima_examples::require(!cfg.url.empty(), "Missing --rtsp <url>");
    sima_examples::require(!cfg.mpk.empty(), "Missing --mpk <path/to/model_mpk.tar.gz>");

    enable_optiview_diagnostics(true);

    RtspRuntime rtsp_runtime = build_rtsp_runtime(cfg);
    OptiViewRuntime optiview_runtime =
        build_optiview_runtime(cfg, rtsp_runtime.frame_w, rtsp_runtime.frame_h,
                               rtsp_runtime.output_fps);
    YoloRuntime yolo_runtime = build_yolo_runtime(cfg, rtsp_runtime.frame_w, rtsp_runtime.frame_h);

    std::optional<int> frame_limit = resolve_frame_limit(cfg);
    std::cout << "mode=optiview"
              << " frame_limit=" << (frame_limit ? std::to_string(*frame_limit) : "inf")
              << " frames_set=" << (cfg.frames_set ? "1" : "0") << "\n";

    // Contract: bounded queue preserves backpressure and producer closes it on exit.
    FrameQueue queue(300);
    ProducerTiming producer_stats;
    ConsumerTiming consumer_stats;
    std::atomic<bool> stop{false};
    std::atomic<int> published{0};
    std::atomic<int> det_outputs{0};
    double producer_start_ms = 0.0;
    double producer_end_ms = 0.0;
    double consumer_start_ms = 0.0;
    double consumer_end_ms = 0.0;

    WorkerSharedState worker_state{
        cfg,
        frame_limit,
        queue,
        producer_stats,
        consumer_stats,
        stop,
        published,
        det_outputs,
        producer_start_ms,
        producer_end_ms,
        consumer_start_ms,
        consumer_end_ms,
    };

    // Contract: start producer first, then consumer; both terminate when queue closes or stop is set.
    std::thread producer_thread(producer_worker, std::ref(rtsp_runtime.run),
                                std::move(rtsp_runtime.first_frame), rtsp_runtime.first_pull_ms,
                                rtsp_runtime.first_pull_ts, std::ref(worker_state));
    std::thread consumer_thread(
        consumer_worker, std::ref(yolo_runtime.run), std::ref(optiview_runtime.video_run),
        std::ref(*optiview_runtime.sender), std::cref(optiview_runtime.labels), rtsp_runtime.frame_w,
        rtsp_runtime.frame_h, yolo_runtime.topk, std::ref(worker_state));

    if (producer_thread.joinable())
      producer_thread.join();
    if (consumer_thread.joinable())
      consumer_thread.join();

    std::cout << "published=" << published.load() << " udp=" << optiview_runtime.host << ":"
              << optiview_runtime.video_port << "\n";
    print_throughput_summary(producer_stats.count, det_outputs.load(), published.load(),
                             producer_start_ms, producer_end_ms, consumer_start_ms,
                             consumer_end_ms);
    print_stream_summary("rtsp", rtsp_runtime.run, true);
    print_stream_summary("yolo", yolo_runtime.run, true);
    print_pipeline_report("yolo", yolo_runtime.run, true);
    print_stream_summary("udp", optiview_runtime.video_run, true);
    print_pipeline_report("udp", optiview_runtime.video_run, true);
    producer_stats.print();
    consumer_stats.print();

    // Contract: join workers before dropping NEAT runs, then release pipelines in deterministic order.
    std::cerr << "[HOLD] pid=" << getpid() << " (sleeping 20s)\n";
    optiview_runtime.video_run = simaai::neat::Run{};
    rtsp_runtime.run = simaai::neat::Run{};
    yolo_runtime.run = simaai::neat::Run{};
    std::this_thread::sleep_for(std::chrono::seconds(20));
    return 0;

  } catch (const std::exception& e) {
    std::cerr << "[ERR] " << e.what() << "\n";
    return 1;
  }
}
