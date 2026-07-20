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

/**
 * @example single-stream-thermal-face-detector.cpp
 * Single-camera RTSP yolov5s-face Insight example.
 *
 *   RTSP decode (NV12) --> branch --> video_sender (H264 RTP/UDP -> Insight)
 *                                \--> model (raw split heads) --> detections
 *
 * The model archive emits six raw FP32 split heads (paired 18-channel box and
 * 30-channel landmark heads at three pyramid levels). decode_type is left
 * Unspecified, so no fused BoxDecode runs -- the box + 5-landmark decode runs on
 * the host (APU) because the NEAT BBOX wire format carries no landmark slots.
 *
 * Each frame's detections are published to Insight as a `pose-estimation`
 * overlay carrying the 5 named facial landmarks (eyes, nose, mouth corners).
 * The Insight viewer renders one metadata type per channel at a time, so the
 * pipeline sends a single type rather than competing overlays.
 */
#include "neat.h"
#include "support/runtime/config_utils.h"
#include "support/runtime/example_utils.h"

#include <nodes/groups/VideoSender.h>
#include <nodes/io/MetadataSender.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

// The model was compiled for an 800x800 canvas (pyramid levels 100/50/25).
constexpr int kInferSize = 800;
constexpr int kNumAnchors = 3;
constexpr int kNumLandmarks = 5;
constexpr int kBoxChan = 18; // 3 anchors x (xywh + obj + cls)
constexpr int kLmChan = 30;  // 3 anchors x (5 landmarks x xy)
constexpr int kPerAnchorBox = 6;
constexpr int kPerAnchorLm = 10;

// yolov5s-face anchors / strides (fixed by the model architecture).
constexpr float kStrides[3] = {8.0f, 16.0f, 32.0f};
constexpr float kAnchors[3][kNumAnchors][2] = {
    {{4.f, 5.f}, {8.f, 10.f}, {13.f, 16.f}},
    {{23.f, 29.f}, {43.f, 55.f}, {73.f, 105.f}},
    {{146.f, 217.f}, {231.f, 300.f}, {335.f, 433.f}},
};

// Landmark names in yolov5s-face output order. The Insight pose overlay draws a
// named dot per keypoint, and joins any keypoints whose names match a COCO body
// skeleton pair (nose/left_eye/right_eye/...). These names deliberately avoid the
// COCO joint names so no skeleton lines are drawn across the face.
const std::array<const char*, kNumLandmarks> kLmNames = {"eye_l", "eye_r", "nose_tip", "mouth_l",
                                                         "mouth_r"};

struct AppConfig {
  std::string model_path;
  fs::path labels_path;
  std::string rtsp_url;
  int latency_ms = 200;
  bool tcp = true;
  int frames = 0;
  double min_score = 0.25;
  double nms_iou = 0.45;
  int max_detections = 50;
  bool profile = false;
  int profile_interval = 100;
  std::string insight_host = "127.0.0.1";
  int video_port = 9000;
  int metadata_port = 9100;
};

struct CliOptions {
  fs::path config_path;
  bool validate_config_only = false;
};

struct Detection {
  float x1 = 0.f, y1 = 0.f, x2 = 0.f, y2 = 0.f;
  float score = 0.f;
  std::array<cv::Point2f, kNumLandmarks> landmarks{};
};

struct ProfileWindow {
  bool enabled = false;
  int interval = 100;
  int frames = 0;
  int faces = 0;
  double start_ms = 0.0;
  double pull_ms = 0.0;
  double decode_ms = 0.0;
  double metadata_ms = 0.0;

  void add(double pull, double decode, double metadata, int face_count) {
    if (!enabled)
      return;
    if (frames == 0)
      start_ms = sima_examples::time_ms();
    ++frames;
    faces += face_count;
    pull_ms += pull;
    decode_ms += decode;
    metadata_ms += metadata;
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
              << " avg_pull_ms=" << avg(pull_ms) << " avg_decode_ms=" << avg(decode_ms)
              << " avg_metadata_ms=" << avg(metadata_ms)
              << " avg_faces=" << static_cast<double>(faces) / static_cast<double>(frames) << "\n";
    frames = 0;
    faces = 0;
    start_ms = 0.0;
    pull_ms = 0.0;
    decode_ms = 0.0;
    metadata_ms = 0.0;
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
  float scale = 1.0f;
  int pad_l = 0;
  int pad_t = 0;
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
}

AppConfig load_app_config(const fs::path& config_path) {
  const auto raw = sima_examples::ScalarConfig::load(config_path);
  AppConfig cfg;
  const auto default_labels =
      fs::path(SIMANEAT_APPS_EXAMPLE_SOURCE_DIR).parent_path() / "common" / "face_label.txt";
  cfg.model_path = raw.string_or("model.path", "");
  cfg.labels_path = raw.string_or("model.labels", default_labels.string());
  cfg.rtsp_url = raw.string_or("source.rtsp_url", "");
  cfg.latency_ms = raw.int_or("source.latency_ms", 200);
  cfg.tcp = raw.bool_or("source.tcp", true);
  cfg.frames = raw.int_or("inference.frames", 0);
  cfg.min_score = raw.double_or("inference.min_score", 0.25);
  cfg.nms_iou = raw.double_or("inference.nms_iou", 0.45);
  cfg.max_detections = raw.int_or("inference.max_detections", 50);
  cfg.profile = raw.bool_or("runtime.profile", false);
  cfg.profile_interval = raw.int_or("runtime.profile_interval", 100);
  cfg.insight_host = raw.string_or("output.insight.host", "");
  cfg.video_port = raw.int_or("output.insight.video_port", 9000);
  cfg.metadata_port = raw.int_or("output.insight.metadata_port", 9100);
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

// Compute the on-device center-letterbox (scale, pad_l, pad_t) without applying
// it. The CVU preproc does the actual scale + pad; we recompute the same params
// only to inverse-map model-canvas coordinates back to stream pixels.
void letterbox_params(int orig_w, int orig_h, int target_w, int target_h, float& scale, int& pad_l,
                      int& pad_t) {
  scale = std::min(static_cast<float>(target_w) / orig_w, static_cast<float>(target_h) / orig_h);
  const int nw = static_cast<int>(std::round(orig_w * scale));
  const int nh = static_cast<int>(std::round(orig_h * scale));
  pad_l = (target_w - nw) / 2;
  pad_t = (target_h - nh) / 2;
}

inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// Owns the runtime's dense byte buffer and exposes it as a channels-last
// (h, w, c) float32 view.
struct HwcTensor {
  int h = 0;
  int w = 0;
  int c = 0;
  std::vector<uint8_t> bytes;
  const float* floats() const { return reinterpret_cast<const float*>(bytes.data()); }
};

HwcTensor tensor_to_hwc(const simaai::neat::Tensor& t) {
  if (t.dtype != simaai::neat::TensorDType::Float32) {
    throw std::runtime_error("expected Float32 tensor");
  }
  if (t.shape.size() != 4 || t.shape[0] != 1) {
    throw std::runtime_error("expected 4D tensor with shape [1,H,W,C]");
  }
  const int c = static_cast<int>(t.shape[3]);
  if (c != kBoxChan && c != kLmChan) {
    throw std::runtime_error("expected NHWC tensor with channel dim " + std::to_string(kBoxChan) +
                             " or " + std::to_string(kLmChan) + ", got channel " +
                             std::to_string(c));
  }
  HwcTensor out;
  out.h = static_cast<int>(t.shape[1]);
  out.w = static_cast<int>(t.shape[2]);
  out.c = c;
  out.bytes = t.copy_dense_bytes_tight();
  return out;
}

float iou(const Detection& a, const Detection& b) {
  const float xx1 = std::max(a.x1, b.x1);
  const float yy1 = std::max(a.y1, b.y1);
  const float xx2 = std::min(a.x2, b.x2);
  const float yy2 = std::min(a.y2, b.y2);
  const float inter = std::max(0.f, xx2 - xx1) * std::max(0.f, yy2 - yy1);
  const float area_a = std::max(0.f, a.x2 - a.x1) * std::max(0.f, a.y2 - a.y1);
  const float area_b = std::max(0.f, b.x2 - b.x1) * std::max(0.f, b.y2 - b.y1);
  const float den = area_a + area_b - inter;
  return den > 0.f ? inter / den : 0.f;
}

// Pair box (18ch) and landmark (30ch) heads per pyramid level, decode
// anchors + grid, score = sigmoid(obj) * sigmoid(cls), NMS. Output order is not
// assumed; heads are grouped by spatial size and channel count.
std::vector<Detection> decode_yolov5face_split(const simaai::neat::TensorList& tensors,
                                               float min_score, float nms_iou_t,
                                               int max_detections) {
  if (tensors.size() != 6) {
    throw std::runtime_error("expected 6 split tensors, got " + std::to_string(tensors.size()));
  }

  struct LevelHeads {
    HwcTensor box;
    HwcTensor lm;
    bool have_box = false;
    bool have_lm = false;
  };
  std::map<int, LevelHeads> groups;
  for (const auto& t : tensors) {
    HwcTensor hwc = tensor_to_hwc(t);
    const int size = std::max(hwc.h, hwc.w);
    if (hwc.c == kBoxChan) {
      groups[size].box = std::move(hwc);
      groups[size].have_box = true;
    } else {
      groups[size].lm = std::move(hwc);
      groups[size].have_lm = true;
    }
  }

  if (groups.size() != 3) {
    throw std::runtime_error("expected 3 pyramid levels, got " + std::to_string(groups.size()));
  }

  std::vector<int> sizes;
  sizes.reserve(groups.size());
  for (const auto& kv : groups) sizes.push_back(kv.first);
  std::sort(sizes.begin(), sizes.end(), std::greater<int>());

  std::vector<Detection> candidates;
  for (size_t lvl = 0; lvl < sizes.size(); ++lvl) {
    const auto& lh = groups[sizes[lvl]];
    if (!lh.have_box || !lh.have_lm) {
      throw std::runtime_error("level missing box or lm head");
    }
    if (lh.box.h != lh.lm.h || lh.box.w != lh.lm.w) {
      throw std::runtime_error("box/lm spatial mismatch at level " + std::to_string(lvl));
    }
    const float stride = kStrides[lvl];
    const int ny = lh.box.h;
    const int nx = lh.box.w;
    const float* box_data = lh.box.floats();
    const float* lm_data = lh.lm.floats();
    for (int y = 0; y < ny; ++y) {
      for (int x = 0; x < nx; ++x) {
        const size_t cell_box_off = (static_cast<size_t>(y) * nx + x) * kBoxChan;
        const size_t cell_lm_off = (static_cast<size_t>(y) * nx + x) * kLmChan;
        for (int a = 0; a < kNumAnchors; ++a) {
          const float* bp = &box_data[cell_box_off + a * kPerAnchorBox];
          const float* lp = &lm_data[cell_lm_off + a * kPerAnchorLm];

          const float obj = sigmoid(bp[4]);
          const float cls = sigmoid(bp[5]);
          const float score = obj * cls;
          if (score < min_score) continue;

          const float aw = kAnchors[lvl][a][0];
          const float ah = kAnchors[lvl][a][1];
          const float sx = sigmoid(bp[0]);
          const float sy = sigmoid(bp[1]);
          const float sw = sigmoid(bp[2]);
          const float sh = sigmoid(bp[3]);

          const float cx = (sx * 2.0f - 0.5f + static_cast<float>(x)) * stride;
          const float cy = (sy * 2.0f - 0.5f + static_cast<float>(y)) * stride;
          const float bw = (sw * 2.0f) * (sw * 2.0f) * aw;
          const float bh = (sh * 2.0f) * (sh * 2.0f) * ah;

          Detection d;
          d.x1 = cx - bw * 0.5f;
          d.y1 = cy - bh * 0.5f;
          d.x2 = cx + bw * 0.5f;
          d.y2 = cy + bh * 0.5f;
          d.score = score;
          for (int k = 0; k < kNumLandmarks; ++k) {
            d.landmarks[k].x = lp[2 * k] * aw + static_cast<float>(x) * stride;
            d.landmarks[k].y = lp[2 * k + 1] * ah + static_cast<float>(y) * stride;
          }
          candidates.push_back(d);
        }
      }
    }
  }

  std::sort(candidates.begin(), candidates.end(),
            [](const Detection& a, const Detection& b) { return a.score > b.score; });
  std::vector<Detection> keep;
  keep.reserve(candidates.size());
  for (const auto& cand : candidates) {
    bool suppressed = false;
    for (const auto& k : keep) {
      if (iou(k, cand) > nms_iou_t) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) {
      keep.push_back(cand);
      // Candidates are score-sorted; stop once the configured cap is reached to
      // bound both the published count and NMS cost on crowded inputs.
      if (max_detections > 0 && static_cast<int>(keep.size()) >= max_detections) {
        break;
      }
    }
  }
  return keep;
}

void unletterbox(std::vector<Detection>& dets, float scale, int pad_l, int pad_t, int orig_w,
                 int orig_h) {
  const float ow = static_cast<float>(orig_w);
  const float oh = static_cast<float>(orig_h);
  for (auto& d : dets) {
    d.x1 = std::clamp((d.x1 - pad_l) / scale, 0.f, ow);
    d.y1 = std::clamp((d.y1 - pad_t) / scale, 0.f, oh);
    d.x2 = std::clamp((d.x2 - pad_l) / scale, 0.f, ow);
    d.y2 = std::clamp((d.y2 - pad_t) / scale, 0.f, oh);
    for (auto& lm : d.landmarks) {
      lm.x = (lm.x - pad_l) / scale;
      lm.y = (lm.y - pad_t) / scale;
    }
  }
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
  simaai::neat::Model::Options model_opt;
  model_opt.preprocess.kind = simaai::neat::InputKind::Image;
  model_opt.preprocess.enable = simaai::neat::AutoFlag::On;
  // The RTSP decoder emits NV12; the model's on-device preproc converts NV12->RGB,
  // letterboxes to the 800x800 canvas, and normalizes /255 (COCO_YOLO preset).
  model_opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::NV12;
  model_opt.preprocess.preset = simaai::neat::NormalizePreset::COCO_YOLO;
  // decode_type stays Unspecified: the model emits raw split heads; the box +
  // landmark decode runs on the host below.
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
  letterbox_params(runtime.frame_w, runtime.frame_h, kInferSize, kInferSize, runtime.scale,
                   runtime.pad_l, runtime.pad_t);

  auto source = simaai::neat::nodes::groups::RtspDecodedInput(source_options);
  auto branch = simaai::neat::graphs::Branch("source", {"video", "model"});

  auto video_options = simaai::neat::nodes::groups::VideoSenderOptions::H264RtpUdpFromRaw(
      runtime.frame_w, runtime.frame_h, runtime.output_fps);
  video_options.host = cfg.insight_host;
  video_options.channel = 0;
  video_options.video_port_base = cfg.video_port;
  video_options.encoder.bitrate_kbps = 4000;
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
  runtime.graph.connect(source, branch);
  runtime.graph.connect(branch, video_graph, live_link_options);
  runtime.graph.connect(branch, model_graph, live_link_options);
  runtime.graph.connect(model_graph, detections_graph);
  if (cfg.profile) {
    std::cout << "Backend:\n" << runtime.graph.describe_backend() << "\n";
  }

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

// Build the pose-estimation data JSON ({"poses":[{id,label,keypoints:[...]}]}) for
// the Insight pose overlay. No support helper exists for keypoints, so build it
// directly. Names are fixed literals; numeric fields are plain floats.
std::string build_poses_json(const std::vector<Detection>& dets, const std::string& label) {
  std::ostringstream out;
  out << "{\"poses\":[";
  for (size_t i = 0; i < dets.size(); ++i) {
    const auto& d = dets[i];
    if (i) out << ",";
    out << "{\"id\":\"face_" << (i + 1) << "\",\"label\":\"" << label << "\",\"keypoints\":[";
    for (int k = 0; k < kNumLandmarks; ++k) {
      if (k) out << ",";
      out << "{\"name\":\"" << kLmNames[k] << "\",\"x\":" << d.landmarks[k].x
          << ",\"y\":" << d.landmarks[k].y << ",\"confidence\":" << d.score << "}";
    }
    out << "]}";
  }
  out << "]}";
  return out.str();
}

void send_metadata(PipelineRuntime& runtime, const simaai::neat::Sample& sample,
                   const std::vector<Detection>& dets) {
  const std::string label = runtime.labels.empty() ? "face" : runtime.labels[0];
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  const int64_t ts_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
  const int64_t frame_id = sample.frame_id >= 0 ? sample.frame_id : 0;
  const std::string frame_id_str = std::to_string(frame_id);

  const std::string poses_json = build_poses_json(dets, label);

  std::string err;
  if (!runtime.metadata_sender->send_metadata("pose-estimation", poses_json, ts_ms, frame_id_str,
                                              &err)) {
    std::cerr << "[warn] insight pose metadata send failed: " << err << "\n";
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
    const double pull_start = sima_examples::time_ms();
    const auto status = runtime.run.pull("detections", 20000, sample, &pull_error);
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

    const auto tensors = simaai::neat::tensors_from_sample(sample, false);
    auto dets = decode_yolov5face_split(tensors, static_cast<float>(cfg.min_score),
                                        static_cast<float>(cfg.nms_iou), cfg.max_detections);
    unletterbox(dets, runtime.scale, runtime.pad_l, runtime.pad_t, runtime.frame_w,
                runtime.frame_h);
    const double decode_end = sima_examples::time_ms();

    send_metadata(runtime, sample, dets);
    const double metadata_end = sima_examples::time_ms();

    ++processed;
    profile.add(pull_end - pull_start, decode_end - pull_end, metadata_end - decode_end,
                static_cast<int>(dets.size()));
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
