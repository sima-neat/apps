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
//
// RF-DETR-Seg (432) single-camera RTSP Insight example. The model ships as a
// three-stage split -- INT8 backbone (MLA), top-k gather (A65, in-process TVM),
// BF16 transformer + seg-head (MLA) -- so, unlike the single-model examples in
// this category, the three stages are driven explicitly here instead of being
// expressed as one graph node.

#include "neat.h"
#include "neat/models.h"
#include "neat/node_groups.h"
#include "neat/nodes.h"
#include "support/runtime/config_utils.h"
#include "support/runtime/example_utils.h"
#include <nodes/groups/VideoSender.h>
#include <nodes/io/MetadataSender.h>

#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>

#include <nlohmann/json.hpp>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace nt = simaai::neat;

using sima_examples::time_ms;

namespace {

// ── model constants (fixed by the compiled RF-DETR-Seg 432 split) ──────────
constexpr int kImageH = 432, kImageW = 432;
constexpr int kNumQueries = 200, kNumClasses = 91, kMaskHW = 108;
const std::array<float, 3> kCocoMean = {0.485f, 0.456f, 0.406f};
const std::array<float, 3> kCocoStd = {0.229f, 0.224f, 0.225f};

const std::vector<int64_t> kFeatureShape = {1, 36, 36, 256};
const std::vector<int64_t> kConcatShape = {1, 1296, 4};
const std::vector<int64_t> kGatherShape = {1, 200, 4};
const std::vector<int64_t> kLogitsShape = {1, 200, 91};
const std::vector<int64_t> kMasksDeviceShape = {108, 108, 200};

const char* kBackboneName = "rfdetr_seg_432_simplified_backbone_before_topk_base_mpk";
const char* kTopkName = "rfdetr_seg_432_simplified_topk_to_gather_base_mpk";
const char* kTopkSo = "rfdetr_seg_432_simplified_topk_to_gather_base_stage1_a65.so";
const char* kTransformerName = "rfdetr_seg_432_simplified_transformer_after_gather_base_mpk";

/// MetadataSender rejects a payload above 65507 bytes, and the rejection surfaces as an error the
/// application has to handle mid-stream. Half of that leaves room for the envelope and keeps the
/// datagram count low enough for Insight to reassemble within its 250 ms window.
constexpr std::size_t kMetadataByteBudget = 32768;

int64_t numel(const std::vector<int64_t>& s) {
  int64_t n = 1;
  for (auto d : s)
    n *= d;
  return n;
}
inline float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }

std::vector<nt::Tensor> collect_tensors(const nt::Sample& s) {
  if (s.kind == nt::SampleKind::Tensor) {
    if (!s.tensor.has_value())
      throw std::runtime_error("tensor sample missing payload");
    return {*s.tensor};
  }
  if (s.kind == nt::SampleKind::TensorSet)
    return s.tensors;
  if (s.kind == nt::SampleKind::Bundle) {
    std::vector<nt::Tensor> out;
    for (const auto& f : s.fields) {
      auto c = collect_tensors(f);
      out.insert(out.end(), c.begin(), c.end());
    }
    return out;
  }
  throw std::runtime_error("unexpected sample kind");
}

std::vector<float> tensor_to_f32(const nt::Tensor& t) {
  if (t.dtype != nt::TensorDType::Float32)
    throw std::runtime_error("expected Float32 tensor");
  const std::vector<uint8_t> raw = t.copy_dense_bytes_tight();
  std::vector<float> out(raw.size() / sizeof(float));
  std::memcpy(out.data(), raw.data(), raw.size());
  return out;
}

bool sample_empty(const nt::Sample& s) {
  try {
    return collect_tensors(s).empty();
  } catch (...) {
    return true;
  }
}

// ── config ───────────────────────────────────────────────────────────────
struct AppConfig {
  std::string model_root;
  fs::path labels_path;
  std::string source_url;
  bool tcp = true;
  int latency_ms = 200;
  int source_fps = 0;
  int frames = 0;
  double score_threshold = 0.30;
  int max_detections = 24;
  bool profile = false;
  int profile_interval = 100;
  std::string insight_host;
  int video_port = 9000;
  int metadata_port = 9100;
  fs::path save_dir;
  int save_every = 0;
  double mask_alpha = 0.55;
  // Sigmoid-space foreground cutoff; see src/common/config.yaml for why 0.08, not 0.5.
  double mask_threshold = 0.08;
  bool draw_boxes = true;
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
      if (i + 1 >= argc)
        throw std::runtime_error("--config requires a path");
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
  sima_examples::require(!cfg.source_url.empty(), "source.url must be set");
  sima_examples::require(!cfg.model_root.empty(), "model.path must be set");
  sima_examples::require(!cfg.labels_path.empty(), "model.labels must be set");
  sima_examples::require(!cfg.insight_host.empty(), "output.insight.host must be set");
  sima_examples::require(cfg.latency_ms >= 0, "source.latency_ms must be >= 0");
  sima_examples::require(cfg.source_fps >= 0, "source.fps must be >= 0");
  sima_examples::require(cfg.frames >= 0, "inference.frames must be >= 0");
  sima_examples::require(cfg.score_threshold >= 0.0 && cfg.score_threshold <= 1.0,
                         "inference.score_threshold must be between 0 and 1");
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
  cfg.model_root = raw.string_or("model.path", "");
  cfg.labels_path = raw.string_or("model.labels", default_labels.string());
  cfg.source_url = raw.string_or("source.url", "");
  cfg.tcp = raw.bool_or("source.tcp", true);
  cfg.latency_ms = raw.int_or("source.latency_ms", 200);
  cfg.source_fps = raw.int_or("source.fps", 0);
  cfg.frames = raw.int_or("inference.frames", 0);
  cfg.score_threshold = raw.double_or("inference.score_threshold", 0.30);
  cfg.max_detections = raw.int_or("inference.max_detections", 24);
  cfg.profile = raw.bool_or("runtime.profile", false);
  cfg.profile_interval = raw.int_or("runtime.profile_interval", 100);
  cfg.insight_host = raw.string_or("output.insight.host", "");
  cfg.video_port = raw.int_or("output.insight.video_port", 9000);
  cfg.metadata_port = raw.int_or("output.insight.metadata_port", 9100);
  cfg.save_dir = raw.string_or("output.save_dir", "");
  cfg.save_every = raw.int_or("output.save_every", 0);
  cfg.mask_alpha = raw.double_or("output.mask_alpha", 0.55);
  cfg.mask_threshold = raw.double_or("output.mask_threshold", 0.08);
  cfg.draw_boxes = raw.bool_or("output.draw_boxes", true);
  validate_config(cfg);
  return cfg;
}

std::vector<std::string> load_labels(const fs::path& labels_path) {
  std::ifstream in(labels_path);
  if (!in.good())
    throw std::runtime_error("labels file does not exist: " + labels_path.string());

  std::vector<std::string> labels;
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty())
      labels.push_back(line);
  }
  if (labels.empty())
    throw std::runtime_error("labels file is empty: " + labels_path.string());
  return labels;
}

// ── model options (preprocess is stretch-resize, matching the compiled graph) ──
nt::Model::Options backbone_options() {
  nt::Model::Options opt;
  opt.preprocess.kind = nt::InputKind::Image;
  opt.preprocess.enable = nt::AutoFlag::On;
  opt.preprocess.color_convert.enable = nt::AutoFlag::On;
  opt.preprocess.color_convert.input_format = nt::PreprocessColorFormat::BGR;
  opt.preprocess.color_convert.output_format = nt::PreprocessColorFormat::RGB;
  opt.preprocess.input_max_width = kImageW;
  opt.preprocess.input_max_height = kImageH;
  opt.preprocess.input_max_depth = 3;
  opt.preprocess.resize.enable = nt::AutoFlag::On;
  opt.preprocess.resize.width = kImageW;
  opt.preprocess.resize.height = kImageH;
  opt.preprocess.resize.mode = nt::ResizeMode::Stretch;
  opt.preprocess.resize.scaling_type = "BILINEAR";
  opt.preprocess.normalize.enable = nt::AutoFlag::On;
  opt.preprocess.normalize.mean = kCocoMean;
  opt.preprocess.normalize.stddev = kCocoStd;
  opt.preprocess.normalize.has_explicit_stats = true;
  opt.processcvu.pre_run_target = "EV74";
  opt.processcvu.post_run_target = "EV74";
  opt.processcvu.async = false;
  return opt;
}

nt::Model::Options transformer_options() {
  nt::Model::Options opt;
  opt.preprocess.kind = nt::InputKind::Tensor;
  opt.preprocess.enable = nt::AutoFlag::Off;
  opt.processcvu.pre_run_target = "A65";
  opt.processcvu.post_run_target = "A65";
  opt.processcvu.async = false;
  return opt;
}

// ── in-process TVM top-k gather (host-side hop between the two MLA stages) ──
class TopkTvm {
public:
  explicit TopkTvm(const std::string& so_path) {
    dev_.device_type = kDLCPU;
    dev_.device_id = 0;
    tvm::runtime::Module lib = tvm::runtime::Module::LoadFromFile(so_path);
    gmod_ = lib.GetFunction("default")(dev_);
    set_input_ = gmod_.GetFunction("set_input");
    run_ = gmod_.GetFunction("run");
    get_output_ = gmod_.GetFunction("get_output");
    if (set_input_ == nullptr || run_ == nullptr || get_output_ == nullptr)
      throw std::runtime_error("topk: graph_executor functions not found");
  }

  std::vector<float> run(const std::vector<float>& concat, const std::vector<float>& reduce) {
    using tvm::runtime::NDArray;
    DLDataType f32{kDLFloat, 32, 1};
    NDArray a0 = NDArray::Empty(tvm::runtime::ShapeTuple({1, 1296, 4}), f32, dev_);
    NDArray a1 = NDArray::Empty(tvm::runtime::ShapeTuple({1, 1296}), f32, dev_);
    a0.CopyFromBytes(concat.data(), concat.size() * sizeof(float));
    a1.CopyFromBytes(reduce.data(), reduce.size() * sizeof(float));
    set_input_("arm_0_i0", a0);
    set_input_("arm_0_i1", a1);
    run_();
    NDArray out = get_output_(0);
    std::vector<float> gather(200 * 4);
    out.CopyToBytes(gather.data(), gather.size() * sizeof(float));
    return gather;
  }

private:
  DLDevice dev_{};
  tvm::runtime::Module gmod_{nullptr};
  tvm::runtime::PackedFunc set_input_, run_, get_output_;
};

struct FrameOut {
  std::vector<float> boxes;     // 200*4, normalized cx,cy,w,h in [0,1]
  std::vector<float> logits;    // 200*91, raw (pre-sigmoid)
  std::vector<float> masks_dev; // raw device (108,108,200), pre-sigmoid
};

// Explicit three-stage split: backbone (MLA) -> top-k gather (A65/TVM) ->
// transformer + seg-head (MLA). Kept as a single-threaded per-frame call so the
// setup/inference/teardown flow stays visible at the call site in main().
class Pipeline {
public:
  Pipeline(const fs::path& root, int timeout_ms) : timeout_ms_(timeout_ms) {
    nt::RunOptions ro;
    ro.queue_depth = 4;
    ro.startup_preflight = false;

    backbone_ = std::make_unique<nt::Model>((root / kBackboneName).string(), backbone_options());
    cv::Mat seed(kImageH, kImageW, CV_8UC3, cv::Scalar(0, 0, 0));
    nt::Tensor seed_img =
        nt::Tensor::from_cv_mat(seed, nt::ImageSpec::PixelFormat::BGR, nt::TensorMemory::EV74);
    br_ = backbone_->build(nt::TensorList{seed_img}, nt::Model::RouteOptions{}, ro);

    transformer_ =
        std::make_unique<nt::Model>((root / kTransformerName).string(), transformer_options());
    nt::Tensor fseed = nt::Tensor::from_vector(std::vector<float>(36 * 36 * 256, 0.f),
                                               {36, 36, 256}, nt::TensorMemory::EV74);
    nt::Tensor gseed =
        nt::Tensor::from_vector(std::vector<float>(200 * 4, 0.f), {1, 200, 4}, nt::TensorMemory::EV74);
    tr_ = transformer_->build(nt::TensorList{fseed, gseed}, nt::Model::RouteOptions{}, ro);

    topk_ = std::make_unique<TopkTvm>((root / kTopkName / "lib" / kTopkSo).string());
  }

  FrameOut run_one(const cv::Mat& bgr_432) {
    nt::Tensor img =
        nt::Tensor::from_cv_mat(bgr_432, nt::ImageSpec::PixelFormat::BGR, nt::TensorMemory::EV74);
    if (!br_.push(nt::TensorList{img}))
      throw std::runtime_error("backbone push failed");
    nt::Sample bs = br_.pull(timeout_ms_);
    if (sample_empty(bs))
      throw std::runtime_error("backbone pull empty");
    std::vector<float> feature, reduce, concat;
    for (const auto& t : collect_tensors(bs)) {
      const auto& s = t.shape;
      const int64_t n = numel(s);
      if (s == kFeatureShape)
        feature = tensor_to_f32(t);
      else if (s == kConcatShape)
        concat = tensor_to_f32(t);
      else if (n == 1296)
        reduce = tensor_to_f32(t);
    }
    if (feature.empty() || reduce.empty() || concat.empty())
      throw std::runtime_error("backbone split failed");

    std::vector<float> gather = topk_->run(concat, reduce);

    nt::Tensor feat_in = nt::Tensor::from_vector(feature, {36, 36, 256}, nt::TensorMemory::EV74);
    nt::Tensor gath_in =
        nt::Tensor::from_vector(gather, {1, 200, 4}, nt::TensorMemory::EV74);
    if (!tr_.push(nt::TensorList{feat_in, gath_in}))
      throw std::runtime_error("transformer push failed");
    nt::Sample to = tr_.pull(timeout_ms_);
    int tries = 0;
    while (sample_empty(to) && tries++ < 5)
      to = tr_.pull(timeout_ms_);
    if (sample_empty(to))
      throw std::runtime_error("transformer pull empty");

    FrameOut fo;
    std::vector<float> masks_dev;
    for (const auto& t : collect_tensors(to)) {
      const auto& s = t.shape;
      if (s == kGatherShape)
        fo.boxes = tensor_to_f32(t);
      else if (s == kLogitsShape)
        fo.logits = tensor_to_f32(t);
      else if (s == kMasksDeviceShape || numel(s) == kNumQueries * kMaskHW * kMaskHW)
        masks_dev = tensor_to_f32(t);
    }
    if (fo.boxes.empty() || fo.logits.empty() || masks_dev.empty())
      throw std::runtime_error("transformer split failed");
    fo.masks_dev = std::move(masks_dev);
    return fo;
  }

  void close() {
    try {
      br_.close();
    } catch (...) {
    }
    try {
      tr_.close();
    } catch (...) {
    }
  }

private:
  int timeout_ms_;
  std::unique_ptr<nt::Model> backbone_, transformer_;
  std::unique_ptr<TopkTvm> topk_;
  nt::Model::Runner br_, tr_;
};

// ── per-query select + stretch-space box/mask projection ───────────────────
// The compiled graph resizes with ResizeMode::Stretch (independent x/y scale,
// no letterbox padding), so, unlike the letterboxed examples in this category,
// mapping a normalized box or the 108x108 mask grid back to frame pixels is a
// plain per-axis scale.
struct Detection {
  int query = 0;
  float score = 0.0f;
  int class_id = -1;
};

std::vector<Detection> select_dets(const FrameOut& fo, double score_threshold, int max_detections) {
  std::vector<Detection> keep;
  for (int q = 0; q < kNumQueries; ++q) {
    int best_c = 0;
    float best = -1e30f;
    for (int c = 0; c < kNumClasses; ++c) {
      const float v = fo.logits[q * kNumClasses + c];
      if (v > best) {
        best = v;
        best_c = c;
      }
    }
    const float score = sigmoidf(best);
    if (score >= score_threshold)
      keep.push_back({q, score, best_c});
  }
  std::stable_sort(keep.begin(), keep.end(),
                   [](const Detection& a, const Detection& b) { return a.score > b.score; });
  if (static_cast<int>(keep.size()) > max_detections)
    keep.resize(max_detections);
  return keep;
}

cv::Rect frame_rect_for_query(const FrameOut& fo, int query, const cv::Size& frame_size) {
  const float cx = fo.boxes[query * 4 + 0], cy = fo.boxes[query * 4 + 1];
  const float bw = fo.boxes[query * 4 + 2], bh = fo.boxes[query * 4 + 3];
  const int x0 = std::clamp(static_cast<int>(std::lround((cx - bw / 2) * frame_size.width)), 0,
                            frame_size.width - 1);
  const int y0 = std::clamp(static_cast<int>(std::lround((cy - bh / 2) * frame_size.height)), 0,
                            frame_size.height - 1);
  const int x1 = std::clamp(static_cast<int>(std::lround((cx + bw / 2) * frame_size.width)), x0 + 1,
                            frame_size.width);
  const int y1 = std::clamp(static_cast<int>(std::lround((cy + bh / 2) * frame_size.height)), y0 + 1,
                            frame_size.height);
  return cv::Rect(x0, y0, x1 - x0, y1 - y0);
}

/// Sigmoid-activated 8-bit mask for query `q`, gathered directly from the raw device
/// (108,108,200) buffer so only the handful of kept queries are ever touched.
cv::Mat mask_for_query(const std::vector<float>& masks_dev, int q) {
  cv::Mat m(kMaskHW, kMaskHW, CV_8UC1);
  for (int hh = 0; hh < kMaskHW; ++hh) {
    uint8_t* dst = m.ptr<uint8_t>(hh);
    const float* base = &masks_dev[(hh * kMaskHW) * kNumQueries + q];
    for (int ww = 0; ww < kMaskHW; ++ww) {
      const float p = sigmoidf(base[ww * kNumQueries]);
      dst[ww] = static_cast<uint8_t>(std::clamp(p * 255.0f, 0.0f, 255.0f));
    }
  }
  return m;
}

/// The 108x108 mask grid covers the whole stretched 432x432 model input with no
/// padding, so a frame rectangle reaches it through a plain per-axis scale.
cv::Rect mask_rect_for_frame_rect(const cv::Rect& frame_rect, const cv::Size& frame_size) {
  const double sx = static_cast<double>(kMaskHW) / frame_size.width;
  const double sy = static_cast<double>(kMaskHW) / frame_size.height;
  const int x0 = std::clamp(static_cast<int>(std::floor(frame_rect.x * sx)), 0, kMaskHW - 1);
  const int y0 = std::clamp(static_cast<int>(std::floor(frame_rect.y * sy)), 0, kMaskHW - 1);
  const int x1 = std::clamp(static_cast<int>(std::ceil((frame_rect.x + frame_rect.width) * sx)),
                            x0 + 1, kMaskHW);
  const int y1 = std::clamp(static_cast<int>(std::ceil((frame_rect.y + frame_rect.height) * sy)),
                            y0 + 1, kMaskHW);
  return cv::Rect(x0, y0, x1 - x0, y1 - y0);
}

cv::Mat project_mask_roi(const cv::Mat& mask, const cv::Rect& frame_rect,
                         const cv::Size& frame_size) {
  const cv::Rect mask_rect = mask_rect_for_frame_rect(frame_rect, frame_size);
  cv::Mat projected;
  cv::resize(mask(mask_rect), projected, frame_rect.size(), 0, 0, cv::INTER_LINEAR);
  return projected;
}

std::string class_name(const std::vector<std::string>& labels, int class_id) {
  return class_id >= 0 && class_id < static_cast<int>(labels.size()) ? labels[class_id]
                                                                     : "unknown";
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

void draw_box(cv::Mat& frame, const cv::Rect& rect, int class_id, float score,
             const std::vector<std::string>& labels) {
  const cv::Scalar color = class_color(class_id);
  cv::rectangle(frame, rect, color, 2);
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%s %.2f", class_name(labels, class_id).c_str(), score);
  cv::putText(frame, buf, cv::Point(rect.x, std::max(0, rect.y - 4)), cv::FONT_HERSHEY_SIMPLEX,
              0.5, color, 1, cv::LINE_AA);
}

cv::Mat overlay_segmentation(const cv::Mat& frame, const FrameOut& fo,
                             const std::vector<Detection>& detections,
                             const std::vector<std::string>& labels, const AppConfig& cfg) {
  cv::Mat annotated = frame.clone();
  for (const auto& det : detections) {
    const cv::Rect rect = frame_rect_for_query(fo, det.query, annotated.size());
    cv::Mat mask108 = mask_for_query(fo.masks_dev, det.query);
    cv::Mat resized_mask = project_mask_roi(mask108, rect, annotated.size());
    cv::Mat binary_mask;
    cv::threshold(resized_mask, binary_mask, cfg.mask_threshold * 255.0, 255, cv::THRESH_BINARY);
    if (cv::countNonZero(binary_mask) > 0) {
      cv::Mat roi = annotated(rect);
      cv::Mat mask_color(rect.size(), annotated.type(), class_color(det.class_id));
      cv::Mat blended;
      cv::addWeighted(roi, 1.0 - cfg.mask_alpha, mask_color, cfg.mask_alpha, 0.0, blended);
      blended.copyTo(roi, binary_mask);

      std::vector<std::vector<cv::Point>> contours;
      cv::findContours(binary_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
      cv::drawContours(roi, contours, -1, class_color(det.class_id), 2);
    }
    if (cfg.draw_boxes)
      draw_box(annotated, rect, det.class_id, det.score, labels);
  }
  return annotated;
}

// ── Insight metadata (segmentation polygons), same wire format as the other
// segmentation examples in this category ────────────────────────────────────
struct MetadataSegment {
  std::string id;
  std::string label;
  float confidence = 0.0f;
  cv::Rect bbox;
  std::vector<cv::Point> polygon;
};

struct EncodedSegments {
  std::string data_json;
  int dropped = 0;
};

std::vector<cv::Point> mask_polygon(const cv::Mat& mask108, const cv::Rect& frame_rect,
                                    const cv::Size& frame_size, double threshold) {
  cv::Mat binary;
  cv::threshold(project_mask_roi(mask108, frame_rect, frame_size), binary, threshold * 255.0, 255,
               cv::THRESH_BINARY);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  if (contours.empty())
    return {};
  const auto& largest =
      *std::max_element(contours.begin(), contours.end(),
                        [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                          return cv::contourArea(a) < cv::contourArea(b);
                        });

  std::vector<cv::Point> polygon;
  cv::approxPolyDP(largest, polygon, 0.004 * cv::arcLength(largest, true), true);
  if (polygon.size() < 3)
    return {};
  for (auto& point : polygon)
    point += frame_rect.tl();
  return polygon;
}

EncodedSegments encode_segments(std::vector<MetadataSegment> segments) {
  std::stable_sort(segments.begin(), segments.end(),
                   [](const MetadataSegment& a, const MetadataSegment& b) {
                     return a.confidence > b.confidence;
                   });

  nlohmann::json entries = nlohmann::json::array();
  std::size_t bytes = sizeof(R"({"segments":[]})") - 1;
  for (const auto& segment : segments) {
    nlohmann::json points = nlohmann::json::array();
    for (const auto& point : segment.polygon)
      points.push_back({point.x, point.y});
    nlohmann::json entry = {
        {"id", segment.id},
        {"label", segment.label},
        {"confidence", segment.confidence},
        {"bbox", {segment.bbox.x, segment.bbox.y, segment.bbox.width, segment.bbox.height}},
        {"mask_format", "polygon"},
        {"mask", std::move(points)},
    };
    const std::size_t entry_bytes = entry.dump().size() + 1;
    if (bytes + entry_bytes > kMetadataByteBudget)
      break;
    bytes += entry_bytes;
    entries.push_back(std::move(entry));
  }
  const int dropped = static_cast<int>(segments.size() - entries.size());
  return {nlohmann::json{{"segments", std::move(entries)}}.dump(), dropped};
}

std::vector<MetadataSegment> build_metadata_segments(const FrameOut& fo,
                                                     const std::vector<Detection>& detections,
                                                     const std::vector<std::string>& labels,
                                                     const cv::Size& frame_size,
                                                     double mask_threshold) {
  std::vector<MetadataSegment> segments;
  segments.reserve(detections.size());
  for (const auto& det : detections) {
    const cv::Rect rect = frame_rect_for_query(fo, det.query, frame_size);
    cv::Mat mask108 = mask_for_query(fo.masks_dev, det.query);
    auto polygon = mask_polygon(mask108, rect, frame_size, mask_threshold);
    if (polygon.empty())
      continue;
    segments.push_back({"seg_" + std::to_string(segments.size() + 1),
                        class_name(labels, det.class_id), det.score, rect, std::move(polygon)});
  }
  return segments;
}

// ── RTSP source + Insight video sender ──────────────────────────────────────
struct SourceGeometry {
  int width = 0;
  int height = 0;
  int fps = 0;
};

SourceGeometry probe_source_geometry(const AppConfig& cfg) {
  sima_examples::RtspStreamInfo probe;
  sima_examples::RtspProbeOptions probe_options;
  probe_options.latency_ms = cfg.latency_ms;
  probe_options.rtsp_tcp = cfg.tcp;
  probe_options.debug = cfg.profile;
  (void)sima_examples::probe_rtsp_stream_info(cfg.source_url, probe_options, probe);

  SourceGeometry geometry;
  geometry.width = probe.width;
  geometry.height = probe.height;
  geometry.fps = cfg.source_fps > 0 ? cfg.source_fps : probe.fps;
  return geometry;
}

nt::nodes::groups::RtspDecodedInputOptions make_rtsp_source_options(const AppConfig& cfg,
                                                                    const SourceGeometry& geometry) {
  nt::nodes::groups::RtspDecodedInputOptions opt;
  opt.url = cfg.source_url;
  opt.latency_ms = cfg.latency_ms;
  opt.tcp = cfg.tcp;
  opt.insert_queue = true;
  opt.out_format = "NV12";
  opt.decoder_name = "decoder";
  opt.decoder_raw_output = true;
  opt.codec = nt::nodes::groups::RtspCodec::H264;
  opt.source_fps = geometry.fps;
  opt.auto_caps_from_stream = true;
  opt.fallback_h264_width = geometry.width;
  opt.fallback_h264_height = geometry.height;
  if (geometry.width > 0 && geometry.height > 0 && geometry.fps > 0) {
    opt.output_caps.enable = true;
    opt.output_caps.format = "NV12";
    opt.output_caps.width = geometry.width;
    opt.output_caps.height = geometry.height;
    opt.output_caps.fps = geometry.fps;
    opt.output_caps.memory = nt::CapsMemory::Any;
  }
  return opt;
}

cv::Mat tensor_bgr_from_decoded(const nt::Tensor& tensor) {
  cv::Mat bgr;
  std::string err;
  if (sima_examples::nv12_to_bgr(tensor, bgr, err))
    return bgr;
  return tensor.to_cv_mat_copy(nt::ImageSpec::PixelFormat::BGR);
}

nt::Tensor frame_tensor_from_sample(const nt::Sample& sample) {
  const auto tensors = collect_tensors(sample);
  if (tensors.empty())
    throw std::runtime_error("decoded frame sample has no tensor");
  return tensors.front();
}

struct ProfileWindow {
  bool enabled = false;
  int interval = 100;
  int frames = 0;
  int instances = 0;
  int dropped_segments = 0;
  double start_ms = 0.0;
  double pull_ms = 0.0;
  double model_ms = 0.0;
  double metadata_ms = 0.0;

  void add(double pull, double model, double metadata, int instance_count, int dropped) {
    if (!enabled)
      return;
    if (frames == 0)
      start_ms = time_ms();
    ++frames;
    instances += instance_count;
    dropped_segments += dropped;
    pull_ms += pull;
    model_ms += model;
    metadata_ms += metadata;
    if (frames >= interval)
      flush();
  }

  void flush() {
    if (!enabled || frames <= 0)
      return;
    const double elapsed_ms = std::max(time_ms() - start_ms, 1e-6);
    const double n = static_cast<double>(frames);
    std::cout << "[profile] frames=" << frames << " output_fps=" << frames * 1000.0 / elapsed_ms
              << " avg_pull_ms=" << pull_ms / n << " avg_model_ms=" << model_ms / n
              << " avg_metadata_ms=" << metadata_ms / n
              << " avg_instances=" << static_cast<double>(instances) / n
              << " dropped_segments=" << dropped_segments << "\n";
    frames = 0;
    instances = 0;
    dropped_segments = 0;
    start_ms = 0.0;
    pull_ms = 0.0;
    model_ms = 0.0;
    metadata_ms = 0.0;
  }
};

struct PipelineRuntime {
  nt::Graph graph;
  nt::Run run;
  std::unique_ptr<nt::MetadataSender> metadata_sender;
  std::vector<std::string> labels;
  int frame_w = 0;
  int frame_h = 0;
  int output_fps = 30;
  int video_port = 0;
};

PipelineRuntime build_graph(const AppConfig& cfg, const SourceGeometry& geometry) {
  PipelineRuntime runtime;
  runtime.frame_w = geometry.width;
  runtime.frame_h = geometry.height;
  runtime.output_fps = geometry.fps;
  sima_examples::require(runtime.frame_w > 0 && runtime.frame_h > 0,
                         "failed to probe source frame dimensions");
  sima_examples::require(runtime.output_fps > 0, "failed to resolve source frame rate");

  runtime.labels = load_labels(cfg.labels_path);

  auto video_options = nt::nodes::groups::VideoSenderOptions::H264RtpUdpFromRaw(
      runtime.frame_w, runtime.frame_h, runtime.output_fps);
  video_options.host = cfg.insight_host;
  video_options.channel = 0;
  video_options.video_port_base = cfg.video_port;
  video_options.encoder.bitrate_kbps = 1000;
  runtime.video_port = video_options.video_port();

  // Insight correlates the RTP timestamp with the metadata timestamp, so the encoder and the
  // segments must stay in one Run and therefore on one GStreamer timeline. The app pulls "frame"
  // to drive the three model stages itself; the "video" branch flows to Insight on its own.
  auto source = nt::nodes::groups::RtspDecodedInput(make_rtsp_source_options(cfg, geometry));
  auto branch = nt::graphs::Branch("source", {"video", "frame"});

  nt::Graph video_graph("video");
  video_graph.connect(nt::nodes::Input("video"), nt::nodes::groups::VideoSender(video_options));

  nt::Graph frame_graph("frame");
  frame_graph.add(nt::nodes::Output("frame", nt::OutputOptions::EveryFrame(1)));

  runtime.graph.connect(source, branch);
  runtime.graph.connect(branch, video_graph);
  runtime.graph.connect(branch, frame_graph);

  if (cfg.profile)
    std::cout << "Backend:\n" << runtime.graph.describe_backend() << "\n";

  nt::RunOptions run_options;
  run_options.preset = nt::RunPreset::Realtime;
  run_options.queue_depth = 3;
  run_options.overflow_policy = nt::OverflowPolicy::KeepLatest;
  run_options.output_memory = nt::OutputMemory::ZeroCopy;
  runtime.run = runtime.graph.build(run_options);

  nt::MetadataSenderOptions metadata_options;
  metadata_options.host = cfg.insight_host;
  metadata_options.channel = 0;
  metadata_options.metadata_port_base = cfg.metadata_port;
  std::string metadata_err;
  runtime.metadata_sender = std::make_unique<nt::MetadataSender>(metadata_options, &metadata_err);
  sima_examples::require(runtime.metadata_sender->ok(), metadata_err);

  std::cout << "source=" << cfg.source_url << " stream=" << runtime.frame_w << "x"
            << runtime.frame_h << "@" << runtime.output_fps << " insight=" << cfg.insight_host
            << " video=" << runtime.video_port
            << " metadata=" << runtime.metadata_sender->metadata_port() << " channel=0\n";
  return runtime;
}

int send_metadata(PipelineRuntime& runtime, const AppConfig& cfg, const nt::Sample& sample,
                  const FrameOut& fo, const std::vector<Detection>& detections) {
  const auto encoded = encode_segments(build_metadata_segments(
      fo, detections, runtime.labels, cv::Size(runtime.frame_w, runtime.frame_h),
      cfg.mask_threshold));
  const int64_t ts_ms = sample.pts_ns >= 0 ? sample.pts_ns / 1'000'000 : -1;
  const std::string frame_id = sample.frame_id >= 0 ? std::to_string(sample.frame_id) : "";
  std::string err;
  if (!runtime.metadata_sender->send_metadata("segmentation", encoded.data_json, ts_ms, frame_id,
                                              &err))
    std::cerr << "[warn] insight metadata send failed: " << err << "\n";
  return encoded.dropped;
}

void maybe_save_frame(const AppConfig& cfg, int processed, const cv::Mat& bgr, const FrameOut& fo,
                      const std::vector<Detection>& detections,
                      const std::vector<std::string>& labels) {
  if (cfg.save_dir.empty() || cfg.save_every <= 0 || processed % cfg.save_every != 0)
    return;
  const cv::Mat annotated = overlay_segmentation(bgr, fo, detections, labels, cfg);
  const auto out_path = cfg.save_dir / ("frame_" + std::to_string(processed) + ".jpg");
  if (!cv::imwrite(out_path.string(), annotated))
    std::cerr << "[warn] failed to write output frame: " << out_path.string() << "\n";
}

int run_pipeline(PipelineRuntime& runtime, Pipeline& pipe, const AppConfig& cfg) {
  ProfileWindow profile;
  profile.enabled = cfg.profile;
  profile.interval = cfg.profile_interval;

  int processed = 0;
  int dropped_total = 0;
  while (cfg.frames <= 0 || processed < cfg.frames) {
    nt::Sample sample;
    nt::PullError pull_error;
    const double pull_start = time_ms();
    const auto status = runtime.run.pull("frame", 20000, sample, &pull_error);
    const double pull_end = time_ms();
    if (status == nt::PullStatus::Timeout) {
      std::cerr << "[warn] timed out waiting for decoded frame\n";
      continue;
    }
    if (status == nt::PullStatus::Closed)
      break;
    if (status != nt::PullStatus::Ok)
      throw std::runtime_error("failed to pull decoded frame: " + pull_error.message);

    const double model_start = time_ms();
    const cv::Mat bgr = tensor_bgr_from_decoded(frame_tensor_from_sample(sample));
    cv::Mat model_in;
    cv::resize(bgr, model_in, cv::Size(kImageW, kImageH), 0, 0, cv::INTER_LINEAR);
    FrameOut fo = pipe.run_one(model_in);
    const auto detections = select_dets(fo, cfg.score_threshold, cfg.max_detections);
    const double model_end = time_ms();

    const double metadata_start = time_ms();
    const int dropped = send_metadata(runtime, cfg, sample, fo, detections);
    const double metadata_end = time_ms();
    if (dropped > 0 && dropped_total == 0)
      std::cerr << "[warn] metadata byte budget exceeded, dropped " << dropped << " segments\n";
    dropped_total += dropped;

    ++processed;
    maybe_save_frame(cfg, processed, bgr, fo, detections, runtime.labels);
    profile.add(pull_end - pull_start, model_end - model_start, metadata_end - metadata_start,
               static_cast<int>(detections.size()), dropped);
  }

  profile.flush();
  std::cout << "processed=" << processed << " dropped_segments=" << dropped_total
            << " video_sender=" << cfg.insight_host << ":" << runtime.video_port << "\n";
  return processed;
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
    if (!cfg.save_dir.empty())
      fs::create_directories(cfg.save_dir);

    const SourceGeometry geometry = probe_source_geometry(cfg);
    // Load the two on-device models before starting the RTSP graph below: model load can take
    // several seconds (real MLA .elf loads), and the RTSP source starts decoding and queueing
    // frames the instant its Run is built. Building it first left that queue filling with no
    // consumer, and the run failed with a backpressure timeout by the time we reached the first
    // pull.
    Pipeline pipe(cfg.model_root, /*timeout_ms=*/30000);
    PipelineRuntime runtime = build_graph(cfg, geometry);

    const int processed = run_pipeline(runtime, pipe, cfg);

    pipe.close();
    runtime.run.close();
    return processed > 0 ? 0 : 3;
  } catch (const std::exception& ex) {
    std::cerr << "Error: " << ex.what() << "\n";
    return 2;
  }
}
