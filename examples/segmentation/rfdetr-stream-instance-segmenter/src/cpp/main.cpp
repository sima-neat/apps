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
// two-stage split -- INT8 backbone (MLA) and a BF16 transformer + seg-head
// (MLA) -- with a host-side top-k+gather hop between them. The backbone runs
// embedded in the same async graph as the RTSP decode and the (passthrough)
// video sender; a bridge thread does the top-k+gather and feeds the
// transformer, which runs as a separately-built Runner. This mirrors the
// object-detection RF-DETR example in this repo (examples/object-detection/
// rfdetr-object-detector), adapted for the segmentation model's extra mask
// output and third (now-removed) compiled top-k stage: the top-k .so this
// model shipped with was verified bit-exact against a plain stable
// argsort-by-score + gather on captured device tensors, so it is not needed.

#include "neat.h"
#include "neat/models.h"
#include "neat/node_groups.h"
#include "neat/nodes.h"
#include "support/runtime/config_utils.h"
#include "support/runtime/example_utils.h"
#include <nodes/groups/VideoSender.h>
#include <nodes/io/MetadataSender.h>

#include <nlohmann/json.hpp>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
namespace nt = simaai::neat;

using sima_examples::time_ms;

namespace {

// ── model constants (fixed by the compiled RF-DETR-Seg 432 backbone/transformer) ──
constexpr int kImageW = 432, kImageH = 432;
constexpr int kNumQueries = 200, kNumClasses = 91, kMaskHW = 108;
const std::array<float, 3> kCocoMean = {0.485f, 0.456f, 0.406f};
const std::array<float, 3> kCocoStd = {0.229f, 0.224f, 0.225f};

const std::vector<int64_t> kFeatureShape = {1, 36, 36, 256};
const std::vector<int64_t> kConcatShape = {1, 1296, 4};
const std::vector<int64_t> kGatherShape = {1, 200, 4};
const std::vector<int64_t> kLogitsShape = {1, 200, 91};
const std::vector<int64_t> kMasksDeviceShape = {108, 108, 200};

const char* kBackboneName = "rfdetr_seg_432_simplified_backbone_before_topk_base_mpk";
const char* kTransformerName = "rfdetr_seg_432_simplified_transformer_after_gather_base_mpk";

/// MetadataSender rejects a payload above 65507 bytes, and the rejection surfaces as an error the
/// application has to handle mid-stream. Half of that leaves room for the envelope and keeps the
/// datagram count low enough for Insight to reassemble within its 250 ms window.
constexpr std::size_t kMetadataByteBudget = 32768;

std::atomic<bool> g_stop{false};
void request_stop(int) { g_stop.store(true); }

int64_t numel(const std::vector<int64_t>& s) {
  int64_t n = 1;
  for (auto d : s)
    n *= d;
  return n;
}
inline float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }

std::vector<nt::Tensor> collect_tensors(const nt::Sample& s) {
  std::vector<nt::Tensor> out;
  if (s.kind == nt::SampleKind::Tensor) {
    if (s.tensor.has_value())
      out.push_back(*s.tensor);
  } else if (s.kind == nt::SampleKind::TensorSet) {
    out = s.tensors;
  }
  for (const auto& f : s.fields) {
    auto nested = collect_tensors(f);
    out.insert(out.end(), nested.begin(), nested.end());
  }
  return out;
}

std::vector<float> tensor_to_f32(const nt::Tensor& t) {
  if (t.dtype != nt::TensorDType::Float32)
    throw std::runtime_error("expected Float32 tensor");
  const std::vector<uint8_t> raw = t.copy_dense_bytes_tight();
  std::vector<float> out(raw.size() / sizeof(float));
  std::memcpy(out.data(), raw.data(), raw.size());
  return out;
}

int64_t identity_key(const nt::Sample& s) { return s.frame_id >= 0 ? s.frame_id : s.input_seq; }

void copy_identity(const nt::Sample& source, nt::Sample& target) {
  target.frame_id = source.frame_id;
  target.stream_id = source.stream_id;
  target.stream_label = source.stream_label;
  target.input_seq = source.input_seq;
  target.orig_input_seq = source.orig_input_seq;
  target.pts_ns = source.pts_ns;
  target.dts_ns = source.dts_ns;
  target.duration_ns = source.duration_ns;
  target.attributes = source.attributes;
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

// ── model options ────────────────────────────────────────────────────────
// The backbone consumes the decoder's NV12 output directly (fed by the graph, not the host), and
// EV74 resizes from the live stream resolution down to the compiled 432x432 input with a plain
// stretch (independent x/y scale, no letterbox padding) -- confirmed on real hardware to work at
// native stream resolution as well as at a pre-resized 432x432 input.
nt::Model::Options backbone_options(int stream_w, int stream_h) {
  nt::Model::Options opt;
  opt.preprocess.kind = nt::InputKind::Image;
  opt.preprocess.enable = nt::AutoFlag::On;
  opt.preprocess.color_convert.enable = nt::AutoFlag::On;
  opt.preprocess.color_convert.input_format = nt::PreprocessColorFormat::NV12;
  opt.preprocess.color_convert.output_format = nt::PreprocessColorFormat::RGB;
  opt.preprocess.input_max_width = stream_w;
  opt.preprocess.input_max_height = stream_h;
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
  opt.processcvu.post_run_target = "A65";
  return opt;
}

nt::Model::Options transformer_options() {
  nt::Model::Options opt;
  opt.preprocess.kind = nt::InputKind::Tensor;
  opt.preprocess.enable = nt::AutoFlag::Off;
  opt.processcvu.pre_run_target = "A65";
  opt.processcvu.post_run_target = "A65";
  return opt;
}

// ── host-side top-k + gather ────────────────────────────────────────────────
// Verified bit-exact against the model's compiled top-k .so on captured device tensors (stable
// argsort by score, descending, then gather the matching boxes) -- see the PR description for the
// comparison. Removing the compiled stage also removes the TVM runtime dependency entirely
// (and, in the Python port, the subprocess needed to keep TVM out of the same process as pyneat).
std::vector<float> stable_topk_gather(const std::vector<float>& reduce_scores,
                                      const std::vector<float>& concat_boxes) {
  if (reduce_scores.size() < static_cast<std::size_t>(kNumQueries) ||
      concat_boxes.size() != reduce_scores.size() * 4U)
    throw std::runtime_error("backbone score and box shapes do not match");
  std::vector<int> indices(reduce_scores.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::stable_sort(indices.begin(), indices.end(), [&](int a, int b) {
    return reduce_scores[static_cast<std::size_t>(a)] > reduce_scores[static_cast<std::size_t>(b)];
  });
  indices.resize(kNumQueries);
  std::vector<float> gathered(static_cast<std::size_t>(kNumQueries) * 4U);
  for (int out = 0; out < kNumQueries; ++out) {
    const auto src = static_cast<std::size_t>(indices[static_cast<std::size_t>(out)]) * 4U;
    std::copy_n(concat_boxes.begin() + static_cast<std::ptrdiff_t>(src), 4,
               gathered.begin() + static_cast<std::ptrdiff_t>(out * 4));
  }
  return gathered;
}

// Move-only: `masks_map` is a live read mapping of the transformer's own mask tensor.
struct FrameOut {
  std::vector<float> boxes;  // 200*4, normalized cx,cy,w,h in [0,1]
  std::vector<float> logits; // 200*91, raw (pre-sigmoid)

  // Raw device (108,108,200) mask logits, pre-sigmoid, read in place. Only the <=max_detections
  // kept queries are ever sampled (see mask_for_query), so materialising all 200 through
  // tensor_to_f32 -- a 9.3 MB allocate-and-copy, done twice, on every single frame -- was the
  // dominant term in the main loop's decode time, and that cost is what put the host behind the
  // stream. `masks_owned` is only populated on the non-contiguous fallback path.
  nt::Tensor masks_tensor;
  nt::Mapping masks_map;
  std::vector<float> masks_owned;
  const float* masks_dev = nullptr;
};

struct BackboneOutputs {
  std::vector<float> feature; // 36*36*256
  std::vector<float> reduce;  // 1296
  std::vector<float> concat;  // 1296*4
};

BackboneOutputs split_backbone(const nt::Sample& sample) {
  BackboneOutputs out;
  for (const auto& t : collect_tensors(sample)) {
    const auto& s = t.shape;
    const int64_t n = numel(s);
    if (s == kFeatureShape)
      out.feature = tensor_to_f32(t);
    else if (s == kConcatShape)
      out.concat = tensor_to_f32(t);
    else if (n == 1296)
      out.reduce = tensor_to_f32(t);
  }
  if (out.feature.empty() || out.reduce.empty() || out.concat.empty())
    throw std::runtime_error("backbone did not produce feature, reduce, and concat tensors");
  return out;
}

FrameOut split_transformer(const nt::Sample& sample) {
  FrameOut fo;
  for (const auto& t : collect_tensors(sample)) {
    const auto& s = t.shape;
    if (s == kGatherShape)
      fo.boxes = tensor_to_f32(t);
    else if (s == kLogitsShape)
      fo.logits = tensor_to_f32(t);
    else if (s == kMasksDeviceShape || numel(s) == kNumQueries * kMaskHW * kMaskHW) {
      if (t.dtype != nt::TensorDType::Float32)
        throw std::runtime_error("expected Float32 mask tensor");
      constexpr std::size_t kMaskBytes =
          static_cast<std::size_t>(kMaskHW) * kMaskHW * kNumQueries * sizeof(float);
      nt::Mapping mapped = t.is_dense() && t.is_contiguous() ? t.view_read() : nt::Mapping{};
      if (mapped.data != nullptr && mapped.size_bytes >= kMaskBytes) {
        fo.masks_tensor = t;
        fo.masks_map = std::move(mapped);
        fo.masks_dev = static_cast<const float*>(fo.masks_map.data);
      } else {
        // Strided or otherwise unmappable output: fall back to the packed copy.
        fo.masks_owned = tensor_to_f32(t);
        fo.masks_dev = fo.masks_owned.data();
      }
    }
  }
  if (fo.boxes.empty() || fo.logits.empty() || fo.masks_dev == nullptr)
    throw std::runtime_error("transformer did not produce box, class, and mask tensors");
  return fo;
}

// ── per-query select + stretch-space box/mask projection ───────────────────
// The compiled graph resizes with ResizeMode::Stretch (independent x/y scale, no letterbox
// padding), so, unlike the letterboxed examples in this category, mapping a normalized box or the
// 108x108 mask grid back to frame pixels is a plain per-axis scale.
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
cv::Mat mask_for_query(const float* masks_dev, int q) {
  cv::Mat m(kMaskHW, kMaskHW, CV_8UC1);
  for (int hh = 0; hh < kMaskHW; ++hh) {
    uint8_t* dst = m.ptr<uint8_t>(hh);
    const float* base = masks_dev + (hh * kMaskHW) * kNumQueries + q;
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

// ── RTSP source (encoded passthrough + separate decode) + Insight video sender ──
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
  sima_examples::require(geometry.width > 0 && geometry.height > 0 && geometry.fps > 0,
                         "failed to resolve RTSP width, height, and FPS; check source.url");
  return geometry;
}

cv::Mat tensor_bgr_from_decoded(const nt::Tensor& tensor) {
  cv::Mat bgr;
  std::string err;
  if (sima_examples::nv12_to_bgr(tensor, bgr, err))
    return bgr;
  return tensor.to_cv_mat_copy(nt::ImageSpec::PixelFormat::BGR);
}

struct ProfileWindow {
  bool enabled = false;
  int interval = 100;
  int frames = 0;
  int instances = 0;
  int dropped_segments = 0;
  double start_ms = 0.0;
  double pull_ms = 0.0;
  double decode_ms = 0.0;
  double metadata_ms = 0.0;

  void add(double pull, double decode, double metadata, int instance_count, int dropped) {
    if (!enabled)
      return;
    if (frames == 0)
      start_ms = time_ms();
    ++frames;
    instances += instance_count;
    dropped_segments += dropped;
    pull_ms += pull;
    decode_ms += decode;
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
              << " avg_pull_ms=" << pull_ms / n << " avg_decode_ms=" << decode_ms / n
              << " avg_metadata_ms=" << metadata_ms / n
              << " avg_instances=" << static_cast<double>(instances) / n
              << " dropped_segments=" << dropped_segments << "\n";
    frames = 0;
    instances = 0;
    dropped_segments = 0;
    start_ms = 0.0;
    pull_ms = 0.0;
    decode_ms = 0.0;
    metadata_ms = 0.0;
  }
};

int run(const AppConfig& cfg) {
  const SourceGeometry geometry = probe_source_geometry(cfg);
  const auto labels = load_labels(cfg.labels_path);
  const bool save_frames = !cfg.save_dir.empty();
  if (save_frames)
    fs::create_directories(cfg.save_dir);

  nt::Model backbone((fs::path(cfg.model_root) / kBackboneName).string(),
                     backbone_options(geometry.width, geometry.height));
  nt::Model transformer((fs::path(cfg.model_root) / kTransformerName).string(),
                        transformer_options());

  // ── RTSP source: keep the encoded bitstream for a true passthrough to Insight (no
  // decode-then-re-encode round trip), and decode once, separately, for the model. ──
  nt::nodes::groups::RtspEncodedInputOptions encoded_options;
  encoded_options.url = cfg.source_url;
  encoded_options.codec = nt::nodes::groups::RtspCodec::H264;
  encoded_options.latency_ms = cfg.latency_ms;
  encoded_options.tcp = cfg.tcp;
  encoded_options.source_fps = geometry.fps;
  encoded_options.fallback_h264_width = geometry.width;
  encoded_options.fallback_h264_height = geometry.height;
  auto source = nt::nodes::groups::RtspEncodedInput(encoded_options);

  nt::SimaDecodeOptions decode_options;
  decode_options.type = nt::SimaDecodeType::H264;
  decode_options.out_format = nt::FormatTag::NV12;
  decode_options.raw_output = true;
  decode_options.dec_width = geometry.width;
  decode_options.dec_height = geometry.height;
  decode_options.dec_fps = geometry.fps;
  nt::Graph decode("decoder");
  decode.add(nt::nodes::SimaDecode(decode_options));

  auto video_options =
      nt::nodes::groups::VideoSenderOptions::Passthrough(nt::nodes::groups::RtspCodec::H264);
  video_options.host = cfg.insight_host;
  video_options.video_port_base = cfg.video_port;
  video_options.channel = 0;
  video_options.async = true;
  auto video = nt::nodes::groups::VideoSender(video_options);

  // ── backbone runs embedded in the same async graph as decode/video, pipelined two
  // frames deep, instead of behind a manual push/pull call in the main loop. ──
  nt::Graph backbone_graph = backbone.graph();
  // Latest(), not EveryFrame(N): a public Output's own drop policy -- not
  // RunOptions::overflow_policy -- governs its terminal appsink, and decode/preproc/MLA/dequant
  // lower into one fused GStreamer segment ending at that appsink. A non-dropping sink therefore
  // back-pressures straight into neatdecoder, which cannot then recycle its zero-copy output
  // buffers; the pool starves, the decoder force-restarts, and the restart races the outstanding
  // loans (output_pool_exhausted). Dropping here bounds the stall at one frame instead.
  nt::Graph backbone_output("backbone_output");
  backbone_output.add(nt::nodes::Output("backbone", nt::OutputOptions::Latest()));

  nt::GraphLinkOptions link;
  link.policy = nt::GraphLinkPolicy::RealtimeLatestByStream;
  link.max_inflight_per_stream = 2;
  link.stream_id = "stream0";

  nt::Graph graph("rfdetr_seg");
  graph.connect(source, decode);
  graph.connect(source, video);
  graph.connect(decode, backbone_graph, link);
  graph.connect(backbone_graph, backbone_output);

  nt::Graph frame_output("frame_output");
  if (save_frames) {
    // Same reasoning as the backbone sink above: this branch hangs off the same decoder, so a
    // non-dropping sink here starves the decoder pool just as effectively.
    frame_output.add(nt::nodes::Output("frame", nt::OutputOptions::Latest()));
    graph.connect(decode, frame_output);
  }

  if (cfg.profile)
    std::cout << "Backend:\n" << graph.describe_backend() << "\n";

  nt::RunOptions transformer_run_options;
  transformer_run_options.preset = nt::RunPreset::Realtime;
  transformer_run_options.queue_depth = 2;
  // KeepLatest, not Block: a blocking push stalls the bridge thread, which is the only drainer of
  // the backbone sink, so main-loop jitter would propagate back into the decoder. Dropping a stale
  // transformer input degrades output FPS instead of killing the pipeline.
  transformer_run_options.overflow_policy = nt::OverflowPolicy::KeepLatest;
  transformer_run_options.output_memory = nt::OutputMemory::Owned;
  nt::Tensor fseed = nt::Tensor::from_vector(std::vector<float>(36 * 36 * 256, 0.f), {36, 36, 256},
                                             nt::TensorMemory::EV74);
  nt::Tensor gseed =
      nt::Tensor::from_vector(std::vector<float>(kNumQueries * 4, 0.f), {1, kNumQueries, 4},
                              nt::TensorMemory::EV74);
  nt::Model::Runner transformer_runner = transformer.build(
      nt::TensorList{fseed, gseed}, nt::Model::RouteOptions{}, transformer_run_options);

  nt::RunOptions source_options;
  source_options.preset = nt::RunPreset::Realtime;
  source_options.queue_depth = 3;
  source_options.overflow_policy = nt::OverflowPolicy::KeepLatest;
  source_options.output_memory = nt::OutputMemory::ZeroCopy;
  source_options.advanced.prepare_output_cpu_visible = true;
  nt::Run source_run = graph.build(source_options);

  nt::MetadataSenderOptions metadata_options;
  metadata_options.host = cfg.insight_host;
  metadata_options.channel = 0;
  metadata_options.metadata_port_base = cfg.metadata_port;
  std::string metadata_err;
  nt::MetadataSender metadata_sender(metadata_options, &metadata_err);
  sima_examples::require(metadata_sender.ok(), metadata_err);

  std::cout << "source=" << cfg.source_url << " stream=" << geometry.width << "x"
            << geometry.height << "@" << geometry.fps << " insight=" << cfg.insight_host
            << " video=" << video_options.video_port()
            << " metadata=" << metadata_sender.metadata_port() << " channel=0\n";

  // ── bridge thread: pulls the backbone's output, does the (now host-side) top-k +
  // gather, and feeds the transformer. Runs concurrently with the main loop pulling
  // transformer output below, so the three stages overlap instead of running one
  // frame fully to completion before the next starts. ──
  std::string bridge_error;
  std::mutex identity_mutex;
  std::map<int64_t, int64_t> source_pts;
  std::map<int64_t, cv::Mat> pending_frames;
  std::thread bridge([&] {
    try {
      while (!g_stop.load()) {
        nt::Sample transformer_sample;

        // Scoped tightly on purpose. `backbone_sample` is a ZERO-COPY view of a buffer from the
        // graph's last CVU stage (detessdequant), whose output pool holds only num-buffers=4.
        // Every instruction executed while this Sample is alive is an instruction that pool
        // buffer is unavailable, and exhausting it is what actually killed the pipeline:
        // detessdequant raises output_pool_exhausted, the decoder's downstream push then fails,
        // it force-restarts, and the restart races its own outstanding zero-copy loans
        // (gst_mini_object_unref refcount assertion). Everything downstream of this block needs
        // only host-side copies, so the loan is returned before the push below -- which can
        // block -- rather than being held across it.
        {
          nt::Sample backbone_sample;
          nt::PullError pull_error;
          const auto status = source_run.pull("backbone", 500, backbone_sample, &pull_error);
          if (status == nt::PullStatus::Timeout)
            continue;
          if (status == nt::PullStatus::Closed)
            break;
          if (status != nt::PullStatus::Ok)
            throw std::runtime_error("failed to pull backbone output: " + pull_error.message);

          const auto backbone_out = split_backbone(backbone_sample);
          const auto gathered = stable_topk_gather(backbone_out.reduce, backbone_out.concat);

          // Fresh CPU tensors, not the live device-backed backbone output: pushing the backbone's
          // own output tensor straight into the transformer's 2-input stage has been observed to
          // scramble that stage's packed multi-input buffer on this NEAT build.
          nt::Tensor feat_in =
              nt::Tensor::from_vector(backbone_out.feature, {36, 36, 256}, nt::TensorMemory::EV74);
          nt::Tensor gath_in = nt::Tensor::from_vector(gathered, {1, kNumQueries, 4},
                                                       nt::TensorMemory::EV74);
          transformer_sample.kind = nt::SampleKind::TensorSet;
          transformer_sample.tensors = {feat_in, gath_in};
          copy_identity(backbone_sample, transformer_sample);

          {
            std::lock_guard<std::mutex> lock(identity_mutex);
            source_pts[identity_key(backbone_sample)] = backbone_sample.pts_ns;
            if (source_pts.size() > 8U)
              source_pts.erase(source_pts.begin());
          }

          if (save_frames) {
            // Same contract on the decoder's own zero-copy output: copy into a cv::Mat and let
            // the Sample die immediately.
            nt::Sample frame_sample;
            nt::PullError frame_error;
            const auto frame_status = source_run.pull("frame", 0, frame_sample, &frame_error);
            if (frame_status == nt::PullStatus::Ok) {
              const auto tensors = collect_tensors(frame_sample);
              if (!tensors.empty()) {
                std::lock_guard<std::mutex> lock(identity_mutex);
                pending_frames[identity_key(frame_sample)] =
                    tensor_bgr_from_decoded(tensors.front());
                if (pending_frames.size() > 8U)
                  pending_frames.erase(pending_frames.begin());
              }
            }
          }
        }

        if (!transformer_runner.push(transformer_sample)) {
          if (!g_stop.load())
            throw std::runtime_error("transformer input closed");
          break;
        }
      }
    } catch (const std::exception& e) {
      if (!g_stop.load())
        bridge_error = e.what();
      g_stop.store(true);
    }
  });

  ProfileWindow profile;
  profile.enabled = cfg.profile;
  profile.interval = cfg.profile_interval;

  int processed = 0;
  int dropped_total = 0;
  try {
    while (!g_stop.load() && (cfg.frames <= 0 || processed < cfg.frames)) {
      const double pull_start = time_ms();
      nt::Sample sample = transformer_runner.pull(500);
      const double pull_end = time_ms();
      if (collect_tensors(sample).empty())
        continue;

      const double decode_start = time_ms();
      const auto fo = split_transformer(sample);
      const auto detections = select_dets(fo, cfg.score_threshold, cfg.max_detections);
      const double decode_end = time_ms();

      const double metadata_start = time_ms();
      const auto encoded = encode_segments(build_metadata_segments(
          fo, detections, labels, cv::Size(geometry.width, geometry.height), cfg.mask_threshold));
      int64_t pts_ns = sample.pts_ns;
      {
        std::lock_guard<std::mutex> lock(identity_mutex);
        const auto found = source_pts.find(identity_key(sample));
        if (found != source_pts.end()) {
          pts_ns = found->second;
          source_pts.erase(found);
        }
      }
      const int64_t ts_ms = pts_ns >= 0 ? pts_ns / 1'000'000 : -1;
      const std::string frame_id = sample.frame_id >= 0 ? std::to_string(sample.frame_id) : "";
      std::string send_err;
      if (!metadata_sender.send_metadata("segmentation", encoded.data_json, ts_ms, frame_id,
                                         &send_err))
        std::cerr << "[warn] insight metadata send failed: " << send_err << "\n";
      const double metadata_end = time_ms();
      if (encoded.dropped > 0 && dropped_total == 0)
        std::cerr << "[warn] metadata byte budget exceeded, dropped " << encoded.dropped
                  << " segments\n";
      dropped_total += encoded.dropped;

      ++processed;
      if (save_frames && cfg.save_every > 0 && processed % cfg.save_every == 0) {
        cv::Mat frame;
        {
          std::lock_guard<std::mutex> lock(identity_mutex);
          const auto found = pending_frames.find(identity_key(sample));
          if (found != pending_frames.end()) {
            frame = found->second;
            pending_frames.erase(found);
          }
        }
        if (!frame.empty()) {
          const cv::Mat annotated = overlay_segmentation(frame, fo, detections, labels, cfg);
          const auto out_path = cfg.save_dir / ("frame_" + std::to_string(processed) + ".jpg");
          if (!cv::imwrite(out_path.string(), annotated))
            std::cerr << "[warn] failed to write output frame: " << out_path.string() << "\n";
        }
      }
      profile.add(pull_end - pull_start, decode_end - decode_start, metadata_end - metadata_start,
                 static_cast<int>(detections.size()), encoded.dropped);
    }
  } catch (...) {
    g_stop.store(true);
    source_run.stop();
    transformer_runner.close();
    bridge.join();
    throw;
  }

  g_stop.store(true);
  source_run.stop();
  transformer_runner.close();
  bridge.join();
  if (!bridge_error.empty())
    throw std::runtime_error(bridge_error);

  profile.flush();
  std::cout << "processed=" << processed << " dropped_segments=" << dropped_total
            << " video_sender=" << cfg.insight_host << ":" << video_options.video_port() << "\n";
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
    std::signal(SIGINT, request_stop);
    std::signal(SIGTERM, request_stop);
    const int processed = run(cfg);
    return processed > 0 ? 0 : 3;
  } catch (const std::exception& ex) {
    std::cerr << "Error: " << ex.what() << "\n";
    return 2;
  }
}
