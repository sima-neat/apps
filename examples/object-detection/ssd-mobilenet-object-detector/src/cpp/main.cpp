/**
 * @example ssd-mobilenet-object-detector.cpp
 * SSD-MobileNetV2 (TF COCO) folder detection with host-side box decode.
 *
 * The compiled pack emits the 12 grouped convolution heads (6 localization + 6
 * classification, feature maps 19/10/5/3/2/1 -> 1917 priors). This example reassembles
 * them in anchor-major layout, decodes with the recovered ssd_anchor_generator priors
 * (FasterRcnnBoxCoder), applies SIGMOID scoring + per-class NMS, and draws detections.
 *
 * Model input is normalized to [-1, 1] via (pixel / 127.5) - 1, with a direct STRETCH
 * resize to 300x300 (fixed_shape_resizer), so boxes back-project with a per-axis scale.
 *
 * Usage: ssd-mobilenet-object-detector [--config <path>]
 */
#include "neat.h"
#include "support/runtime/example_utils.h"
#include "support/runtime/config_utils.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;
namespace neat = simaai::neat;

namespace {

constexpr int kModelSize = 300;
constexpr int kNumClasses = 91;  // index 0 = background, 1..90 = COCO ids.
constexpr int kDefaultTimeoutMs = 20000;
constexpr float kScaleY = 10.0f;
constexpr float kScaleX = 10.0f;
constexpr float kScaleH = 5.0f;
constexpr float kScaleW = 5.0f;

// Feature levels largest -> smallest, matching the head/anchor ordering.
struct Level {
  int feat;
  int anchors_per_cell;
};
constexpr std::array<Level, 6> kLevels = {
    Level{19, 3}, Level{10, 6}, Level{5, 6}, Level{3, 6}, Level{2, 6}, Level{1, 6}};

// Per-level SSD priors (w, h) in normalized image coordinates, largest -> smallest
// feature map. Anchor centers are (cell + 0.5) / feature_size. This reproduces the
// ssd_anchor_generator anchors of ssd_mobilenet_v2_coco_2018_03_29 -- the same per-level
// prior table baked into the on-device SSD decode kernel -- so no anchor asset is shipped.
struct PriorLevel {
  int feat;
  std::vector<std::array<float, 2>> priors;  // each entry is (w, h)
};

std::vector<PriorLevel> prior_levels() {
  return {
      PriorLevel{19, {{0.1f, 0.099999994f}, {0.2828427f, 0.14142135f}, {0.14142135f, 0.28284273f}}},
      PriorLevel{10, {{0.34999999f, 0.35000002f}, {0.49497473f, 0.2474874f}, {0.24748738f, 0.49497464f}, {0.60621786f, 0.20207258f}, {0.20206252f, 0.60624808f}, {0.41833004f, 0.41833001f}}},
      PriorLevel{5, {{0.5f, 0.49999997f}, {0.70710671f, 0.35355341f}, {0.35355338f, 0.70710677f}, {0.86602533f, 0.28867519f}, {0.28866071f, 0.86606878f}, {0.57008761f, 0.57008779f}}},
      PriorLevel{3, {{0.64999992f, 0.65000004f}, {0.91923869f, 0.45961937f}, {0.45961937f, 0.91923869f}, {1.1258328f, 0.37527764f}, {0.37525886f, 1.1258892f}, {0.72111022f, 0.72111011f}}},
      PriorLevel{2, {{0.79999995f, 0.79999995f}, {1.131371f, 0.56568551f}, {0.56568539f, 1.131371f}, {1.3856405f, 0.46188015f}, {0.46185714f, 1.3857099f}, {0.87177998f, 0.87177974f}}},
      PriorLevel{1, {{0.95000011f, 0.94999999f}, {1.343503f, 0.67175144f}, {0.67175162f, 1.343503f}, {1.6454482f, 0.54848289f}, {0.54845542f, 1.6455302f}, {0.97467947f, 0.97467953f}}},
  };
}

struct Config {
  std::string model = "assets/models/ssd_mobilenet_v2_heads_mpk.tar.gz";
  fs::path labels;
  fs::path input_dir;
  fs::path output_dir;
  float score_threshold = 0.30f;
  float nms_iou = 0.60f;
  int max_detections = 100;
  int timeout_ms = kDefaultTimeoutMs;
  int num_runs = 1;
  bool profile = false;
};

struct Detection {
  float x1 = 0.0f;
  float y1 = 0.0f;
  float x2 = 0.0f;
  float y2 = 0.0f;
  float score = 0.0f;
  int class_id = -1;
};

struct Stats {
  double mean = 0.0;
  double min = 0.0;
  double max = 0.0;
  double sum = 0.0;
};

Stats compute_stats(const std::vector<double>& values) {
  Stats s;
  if (values.empty()) {
    return s;
  }
  s.min = values.front();
  s.max = values.front();
  for (double x : values) {
    s.sum += x;
    s.min = std::min(s.min, x);
    s.max = std::max(s.max, x);
  }
  s.mean = s.sum / static_cast<double>(values.size());
  return s;
}

float sigmoid(float x) {
  return 1.0f / (1.0f + std::exp(-x));
}

bool is_image_file(const fs::path& path) {
  std::string ext = path.extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp";
}

std::vector<fs::path> image_paths_in_dir(const fs::path& input_dir) {
  std::vector<fs::path> images;
  for (const auto& entry : fs::directory_iterator(input_dir)) {
    if (entry.is_regular_file() && is_image_file(entry.path())) {
      images.push_back(entry.path());
    }
  }
  std::sort(images.begin(), images.end());
  return images;
}

// Resolve a labels/anchors asset: use the configured path if it exists, otherwise fall
// back to the copy shipped next to this example under src/common.
fs::path resolve_asset(const std::string& configured, const char* default_name) {
  if (!configured.empty() && fs::exists(configured)) {
    return configured;
  }
  const fs::path fallback =
      fs::path(SIMANEAT_APPS_EXAMPLE_SOURCE_DIR) / ".." / "common" / default_name;
  if (fs::exists(fallback)) {
    return fallback;
  }
  return configured;
}

Config load_config(const fs::path& path) {
  const auto raw = sima_examples::ScalarConfig::load(path);
  Config cfg;
  cfg.model = raw.string_or("model.path", "assets/models/ssd_mobilenet_v2_heads_mpk.tar.gz");
  cfg.labels = resolve_asset(raw.string_or("model.labels", ""), "coco_labels.txt");
  cfg.input_dir = raw.string_or("io.input_dir", "assets/test_images");
  cfg.output_dir = raw.string_or("io.output_dir", "sandbox/ssd-mobilenet-object-detector");
  cfg.score_threshold = static_cast<float>(raw.double_or("decode.score_threshold", 0.30));
  cfg.nms_iou = static_cast<float>(raw.double_or("decode.nms_iou", 0.60));
  cfg.max_detections = raw.int_or("decode.max_detections", 100);
  cfg.timeout_ms = raw.int_or("runtime.timeout_ms", kDefaultTimeoutMs);
  cfg.num_runs = raw.int_or("runtime.num_runs", 1);
  cfg.profile = raw.bool_or("runtime.profile", false);
  if (cfg.score_threshold < 0.0f || cfg.score_threshold > 1.0f) {
    throw std::runtime_error("decode.score_threshold must be in [0.0, 1.0]");
  }
  if (cfg.num_runs < 1) {
    throw std::runtime_error("runtime.num_runs must be >= 1");
  }
  if (cfg.timeout_ms <= 0) {
    throw std::runtime_error("runtime.timeout_ms must be > 0");
  }
  return cfg;
}

Config parse_config(int argc, char** argv) {
  fs::path config_path = sima_examples::default_config_path(SIMANEAT_APPS_EXAMPLE_SOURCE_DIR);
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--config") {
      if (i + 1 >= argc) {
        throw std::runtime_error("--config requires a path");
      }
      config_path = argv[++i];
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: " << argv[0] << " [--config <path>]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  return load_config(config_path);
}

std::vector<std::string> load_labels(const fs::path& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("failed to open labels file: " + path.string());
  }
  std::vector<std::string> labels;
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    labels.push_back(line);
  }
  if (labels.empty()) {
    throw std::runtime_error("labels file is empty: " + path.string());
  }
  return labels;
}

// Build the flat 1917x4 SSD priors (ycenter, xcenter, h, w) from the prior table.
std::vector<float> generate_anchors() {
  std::vector<float> anchors;
  for (const PriorLevel& level : prior_levels()) {
    const int feat = level.feat;
    for (int y = 0; y < feat; ++y) {
      const float cy = (static_cast<float>(y) + 0.5f) / static_cast<float>(feat);
      for (int x = 0; x < feat; ++x) {
        const float cx = (static_cast<float>(x) + 0.5f) / static_cast<float>(feat);
        for (const std::array<float, 2>& wh : level.priors) {
          anchors.push_back(cy);      // ycenter
          anchors.push_back(cx);      // xcenter
          anchors.push_back(wh[1]);   // h
          anchors.push_back(wh[0]);   // w
        }
      }
    }
  }
  return anchors;
}

std::string class_name(const std::vector<std::string>& labels, int class_id) {
  if (class_id >= 0 && static_cast<size_t>(class_id) < labels.size()) {
    const std::string& name = labels[static_cast<size_t>(class_id)];
    if (!name.empty() && name != "N/A") {
      return name;
    }
  }
  return "class_" + std::to_string(class_id);
}

cv::Scalar class_color(int class_id) {
  static const std::array<cv::Scalar, 8> kColors = {
      cv::Scalar(0, 255, 0),   cv::Scalar(255, 0, 0),   cv::Scalar(0, 0, 255),
      cv::Scalar(255, 255, 0), cv::Scalar(255, 0, 255), cv::Scalar(0, 255, 255),
      cv::Scalar(128, 255, 0), cv::Scalar(255, 128, 0),
  };
  const size_t idx = static_cast<size_t>(class_id >= 0 ? class_id : -class_id) % kColors.size();
  return kColors[idx];
}

cv::Mat preprocess_to_tensor_input(const cv::Mat& bgr_u8) {
  if (bgr_u8.empty()) {
    throw std::runtime_error("preprocess_to_tensor_input: empty image");
  }
  cv::Mat resized;
  cv::resize(bgr_u8, resized, cv::Size(kModelSize, kModelSize), 0.0, 0.0, cv::INTER_LINEAR);
  cv::Mat rgb;
  cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
  cv::Mat normalized;
  // (pixel / 127.5) - 1.0
  rgb.convertTo(normalized, CV_32FC3, 1.0 / 127.5, -1.0);
  return normalized;
}

neat::Tensor tensor_from_hwc_f32(const cv::Mat& hwc_f32) {
  if (hwc_f32.empty() || hwc_f32.type() != CV_32FC3) {
    throw std::runtime_error("tensor_from_hwc_f32: expected non-empty CV_32FC3");
  }
  const int h = hwc_f32.rows;
  const int w = hwc_f32.cols;
  const int c = 3;
  const size_t elems = static_cast<size_t>(h) * static_cast<size_t>(w) * static_cast<size_t>(c);
  std::vector<float> data(elems);
  std::memcpy(data.data(), hwc_f32.ptr<float>(), elems * sizeof(float));

  neat::Tensor t = neat::Tensor::from_vector(data, {h, w, c}, neat::TensorMemory::EV74);
  t.layout = neat::TensorLayout::HWC;
  t.strides_bytes = {static_cast<int64_t>(w * c * sizeof(float)),
                     static_cast<int64_t>(c * sizeof(float)), static_cast<int64_t>(sizeof(float))};
  return t;
}

std::vector<neat::Tensor> collect_tensors(const neat::Sample& sample) {
  if (sample.kind == neat::SampleKind::Tensor) {
    if (!sample.tensor.has_value()) {
      throw std::runtime_error("tensor sample missing payload");
    }
    return {*sample.tensor};
  }
  if (sample.kind == neat::SampleKind::TensorSet) {
    return sample.tensors;
  }
  if (sample.kind == neat::SampleKind::Bundle) {
    std::vector<neat::Tensor> out;
    for (const auto& field : sample.fields) {
      auto child = collect_tensors(field);
      out.insert(out.end(), child.begin(), child.end());
    }
    return out;
  }
  throw std::runtime_error("unexpected sample kind");
}

std::vector<float> tensor_to_f32(const neat::Tensor& t) {
  if (t.dtype != neat::TensorDType::Float32) {
    throw std::runtime_error("expected Float32 tensor");
  }
  const std::vector<uint8_t> raw = t.copy_dense_bytes_tight();
  if (raw.size() % sizeof(float) != 0) {
    throw std::runtime_error("tensor raw byte size is not float-aligned");
  }
  std::vector<float> out(raw.size() / sizeof(float));
  std::memcpy(out.data(), raw.data(), raw.size());
  return out;
}

// A single decoded head, resolved to logical (feature map size, channels, layout).
struct HeadView {
  int feat = 0;
  int channels = 0;
  bool nhwc = true;  // true: data laid out [H,W,C]; false: [C,H,W].
  std::vector<float> data;

  float at(int y, int x, int ch) const {
    if (nhwc) {
      return data[(static_cast<size_t>(y) * feat + x) * channels + ch];
    }
    return data[(static_cast<size_t>(ch) * feat + y) * feat + x];
  }
};

HeadView make_head_view(const neat::Tensor& t) {
  std::vector<int64_t> dims = t.shape;
  if (dims.size() == 4 && dims.front() == 1) {
    dims.erase(dims.begin());
  }
  if (dims.size() != 3) {
    throw std::runtime_error("expected a rank-3 head after squeezing batch");
  }
  const int d0 = static_cast<int>(dims[0]);
  const int d1 = static_cast<int>(dims[1]);
  const int d2 = static_cast<int>(dims[2]);

  HeadView view;
  view.data = tensor_to_f32(t);
  if (d0 == d1 && d0 != d2) {  // HWC
    view.nhwc = true;
    view.feat = d0;
    view.channels = d2;
  } else if (d1 == d2 && d1 != d0) {  // CHW
    view.nhwc = false;
    view.feat = d1;
    view.channels = d0;
  } else {
    throw std::runtime_error("cannot infer head layout from shape");
  }
  return view;
}

// Reassemble the 12 heads into anchor-major encodings [N*4] and class logits [N*91].
void reassemble_heads(const std::vector<neat::Tensor>& tensors, std::vector<float>& encodings,
                      std::vector<float>& logits, int num_priors) {
  std::unordered_map<int, std::pair<int, int>> level_of;  // feat -> (anchors_per_cell, offset)
  int off = 0;
  for (const Level& lvl : kLevels) {
    level_of[lvl.feat] = {lvl.anchors_per_cell, off};
    off += lvl.feat * lvl.feat * lvl.anchors_per_cell;
  }
  encodings.assign(static_cast<size_t>(num_priors) * 4, 0.0f);
  logits.assign(static_cast<size_t>(num_priors) * kNumClasses, 0.0f);

  int loc_heads = 0;
  int conf_heads = 0;
  for (const neat::Tensor& t : tensors) {
    const HeadView head = make_head_view(t);
    auto it = level_of.find(head.feat);
    if (it == level_of.end()) {
      throw std::runtime_error("unexpected head feature map size");
    }
    const int anchors_per_cell = it->second.first;
    const int base = it->second.second;

    if (head.channels % kNumClasses == 0) {  // classification head
      if (head.channels / kNumClasses != anchors_per_cell) {
        throw std::runtime_error("conf head channel/anchor mismatch");
      }
      for (int y = 0; y < head.feat; ++y) {
        for (int x = 0; x < head.feat; ++x) {
          for (int a = 0; a < anchors_per_cell; ++a) {
            const int idx = base + (y * head.feat + x) * anchors_per_cell + a;
            for (int c = 0; c < kNumClasses; ++c) {
              logits[static_cast<size_t>(idx) * kNumClasses + c] = head.at(y, x, a * kNumClasses + c);
            }
          }
        }
      }
      ++conf_heads;
    } else if (head.channels % 4 == 0) {  // localization head
      if (head.channels / 4 != anchors_per_cell) {
        throw std::runtime_error("loc head channel/anchor mismatch");
      }
      for (int y = 0; y < head.feat; ++y) {
        for (int x = 0; x < head.feat; ++x) {
          for (int a = 0; a < anchors_per_cell; ++a) {
            const int idx = base + (y * head.feat + x) * anchors_per_cell + a;
            for (int k = 0; k < 4; ++k) {
              encodings[static_cast<size_t>(idx) * 4 + k] = head.at(y, x, a * 4 + k);
            }
          }
        }
      }
      ++loc_heads;
    } else {
      throw std::runtime_error("unrecognized head channel depth");
    }
  }
  if (loc_heads != static_cast<int>(kLevels.size()) ||
      conf_heads != static_cast<int>(kLevels.size())) {
    throw std::runtime_error("expected 6 localization and 6 classification heads");
  }
}

// NMS over corner boxes (ymin,xmin,ymax,xmax); returns kept indices into `order` inputs.
std::vector<int> nms(const std::vector<std::array<float, 4>>& boxes,
                     const std::vector<float>& scores, float iou_thr) {
  std::vector<int> order(boxes.size());
  for (size_t i = 0; i < order.size(); ++i) {
    order[i] = static_cast<int>(i);
  }
  std::sort(order.begin(), order.end(),
            [&](int a, int b) { return scores[a] > scores[b]; });

  std::vector<char> removed(boxes.size(), 0);
  std::vector<int> keep;
  for (size_t oi = 0; oi < order.size(); ++oi) {
    const int i = order[oi];
    if (removed[i]) {
      continue;
    }
    keep.push_back(i);
    const auto& bi = boxes[i];
    const float ai =
        std::max(0.0f, bi[2] - bi[0]) * std::max(0.0f, bi[3] - bi[1]);
    for (size_t oj = oi + 1; oj < order.size(); ++oj) {
      const int j = order[oj];
      if (removed[j]) {
        continue;
      }
      const auto& bj = boxes[j];
      const float yy1 = std::max(bi[0], bj[0]);
      const float xx1 = std::max(bi[1], bj[1]);
      const float yy2 = std::min(bi[2], bj[2]);
      const float xx2 = std::min(bi[3], bj[3]);
      const float inter = std::max(0.0f, yy2 - yy1) * std::max(0.0f, xx2 - xx1);
      const float aj =
          std::max(0.0f, bj[2] - bj[0]) * std::max(0.0f, bj[3] - bj[1]);
      const float iou = inter / (ai + aj - inter + 1e-9f);
      if (iou > iou_thr) {
        removed[j] = 1;
      }
    }
  }
  return keep;
}

std::vector<Detection> decode_ssd_outputs(const std::vector<float>& encodings,
                                          const std::vector<float>& logits,
                                          const std::vector<float>& anchors, int num_priors,
                                          int orig_w, int orig_h, const Config& cfg) {
  // Decode all priors to normalized corner boxes once.
  std::vector<std::array<float, 4>> boxes(static_cast<size_t>(num_priors));
  for (int j = 0; j < num_priors; ++j) {
    const float ty = encodings[static_cast<size_t>(j) * 4 + 0];
    const float tx = encodings[static_cast<size_t>(j) * 4 + 1];
    const float th = encodings[static_cast<size_t>(j) * 4 + 2];
    const float tw = encodings[static_cast<size_t>(j) * 4 + 3];
    const float ay = anchors[static_cast<size_t>(j) * 4 + 0];
    const float ax = anchors[static_cast<size_t>(j) * 4 + 1];
    const float ah = anchors[static_cast<size_t>(j) * 4 + 2];
    const float aw = anchors[static_cast<size_t>(j) * 4 + 3];
    const float yc = ty / kScaleY * ah + ay;
    const float xc = tx / kScaleX * aw + ax;
    const float h = std::exp(th / kScaleH) * ah;
    const float w = std::exp(tw / kScaleW) * aw;
    boxes[j] = {yc - h / 2.0f, xc - w / 2.0f, yc + h / 2.0f, xc + w / 2.0f};
  }

  std::vector<Detection> dets;
  const int foreground = kNumClasses - 1;  // drop background
  for (int c = 0; c < foreground; ++c) {
    std::vector<std::array<float, 4>> cand_boxes;
    std::vector<float> cand_scores;
    std::vector<int> cand_priors;
    for (int j = 0; j < num_priors; ++j) {
      const float score = sigmoid(logits[static_cast<size_t>(j) * kNumClasses + (c + 1)]);
      if (score >= cfg.score_threshold) {
        cand_boxes.push_back(boxes[j]);
        cand_scores.push_back(score);
        cand_priors.push_back(j);
      }
    }
    if (cand_boxes.empty()) {
      continue;
    }
    const std::vector<int> keep = nms(cand_boxes, cand_scores, cfg.nms_iou);
    for (int k : keep) {
      const auto& b = cand_boxes[static_cast<size_t>(k)];
      float x1 = std::clamp(b[1] * orig_w, 0.0f, static_cast<float>(orig_w));
      float y1 = std::clamp(b[0] * orig_h, 0.0f, static_cast<float>(orig_h));
      float x2 = std::clamp(b[3] * orig_w, 0.0f, static_cast<float>(orig_w));
      float y2 = std::clamp(b[2] * orig_h, 0.0f, static_cast<float>(orig_h));
      if (x2 <= x1 || y2 <= y1) {
        continue;
      }
      dets.push_back(Detection{x1, y1, x2, y2, cand_scores[static_cast<size_t>(k)], c + 1});
    }
  }

  std::sort(dets.begin(), dets.end(),
            [](const Detection& a, const Detection& b) { return a.score > b.score; });
  if (cfg.max_detections > 0 && static_cast<int>(dets.size()) > cfg.max_detections) {
    dets.resize(static_cast<size_t>(cfg.max_detections));
  }
  return dets;
}

void draw_detections(cv::Mat& bgr, const std::vector<Detection>& dets,
                     const std::vector<std::string>& labels) {
  for (const Detection& d : dets) {
    const int x1 = std::max(0, std::min(bgr.cols - 1, static_cast<int>(std::lround(d.x1))));
    const int y1 = std::max(0, std::min(bgr.rows - 1, static_cast<int>(std::lround(d.y1))));
    const int x2 = std::max(0, std::min(bgr.cols - 1, static_cast<int>(std::lround(d.x2))));
    const int y2 = std::max(0, std::min(bgr.rows - 1, static_cast<int>(std::lround(d.y2))));
    if (x2 <= x1 || y2 <= y1) {
      continue;
    }
    const cv::Scalar color = class_color(d.class_id);
    const std::string text = class_name(labels, d.class_id) + " " + cv::format("%.2f", d.score);
    cv::rectangle(bgr, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);
    cv::putText(bgr, text, cv::Point(x1, std::max(0, y1 - 4)), cv::FONT_HERSHEY_SIMPLEX, 0.5, color,
                2, cv::LINE_AA);
  }
}

int num_priors_total() {
  int total = 0;
  for (const Level& lvl : kLevels) {
    total += lvl.feat * lvl.feat * lvl.anchors_per_cell;
  }
  return total;
}

} // namespace

int main(int argc, char** argv) {
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);

  try {
    const Config cfg = parse_config(argc, argv);
    if (!fs::is_directory(cfg.input_dir)) {
      throw std::runtime_error("input_dir does not exist: " + cfg.input_dir.string());
    }
    if (!fs::exists(cfg.model)) {
      throw std::runtime_error("model does not exist: " + cfg.model);
    }

    const std::vector<std::string> labels = load_labels(cfg.labels);
    const std::vector<float> anchors = generate_anchors();
    const int num_priors = num_priors_total();
    if (static_cast<int>(anchors.size()) != num_priors * 4) {
      throw std::runtime_error("generated anchors count does not match expected priors");
    }

    const std::vector<fs::path> image_paths = image_paths_in_dir(cfg.input_dir);
    if (image_paths.empty()) {
      throw std::runtime_error("no images found in: " + cfg.input_dir.string());
    }

    neat::Model::Options model_opt;
    model_opt.preprocess.kind = neat::InputKind::Tensor;
    model_opt.preprocess.input_max_width = kModelSize;
    model_opt.preprocess.input_max_height = kModelSize;
    model_opt.preprocess.input_max_depth = 3;

    neat::Model model(cfg.model, model_opt);

    neat::Graph graph;
    graph.add(neat::nodes::Input(model.input_appsrc_options(true)));
    graph.add(neat::nodes::QuantTess(neat::QuantTessOptions(model)));
    graph.add(neat::nodes::groups::MLA(model));
    graph.add(neat::nodes::DetessDequant(neat::DetessDequantOptions(model)));
    graph.add(neat::nodes::Output());

    cv::Mat dummy(kModelSize, kModelSize, CV_32FC3, cv::Scalar(0.0f, 0.0f, 0.0f));
    auto run = graph.build(neat::TensorList{tensor_from_hwc_f32(dummy)});

    if (cfg.profile) {
      const fs::path& image_path = image_paths.front();
      cv::Mat bgr_u8 = cv::imread(image_path.string(), cv::IMREAD_COLOR);
      if (bgr_u8.empty()) {
        throw std::runtime_error("failed to read image: " + image_path.string());
      }
      cv::Mat input_f32 = preprocess_to_tensor_input(bgr_u8);
      neat::Tensor input_t = tensor_from_hwc_f32(input_f32);

      const int runs = std::max(1, cfg.num_runs);
      std::vector<double> graph_times;
      std::vector<double> post_times;
      std::vector<Detection> last_dets;
      std::vector<float> encodings;
      std::vector<float> logits;

      for (int i = 0; i < runs; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        if (!run.push(neat::TensorList{input_t})) {
          throw std::runtime_error("run.push failed during profiling");
        }
        auto out_sample = run.pull(cfg.timeout_ms);
        const auto t1 = std::chrono::steady_clock::now();
        if (!out_sample.has_value()) {
          throw std::runtime_error("run.pull returned no sample during profiling");
        }
        const std::vector<neat::Tensor> tensors = collect_tensors(*out_sample);
        reassemble_heads(tensors, encodings, logits, num_priors);
        last_dets = decode_ssd_outputs(encodings, logits, anchors, num_priors, bgr_u8.cols,
                                       bgr_u8.rows, cfg);
        const auto t2 = std::chrono::steady_clock::now();
        graph_times.push_back(std::chrono::duration<double>(t1 - t0).count());
        post_times.push_back(std::chrono::duration<double>(t2 - t1).count());
      }

      const Stats graph_stats = compute_stats(graph_times);
      const Stats post = compute_stats(post_times);
      const double runs_d = static_cast<double>(graph_times.size());
      std::cout << "Profiling over " << graph_times.size() << " runs (image='"
                << image_path.string() << "'):\n";
      std::cout << "  Graph run (push+pull): mean=" << graph_stats.mean
                << "s, min=" << graph_stats.min << "s, max=" << graph_stats.max
                << "s, FPS=" << (runs_d / graph_stats.sum) << "\n";
      std::cout << "  Postprocessing (decode+NMS): mean=" << post.mean << "s, min=" << post.min
                << "s, max=" << post.max << "s, FPS=" << (runs_d / post.sum) << "\n";
      std::cout << "Last run detections: " << last_dets.size() << "\n";
      for (size_t i = 0; i < std::min<size_t>(last_dets.size(), 20); ++i) {
        const auto& d = last_dets[i];
        std::cout << "  [" << i << "] class=" << class_name(labels, d.class_id) << "("
                  << d.class_id << ") score=" << d.score << " box=[" << d.x1 << "," << d.y1 << ","
                  << d.x2 << "," << d.y2 << "]\n";
      }
      run.close();
      return 0;
    }

    fs::create_directories(cfg.output_dir);
    std::vector<float> encodings;
    std::vector<float> logits;
    int processed = 0;
    for (const fs::path& image_path : image_paths) {
      cv::Mat bgr_u8 = cv::imread(image_path.string(), cv::IMREAD_COLOR);
      if (bgr_u8.empty()) {
        throw std::runtime_error("failed to read image: " + image_path.string());
      }
      cv::Mat input_f32 = preprocess_to_tensor_input(bgr_u8);
      neat::Tensor input_t = tensor_from_hwc_f32(input_f32);

      if (!run.push(neat::TensorList{input_t})) {
        throw std::runtime_error("run.push failed");
      }
      auto out_sample = run.pull(cfg.timeout_ms);
      if (!out_sample.has_value()) {
        throw std::runtime_error("run.pull returned no sample");
      }

      const std::vector<neat::Tensor> tensors = collect_tensors(*out_sample);
      reassemble_heads(tensors, encodings, logits, num_priors);
      const std::vector<Detection> dets = decode_ssd_outputs(
          encodings, logits, anchors, num_priors, bgr_u8.cols, bgr_u8.rows, cfg);

      cv::Mat overlay = bgr_u8.clone();
      draw_detections(overlay, dets, labels);
      const fs::path output_path = cfg.output_dir / (image_path.stem().string() + "_ssd.png");
      if (!cv::imwrite(output_path.string(), overlay)) {
        throw std::runtime_error("failed to write: " + output_path.string());
      }
      ++processed;
      std::cout << "[" << processed << "/" << image_paths.size() << "] "
                << image_path.filename().string() << ": " << dets.size() << " detections -> "
                << output_path.filename().string() << "\n";
    }
    std::cout << "Done: " << processed << " images processed\n";

    run.close();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 2;
  }
}
