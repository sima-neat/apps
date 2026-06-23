/**
 * @example faster-rcnn-object-detector.cpp
 * Faster R-CNN folder object detection using two compiled NEAT model packages.
 *
 * Usage: faster-rcnn-object-detector [--config <path>]
 */
#include "neat.h"
#include "neat/models.h"
#include "neat/nodes.h"
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
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr int kInferW = 640;
constexpr int kInferH = 640;
constexpr int kDefaultTimeoutMs = 20000;
constexpr int kRpnPreNmsTopK = 1000;
constexpr int kRpnPostNmsTopK = 1000;
constexpr int kRoiNumClasses = 80;
constexpr int kMaxDetections = 100;
constexpr float kRpnNmsThresh = 0.7f;
constexpr float kRoiNmsThresh = 0.5f;
constexpr int kRoiOutputSize = 7;
constexpr int kFpnChannels = 256;
constexpr int kHeadInputDepth = kFpnChannels * kRoiOutputSize * kRoiOutputSize;

constexpr std::array<int, 5> kAnchorSizes = {32, 64, 128, 256, 512};
constexpr std::array<float, 3> kAnchorAspectRatios = {0.5f, 1.0f, 2.0f};
constexpr std::array<int, 5> kFpnStrides = {4, 8, 16, 32, 64};
constexpr std::array<std::pair<int, int>, 5> kFpnLevelHw = {{{160, 160}, {80, 80}, {40, 40}, {20, 20}, {10, 10}}};
constexpr std::array<float, 4> kRpnWeights = {1.0f, 1.0f, 1.0f, 1.0f};
constexpr std::array<float, 4> kRoiWeights = {10.0f, 10.0f, 5.0f, 5.0f};
constexpr std::array<float, 3> kPixelMean = {103.530f, 116.280f, 123.675f};

constexpr std::array<std::string_view, 80> kCocoClasses = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
    "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
};

struct Box {
  float x1 = 0.0f;
  float y1 = 0.0f;
  float x2 = 0.0f;
  float y2 = 0.0f;
};

struct Detection {
  Box box;
  float score = 0.0f;
  int class_id = -1;
};

struct Config {
  fs::path backbone_rpn_model = "assets/models/backbone_rpn_head_640_640_mpk.tar.gz";
  fs::path head_predictor_model = "assets/models/box_head_predictor_640_640_mpk.tar.gz";
  fs::path input_dir = "assets/test_images";
  fs::path output_dir = "sandbox/faster-rcnn-object-detector";
  float conf = 0.5f;
  int max_draw = 50;
  bool profile = false;
  int num_runs = 100;
  int timeout_ms = kDefaultTimeoutMs;
};

struct TensorData {
  int n = 1;
  int h = 0;
  int w = 0;
  int c = 0;
  std::vector<int64_t> shape;
  std::vector<float> data;

  float at(int ni, int yi, int xi, int ci) const {
    const size_t idx = (((static_cast<size_t>(ni) * h + yi) * w + xi) * c + ci);
    return data[idx];
  }
};

struct BackboneOutputs {
  std::array<TensorData, 4> features;
  TensorData logits_concat;
  TensorData deltas_concat;
};

struct HeadOutputs {
  TensorData scores;
  TensorData deltas;
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

Config load_config(const fs::path& path) {
  const auto raw = sima_examples::ScalarConfig::load(path);
  Config cfg;
  cfg.backbone_rpn_model = raw.string_or("models.backbone_rpn.path", cfg.backbone_rpn_model.string());
  cfg.head_predictor_model = raw.string_or("models.head_predictor.path", cfg.head_predictor_model.string());
  cfg.input_dir = raw.string_or("io.input_dir", cfg.input_dir.string());
  cfg.output_dir = raw.string_or("io.output_dir", cfg.output_dir.string());
  cfg.conf = static_cast<float>(raw.double_or("decode.confidence_threshold", 0.5));
  cfg.max_draw = raw.int_or("decode.max_draw", 50);
  cfg.profile = raw.bool_or("runtime.profile", false);
  cfg.num_runs = raw.int_or("runtime.num_runs", 100);
  cfg.timeout_ms = raw.int_or("runtime.timeout_ms", kDefaultTimeoutMs);
  if (cfg.conf < 0.0f || cfg.conf > 1.0f) {
    throw std::runtime_error("decode.confidence_threshold must be in [0.0, 1.0]");
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

simaai::neat::Tensor tensor_from_hwc_f32(const std::vector<float>& data, int h, int w, int c) {
  if (h <= 0 || w <= 0 || c <= 0) {
    throw std::runtime_error("tensor_from_hwc_f32: invalid shape");
  }
  const size_t expected = static_cast<size_t>(h) * static_cast<size_t>(w) * static_cast<size_t>(c);
  if (data.size() != expected) {
    throw std::runtime_error("tensor_from_hwc_f32: data size does not match shape");
  }
  simaai::neat::Tensor t =
      simaai::neat::Tensor::from_vector(data, {h, w, c}, simaai::neat::TensorMemory::EV74);
  t.layout = simaai::neat::TensorLayout::HWC;
  t.strides_bytes = {static_cast<int64_t>(w * c * sizeof(float)),
                     static_cast<int64_t>(c * sizeof(float)), static_cast<int64_t>(sizeof(float))};
  return t;
}

std::vector<float> preprocess_image(const cv::Mat& bgr_u8) {
  if (bgr_u8.empty()) {
    throw std::runtime_error("preprocess_image: empty image");
  }
  cv::Mat resized;
  cv::resize(bgr_u8, resized, cv::Size(kInferW, kInferH), 0.0, 0.0, cv::INTER_LINEAR);
  cv::Mat f32;
  resized.convertTo(f32, CV_32FC3);

  std::vector<float> out(static_cast<size_t>(kInferH) * kInferW * 3U);
  for (int y = 0; y < kInferH; ++y) {
    const cv::Vec3f* row = f32.ptr<cv::Vec3f>(y);
    for (int x = 0; x < kInferW; ++x) {
      const size_t base = (static_cast<size_t>(y) * kInferW + x) * 3U;
      out[base + 0] = row[x][0] - kPixelMean[0];
      out[base + 1] = row[x][1] - kPixelMean[1];
      out[base + 2] = row[x][2] - kPixelMean[2];
    }
  }
  return out;
}

std::vector<simaai::neat::Tensor> collect_tensors(const simaai::neat::Sample& sample) {
  if (sample.kind == simaai::neat::SampleKind::Tensor) {
    if (!sample.tensor.has_value()) {
      throw std::runtime_error("tensor sample missing payload");
    }
    return {*sample.tensor};
  }
  if (sample.kind == simaai::neat::SampleKind::TensorSet) {
    return sample.tensors;
  }
  if (sample.kind == simaai::neat::SampleKind::Bundle) {
    std::vector<simaai::neat::Tensor> out;
    for (const auto& field : sample.fields) {
      auto child = collect_tensors(field);
      out.insert(out.end(), child.begin(), child.end());
    }
    return out;
  }
  throw std::runtime_error("unexpected sample kind");
}

TensorData tensor_to_data(const simaai::neat::Tensor& t, const std::string& name) {
  if (t.dtype != simaai::neat::TensorDType::Float32) {
    throw std::runtime_error(name + ": expected Float32 tensor");
  }
  std::vector<uint8_t> raw = t.copy_dense_bytes_tight();
  if (raw.size() % sizeof(float) != 0) {
    throw std::runtime_error(name + ": dense byte size is not float-aligned");
  }
  TensorData out;
  out.shape = t.shape;
  if (out.shape.size() == 3U) {
    out.n = 1;
    out.h = static_cast<int>(out.shape[0]);
    out.w = static_cast<int>(out.shape[1]);
    out.c = static_cast<int>(out.shape[2]);
  } else if (out.shape.size() == 4U) {
    out.n = static_cast<int>(out.shape[0]);
    out.h = static_cast<int>(out.shape[1]);
    out.w = static_cast<int>(out.shape[2]);
    out.c = static_cast<int>(out.shape[3]);
  } else {
    throw std::runtime_error(name + ": expected rank-3 or rank-4 NHWC tensor");
  }
  const size_t elems = raw.size() / sizeof(float);
  const size_t expected = static_cast<size_t>(out.n) * out.h * out.w * out.c;
  if (elems != expected) {
    throw std::runtime_error(name + ": dense size does not match shape");
  }
  out.data.resize(elems);
  std::memcpy(out.data.data(), raw.data(), raw.size());
  return out;
}

std::vector<TensorData> run_neat_runner(simaai::neat::Run& run, const std::vector<float>& input,
                                        int h, int w, int c, int timeout_ms,
                                        const std::string& stage) {
  simaai::neat::Tensor tensor = tensor_from_hwc_f32(input, h, w, c);
  if (!run.push(simaai::neat::TensorList{tensor})) {
    throw std::runtime_error(stage + ": Run.push returned false");
  }
  auto sample = run.pull(timeout_ms);
  if (!sample.has_value()) {
    throw std::runtime_error(stage + ": Run.pull returned no sample");
  }
  const auto tensors = collect_tensors(*sample);
  std::vector<TensorData> outputs;
  outputs.reserve(tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    outputs.push_back(tensor_to_data(tensors[i], stage + ": output " + std::to_string(i)));
  }
  return outputs;
}

simaai::neat::Run build_tensor_runner(const fs::path& model_path, int h, int w, int c) {
  simaai::neat::Model::Options opt;
  opt.preprocess.kind = simaai::neat::InputKind::Tensor;
  opt.preprocess.input_max_width = w;
  opt.preprocess.input_max_height = h;
  opt.preprocess.input_max_depth = c;

  simaai::neat::Model model(model_path.string(), opt);
  simaai::neat::Graph graph;
  graph.add(simaai::neat::nodes::Input(model.input_appsrc_options(true)));
  graph.add(simaai::neat::nodes::QuantTess(simaai::neat::QuantTessOptions(model)));
  graph.add(simaai::neat::nodes::groups::MLA(model));
  graph.add(simaai::neat::nodes::DetessDequant(simaai::neat::DetessDequantOptions(model)));
  graph.add(simaai::neat::nodes::Output());

  simaai::neat::RunOptions run_options;
  run_options.output_memory = simaai::neat::OutputMemory::Owned;
  std::vector<float> dummy(static_cast<size_t>(h) * w * c, 0.0f);
  return graph.build(simaai::neat::TensorList{tensor_from_hwc_f32(dummy, h, w, c)}, run_options);
}

BackboneOutputs parse_backbone_outputs(const std::vector<TensorData>& outputs) {
  if (outputs.size() != 6U) {
    throw std::runtime_error("expected 6 backbone+RPN outputs, got " + std::to_string(outputs.size()));
  }
  BackboneOutputs parsed{{outputs[0], outputs[1], outputs[2], outputs[3]}, outputs[4], outputs[5]};
  const std::array<std::pair<int, int>, 4> expected_hw = {{{160, 160}, {80, 80}, {40, 40}, {20, 20}}};
  for (size_t i = 0; i < parsed.features.size(); ++i) {
    const auto& f = parsed.features[i];
    if (f.n != 1 || f.h != expected_hw[i].first || f.w != expected_hw[i].second || f.c != kFpnChannels) {
      throw std::runtime_error("unexpected FPN feature shape at output " + std::to_string(i));
    }
  }
  if (parsed.logits_concat.n != 1 || parsed.logits_concat.h != 34100 || parsed.logits_concat.w != 1 ||
      parsed.logits_concat.c < 3) {
    throw std::runtime_error("unexpected logits_concat shape");
  }
  if (parsed.deltas_concat.n != 1 || parsed.deltas_concat.h != 34100 || parsed.deltas_concat.w != 1 ||
      parsed.deltas_concat.c < 12) {
    throw std::runtime_error("unexpected deltas_concat shape");
  }
  return parsed;
}

HeadOutputs parse_head_outputs(const std::vector<TensorData>& outputs) {
  if (outputs.size() != 2U) {
    throw std::runtime_error("expected 2 head+predictor outputs, got " + std::to_string(outputs.size()));
  }
  HeadOutputs out{outputs[0], outputs[1]};
  if (out.scores.n != 1 || out.scores.h != kRpnPostNmsTopK || out.scores.w != 1 || out.scores.c < kRoiNumClasses) {
    throw std::runtime_error("unexpected scores output shape");
  }
  if (out.deltas.n != 1 || out.deltas.h != kRpnPostNmsTopK || out.deltas.w != 1 ||
      out.deltas.c < kRoiNumClasses * 4) {
    throw std::runtime_error("unexpected box deltas output shape");
  }
  return out;
}

std::vector<Box> cell_anchors(int size) {
  const float area = static_cast<float>(size * size);
  std::vector<Box> anchors;
  anchors.reserve(kAnchorAspectRatios.size());
  for (float ratio : kAnchorAspectRatios) {
    const float w = std::sqrt(area / ratio);
    const float h = ratio * w;
    anchors.push_back(Box{-w * 0.5f, -h * 0.5f, w * 0.5f, h * 0.5f});
  }
  return anchors;
}

std::vector<Box> generate_level_anchors(int stride, int size) {
  const auto cell = cell_anchors(size);
  const int feat_h = kInferH / stride;
  const int feat_w = kInferW / stride;
  std::vector<Box> anchors;
  anchors.reserve(static_cast<size_t>(feat_h) * feat_w * cell.size());
  for (int y = 0; y < feat_h; ++y) {
    for (int x = 0; x < feat_w; ++x) {
      const float shift_x = static_cast<float>(x * stride);
      const float shift_y = static_cast<float>(y * stride);
      for (const auto& a : cell) {
        anchors.push_back(Box{a.x1 + shift_x, a.y1 + shift_y, a.x2 + shift_x, a.y2 + shift_y});
      }
    }
  }
  return anchors;
}

Box apply_delta(const Box& box, const std::array<float, 4>& d, const std::array<float, 4>& weights) {
  const float bw = box.x2 - box.x1;
  const float bh = box.y2 - box.y1;
  const float bcx = box.x1 + 0.5f * bw;
  const float bcy = box.y1 + 0.5f * bh;
  const float dx = d[0] / weights[0];
  const float dy = d[1] / weights[1];
  const float dw = std::min(d[2] / weights[2], std::log(1000.0f / 16.0f));
  const float dh = std::min(d[3] / weights[3], std::log(1000.0f / 16.0f));
  const float pred_cx = dx * bw + bcx;
  const float pred_cy = dy * bh + bcy;
  const float pred_w = std::exp(dw) * bw;
  const float pred_h = std::exp(dh) * bh;
  return Box{pred_cx - 0.5f * pred_w, pred_cy - 0.5f * pred_h,
             pred_cx + 0.5f * pred_w, pred_cy + 0.5f * pred_h};
}

Box clip_box(Box b, float h, float w) {
  b.x1 = std::clamp(b.x1, 0.0f, w);
  b.y1 = std::clamp(b.y1, 0.0f, h);
  b.x2 = std::clamp(b.x2, 0.0f, w);
  b.y2 = std::clamp(b.y2, 0.0f, h);
  return b;
}

float box_iou(const Box& a, const Box& b) {
  const float xx1 = std::max(a.x1, b.x1);
  const float yy1 = std::max(a.y1, b.y1);
  const float xx2 = std::min(a.x2, b.x2);
  const float yy2 = std::min(a.y2, b.y2);
  const float inter = std::max(0.0f, xx2 - xx1) * std::max(0.0f, yy2 - yy1);
  const float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
  const float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
  const float denom = area_a + area_b - inter + 1e-8f;
  return denom > 0.0f ? inter / denom : 0.0f;
}

std::vector<int> nms(const std::vector<Box>& boxes, const std::vector<float>& scores, float thresh) {
  std::vector<int> order(boxes.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) { return scores[a] > scores[b]; });
  std::vector<int> keep;
  keep.reserve(order.size());
  while (!order.empty()) {
    const int i = order.front();
    keep.push_back(i);
    std::vector<int> next;
    next.reserve(order.size());
    for (size_t pos = 1; pos < order.size(); ++pos) {
      const int j = order[pos];
      if (box_iou(boxes[i], boxes[j]) <= thresh) {
        next.push_back(j);
      }
    }
    order.swap(next);
  }
  return keep;
}

std::vector<Box> decode_rpn(const TensorData& logits_concat, const TensorData& deltas_concat) {
  std::vector<Box> all_proposals;
  std::vector<float> all_scores;
  all_proposals.reserve(kFpnLevelHw.size() * kRpnPreNmsTopK);
  all_scores.reserve(kFpnLevelHw.size() * kRpnPreNmsTopK);

  int offset = 0;
  for (size_t lvl = 0; lvl < kFpnLevelHw.size(); ++lvl) {
    const int h = kFpnLevelHw[lvl].first;
    const int w = kFpnLevelHw[lvl].second;
    const auto anchors = generate_level_anchors(kFpnStrides[lvl], kAnchorSizes[lvl]);
    const int level_locs = h * w;
    const int total = level_locs * static_cast<int>(kAnchorAspectRatios.size());

    std::vector<int> order(total);
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + std::min(kRpnPreNmsTopK, total), order.end(),
                      [&](int a, int b) {
                        const int posa = a / static_cast<int>(kAnchorAspectRatios.size());
                        const int anchora = a % static_cast<int>(kAnchorAspectRatios.size());
                        const int posb = b / static_cast<int>(kAnchorAspectRatios.size());
                        const int anchorb = b % static_cast<int>(kAnchorAspectRatios.size());
                        return logits_concat.at(0, offset + posa, 0, anchora) >
                               logits_concat.at(0, offset + posb, 0, anchorb);
                      });

    const int topk = std::min(kRpnPreNmsTopK, total);
    for (int rank = 0; rank < topk; ++rank) {
      const int flat = order[rank];
      const int pos = flat / static_cast<int>(kAnchorAspectRatios.size());
      const int anchor = flat % static_cast<int>(kAnchorAspectRatios.size());
      const float score = logits_concat.at(0, offset + pos, 0, anchor);
      std::array<float, 4> d = {
          deltas_concat.at(0, offset + pos, 0, anchor * 4 + 0),
          deltas_concat.at(0, offset + pos, 0, anchor * 4 + 1),
          deltas_concat.at(0, offset + pos, 0, anchor * 4 + 2),
          deltas_concat.at(0, offset + pos, 0, anchor * 4 + 3),
      };
      all_proposals.push_back(clip_box(apply_delta(anchors[static_cast<size_t>(flat)], d, kRpnWeights),
                                       static_cast<float>(kInferH), static_cast<float>(kInferW)));
      all_scores.push_back(score);
    }
    offset += level_locs;
  }

  const auto keep = nms(all_proposals, all_scores, kRpnNmsThresh);
  std::vector<Box> proposals;
  proposals.reserve(kRpnPostNmsTopK);
  for (int idx : keep) {
    proposals.push_back(all_proposals[static_cast<size_t>(idx)]);
    if (static_cast<int>(proposals.size()) >= kRpnPostNmsTopK) {
      break;
    }
  }
  while (static_cast<int>(proposals.size()) < kRpnPostNmsTopK) {
    proposals.push_back(Box{});
  }
  return proposals;
}

std::vector<float> roi_align(const std::array<TensorData, 4>& features, const std::vector<Box>& proposals) {
  std::vector<float> output(static_cast<size_t>(kRpnPostNmsTopK) * kHeadInputDepth, 0.0f);
  for (size_t roi_idx = 0; roi_idx < proposals.size(); ++roi_idx) {
    const Box& roi = proposals[roi_idx];
    const float width = roi.x2 - roi.x1;
    const float height = roi.y2 - roi.y1;
    const float area = std::max(width * height, 1e-6f);
    int level = static_cast<int>(std::floor(4.0f + std::log2(std::sqrt(area) / 224.0f + 1e-8f)));
    level = std::clamp(level, 2, 5);
    const int feat_idx = level - 2;
    if (feat_idx < 0 || feat_idx >= 4) {
      continue;
    }
    const TensorData& feat = features[static_cast<size_t>(feat_idx)];
    const float spatial_scale = 1.0f / static_cast<float>(kFpnStrides[static_cast<size_t>(feat_idx)]);

    float rx1 = roi.x1 * spatial_scale - 0.5f;
    float ry1 = roi.y1 * spatial_scale - 0.5f;
    float rx2 = roi.x2 * spatial_scale - 0.5f;
    float ry2 = roi.y2 * spatial_scale - 0.5f;
    const float roi_w = rx2 - rx1;
    const float roi_h = ry2 - ry1;
    const float bin_w = roi_w / static_cast<float>(kRoiOutputSize);
    const float bin_h = roi_h / static_cast<float>(kRoiOutputSize);
    const int grid_h = std::max(1, static_cast<int>(std::ceil(bin_h)));
    const int grid_w = std::max(1, static_cast<int>(std::ceil(bin_w)));
    const float inv_grid = 1.0f / static_cast<float>(grid_h * grid_w);

    for (int ph = 0; ph < kRoiOutputSize; ++ph) {
      for (int pw = 0; pw < kRoiOutputSize; ++pw) {
        std::array<float, kFpnChannels> accum{};
        for (int iy = 0; iy < grid_h; ++iy) {
          const float sy_raw = ry1 + bin_h * (static_cast<float>(ph) + (static_cast<float>(iy) + 0.5f) / grid_h);
          const float sy = std::clamp(sy_raw, 0.0f, static_cast<float>(feat.h - 1));
          const int y0 = static_cast<int>(std::floor(sy));
          const int y1 = std::min(y0 + 1, feat.h - 1);
          const float ly = sy - static_cast<float>(y0);
          const float hy = 1.0f - ly;
          for (int ix = 0; ix < grid_w; ++ix) {
            const float sx_raw = rx1 + bin_w * (static_cast<float>(pw) + (static_cast<float>(ix) + 0.5f) / grid_w);
            const float sx = std::clamp(sx_raw, 0.0f, static_cast<float>(feat.w - 1));
            const int x0 = static_cast<int>(std::floor(sx));
            const int x1 = std::min(x0 + 1, feat.w - 1);
            const float lx = sx - static_cast<float>(x0);
            const float hx = 1.0f - lx;
            for (int ch = 0; ch < kFpnChannels; ++ch) {
              accum[static_cast<size_t>(ch)] +=
                  hy * hx * feat.at(0, y0, x0, ch) + hy * lx * feat.at(0, y0, x1, ch) +
                  ly * hx * feat.at(0, y1, x0, ch) + ly * lx * feat.at(0, y1, x1, ch);
            }
          }
        }
        for (int ch = 0; ch < kFpnChannels; ++ch) {
          const size_t out_idx =
              ((roi_idx * kFpnChannels + static_cast<size_t>(ch)) * kRoiOutputSize + ph) *
                  kRoiOutputSize +
              pw;
          output[out_idx] = accum[static_cast<size_t>(ch)] * inv_grid;
        }
      }
    }
  }
  return output;
}

std::vector<float> head_input_from_roi_features(const std::vector<float>& roi_features) {
  if (roi_features.size() != static_cast<size_t>(kRpnPostNmsTopK) * kHeadInputDepth) {
    throw std::runtime_error("unexpected ROI feature size");
  }
  return roi_features;
}

std::vector<Detection> decode_final_detections(const TensorData& scores, const TensorData& deltas,
                                               const std::vector<Box>& proposals, float conf) {
  std::vector<Box> boxes_f;
  std::vector<float> scores_f;
  std::vector<int> classes_f;

  for (int i = 0; i < kRpnPostNmsTopK; ++i) {
    float max_logit = -std::numeric_limits<float>::infinity();
    for (int c = 0; c < scores.c; ++c) {
      max_logit = std::max(max_logit, scores.at(0, i, 0, c));
    }
    float denom = 0.0f;
    for (int c = 0; c < scores.c; ++c) {
      denom += std::exp(scores.at(0, i, 0, c) - max_logit);
    }
    if (denom <= 0.0f) {
      continue;
    }
    for (int cls = 0; cls < kRoiNumClasses; ++cls) {
      const float score = std::exp(scores.at(0, i, 0, cls) - max_logit) / denom;
      if (score <= conf) {
        continue;
      }
      std::array<float, 4> d = {
          deltas.at(0, i, 0, cls * 4 + 0), deltas.at(0, i, 0, cls * 4 + 1),
          deltas.at(0, i, 0, cls * 4 + 2), deltas.at(0, i, 0, cls * 4 + 3)};
      boxes_f.push_back(clip_box(apply_delta(proposals[static_cast<size_t>(i)], d, kRoiWeights),
                                 static_cast<float>(kInferH), static_cast<float>(kInferW)));
      scores_f.push_back(score);
      classes_f.push_back(cls);
    }
  }

  if (boxes_f.empty()) {
    return {};
  }

  float max_box = 0.0f;
  for (const auto& b : boxes_f) {
    max_box = std::max(max_box, std::max(std::max(b.x1, b.y1), std::max(b.x2, b.y2)));
  }
  std::vector<Box> offset_boxes = boxes_f;
  for (size_t i = 0; i < offset_boxes.size(); ++i) {
    const float offset = static_cast<float>(classes_f[i]) * (max_box + 1.0f);
    offset_boxes[i].x1 += offset;
    offset_boxes[i].x2 += offset;
    offset_boxes[i].y1 += offset;
    offset_boxes[i].y2 += offset;
  }

  const auto keep = nms(offset_boxes, scores_f, kRoiNmsThresh);
  std::vector<Detection> dets;
  dets.reserve(std::min(static_cast<size_t>(kMaxDetections), keep.size()));
  for (int idx : keep) {
    dets.push_back(Detection{boxes_f[static_cast<size_t>(idx)], scores_f[static_cast<size_t>(idx)],
                             classes_f[static_cast<size_t>(idx)]});
    if (static_cast<int>(dets.size()) >= kMaxDetections) {
      break;
    }
  }
  return dets;
}

void scale_detections_to_original(std::vector<Detection>& dets, int orig_h, int orig_w) {
  const float scale_w = static_cast<float>(kInferW) / static_cast<float>(orig_w);
  const float scale_h = static_cast<float>(kInferH) / static_cast<float>(orig_h);
  for (auto& d : dets) {
    d.box.x1 = std::clamp(d.box.x1 / scale_w, 0.0f, static_cast<float>(orig_w));
    d.box.y1 = std::clamp(d.box.y1 / scale_h, 0.0f, static_cast<float>(orig_h));
    d.box.x2 = std::clamp(d.box.x2 / scale_w, 0.0f, static_cast<float>(orig_w));
    d.box.y2 = std::clamp(d.box.y2 / scale_h, 0.0f, static_cast<float>(orig_h));
  }
}

std::string class_name(int class_id) {
  if (class_id >= 0 && static_cast<size_t>(class_id) < kCocoClasses.size()) {
    return std::string(kCocoClasses[static_cast<size_t>(class_id)]);
  }
  return "class_" + std::to_string(class_id);
}

cv::Scalar class_color(int class_id) {
  static const std::array<cv::Scalar, 8> kColors = {
      cv::Scalar(0, 255, 0),   cv::Scalar(255, 0, 0),   cv::Scalar(0, 0, 255),
      cv::Scalar(255, 255, 0), cv::Scalar(255, 0, 255), cv::Scalar(0, 255, 255),
      cv::Scalar(128, 255, 0), cv::Scalar(255, 128, 0),
  };
  const int safe = class_id >= 0 ? class_id : -class_id;
  return kColors[static_cast<size_t>(safe) % kColors.size()];
}

void draw_detections(cv::Mat& bgr, const std::vector<Detection>& dets, int max_draw) {
  const size_t limit = max_draw > 0 ? std::min(dets.size(), static_cast<size_t>(max_draw)) : dets.size();
  for (size_t i = 0; i < limit; ++i) {
    const auto& d = dets[i];
    const int x1 = std::max(0, std::min(bgr.cols - 1, static_cast<int>(d.box.x1)));
    const int y1 = std::max(0, std::min(bgr.rows - 1, static_cast<int>(d.box.y1)));
    const int x2 = std::max(0, std::min(bgr.cols - 1, static_cast<int>(d.box.x2)));
    const int y2 = std::max(0, std::min(bgr.rows - 1, static_cast<int>(d.box.y2)));
    if (x2 <= x1 || y2 <= y1) {
      continue;
    }
    const auto color = class_color(d.class_id);
    const std::string label = class_name(d.class_id) + " " + cv::format("%.2f", d.score);
    cv::rectangle(bgr, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);
    cv::putText(bgr, label, cv::Point(x1, std::max(0, y1 - 4)), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                color, 2, cv::LINE_AA);
  }
}

std::vector<Detection> run_faster_rcnn(simaai::neat::Run& backbone_rpn,
                                       simaai::neat::Run& head_predictor, const cv::Mat& bgr,
                                       int timeout_ms, float conf) {
  const auto input = preprocess_image(bgr);
  const auto backbone_outputs = run_neat_runner(backbone_rpn, input, kInferH, kInferW, 3, timeout_ms,
                                               "backbone_rpn");
  const BackboneOutputs parsed = parse_backbone_outputs(backbone_outputs);
  const std::vector<Box> proposals = decode_rpn(parsed.logits_concat, parsed.deltas_concat);
  const std::vector<float> roi_features = roi_align(parsed.features, proposals);
  const std::vector<float> head_input = head_input_from_roi_features(roi_features);
  const auto head_outputs = run_neat_runner(head_predictor, head_input, kRpnPostNmsTopK, 1,
                                           kHeadInputDepth, timeout_ms, "head_predictor");
  const HeadOutputs head = parse_head_outputs(head_outputs);
  auto dets = decode_final_detections(head.scores, head.deltas, proposals, conf);
  scale_detections_to_original(dets, bgr.rows, bgr.cols);
  return dets;
}

struct ProfileResult {
  std::vector<Detection> detections;
  double preprocess = 0.0;
  double backbone_rpn = 0.0;
  double rpn_decode = 0.0;
  double roi_align = 0.0;
  double head_predictor = 0.0;
  double final_decode = 0.0;
  double neat_total = 0.0;
  double cpu_glue_total = 0.0;
  double pipeline_run = 0.0;
};

ProfileResult run_faster_rcnn_profiled(simaai::neat::Run& backbone_rpn,
                                       simaai::neat::Run& head_predictor, const cv::Mat& bgr,
                                       int timeout_ms, float conf) {
  ProfileResult result;
  const auto t0 = std::chrono::steady_clock::now();
  const auto input = preprocess_image(bgr);
  const auto t1 = std::chrono::steady_clock::now();

  const auto backbone_outputs = run_neat_runner(backbone_rpn, input, kInferH, kInferW, 3, timeout_ms,
                                               "backbone_rpn");
  const auto t2 = std::chrono::steady_clock::now();

  const BackboneOutputs parsed = parse_backbone_outputs(backbone_outputs);
  const std::vector<Box> proposals = decode_rpn(parsed.logits_concat, parsed.deltas_concat);
  const auto t3 = std::chrono::steady_clock::now();

  const std::vector<float> roi_features = roi_align(parsed.features, proposals);
  const std::vector<float> head_input = head_input_from_roi_features(roi_features);
  const auto t4 = std::chrono::steady_clock::now();

  const auto head_outputs = run_neat_runner(head_predictor, head_input, kRpnPostNmsTopK, 1,
                                           kHeadInputDepth, timeout_ms, "head_predictor");
  const auto t5 = std::chrono::steady_clock::now();

  const HeadOutputs head = parse_head_outputs(head_outputs);
  result.detections = decode_final_detections(head.scores, head.deltas, proposals, conf);
  scale_detections_to_original(result.detections, bgr.rows, bgr.cols);
  const auto t6 = std::chrono::steady_clock::now();

  result.preprocess = std::chrono::duration<double>(t1 - t0).count();
  result.backbone_rpn = std::chrono::duration<double>(t2 - t1).count();
  result.rpn_decode = std::chrono::duration<double>(t3 - t2).count();
  result.roi_align = std::chrono::duration<double>(t4 - t3).count();
  result.head_predictor = std::chrono::duration<double>(t5 - t4).count();
  result.final_decode = std::chrono::duration<double>(t6 - t5).count();
  result.neat_total = result.backbone_rpn + result.head_predictor;
  result.cpu_glue_total = result.preprocess + result.rpn_decode + result.roi_align + result.final_decode;
  result.pipeline_run = std::chrono::duration<double>(t6 - t0).count();
  return result;
}

void print_profile_stats(const std::string& name, const std::vector<double>& values) {
  const Stats s = compute_stats(values);
  const double fps = values.empty() || s.sum <= 0.0 ? 0.0 : static_cast<double>(values.size()) / s.sum;
  std::cout << "  " << name << ": mean=" << s.mean << "s, min=" << s.min << "s, max=" << s.max
            << "s, FPS=" << fps << "\n";
}

} // namespace

int main(int argc, char** argv) {
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);

  try {
    const Config cfg = parse_config(argc, argv);
    if (!fs::exists(cfg.backbone_rpn_model)) {
      throw std::runtime_error("backbone_rpn model does not exist: " + cfg.backbone_rpn_model.string());
    }
    if (!fs::exists(cfg.head_predictor_model)) {
      throw std::runtime_error("head_predictor model does not exist: " + cfg.head_predictor_model.string());
    }
    if (!fs::is_directory(cfg.input_dir)) {
      throw std::runtime_error("input_dir does not exist: " + cfg.input_dir.string());
    }

    const std::vector<fs::path> images = image_paths_in_dir(cfg.input_dir);
    if (images.empty()) {
      throw std::runtime_error("no images found in: " + cfg.input_dir.string());
    }

    auto backbone_rpn = build_tensor_runner(cfg.backbone_rpn_model, kInferH, kInferW, 3);
    auto head_predictor = build_tensor_runner(cfg.head_predictor_model, kRpnPostNmsTopK, 1, kHeadInputDepth);

    if (cfg.profile) {
      const fs::path image_path = images.front();
      cv::Mat bgr = cv::imread(image_path.string(), cv::IMREAD_COLOR);
      if (bgr.empty()) {
        throw std::runtime_error("failed to read image: " + image_path.string());
      }
      std::vector<double> preprocess_times;
      std::vector<double> backbone_rpn_times;
      std::vector<double> rpn_decode_times;
      std::vector<double> roi_align_times;
      std::vector<double> head_predictor_times;
      std::vector<double> final_decode_times;
      std::vector<double> neat_total_times;
      std::vector<double> cpu_glue_total_times;
      std::vector<double> pipeline_times;
      std::vector<double> visualization_times;
      std::vector<Detection> last_dets;
      const int runs = std::max(1, cfg.num_runs);
      preprocess_times.reserve(runs);
      backbone_rpn_times.reserve(runs);
      rpn_decode_times.reserve(runs);
      roi_align_times.reserve(runs);
      head_predictor_times.reserve(runs);
      final_decode_times.reserve(runs);
      neat_total_times.reserve(runs);
      cpu_glue_total_times.reserve(runs);
      pipeline_times.reserve(runs);
      visualization_times.reserve(runs);

      for (int i = 0; i < runs; ++i) {
        ProfileResult profile = run_faster_rcnn_profiled(backbone_rpn, head_predictor, bgr, cfg.timeout_ms, cfg.conf);
        const auto t0 = std::chrono::steady_clock::now();
        cv::Mat overlay = bgr.clone();
        draw_detections(overlay, profile.detections, cfg.max_draw);
        const auto t1 = std::chrono::steady_clock::now();

        preprocess_times.push_back(profile.preprocess);
        backbone_rpn_times.push_back(profile.backbone_rpn);
        rpn_decode_times.push_back(profile.rpn_decode);
        roi_align_times.push_back(profile.roi_align);
        head_predictor_times.push_back(profile.head_predictor);
        final_decode_times.push_back(profile.final_decode);
        neat_total_times.push_back(profile.neat_total);
        cpu_glue_total_times.push_back(profile.cpu_glue_total);
        pipeline_times.push_back(profile.pipeline_run);
        visualization_times.push_back(std::chrono::duration<double>(t1 - t0).count());
        last_dets = std::move(profile.detections);
      }

      std::vector<double> total_times;
      total_times.reserve(pipeline_times.size());
      for (size_t i = 0; i < pipeline_times.size(); ++i) {
        total_times.push_back(pipeline_times[i] + visualization_times[i]);
      }
      std::cout << "Profiling over " << pipeline_times.size() << " runs (image='" << image_path.string()
                << "'):\n";
      print_profile_stats("Preprocess", preprocess_times);
      print_profile_stats("Backbone+RPN NEAT", backbone_rpn_times);
      print_profile_stats("RPN decode+NMS", rpn_decode_times);
      print_profile_stats("ROI Align", roi_align_times);
      print_profile_stats("Box head NEAT", head_predictor_times);
      print_profile_stats("Final decode+NMS", final_decode_times);
      print_profile_stats("NEAT total", neat_total_times);
      print_profile_stats("CPU glue total", cpu_glue_total_times);
      print_profile_stats("Pipeline run", pipeline_times);
      print_profile_stats("Visualization", visualization_times);
      print_profile_stats("Overall", total_times);
      std::cout << "Last run detections: " << last_dets.size() << "\n";
      backbone_rpn.close();
      head_predictor.close();
      return 0;
    }

    fs::create_directories(cfg.output_dir);
    int processed = 0;
    for (const fs::path& image_path : images) {
      cv::Mat bgr = cv::imread(image_path.string(), cv::IMREAD_COLOR);
      if (bgr.empty()) {
        throw std::runtime_error("failed to read image: " + image_path.string());
      }
      const std::vector<Detection> dets =
          run_faster_rcnn(backbone_rpn, head_predictor, bgr, cfg.timeout_ms, cfg.conf);
      cv::Mat overlay = bgr.clone();
      draw_detections(overlay, dets, cfg.max_draw);
      const fs::path output_path = cfg.output_dir / (image_path.stem().string() + "_faster_rcnn.png");
      if (!cv::imwrite(output_path.string(), overlay)) {
        throw std::runtime_error("failed to write: " + output_path.string());
      }
      ++processed;
      std::cout << "[" << processed << "/" << images.size() << "] " << image_path.filename().string()
                << ": " << dets.size() << " detections -> " << output_path.filename().string() << "\n";
    }
    std::cout << "Done: " << processed << " images processed\n";
    backbone_rpn.close();
    head_predictor.close();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 2;
  }
}
