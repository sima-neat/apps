// Copyright 2026 SiMa Technologies, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "neat.h"
#include "neat/models.h"
#include "neat/node_groups.h"
#include "neat/nodes.h"
#include "support/runtime/config_utils.h"
#include "support/runtime/example_utils.h"

#include <nodes/groups/VideoSender.h>
#include <nodes/io/MetadataSender.h>

#include <nlohmann/json.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstddef>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
namespace neat = simaai::neat;

namespace {

constexpr int kNumClasses = 91;
constexpr int kMaskSize = 108;
constexpr int kClassificationTopK = 300;
constexpr std::size_t kMetadataByteBudget = 32'768;
std::atomic<bool> g_stop{false};

void request_stop(int) {
  g_stop.store(true);
}

enum class SourceCodec { H264, H265, Mjpeg };
enum class Task { Detection, Segmentation };

struct SourceGeometry {
  int width = 0;
  int height = 0;
  int fps = 0;
};

std::string lower_copy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

SourceCodec parse_source_codec(const std::string& value) {
  const std::string codec = lower_copy(value);
  if (codec == "h264" || codec == "avc" || codec == "h.264") {
    return SourceCodec::H264;
  }
  if (codec == "h265" || codec == "hevc" || codec == "h.265") {
    return SourceCodec::H265;
  }
  if (codec == "mjpeg" || codec == "jpeg") {
    return SourceCodec::Mjpeg;
  }
  throw std::runtime_error("source.codec must be h264/avc, h265/hevc, or mjpeg");
}

const char* source_codec_name(SourceCodec codec) {
  if (codec == SourceCodec::H264) {
    return "h264";
  }
  return codec == SourceCodec::H265 ? "h265" : "mjpeg";
}

Task parse_task(const std::string& value) {
  const std::string task = lower_copy(value);
  if (task == "detection") {
    return Task::Detection;
  }
  if (task == "segmentation") {
    return Task::Segmentation;
  }
  throw std::runtime_error("model.task must be detection or segmentation");
}

const char* task_name(Task task) {
  return task == Task::Detection ? "detection" : "segmentation";
}

SourceGeometry resolve_geometry(const SourceGeometry& probed, const SourceGeometry& configured) {
  return {
      probed.width > 0 ? probed.width : configured.width,
      probed.height > 0 ? probed.height : configured.height,
      configured.fps > 0 ? configured.fps : probed.fps,
  };
}

struct Config {
  Task task = Task::Detection;
  std::string variant;
  std::string backbone;
  std::string transformer;
  int input_size = 0;
  int feature_size = 0;
  int top_k = 0;
  fs::path labels;
  std::string rtsp_url;
  SourceCodec codec = SourceCodec::H264;
  bool tcp = true;
  int latency_ms = 100;
  int fallback_width = 0;
  int fallback_height = 0;
  int source_fps = 0;
  int frames = 0;
  float min_score = 0.5F;
  int max_results = 100;
  float mask_threshold = 0.08F;
  std::string insight_host;
  int video_port = 9000;
  int metadata_port = 9100;
};

struct CliOptions {
  fs::path config_path;
  bool validate_config_only = false;
};

CliOptions parse_args(int argc, char** argv) {
  CliOptions options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--config") {
      if (++index >= argc) {
        throw std::runtime_error("--config requires a path");
      }
      options.config_path = argv[index];
    } else if (argument == "--validate-config-only") {
      options.validate_config_only = true;
    } else if (argument == "--help" || argument == "-h") {
      std::cout << "Usage: " << argv[0] << " [--config <path>] [--validate-config-only]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + argument);
    }
  }
  if (options.config_path.empty()) {
    throw std::runtime_error("--config is required");
  }
  return options;
}

Config load_config(const fs::path& path) {
  const auto raw = sima_examples::ScalarConfig::load(path);
  Config cfg;
  cfg.task = parse_task(raw.string_or("model.task", "detection"));
  std::string model_prefix;
  std::string inference_prefix;
  if (cfg.task == Task::Detection) {
    cfg.variant = lower_copy(raw.string_or("model.detection.variant", "small"));
    if (cfg.variant != "small" && cfg.variant != "medium") {
      throw std::runtime_error("model.detection.variant must be small or medium");
    }
    model_prefix = "model.detection." + cfg.variant + ".";
    inference_prefix = "inference.detection.";
    cfg.input_size = cfg.variant == "small" ? 512 : 576;
    cfg.feature_size = cfg.input_size / 16;
    cfg.top_k = 300;
  } else {
    cfg.variant = "segmentation";
    model_prefix = "model.segmentation.";
    inference_prefix = "inference.segmentation.";
    cfg.input_size = 432;
    cfg.feature_size = 36;
    cfg.top_k = 200;
  }
  cfg.backbone = raw.string_or(model_prefix + "backbone", "");
  cfg.transformer = raw.string_or(model_prefix + "transformer", "");
  cfg.labels = raw.string_or("model.labels", "");
  cfg.rtsp_url = raw.string_or("source.rtsp_url", "");
  cfg.codec = parse_source_codec(raw.string_or("source.codec", "h264"));
  cfg.tcp = raw.bool_or("source.tcp", true);
  cfg.latency_ms = raw.int_or("source.latency_ms", 100);
  cfg.fallback_width = raw.int_or("source.width", 0);
  cfg.fallback_height = raw.int_or("source.height", 0);
  cfg.source_fps = raw.int_or("source.fps", 0);
  cfg.frames = raw.int_or("inference.frames", 0);
  cfg.min_score = static_cast<float>(
      raw.double_or(inference_prefix + "min_score", cfg.task == Task::Detection ? 0.5 : 0.3));
  cfg.max_results = raw.int_or(
      inference_prefix + (cfg.task == Task::Detection ? "max_detections" : "max_segments"),
      cfg.task == Task::Detection ? 100 : 24);
  if (cfg.task == Task::Segmentation) {
    cfg.mask_threshold =
        static_cast<float>(raw.double_or("inference.segmentation.mask_threshold", 0.08));
  }
  cfg.insight_host = raw.string_or("output.insight.host", "");
  cfg.video_port = raw.int_or("output.insight.video_port", 9000);
  cfg.metadata_port = raw.int_or("output.insight.metadata_port", 9100);

  sima_examples::require(!cfg.backbone.empty() && !cfg.transformer.empty(),
                         model_prefix + "backbone and transformer must be set");
  sima_examples::require(!cfg.labels.empty(), "model.labels must be set");
  sima_examples::require(cfg.rtsp_url.rfind("rtsp://", 0) == 0,
                         "source.rtsp_url must be an RTSP URL");
  sima_examples::require(cfg.latency_ms >= 0 && cfg.frames >= 0,
                         "source.latency_ms and inference.frames must be >= 0");
  sima_examples::require(cfg.fallback_width >= 0 && cfg.fallback_height >= 0 && cfg.source_fps >= 0,
                         "source.width, source.height, and source.fps must be >= 0");
  sima_examples::require(cfg.min_score >= 0.0F && cfg.min_score <= 1.0F,
                         inference_prefix + "min_score must be in [0, 1]");
  sima_examples::require(cfg.max_results > 0,
                         inference_prefix + "max_detections/max_segments must be > 0");
  if (cfg.task == Task::Segmentation) {
    sima_examples::require(cfg.mask_threshold >= 0.0F && cfg.mask_threshold <= 1.0F,
                           "inference.segmentation.mask_threshold must be in [0, 1]");
  }
  sima_examples::require(!cfg.insight_host.empty(), "output.insight.host must be set");
  sima_examples::require(cfg.video_port > 0 && cfg.video_port <= 65535 && cfg.metadata_port > 0 &&
                             cfg.metadata_port <= 65535,
                         "Insight ports must be in [1, 65535]");
  return cfg;
}

std::vector<std::string> load_labels(const fs::path& path) {
  std::ifstream input(path);
  if (!input.is_open()) {
    throw std::runtime_error("failed to open labels: " + path.string());
  }
  std::vector<std::string> labels;
  std::string line;
  while (std::getline(input, line)) {
    labels.push_back(sima_examples::trim_copy(line));
  }
  if (labels.size() != 91U ||
      std::any_of(labels.begin(), labels.end(), [](const auto& label) { return label.empty(); })) {
    throw std::runtime_error("model.labels must contain exactly 91 non-empty COCO labels");
  }
  return labels;
}

std::size_t element_count(const std::vector<int64_t>& shape) {
  return std::accumulate(shape.begin(), shape.end(), std::size_t{1},
                         [](std::size_t total, int64_t dimension) {
                           if (dimension <= 0) {
                             throw std::runtime_error("model tensor shape must be static");
                           }
                           return total * static_cast<std::size_t>(dimension);
                         });
}

std::vector<neat::Tensor> collect_tensors(const neat::Sample& sample) {
  std::vector<neat::Tensor> tensors;
  if (sample.kind == neat::SampleKind::Tensor && sample.tensor.has_value()) {
    tensors.push_back(*sample.tensor);
  } else if (sample.kind == neat::SampleKind::TensorSet) {
    tensors.insert(tensors.end(), sample.tensors.begin(), sample.tensors.end());
  }
  for (const auto& field : sample.fields) {
    auto nested = collect_tensors(field);
    tensors.insert(tensors.end(), nested.begin(), nested.end());
  }
  return tensors;
}

std::vector<float> read_floats(const neat::Tensor& tensor) {
  if (tensor.dtype != neat::TensorDType::Float32 || !tensor.is_dense()) {
    throw std::runtime_error("RF-DETR output must be a dense float32 tensor");
  }
  auto mapping = tensor.map(neat::MapMode::Read);
  if (mapping.data == nullptr || mapping.size_bytes % sizeof(float) != 0) {
    throw std::runtime_error("RF-DETR tensor is not CPU-readable");
  }
  std::vector<float> values(mapping.size_bytes / sizeof(float));
  std::memcpy(values.data(), mapping.data, mapping.size_bytes);
  return values;
}

std::vector<int> stable_topk_indices(const std::vector<float>& values, std::size_t count) {
  count = std::min(count, values.size());
  std::vector<int> indices(values.size());
  std::iota(indices.begin(), indices.end(), 0);
  const auto higher_score = [&values](int left, int right) {
    const float left_score = values[static_cast<std::size_t>(left)];
    const float right_score = values[static_cast<std::size_t>(right)];
    return left_score != right_score ? left_score > right_score : left < right;
  };
  std::partial_sort(indices.begin(), indices.begin() + static_cast<std::ptrdiff_t>(count),
                    indices.end(), higher_score);
  indices.resize(count);
  return indices;
}

std::vector<float> stable_topk_gather(const std::vector<float>& scores,
                                      const std::vector<float>& proposals, int top_k) {
  if (scores.size() < static_cast<std::size_t>(top_k) || proposals.size() != scores.size() * 4U) {
    throw std::runtime_error("backbone score and proposal shapes do not match");
  }
  if (!std::all_of(scores.begin(), scores.end(),
                   [](float value) { return std::isfinite(value); }) ||
      !std::all_of(proposals.begin(), proposals.end(),
                   [](float value) { return std::isfinite(value); })) {
    throw std::runtime_error("backbone output contains non-finite values");
  }
  const auto indices = stable_topk_indices(scores, static_cast<std::size_t>(top_k));
  std::vector<float> gathered(static_cast<std::size_t>(top_k) * 4U);
  for (int output = 0; output < top_k; ++output) {
    const auto source = static_cast<std::size_t>(indices[static_cast<std::size_t>(output)]) * 4U;
    std::copy_n(proposals.begin() + static_cast<std::ptrdiff_t>(source), 4,
                gathered.begin() + static_cast<std::ptrdiff_t>(output * 4));
  }
  return gathered;
}

struct BackboneOutputs {
  neat::Tensor feature;
  neat::Tensor scores;
  neat::Tensor proposals;
};

BackboneOutputs split_backbone(const neat::Sample& sample, int proposal_count) {
  BackboneOutputs output;
  bool have_feature = false;
  bool have_scores = false;
  bool have_proposals = false;
  for (const auto& tensor : collect_tensors(sample)) {
    const auto elements = element_count(tensor.shape);
    if (tensor.shape.size() >= 3U && tensor.shape.back() == 256) {
      output.feature = tensor;
      have_feature = true;
    } else if (!tensor.shape.empty() && tensor.shape.back() == 4 &&
               elements == static_cast<std::size_t>(proposal_count) * 4U) {
      output.proposals = tensor;
      have_proposals = true;
    } else if (elements == static_cast<std::size_t>(proposal_count)) {
      output.scores = tensor;
      have_scores = true;
    }
  }
  if (!have_feature || !have_scores || !have_proposals) {
    throw std::runtime_error("backbone did not produce feature, score, and proposal tensors");
  }
  return output;
}

struct TransformerOutputs {
  neat::Tensor boxes;
  neat::Tensor logits;
  std::optional<neat::Tensor> masks;
};

TransformerOutputs split_transformer(const neat::Sample& sample, const Config& cfg) {
  TransformerOutputs output;
  bool have_boxes = false;
  bool have_logits = false;
  for (const auto& tensor : collect_tensors(sample)) {
    const auto elements = element_count(tensor.shape);
    if (elements == static_cast<std::size_t>(cfg.top_k) * 4U) {
      output.boxes = tensor;
      have_boxes = true;
    } else if (elements == static_cast<std::size_t>(cfg.top_k) * kNumClasses) {
      output.logits = tensor;
      have_logits = true;
    } else if (cfg.task == Task::Segmentation &&
               elements == static_cast<std::size_t>(kMaskSize) * kMaskSize * cfg.top_k) {
      output.masks = tensor;
    }
  }
  if (!have_boxes || !have_logits ||
      (cfg.task == Task::Segmentation && !output.masks.has_value())) {
    throw std::runtime_error("transformer did not produce the expected output tensors");
  }
  return output;
}

void copy_identity(const neat::Sample& source, neat::Sample& target) {
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

int64_t identity_key(const neat::Sample& sample) {
  return sample.frame_id >= 0 ? sample.frame_id : sample.input_seq;
}

neat::TensorList transformer_inputs(const neat::Model& model, const neat::Tensor& feature,
                                    const neat::Tensor& gathered, int top_k) {
  neat::TensorList inputs;
  for (const auto& spec : model.input_specs()) {
    neat::Tensor tensor =
        element_count(spec.shape) == static_cast<std::size_t>(top_k) * 4U ? gathered : feature;
    if (tensor.shape != spec.shape) {
      const bool has_batch_one =
          tensor.shape.size() == spec.shape.size() + 1U && tensor.shape.front() == 1 &&
          std::equal(spec.shape.begin(), spec.shape.end(), tensor.shape.begin() + 1);
      if (!has_batch_one) {
        throw std::runtime_error("transformer input shape does not match model contract");
      }
      tensor.shape.erase(tensor.shape.begin());
      if (tensor.strides_bytes.size() == tensor.shape.size() + 1U) {
        tensor.strides_bytes.erase(tensor.strides_bytes.begin());
      }
      if (tensor.axis_semantics.size() == tensor.shape.size() + 1U) {
        tensor.axis_semantics.erase(tensor.axis_semantics.begin());
      }
    }
    inputs.push_back(std::move(tensor));
  }
  if (inputs.size() != 2U) {
    throw std::runtime_error("unexpected transformer input contract");
  }
  return inputs;
}

std::vector<sima_examples::MetadataBox>
postprocess(const std::vector<float>& boxes, const std::vector<float>& logits, int frame_width,
            int frame_height, const std::vector<std::string>& labels, float min_score,
            int max_detections, int top_k) {
  if (boxes.size() != static_cast<std::size_t>(top_k) * 4U ||
      logits.size() != static_cast<std::size_t>(top_k) * kNumClasses) {
    throw std::runtime_error("unexpected transformer output shape");
  }
  std::vector<float> probabilities(logits.size());
  std::transform(logits.begin(), logits.end(), probabilities.begin(), [](float value) {
    value = std::clamp(value, -80.0F, 80.0F);
    return 1.0F / (1.0F + std::exp(-value));
  });
  const auto ranking = stable_topk_indices(probabilities, kClassificationTopK);

  std::vector<sima_examples::MetadataBox> objects;
  objects.reserve(static_cast<std::size_t>(max_detections));
  for (const int flat_index : ranking) {
    const float score = probabilities[static_cast<std::size_t>(flat_index)];
    if (score < min_score || objects.size() >= static_cast<std::size_t>(max_detections)) {
      break;
    }
    const int query = flat_index / kNumClasses;
    const int class_id = flat_index % kNumClasses;
    if (class_id == 0 || labels[static_cast<std::size_t>(class_id)] == "unused") {
      continue;
    }
    const auto offset = static_cast<std::size_t>(query) * 4U;
    const float cx = boxes[offset];
    const float cy = boxes[offset + 1U];
    const float box_width = boxes[offset + 2U];
    const float box_height = boxes[offset + 3U];
    const float x =
        std::clamp((cx - box_width / 2.0F) * frame_width, 0.0F, static_cast<float>(frame_width));
    const float y =
        std::clamp((cy - box_height / 2.0F) * frame_height, 0.0F, static_cast<float>(frame_height));
    const float x2 =
        std::clamp((cx + box_width / 2.0F) * frame_width, x, static_cast<float>(frame_width));
    const float y2 =
        std::clamp((cy + box_height / 2.0F) * frame_height, y, static_cast<float>(frame_height));
    objects.push_back({"obj_" + std::to_string(objects.size() + 1U),
                       labels[static_cast<std::size_t>(class_id)], score, x, y, x2 - x, y2 - y});
  }
  return objects;
}

float sigmoid(float value) {
  value = std::clamp(value, -80.0F, 80.0F);
  return 1.0F / (1.0F + std::exp(-value));
}

cv::Rect frame_rect(const std::vector<float>& boxes, int query, int width, int height) {
  const auto offset = static_cast<std::size_t>(query) * 4U;
  const float cx = boxes[offset];
  const float cy = boxes[offset + 1U];
  const float box_width = boxes[offset + 2U];
  const float box_height = boxes[offset + 3U];
  const int x0 =
      std::clamp(static_cast<int>(std::nearbyint((cx - box_width / 2.0F) * width)), 0, width - 1);
  const int y0 = std::clamp(static_cast<int>(std::nearbyint((cy - box_height / 2.0F) * height)), 0,
                            height - 1);
  const int x1 =
      std::clamp(static_cast<int>(std::nearbyint((cx + box_width / 2.0F) * width)), x0 + 1, width);
  const int y1 = std::clamp(static_cast<int>(std::nearbyint((cy + box_height / 2.0F) * height)),
                            y0 + 1, height);
  return {x0, y0, x1 - x0, y1 - y0};
}

std::vector<cv::Point> mask_polygon(const float* masks, int query, int top_k, const cv::Rect& box,
                                    int frame_width, int frame_height, float threshold) {
  const double mask_scale_x = static_cast<double>(kMaskSize) / frame_width;
  const double mask_scale_y = static_cast<double>(kMaskSize) / frame_height;
  const int x0 = std::clamp(static_cast<int>(std::floor(box.x * mask_scale_x)), 0, kMaskSize - 1);
  const int y0 = std::clamp(static_cast<int>(std::floor(box.y * mask_scale_y)), 0, kMaskSize - 1);
  const int x1 = std::clamp(static_cast<int>(std::ceil((box.x + box.width) * mask_scale_x)), x0 + 1,
                            kMaskSize);
  const int y1 = std::clamp(static_cast<int>(std::ceil((box.y + box.height) * mask_scale_y)),
                            y0 + 1, kMaskSize);
  cv::Mat mask(y1 - y0, x1 - x0, CV_32FC1);
  for (int y = y0; y < y1; ++y) {
    auto* row = mask.ptr<float>(y - y0);
    for (int x = x0; x < x1; ++x) {
      row[x - x0] = sigmoid(masks[(y * kMaskSize + x) * top_k + query]);
    }
  }
  cv::Mat binary;
  cv::compare(mask, threshold, binary, cv::CMP_GE);
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  if (contours.empty()) {
    return {};
  }
  const auto largest =
      std::max_element(contours.begin(), contours.end(), [](const auto& left, const auto& right) {
        return cv::contourArea(left) < cv::contourArea(right);
      });
  std::vector<cv::Point> polygon;
  cv::approxPolyDP(*largest, polygon, 0.004 * cv::arcLength(*largest, true), true);
  if (polygon.size() < 3U) {
    return {};
  }
  const double scale_x = static_cast<double>(box.width - 1) / std::max(mask.cols - 1, 1);
  const double scale_y = static_cast<double>(box.height - 1) / std::max(mask.rows - 1, 1);
  for (auto& point : polygon) {
    point.x = std::clamp(box.x + static_cast<int>(std::nearbyint(point.x * scale_x)), box.x,
                         box.x + box.width - 1);
    point.y = std::clamp(box.y + static_cast<int>(std::nearbyint(point.y * scale_y)), box.y,
                         box.y + box.height - 1);
  }
  return polygon;
}

std::string segmentation_metadata(const TransformerOutputs& output, int frame_width,
                                  int frame_height, const std::vector<std::string>& labels,
                                  const Config& cfg) {
  const auto boxes = read_floats(output.boxes);
  const auto logits = read_floats(output.logits);
  const auto& masks_tensor = *output.masks;
  if (masks_tensor.dtype != neat::TensorDType::Float32) {
    throw std::runtime_error("segmentation mask output must be float32");
  }
  neat::Mapping masks_map = masks_tensor.is_dense() && masks_tensor.is_contiguous()
                                ? masks_tensor.view_read()
                                : neat::Mapping{};
  std::vector<float> masks_copy;
  const float* masks = nullptr;
  const std::size_t expected_bytes =
      static_cast<std::size_t>(kMaskSize) * kMaskSize * cfg.top_k * sizeof(float);
  if (masks_map.data != nullptr && masks_map.size_bytes >= expected_bytes) {
    masks = static_cast<const float*>(masks_map.data);
  } else {
    masks_copy = read_floats(masks_tensor);
    masks = masks_copy.data();
  }

  std::vector<float> probabilities(logits.size());
  std::transform(logits.begin(), logits.end(), probabilities.begin(), sigmoid);
  const auto ranking = stable_topk_indices(probabilities, kClassificationTopK);

  nlohmann::json segments = nlohmann::json::array();
  std::size_t encoded_bytes = sizeof(R"({"segments":[]})") - 1U;
  for (const int flat_index : ranking) {
    const float score = probabilities[static_cast<std::size_t>(flat_index)];
    if (score < cfg.min_score || segments.size() >= static_cast<std::size_t>(cfg.max_results)) {
      break;
    }
    const int query = flat_index / kNumClasses;
    const int class_id = flat_index % kNumClasses;
    if (class_id == 0 || labels[static_cast<std::size_t>(class_id)] == "unused") {
      continue;
    }
    const cv::Rect box = frame_rect(boxes, query, frame_width, frame_height);
    const auto polygon =
        mask_polygon(masks, query, cfg.top_k, box, frame_width, frame_height, cfg.mask_threshold);
    if (polygon.empty()) {
      continue;
    }
    nlohmann::json points = nlohmann::json::array();
    for (const auto& point : polygon) {
      points.push_back({point.x, point.y});
    }
    nlohmann::json segment = {
        {"id", "seg_" + std::to_string(segments.size() + 1U)},
        {"label", labels[static_cast<std::size_t>(class_id)]},
        {"confidence", score},
        {"bbox", {box.x, box.y, box.width, box.height}},
        {"mask_format", "polygon"},
        {"mask", std::move(points)},
    };
    const std::size_t segment_bytes = segment.dump().size() + 1U;
    if (encoded_bytes + segment_bytes > kMetadataByteBudget) {
      break;
    }
    encoded_bytes += segment_bytes;
    segments.push_back(std::move(segment));
  }
  return nlohmann::json{{"segments", std::move(segments)}}.dump();
}

SourceGeometry probe_source_geometry(const Config& cfg) {
  sima_examples::RtspStreamInfo stream;
  sima_examples::RtspProbeOptions options;
  options.latency_ms = cfg.latency_ms;
  options.rtsp_tcp = cfg.tcp;
  (void)sima_examples::probe_rtsp_stream_info(cfg.rtsp_url, options, stream);

  const SourceGeometry geometry =
      resolve_geometry({stream.width, stream.height, stream.fps},
                       {cfg.fallback_width, cfg.fallback_height, cfg.source_fps});
  if (cfg.codec == SourceCodec::Mjpeg && geometry.fps <= 0) {
    throw std::runtime_error("MJPEG source did not provide an FPS; set source.fps");
  }
  sima_examples::require(
      geometry.width > 0 && geometry.height > 0 && geometry.fps > 0,
      "failed to resolve RTSP width, height, and FPS; set source fallbacks if probing fails");
  return geometry;
}

neat::nodes::groups::RtspCodec rtsp_codec(SourceCodec codec) {
  if (codec == SourceCodec::H264) {
    return neat::nodes::groups::RtspCodec::H264;
  }
  return codec == SourceCodec::H265 ? neat::nodes::groups::RtspCodec::H265
                                    : neat::nodes::groups::RtspCodec::MJPEG;
}

neat::SimaDecodeType decode_type(SourceCodec codec) {
  if (codec == SourceCodec::H264) {
    return neat::SimaDecodeType::H264;
  }
  return codec == SourceCodec::H265 ? neat::SimaDecodeType::H265 : neat::SimaDecodeType::MJPEG;
}

int run(const Config& cfg) {
  const SourceGeometry geometry = probe_source_geometry(cfg);
  const auto labels = load_labels(cfg.labels);
  neat::Model::Options backbone_options;
  backbone_options.preprocess.kind = neat::InputKind::Image;
  backbone_options.preprocess.enable = neat::AutoFlag::On;
  backbone_options.preprocess.input_max_width = geometry.width;
  backbone_options.preprocess.input_max_height = geometry.height;
  backbone_options.preprocess.input_max_depth = 3;
  backbone_options.preprocess.resize.enable = neat::AutoFlag::On;
  backbone_options.preprocess.resize.mode = neat::ResizeMode::Stretch;
  backbone_options.preprocess.color_convert.enable = neat::AutoFlag::On;
  backbone_options.preprocess.color_convert.input_format = neat::PreprocessColorFormat::NV12;
  backbone_options.preprocess.color_convert.output_format = neat::PreprocessColorFormat::RGB;
  if (cfg.task == Task::Segmentation) {
    backbone_options.preprocess.resize.width = cfg.input_size;
    backbone_options.preprocess.resize.height = cfg.input_size;
    backbone_options.preprocess.resize.scaling_type = "BILINEAR";
    backbone_options.preprocess.normalize.enable = neat::AutoFlag::On;
    backbone_options.preprocess.normalize.mean = {0.485F, 0.456F, 0.406F};
    backbone_options.preprocess.normalize.stddev = {0.229F, 0.224F, 0.225F};
    backbone_options.preprocess.normalize.has_explicit_stats = true;
  } else {
    backbone_options.preprocess.preset = neat::NormalizePreset::ImageNet;
  }
  backbone_options.processcvu.pre_run_target = "EV74";
  backbone_options.processcvu.post_run_target = "A65";
  neat::Model backbone(cfg.backbone, backbone_options);

  neat::Model::Options transformer_options;
  transformer_options.preprocess.kind = neat::InputKind::Tensor;
  transformer_options.preprocess.enable = neat::AutoFlag::Off;
  transformer_options.processcvu.pre_run_target = "A65";
  transformer_options.processcvu.post_run_target = "A65";
  neat::Model transformer(cfg.transformer, transformer_options);

  const auto has_specs = [](const auto& specs, const std::vector<std::vector<int64_t>>& shapes) {
    if (specs.size() != shapes.size()) {
      return false;
    }
    for (std::size_t index = 0; index < specs.size(); ++index) {
      if (specs[index].shape != shapes[index] || specs[index].dtypes.size() != 1U ||
          specs[index].dtypes.front() != neat::TensorDType::Float32) {
        return false;
      }
    }
    return true;
  };
  const int side = cfg.feature_size;
  std::vector<std::vector<int64_t>> transformer_outputs = {{1, cfg.top_k, 4},
                                                           {1, cfg.top_k, kNumClasses}};
  if (cfg.task == Task::Segmentation) {
    transformer_outputs.push_back({kMaskSize, kMaskSize, cfg.top_k});
  }
  const auto backbone_inputs = backbone.input_specs();
  const bool valid_contract =
      backbone_inputs.size() == 1U &&
      backbone_inputs.front().shape == std::vector<int64_t>{-1, -1, 3} &&
      backbone_inputs.front().dtypes == std::vector<neat::TensorDType>{neat::TensorDType::UInt8} &&
      has_specs(backbone.output_specs(),
                {{1, side, side, 256}, {1, side * side}, {1, side * side, 4}}) &&
      has_specs(transformer.input_specs(), {{side, side, 256}, {1, cfg.top_k, 4}}) &&
      has_specs(transformer.output_specs(), transformer_outputs);
  sima_examples::require(valid_contract,
                         "selected RF-DETR model pair has an unexpected I/O contract");

  neat::nodes::groups::RtspEncodedInputOptions encoded_options;
  encoded_options.url = cfg.rtsp_url;
  encoded_options.codec = rtsp_codec(cfg.codec);
  encoded_options.latency_ms = cfg.latency_ms;
  encoded_options.tcp = cfg.tcp;
  encoded_options.source_fps = geometry.fps;
  if (cfg.codec == SourceCodec::H264) {
    encoded_options.fallback_h264_width = geometry.width;
    encoded_options.fallback_h264_height = geometry.height;
  }
  auto source = neat::nodes::groups::RtspEncodedInput(encoded_options);

  neat::SimaDecodeOptions decode_options;
  decode_options.type = decode_type(cfg.codec);
  decode_options.out_format = neat::FormatTag::NV12;
  decode_options.raw_output = true;
  decode_options.dec_width = geometry.width;
  decode_options.dec_height = geometry.height;
  decode_options.dec_fps = geometry.fps;
  neat::Graph decode("decoder");
  decode.add(neat::nodes::SimaDecode(decode_options));

  auto video_options =
      cfg.codec == SourceCodec::Mjpeg
          ? neat::nodes::groups::VideoSenderOptions::H264RtpUdpFromRaw(
                geometry.width, geometry.height, geometry.fps)
          : neat::nodes::groups::VideoSenderOptions::Passthrough(rtsp_codec(cfg.codec));
  video_options.host = cfg.insight_host;
  video_options.video_port_base = cfg.video_port;
  video_options.channel = 0;
  video_options.async = true;
  auto video = neat::nodes::groups::VideoSender(video_options);

  neat::QueueOptions queue_options;
  queue_options.max_buffers = 1;
  queue_options.overflow_policy = neat::OverflowPolicy::KeepLatest;
  neat::Graph inference("inference");
  inference.add(neat::nodes::Queue(queue_options));
  inference.add(backbone.graph());
  neat::Graph backbone_output("backbone_output");
  backbone_output.add(neat::nodes::Output("backbone", neat::OutputOptions::Latest()));
  inference.add(backbone_output);

  neat::Graph source_graph("rfdetr_source");
  source_graph.connect(source, decode);
  if (cfg.codec == SourceCodec::Mjpeg) {
    source_graph.connect(decode, video);
  } else {
    source_graph.connect(source, video);
  }
  source_graph.connect(decode, inference);

  neat::RunOptions transformer_run_options;
  transformer_run_options.preset = neat::RunPreset::Realtime;
  transformer_run_options.queue_depth = 1;
  transformer_run_options.overflow_policy = neat::OverflowPolicy::Block;
  transformer_run_options.output_memory = neat::OutputMemory::Owned;
  neat::TensorList transformer_seed;
  for (const auto& spec : transformer.input_specs()) {
    transformer_seed.push_back(neat::Tensor::from_vector(
        std::vector<float>(element_count(spec.shape)), spec.shape, neat::TensorMemory::EV74));
  }
  neat::Model::Runner transformer_runner =
      transformer.build(transformer_seed, neat::Model::RouteOptions{}, transformer_run_options);

  neat::RunOptions source_run_options;
  source_run_options.preset = neat::RunPreset::Realtime;
  source_run_options.output_memory = neat::OutputMemory::ZeroCopy;
  source_run_options.advanced.prepare_output_cpu_visible = true;
  neat::Run source_run = source_graph.build(source_run_options);

  neat::MetadataSenderOptions metadata_options;
  metadata_options.host = cfg.insight_host;
  metadata_options.metadata_port_base = cfg.metadata_port;
  metadata_options.channel = 0;
  std::string metadata_error;
  neat::MetadataSender metadata_sender(metadata_options, &metadata_error);
  sima_examples::require(metadata_sender.ok(), metadata_error);
  std::cout << "RF-DETR " << task_name(cfg.task) << " " << cfg.variant << " "
            << source_codec_name(cfg.codec) << ": " << cfg.rtsp_url << " (" << geometry.width << "x"
            << geometry.height << "@" << geometry.fps
            << ") -> Insight video=" << video_options.video_port()
            << " metadata=" << metadata_sender.metadata_port() << "\n";

  const int proposal_count = cfg.feature_size * cfg.feature_size;
  std::string transformer_bridge_error;
  std::mutex identity_mutex;
  std::map<int64_t, int64_t> source_pts;
  std::thread transformer_bridge([&] {
    try {
      while (!g_stop.load()) {
        auto sample = source_run.pull("backbone", 500);
        if (!sample.has_value()) {
          continue;
        }
        const auto outputs = split_backbone(*sample, proposal_count);
        auto gathered = stable_topk_gather(read_floats(outputs.scores),
                                           read_floats(outputs.proposals), cfg.top_k);
        neat::Tensor gathered_tensor =
            neat::Tensor::from_vector(gathered, {1, cfg.top_k, 4}, neat::TensorMemory::EV74);
        neat::Sample transformer_sample;
        transformer_sample.kind = neat::SampleKind::TensorSet;
        transformer_sample.tensors =
            transformer_inputs(transformer, outputs.feature, gathered_tensor, cfg.top_k);
        copy_identity(*sample, transformer_sample);
        {
          std::lock_guard lock(identity_mutex);
          source_pts[identity_key(*sample)] = sample->pts_ns;
          if (source_pts.size() > 8U) {
            source_pts.erase(source_pts.begin());
          }
        }
        if (!transformer_runner.push(transformer_sample)) {
          if (!g_stop.load()) {
            throw std::runtime_error("transformer input closed");
          }
          break;
        }
      }
    } catch (const std::exception& error) {
      if (!g_stop.load()) {
        transformer_bridge_error = error.what();
      }
      g_stop.store(true);
    }
  });

  int processed = 0;
  std::chrono::steady_clock::time_point first_completed_at;
  std::chrono::steady_clock::time_point last_completed_at;
  try {
    while (!g_stop.load() && (cfg.frames == 0 || processed < cfg.frames)) {
      const auto sample = transformer_runner.pull(500);
      if (sample.empty()) {
        continue;
      }
      const auto output = split_transformer(sample, cfg);
      const std::string data =
          cfg.task == Task::Detection
              ? sima_examples::metadata_boxes_data_json(
                    "objects", postprocess(read_floats(output.boxes), read_floats(output.logits),
                                           geometry.width, geometry.height, labels, cfg.min_score,
                                           cfg.max_results, cfg.top_k))
              : segmentation_metadata(output, geometry.width, geometry.height, labels, cfg);
      const int64_t source_frame_id = sample.frame_id;
      int64_t source_pts_ns = sample.pts_ns;
      {
        std::lock_guard lock(identity_mutex);
        const auto found = source_pts.find(identity_key(sample));
        if (found != source_pts.end()) {
          source_pts_ns = found->second;
          source_pts.erase(found);
        }
      }
      const int64_t timestamp_ms = source_pts_ns >= 0 ? source_pts_ns / 1'000'000 : -1;
      const std::string frame_id = source_frame_id >= 0 ? std::to_string(source_frame_id) : "";
      std::string error;
      const char* metadata_type = cfg.task == Task::Detection ? "object-detection" : "segmentation";
      if (!metadata_sender.send_metadata(metadata_type, data, timestamp_ms, frame_id, &error)) {
        std::cerr << "[warn] Insight metadata send failed: " << error << "\n";
      }
      ++processed;
      last_completed_at = std::chrono::steady_clock::now();
      if (processed == 1) {
        first_completed_at = last_completed_at;
      }
    }
  } catch (...) {
    g_stop.store(true);
    source_run.stop();
    transformer_runner.close();
    transformer_bridge.join();
    throw;
  }

  g_stop.store(true);
  source_run.stop();
  transformer_runner.close();
  transformer_bridge.join();
  if (!transformer_bridge_error.empty()) {
    throw std::runtime_error(transformer_bridge_error);
  }
  const double elapsed_seconds =
      processed > 1 ? std::chrono::duration<double>(last_completed_at - first_completed_at).count()
                    : 0.0;
  const double output_fps = elapsed_seconds > 0.0 ? (processed - 1) / elapsed_seconds : 0.0;
  std::cout << "RF-DETR " << task_name(cfg.task) << ": completed=" << processed
            << " output_fps=" << output_fps << "\n";
  return 0;
}

} // namespace

int main(int argc, char** argv) {
  try {
    const auto cli = parse_args(argc, argv);
    const Config cfg = load_config(cli.config_path);
    if (cli.validate_config_only) {
      (void)load_labels(cfg.labels);
      std::cout << "RF-DETR " << task_name(cfg.task) << " configuration is valid\n";
      return 0;
    }
    std::signal(SIGINT, request_stop);
    std::signal(SIGTERM, request_stop);
    return run(cfg);
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << "\n";
    return 1;
  }
}
