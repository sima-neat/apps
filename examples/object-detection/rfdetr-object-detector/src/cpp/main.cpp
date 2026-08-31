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

#include <algorithm>
#include <atomic>
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
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
namespace neat = simaai::neat;

namespace {

constexpr int kTopK = 300;
std::atomic<bool> g_stop{false};

void request_stop(int) {
  g_stop.store(true);
}

enum class SourceCodec { H264, H265 };

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
  throw std::runtime_error("source.codec must be h264/avc or h265/hevc");
}

const char* source_codec_name(SourceCodec codec) {
  return codec == SourceCodec::H264 ? "h264" : "h265";
}

SourceGeometry resolve_geometry(const SourceGeometry& probed, const SourceGeometry& fallback) {
  return {
      probed.width > 0 ? probed.width : fallback.width,
      probed.height > 0 ? probed.height : fallback.height,
      probed.fps > 0 ? probed.fps : fallback.fps,
  };
}

struct Config {
  std::string variant;
  std::string backbone;
  std::string transformer;
  int input_size = 0;
  fs::path labels;
  std::string rtsp_url;
  SourceCodec codec = SourceCodec::H264;
  bool tcp = true;
  int latency_ms = 100;
  int fallback_width = 0;
  int fallback_height = 0;
  int fallback_fps = 0;
  int frames = 0;
  float min_score = 0.5F;
  int max_detections = 100;
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
  cfg.variant = lower_copy(raw.string_or("model.variant", "small"));
  if (cfg.variant != "small" && cfg.variant != "medium") {
    throw std::runtime_error("model.variant must be small or medium");
  }
  const std::string prefix = "model." + cfg.variant + ".";
  cfg.backbone = raw.string_or(prefix + "backbone", "");
  cfg.transformer = raw.string_or(prefix + "transformer", "");
  cfg.input_size = raw.int_or(prefix + "input_size", 0);
  cfg.labels = raw.string_or("model.labels", "");
  cfg.rtsp_url = raw.string_or("source.rtsp_url", "");
  cfg.codec = parse_source_codec(raw.string_or("source.codec", "h264"));
  cfg.tcp = raw.bool_or("source.tcp", true);
  cfg.latency_ms = raw.int_or("source.latency_ms", 100);
  cfg.fallback_width = raw.int_or("source.width", 0);
  cfg.fallback_height = raw.int_or("source.height", 0);
  cfg.fallback_fps = raw.int_or("source.fps", 0);
  cfg.frames = raw.int_or("inference.frames", 0);
  cfg.min_score = static_cast<float>(raw.double_or("inference.min_score", 0.5));
  cfg.max_detections = raw.int_or("inference.max_detections", 100);
  cfg.insight_host = raw.string_or("output.insight.host", "");
  cfg.video_port = raw.int_or("output.insight.video_port", 9000);
  cfg.metadata_port = raw.int_or("output.insight.metadata_port", 9100);

  sima_examples::require(!cfg.backbone.empty() && !cfg.transformer.empty(),
                         prefix + "backbone and transformer must be set");
  sima_examples::require(cfg.input_size > 0 && cfg.input_size % 16 == 0,
                         prefix + "input_size must be a positive multiple of 16");
  sima_examples::require(!cfg.labels.empty(), "model.labels must be set");
  sima_examples::require(cfg.rtsp_url.rfind("rtsp://", 0) == 0,
                         "source.rtsp_url must be an RTSP URL");
  sima_examples::require(cfg.latency_ms >= 0 && cfg.frames >= 0,
                         "source.latency_ms and inference.frames must be >= 0");
  sima_examples::require(cfg.fallback_width >= 0 && cfg.fallback_height >= 0 &&
                             cfg.fallback_fps >= 0,
                         "source.width, source.height, and source.fps must be >= 0");
  sima_examples::require(cfg.min_score >= 0.0F && cfg.min_score <= 1.0F,
                         "inference.min_score must be in [0, 1]");
  sima_examples::require(cfg.max_detections > 0, "inference.max_detections must be > 0");
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

std::pair<std::vector<float>, std::vector<int>>
stable_topk_gather(const std::vector<float>& scores, const std::vector<float>& proposals) {
  if (scores.size() < static_cast<std::size_t>(kTopK) || proposals.size() != scores.size() * 4U) {
    throw std::runtime_error("backbone score and proposal shapes do not match");
  }
  if (!std::all_of(scores.begin(), scores.end(),
                   [](float value) { return std::isfinite(value); }) ||
      !std::all_of(proposals.begin(), proposals.end(),
                   [](float value) { return std::isfinite(value); })) {
    throw std::runtime_error("backbone output contains non-finite values");
  }
  std::vector<int> indices(scores.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::stable_sort(indices.begin(), indices.end(), [&scores](int left, int right) {
    return scores[static_cast<std::size_t>(left)] > scores[static_cast<std::size_t>(right)];
  });
  indices.resize(kTopK);
  std::vector<float> gathered(static_cast<std::size_t>(kTopK) * 4U);
  for (int output = 0; output < kTopK; ++output) {
    const auto source = static_cast<std::size_t>(indices[static_cast<std::size_t>(output)]) * 4U;
    std::copy_n(proposals.begin() + static_cast<std::ptrdiff_t>(source), 4,
                gathered.begin() + static_cast<std::ptrdiff_t>(output * 4));
  }
  return {std::move(gathered), std::move(indices)};
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

std::pair<neat::Tensor, neat::Tensor> split_transformer(const neat::Sample& sample) {
  neat::Tensor boxes;
  neat::Tensor logits;
  bool have_boxes = false;
  bool have_logits = false;
  for (const auto& tensor : collect_tensors(sample)) {
    const auto elements = element_count(tensor.shape);
    if (elements == static_cast<std::size_t>(kTopK) * 4U) {
      boxes = tensor;
      have_boxes = true;
    } else if (elements == static_cast<std::size_t>(kTopK) * 91U) {
      logits = tensor;
      have_logits = true;
    }
  }
  if (!have_boxes || !have_logits) {
    throw std::runtime_error("transformer did not produce box and class tensors");
  }
  return {std::move(boxes), std::move(logits)};
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
                                    const neat::Tensor& gathered) {
  neat::TensorList inputs;
  for (const auto& spec : model.input_specs()) {
    neat::Tensor tensor =
        element_count(spec.shape) == static_cast<std::size_t>(kTopK) * 4U ? gathered : feature;
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

std::vector<sima_examples::MetadataBox> postprocess(const std::vector<float>& boxes,
                                                    const std::vector<float>& logits,
                                                    int frame_width, int frame_height,
                                                    const std::vector<std::string>& labels,
                                                    float min_score, int max_detections) {
  if (boxes.size() != static_cast<std::size_t>(kTopK) * 4U ||
      logits.size() != static_cast<std::size_t>(kTopK) * 91U) {
    throw std::runtime_error("unexpected transformer output shape");
  }
  std::vector<float> probabilities(logits.size());
  std::transform(logits.begin(), logits.end(), probabilities.begin(), [](float value) {
    value = std::clamp(value, -80.0F, 80.0F);
    return 1.0F / (1.0F + std::exp(-value));
  });
  std::vector<int> ranking(probabilities.size());
  std::iota(ranking.begin(), ranking.end(), 0);
  std::stable_sort(ranking.begin(), ranking.end(), [&probabilities](int left, int right) {
    return probabilities[static_cast<std::size_t>(left)] >
           probabilities[static_cast<std::size_t>(right)];
  });

  std::vector<sima_examples::MetadataBox> objects;
  objects.reserve(static_cast<std::size_t>(max_detections));
  for (const int flat_index : ranking) {
    const float score = probabilities[static_cast<std::size_t>(flat_index)];
    if (score < min_score || objects.size() >= static_cast<std::size_t>(max_detections)) {
      break;
    }
    const int query = flat_index / 91;
    const int class_id = flat_index % 91;
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

SourceGeometry probe_source_geometry(const Config& cfg) {
  sima_examples::RtspStreamInfo stream;
  sima_examples::RtspProbeOptions options;
  options.latency_ms = cfg.latency_ms;
  options.rtsp_tcp = cfg.tcp;
  (void)sima_examples::probe_rtsp_stream_info(cfg.rtsp_url, options, stream);

  const SourceGeometry geometry =
      resolve_geometry({stream.width, stream.height, stream.fps},
                       {cfg.fallback_width, cfg.fallback_height, cfg.fallback_fps});
  sima_examples::require(
      geometry.width > 0 && geometry.height > 0 && geometry.fps > 0,
      "failed to resolve RTSP width, height, and FPS; set source fallbacks if probing fails");
  return geometry;
}

neat::nodes::groups::RtspCodec rtsp_codec(SourceCodec codec) {
  return codec == SourceCodec::H264 ? neat::nodes::groups::RtspCodec::H264
                                    : neat::nodes::groups::RtspCodec::H265;
}

neat::SimaDecodeType decode_type(SourceCodec codec) {
  return codec == SourceCodec::H264 ? neat::SimaDecodeType::H264 : neat::SimaDecodeType::H265;
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
  backbone_options.preprocess.preset = neat::NormalizePreset::ImageNet;
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
  const int side = cfg.input_size / 16;
  const auto backbone_inputs = backbone.input_specs();
  const bool valid_contract =
      backbone_inputs.size() == 1U &&
      backbone_inputs.front().shape == std::vector<int64_t>{-1, -1, 3} &&
      backbone_inputs.front().dtypes == std::vector<neat::TensorDType>{neat::TensorDType::UInt8} &&
      has_specs(backbone.output_specs(),
                {{1, side, side, 256}, {1, side * side}, {1, side * side, 4}}) &&
      has_specs(transformer.input_specs(), {{side, side, 256}, {1, kTopK, 4}}) &&
      has_specs(transformer.output_specs(), {{1, kTopK, 4}, {1, kTopK, 91}});
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
  decode.add(neat::nodes::CapsRaw("NV12", geometry.width, geometry.height, geometry.fps,
                                  neat::CapsMemory::Any));

  auto video_options = neat::nodes::groups::VideoSenderOptions::Passthrough(rtsp_codec(cfg.codec));
  video_options.host = cfg.insight_host;
  video_options.video_port_base = cfg.video_port;
  video_options.channel = 0;
  video_options.async = true;
  auto video = neat::nodes::groups::VideoSender(video_options);

  neat::Graph backbone_graph = backbone.graph();
  neat::Graph backbone_output("backbone_output");
  backbone_output.add(neat::nodes::Output("backbone", neat::OutputOptions::EveryFrame(2)));

  neat::GraphLinkOptions link;
  link.policy = neat::GraphLinkPolicy::RealtimeLatestByStream;
  link.max_inflight_per_stream = 2;
  link.stream_id = "stream0";
  neat::Graph graph("rfdetr");
  graph.connect(source, decode);
  graph.connect(source, video, link);
  graph.connect(decode, backbone_graph, link);
  graph.connect(backbone_graph, backbone_output);

  neat::RunOptions transformer_run_options;
  transformer_run_options.preset = neat::RunPreset::Realtime;
  transformer_run_options.queue_depth = 2;
  transformer_run_options.overflow_policy = neat::OverflowPolicy::Block;
  transformer_run_options.output_memory = neat::OutputMemory::Owned;
  neat::TensorList transformer_seed;
  for (const auto& spec : transformer.input_specs()) {
    transformer_seed.push_back(neat::Tensor::from_vector(
        std::vector<float>(element_count(spec.shape)), spec.shape, neat::TensorMemory::EV74));
  }
  neat::Model::Runner transformer_runner =
      transformer.build(transformer_seed, neat::Model::RouteOptions{}, transformer_run_options);

  neat::RunOptions source_options;
  source_options.preset = neat::RunPreset::Realtime;
  source_options.queue_depth = 3;
  source_options.overflow_policy = neat::OverflowPolicy::KeepLatest;
  source_options.output_memory = neat::OutputMemory::ZeroCopy;
  source_options.advanced.prepare_output_cpu_visible = true;
  neat::Run source_run = graph.build(source_options);

  neat::MetadataSenderOptions metadata_options;
  metadata_options.host = cfg.insight_host;
  metadata_options.metadata_port_base = cfg.metadata_port;
  metadata_options.channel = 0;
  std::string metadata_error;
  neat::MetadataSender metadata_sender(metadata_options, &metadata_error);
  sima_examples::require(metadata_sender.ok(), metadata_error);
  std::cout << "RF-DETR " << cfg.variant << " " << source_codec_name(cfg.codec) << ": "
            << cfg.rtsp_url << " (" << geometry.width << "x" << geometry.height << "@"
            << geometry.fps << ") -> Insight video=" << video_options.video_port()
            << " metadata=" << metadata_sender.metadata_port() << "\n";

  const int proposal_count = (cfg.input_size / 16) * (cfg.input_size / 16);
  std::string bridge_error;
  std::mutex identity_mutex;
  std::map<int64_t, int64_t> source_pts;
  std::thread bridge([&] {
    try {
      while (!g_stop.load()) {
        auto sample = source_run.pull("backbone", 500);
        if (!sample.has_value()) {
          continue;
        }
        const auto outputs = split_backbone(*sample, proposal_count);
        auto [gathered, indices] =
            stable_topk_gather(read_floats(outputs.scores), read_floats(outputs.proposals));
        (void)indices;
        neat::Tensor gathered_tensor =
            neat::Tensor::from_vector(gathered, {1, kTopK, 4}, neat::TensorMemory::EV74);
        neat::Sample transformer_sample;
        transformer_sample.kind = neat::SampleKind::TensorSet;
        transformer_sample.tensors =
            transformer_inputs(transformer, outputs.feature, gathered_tensor);
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
        bridge_error = error.what();
      }
      g_stop.store(true);
    }
  });

  int processed = 0;
  try {
    while (!g_stop.load() && (cfg.frames == 0 || processed < cfg.frames)) {
      const auto sample = transformer_runner.pull(500);
      if (sample.empty()) {
        continue;
      }
      auto [box_tensor, logit_tensor] = split_transformer(sample);
      const auto objects =
          postprocess(read_floats(box_tensor), read_floats(logit_tensor), geometry.width,
                      geometry.height, labels, cfg.min_score, cfg.max_detections);
      const auto data = sima_examples::metadata_boxes_data_json("objects", objects);
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
      if (!metadata_sender.send_metadata("object-detection", data, timestamp_ms, frame_id,
                                         &error)) {
        std::cerr << "[warn] Insight metadata send failed: " << error << "\n";
      }
      ++processed;
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
  if (!bridge_error.empty()) {
    throw std::runtime_error(bridge_error);
  }
  std::cout << "RF-DETR " << cfg.variant << ": completed " << processed << " detections\n";
  return 0;
}

} // namespace

int main(int argc, char** argv) {
  try {
    const auto cli = parse_args(argc, argv);
    const Config cfg = load_config(cli.config_path);
    if (cli.validate_config_only) {
      (void)load_labels(cfg.labels);
      std::cout << "RF-DETR " << cfg.variant << " configuration is valid\n";
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
