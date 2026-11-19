// Copyright 2026 SiMa Technologies, Inc.
// SPDX-License-Identifier: Apache-2.0

// Detects or segments objects in one RTSP stream with RF-DETR and shows them in Insight. The
// backbone runs in the graph; the host picks its best proposals and hands them, with the
// backbone's features, to the transformer. Insight receives the video and, for every result,
// the detection boxes or the segmentation polygons.

#include "neat.h"
#include "support/cpp/config.h"
#include "support/cpp/example_utils.h"
#include "support/cpp/labels.h"
#include "support/cpp/pull_status.h"
#include "support/cpp/rtsp_probe.h"
#include "utils/postprocess.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <csignal>
#include <exception>
#include <filesystem>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
namespace neat = simaai::neat;
namespace groups = simaai::neat::nodes::groups;

namespace {

constexpr int kPullTimeoutMs = 100;
const std::vector<std::string> kModelIds = {"det-nano", "det-small", "det-medium", "det-large",
                                            "seg-nano", "seg-small", "seg-medium"};

std::atomic<bool> g_stop_requested{false};

void request_stop(int) {
  g_stop_requested = true;
}

struct AppConfig {
  std::string model_id;
  bool segmentation = false;
  fs::path model_path;
  fs::path transformer_path;
  fs::path labels_path;
  std::string source;
  std::string codec;
  int width = 0;
  int height = 0;
  int fps = 0; // 0 reads the stream's frame rate with a probe.
  bool rtsp_tcp = true;
  int rtsp_latency_ms = 0;
  int frames = 0; // 0 runs until Ctrl+C.
  double min_score = 0.0;
  int max_detections = 0;
  std::string insight_host;
  int video_port_base = 0;
  int metadata_port_base = 0;
  double mask_threshold = 0.0;
  int mask_grid_size = 0;
};

struct CliOptions {
  fs::path config_path;
  bool validate_config_only = false;
};

CliOptions parse_args(int argc, char** argv) {
  CliOptions options;
  const fs::path adjacent_config = fs::path(argv[0]).parent_path() / "config.yaml";
  options.config_path = fs::exists(adjacent_config)
                            ? adjacent_config
                            : sima_examples::default_config_path(SIMANEAT_APPS_EXAMPLE_SOURCE_DIR);
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

// The decoder package next to the backbone: _backbone_mpk becomes _decoder_mpk.
fs::path transformer_path(const fs::path& model_path) {
  const std::string backbone = "_backbone_mpk";
  std::string name = model_path.filename().string();
  const auto at = name.find(backbone);
  if (at == std::string::npos) {
    throw std::runtime_error(
        "model.path must name an RF-DETR backbone package (*_backbone_mpk.tar.gz): " +
        model_path.string());
  }
  return model_path.parent_path() / name.replace(at, backbone.size(), "_decoder_mpk");
}

AppConfig load_app_config(const fs::path& path) {
  const auto config = sima_examples::Config::load(path);
  AppConfig cfg;
  cfg.model_id = config.string("model.id");
  cfg.segmentation = cfg.model_id.starts_with("seg-");
  cfg.model_path =
      (fs::absolute(path).parent_path() / config.string("model.path")).lexically_normal();
  cfg.transformer_path = transformer_path(cfg.model_path);
  cfg.labels_path = config.string("model.labels");
  cfg.source = config.string("input.source");
  cfg.codec = config.string("input.codec");
  cfg.width = config.integer("input.width");
  cfg.height = config.integer("input.height");
  cfg.fps = config.integer("input.fps");
  cfg.rtsp_tcp = config.boolean("input.rtsp.tcp");
  cfg.rtsp_latency_ms = config.integer("input.rtsp.latency_ms");
  cfg.frames = config.integer("inference.frames");
  cfg.min_score = config.number("inference.min_score");
  cfg.max_detections = config.integer("inference.max_detections");
  cfg.insight_host = config.string("output.insight.host");
  cfg.video_port_base = config.integer("output.insight.video_port_base");
  cfg.metadata_port_base = config.integer("output.insight.metadata_port_base");
  cfg.mask_threshold = config.number("app.mask_threshold");
  cfg.mask_grid_size = config.integer("app.mask_grid_size");
  config.reject_unread_keys();
  if (std::find(kModelIds.begin(), kModelIds.end(), cfg.model_id) == kModelIds.end()) {
    throw std::runtime_error("model.id must be one of det-nano, det-small, det-medium, "
                             "det-large, seg-nano, seg-small, seg-medium");
  }
  if (cfg.codec != "h264" && cfg.codec != "h265" && cfg.codec != "mjpeg") {
    throw std::runtime_error("input.codec must be h264, h265, or mjpeg");
  }
  return cfg;
}

groups::RtspCodec rtsp_codec(const AppConfig& cfg) {
  if (cfg.codec == "h265") {
    return groups::RtspCodec::H265;
  }
  return cfg.codec == "mjpeg" ? groups::RtspCodec::MJPEG : groups::RtspCodec::H264;
}

neat::SimaDecodeType decode_type(const AppConfig& cfg) {
  if (cfg.codec == "h265") {
    return neat::SimaDecodeType::H265;
  }
  return cfg.codec == "mjpeg" ? neat::SimaDecodeType::MJPEG : neat::SimaDecodeType::H264;
}

// EV74 converts each NV12 frame to RGB, stretches it to the model input, and normalizes it.
neat::Model::Options backbone_options(const AppConfig& cfg) {
  neat::Model::Options options;
  options.preprocess.kind = neat::InputKind::Image;
  options.preprocess.enable = neat::AutoFlag::On;
  options.preprocess.input_max_width = cfg.width;
  options.preprocess.input_max_height = cfg.height;
  options.preprocess.input_max_depth = 3;
  options.preprocess.resize.enable = neat::AutoFlag::On;
  options.preprocess.resize.mode = neat::ResizeMode::Stretch;
  options.preprocess.color_convert.enable = neat::AutoFlag::On;
  options.preprocess.color_convert.input_format = neat::PreprocessColorFormat::NV12;
  options.preprocess.color_convert.output_format = neat::PreprocessColorFormat::RGB;
  options.preprocess.preset = neat::NormalizePreset::ImageNet;
  options.processcvu.pre_run_target = "EV74";
  options.processcvu.post_run_target = "A65";
  return options;
}

neat::Model::Options transformer_options() {
  neat::Model::Options options;
  options.processcvu.pre_run_target = "A65";
  options.processcvu.post_run_target = "A65";
  return options;
}

groups::RtspEncodedInputOptions rtsp_options(const AppConfig& cfg, int fps) {
  groups::RtspEncodedInputOptions options;
  options.url = cfg.source;
  options.codec = rtsp_codec(cfg);
  options.tcp = cfg.rtsp_tcp;
  options.latency_ms = cfg.rtsp_latency_ms;
  if (cfg.rtsp_tcp) {
    // Keep the RTP timestamps; arrival-time corrections drift during high-rate replay.
    options.buffer_mode = "none";
  }
  options.source_fps = fps;
  if (cfg.codec == "h264") {
    options.fallback_h264_width = cfg.width;
    options.fallback_h264_height = cfg.height;
  }
  return options;
}

// Hardware decode to NV12 frames at the stream's size.
neat::Graph decoder_graph(const AppConfig& cfg, int fps) {
  neat::SimaDecodeOptions decode;
  decode.type = decode_type(cfg);
  decode.dec_width = cfg.width;
  decode.dec_height = cfg.height;
  decode.dec_fps = fps;
  if (cfg.codec == "mjpeg") {
    // Leaves free decode surfaces while inference and the video sender hold frames.
    decode.num_buffers = 32;
  }
  neat::Graph graph("decoder");
  graph.add(neat::nodes::SimaDecode(std::move(decode)));
  return graph;
}

// Send H.264 and H.265 to Insight unchanged; re-encode decoded MJPEG frames as H.264.
groups::VideoSenderOptions video_options(const AppConfig& cfg, int fps) {
  auto options = cfg.codec == "mjpeg" ? groups::VideoSenderOptions::FromRaw({.fps = fps})
                                      : groups::VideoSenderOptions::Passthrough(rtsp_codec(cfg));
  options.host = cfg.insight_host;
  options.video_port_base = cfg.video_port_base;
  return options;
}

// Bound the decoded frames the MJPEG video sender holds, so it cannot starve the decoder.
neat::GraphLinkOptions video_link() {
  neat::GraphLinkOptions link;
  link.policy = neat::GraphLinkPolicy::RealtimeLatestByStream;
  link.max_inflight_per_stream = 2;
  link.max_inflight_total = 2;
  return link;
}

// Newest decoded frames -> backbone -> "backbone" output. The queue replaces its oldest frame
// instead of slowing the decoder when inference lags.
neat::Graph backbone_graph(neat::Model& backbone) {
  neat::QueueOptions newest_frames;
  newest_frames.max_buffers = 4;
  newest_frames.overflow_policy = neat::OverflowPolicy::KeepLatest;
  neat::Graph graph("inference");
  graph.add(neat::nodes::Queue(newest_frames));
  graph.add(backbone.graph());
  graph.add(neat::nodes::Output("backbone", neat::OutputOptions::EveryFrame(8)));
  return graph;
}

neat::MetadataSenderOptions metadata_options(const AppConfig& cfg) {
  neat::MetadataSenderOptions options;
  options.host = cfg.insight_host;
  options.metadata_port_base = cfg.metadata_port_base;
  return options;
}

neat::RunOptions run_options() {
  neat::RunOptions options;
  options.preset = neat::RunPreset::Realtime;
  options.output_memory = neat::OutputMemory::ZeroCopy;
  return options;
}

// Copy each result to host memory, so reading it holds none of the transformer's buffers.
neat::RunOptions transformer_run_options() {
  neat::RunOptions options;
  options.output_memory = neat::OutputMemory::Owned;
  return options;
}

// Zero tensors shaped like the transformer's inputs, which the build runs once.
neat::TensorList transformer_seed(neat::Model& transformer) {
  neat::TensorList seed;
  for (const auto& spec : transformer.input_specs()) {
    std::size_t elements = 1;
    for (const auto dimension : spec.shape) {
      elements *= static_cast<std::size_t>(dimension);
    }
    seed.push_back(neat::Tensor::from_vector(std::vector<float>(elements), spec.shape,
                                             neat::TensorMemory::EV74));
  }
  return seed;
}

rfdetr::PostprocessOptions postprocess_options(const AppConfig& cfg) {
  rfdetr::PostprocessOptions options;
  options.width = cfg.width;
  options.height = cfg.height;
  options.min_score = static_cast<float>(cfg.min_score);
  options.max_detections = cfg.max_detections;
  options.mask_threshold = static_cast<float>(cfg.mask_threshold);
  options.mask_grid_size = cfg.mask_grid_size;
  return options;
}

// The stream's frame rate: the configured one, or a probe when input.fps is 0.
int stream_fps(const AppConfig& cfg) {
  if (cfg.fps > 0) {
    return cfg.fps;
  }
  sima_examples::RtspProbeOptions probe;
  probe.rtsp_tcp = cfg.rtsp_tcp;
  sima_examples::RtspStreamInfo info;
  sima_examples::probe_rtsp_stream_info(cfg.source, probe, info);
  sima_examples::require(info.fps > 0, "failed to probe RTSP frame rate: " + cfg.source);
  return info.fps;
}

// The values of a float32 output tensor, copied to host memory. On the DevKit, data_ptr()
// throws "tensor storage is not mappable" for these outputs, while this copy reads them.
std::vector<float> float_values(const neat::Tensor& tensor) {
  std::vector<float> values(tensor.dense_bytes_tight() / sizeof(float));
  sima_examples::require(tensor.copy_dense_bytes_tight_to(reinterpret_cast<uint8_t*>(values.data()),
                                                          values.size() * sizeof(float)),
                         "failed to read an RF-DETR output tensor");
  return values;
}

// The transformer's input for one frame: the backbone features and the best proposals.
neat::Sample transformer_sample(const neat::Sample& backbone_result,
                                const std::vector<int64_t>& proposals_shape) {
  neat::Tensor feature = backbone_result.tensors.at(0);
  if (feature.storage && (feature.storage->kind == neat::StorageKind::CpuOwned ||
                          feature.storage->kind == neat::StorageKind::CpuExternal)) {
    // The transformer reads its features from device memory; a device-backed feature map
    // passes without a copy.
    feature = feature.cvu();
  }
  const auto best = rfdetr::top_proposals(float_values(backbone_result.tensors.at(1)),
                                          float_values(backbone_result.tensors.at(2)),
                                          static_cast<int>(proposals_shape.at(1)));
  neat::Sample sample;
  sample.kind = neat::SampleKind::TensorSet;
  sample.tensors = {feature,
                    neat::Tensor::from_vector(best, proposals_shape, neat::TensorMemory::EV74)};
  sample.frame_id = backbone_result.frame_id;
  sample.stream_id = backbone_result.stream_id;
  sample.stream_label = backbone_result.stream_label;
  sample.input_seq = backbone_result.input_seq;
  sample.orig_input_seq = backbone_result.orig_input_seq;
  sample.pts_ns = backbone_result.pts_ns;
  sample.dts_ns = backbone_result.dts_ns;
  sample.duration_ns = backbone_result.duration_ns;
  sample.attributes = backbone_result.attributes;
  return sample;
}

// The metadata of one transformer result: detection boxes, or segmentation polygons.
std::string result_json(const neat::Sample& result, const std::vector<std::string>& labels,
                        int mask_size, const rfdetr::PostprocessOptions& options) {
  std::vector<std::vector<float>> values;
  for (const auto& tensor : result.tensors) {
    values.push_back(float_values(tensor));
  }
  if (mask_size > 0) {
    return rfdetr::segmentation_json(values.at(0), values.at(1), values.at(2), mask_size, labels,
                                     options);
  }
  return rfdetr::detection_json(values.at(0), values.at(1), labels, options);
}

// Hand every backbone result to the transformer until the run stops. A push waits while the
// transformer's input queue is full, so no backbone result is lost.
void feed_transformer(neat::Run& source_run, neat::Model::Runner& transformer_runner,
                      const std::vector<int64_t>& proposals_shape, std::string& error) {
  try {
    while (!g_stop_requested) {
      neat::Sample sample;
      neat::PullError pull_error;
      const auto status = source_run.pull("backbone", kPullTimeoutMs, sample, &pull_error);
      if (!sima_examples::pull_status_has_sample(status, "backbone", pull_error,
                                                 source_run.last_error())) {
        continue;
      }
      if (!transformer_runner.push(transformer_sample(sample, proposals_shape))) {
        throw std::runtime_error("the transformer rejected its input");
      }
    }
  } catch (const std::exception& failure) {
    if (!g_stop_requested) {
      error = failure.what();
    }
  }
  g_stop_requested = true;
}

// Send every transformer result to Insight until inference.frames or a stop; return the count.
int send_results(neat::Model::Runner& transformer_runner,
                 const neat::MetadataSender& metadata_sender,
                 const std::vector<std::string>& labels, int mask_size, const AppConfig& cfg) {
  const std::string metadata_type = cfg.segmentation ? "segmentation" : "object-detection";
  const auto options = postprocess_options(cfg);
  int sent = 0;
  while (!g_stop_requested && (cfg.frames == 0 || sent < cfg.frames)) {
    const neat::Sample result = transformer_runner.pull(kPullTimeoutMs);
    if (result.empty()) {
      continue;
    }
    // A nonblocking send drops the message while the socket buffer is full; the drops are
    // counted in the sender's stats and reported when the run ends.
    (void)metadata_sender.send_metadata(
        metadata_type, result_json(result, labels, mask_size, options),
        result.pts_ns >= 0 ? result.pts_ns / 1'000'000 : -1,
        result.frame_id >= 0 ? std::to_string(result.frame_id) : "");
    ++sent;
  }
  return sent;
}

int run_app(const CliOptions& cli) {
  if (!fs::exists(cli.config_path)) {
    std::cerr << "Error: config file not found: " << cli.config_path.string() << "\n";
    return 2;
  }
  const AppConfig cfg = load_app_config(cli.config_path);
  if (cli.validate_config_only) {
    std::cout << "Config validated: " << cli.config_path.string() << " (model=" << cfg.model_id
              << ")\n";
    return 0;
  }
  sima_examples::require(fs::exists(cfg.transformer_path),
                         "decoder package not found: " + cfg.transformer_path.string());

  const auto labels = sima_examples::load_labels(cfg.labels_path);
  const int fps = stream_fps(cfg);
  neat::Model backbone(cfg.model_path.string(), backbone_options(cfg));
  neat::Model transformer(cfg.transformer_path.string(), transformer_options());
  const std::vector<int64_t> proposals_shape = transformer.input_specs().at(1).shape;
  const int mask_size =
      cfg.segmentation ? static_cast<int>(transformer.output_specs().at(2).shape.at(1)) : 0;

  auto rtsp = groups::RtspEncodedInput(rtsp_options(cfg, fps));
  const neat::Graph decoder = decoder_graph(cfg, fps);
  auto video = groups::VideoSender(video_options(cfg, fps));
  neat::Graph graph;
  graph.connect(rtsp, decoder);
  if (cfg.codec == "mjpeg") {
    graph.connect(decoder, video, video_link());
  } else {
    // A plain link lets Core tee the encoded stream to the decoder and the video sender.
    graph.connect(rtsp, video);
  }
  graph.connect(decoder, backbone_graph(backbone));

  neat::Model::Runner transformer_runner = transformer.build(
      transformer_seed(transformer), neat::Model::RouteOptions{}, transformer_run_options());
  neat::Run source_run = graph.build(run_options());
  std::string metadata_error;
  neat::MetadataSender metadata_sender(metadata_options(cfg), &metadata_error);
  sima_examples::require(metadata_sender.ok(), metadata_error);
  std::cout << "RF-DETR " << cfg.model_id << " " << cfg.codec << ": " << cfg.source << " ("
            << cfg.width << "x" << cfg.height << "@" << fps << ") -> Insight " << cfg.insight_host
            << " video=" << cfg.video_port_base << " metadata=" << cfg.metadata_port_base << "\n";

  std::signal(SIGINT, request_stop);
  std::string feed_error;
  std::thread feeder(feed_transformer, std::ref(source_run), std::ref(transformer_runner),
                     std::cref(proposals_shape), std::ref(feed_error));
  int sent = 0;
  std::exception_ptr send_error;
  try {
    sent = send_results(transformer_runner, metadata_sender, labels, mask_size, cfg);
  } catch (...) {
    send_error = std::current_exception();
  }
  g_stop_requested = true;
  transformer_runner.close_input(); // Ends a push the feeder may be blocked in.
  feeder.join();
  transformer_runner.close();
  source_run.close();
  if (send_error) {
    std::rethrow_exception(send_error);
  }
  if (!feed_error.empty()) {
    throw std::runtime_error(feed_error);
  }

  std::cout << "RF-DETR " << cfg.model_id << ": sent " << sent << " results to Insight\n";
  const auto failures = metadata_sender.stats().send_failures;
  if (failures > 0) {
    std::cout << failures << " metadata messages dropped\n";
  }
  return 0;
}

} // namespace

int main(int argc, char** argv) {
  try {
    return run_app(parse_args(argc, argv));
  } catch (const std::exception& error) {
    std::cerr << "[ERR] " << error.what() << "\n";
    return 1;
  }
}
