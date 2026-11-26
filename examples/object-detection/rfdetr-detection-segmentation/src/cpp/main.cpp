// Copyright 2026 SiMa Technologies, Inc.
// SPDX-License-Identifier: Apache-2.0

// Detects or segments objects in one RTSP stream with RF-DETR and shows them in Insight. The
// backbone runs in the graph; the host picks its best proposals and hands them, with the
// backbone's features, to the transformer. Insight receives the video and, for every result,
// the detection boxes or the segmentation polygons.

#include "neat.h"
#include "sima_examples/config.h"
#include "sima_examples/example_utils.h"
#include "sima_examples/labels.h"
#include "sima_examples/pull_status.h"
#include "sima_examples/rtsp_probe.h"
#include "sima_examples/stop_signals.h"
#include "utils/postprocess.h"

#include <atomic>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <functional>
#include <map>
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
const std::map<std::string, groups::RtspCodec> kRtspCodecs = {
    {"h264", groups::RtspCodec::H264},
    {"h265", groups::RtspCodec::H265},
    {"mjpeg", groups::RtspCodec::MJPEG},
};
const std::map<std::string, neat::SimaDecodeType> kDecodeTypes = {
    {"h264", neat::SimaDecodeType::H264},
    {"h265", neat::SimaDecodeType::H265},
    {"mjpeg", neat::SimaDecodeType::MJPEG},
};

fs::path parse_args(int argc, char** argv) {
  const fs::path adjacent_config = fs::path(argv[0]).parent_path() / "config.yaml";
  fs::path config_path = fs::exists(adjacent_config)
                             ? adjacent_config
                             : sima_examples::default_config_path(SIMANEAT_APPS_EXAMPLE_SOURCE_DIR);
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--config" && i + 1 < argc) {
      config_path = argv[++i];
    } else {
      throw std::runtime_error("usage: " + std::string(argv[0]) + " [--config <path>]");
    }
  }
  return config_path;
}

// EV74 converts each NV12 frame to RGB, stretches it to the model input, and normalizes it.
neat::Model::Options backbone_options(const sima_examples::RtspStreamInfo& stream) {
  neat::Model::Options options;
  options.preprocess.kind = neat::InputKind::Image;
  options.preprocess.enable = neat::AutoFlag::On;
  options.preprocess.input_max_width = stream.width;
  options.preprocess.input_max_height = stream.height;
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

groups::RtspEncodedInputOptions rtsp_options(const sima_examples::Config& config,
                                             const sima_examples::RtspStreamInfo& stream) {
  groups::RtspEncodedInputOptions options;
  options.url = config.string("input.source");
  options.codec = config.choice("input.codec", kRtspCodecs);
  options.tcp = config.boolean("input.rtsp.tcp");
  options.latency_ms = config.integer("input.rtsp.latency_ms");
  if (options.tcp) {
    // Keep the RTP timestamps; arrival-time corrections drift during high-rate replay.
    options.buffer_mode = "none";
  }
  options.source_fps = stream.fps;
  if (options.codec == groups::RtspCodec::H264) {
    options.fallback_h264_width = stream.width;
    options.fallback_h264_height = stream.height;
  }
  return options;
}

// Hardware decode to NV12 frames at the stream's size.
neat::Graph decoder_graph(const sima_examples::Config& config,
                          const sima_examples::RtspStreamInfo& stream) {
  neat::SimaDecodeOptions decode;
  decode.type = config.choice("input.codec", kDecodeTypes);
  decode.dec_width = stream.width;
  decode.dec_height = stream.height;
  decode.dec_fps = stream.fps;
  neat::Graph graph("decoder");
  graph.add(neat::nodes::SimaDecode(std::move(decode)));
  return graph;
}

// Send the stream to Insight as received.
groups::VideoSenderOptions video_options(const sima_examples::Config& config) {
  auto options = groups::VideoSenderOptions::Passthrough(config.choice("input.codec", kRtspCodecs));
  options.host = config.string("output.insight.host");
  options.video_port_base = config.integer("output.insight.video_port_base");
  return options;
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

neat::MetadataSenderOptions metadata_options(const sima_examples::Config& config) {
  neat::MetadataSenderOptions options;
  options.host = config.string("output.insight.host");
  options.metadata_port_base = config.integer("output.insight.metadata_port_base");
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

rfdetr::PostprocessOptions postprocess_options(const sima_examples::Config& config,
                                               const sima_examples::RtspStreamInfo& stream) {
  rfdetr::PostprocessOptions options;
  options.width = stream.width;
  options.height = stream.height;
  options.min_score = static_cast<float>(config.number("inference.min_score"));
  options.max_detections = config.integer("inference.max_detections");
  options.mask_threshold = static_cast<float>(config.number("inference.mask_threshold"));
  options.mask_grid_size = config.integer("app.mask_grid_size");
  return options;
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
                      const std::vector<int64_t>& proposals_shape, std::atomic<bool>& stop,
                      std::string& error) {
  try {
    while (!stop) {
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
    if (!stop) {
      error = failure.what();
    }
  }
  stop = true;
}

// Send every transformer result to Insight until `frames` (0: no limit) or a stop; return the
// count.
int send_results(neat::Model::Runner& transformer_runner,
                 const neat::MetadataSender& metadata_sender,
                 const std::vector<std::string>& labels, int mask_size,
                 const rfdetr::PostprocessOptions& options, int frames,
                 const std::atomic<bool>& stop) {
  const std::string metadata_type = mask_size > 0 ? "segmentation" : "object-detection";
  int sent = 0;
  while (!stop && (frames == 0 || sent < frames)) {
    const neat::Sample result = transformer_runner.pull(kPullTimeoutMs);
    if (result.empty()) {
      continue;
    }
    // A nonblocking send drops the message while the socket buffer is full; the drops are
    // counted in the sender's stats and reported when the run ends. Any other failure throws.
    std::string send_error;
    if (!metadata_sender.send_metadata(
            metadata_type, result_json(result, labels, mask_size, options),
            result.pts_ns >= 0 ? result.pts_ns / 1'000'000 : -1,
            result.frame_id >= 0 ? std::to_string(result.frame_id) : "", &send_error) &&
        !send_error.empty()) {
      throw std::runtime_error("metadata send failed: " + send_error);
    }
    ++sent;
  }
  return sent;
}

int run_app(const fs::path& config_path) {
  const auto config = sima_examples::Config::load(config_path);
  const auto stream = sima_examples::probe_rtsp_stream_info(
      config.string("input.source"), config.boolean("input.rtsp.tcp"),
      {config.integer("input.width"), config.integer("input.height"), config.integer("input.fps")});
  const std::string model_id = config.string("model.id");
  // A model comes as two packages: the backbone that model.path names, and the decoder, which
  // the code calls the transformer, next to it.
  const fs::path backbone_path =
      fs::absolute(config_path).parent_path() / config.string("model.path");
  std::string transformer_name = backbone_path.filename().string();
  transformer_name.replace(transformer_name.find("_backbone_mpk"), 13, "_decoder_mpk");
  neat::Model backbone(backbone_path.string(), backbone_options(stream));
  neat::Model transformer((backbone_path.parent_path() / transformer_name).string(),
                          transformer_options());
  const std::vector<int64_t> proposals_shape = transformer.input_specs().at(1).shape;
  const int mask_size = model_id.starts_with("seg-")
                            ? static_cast<int>(transformer.output_specs().at(2).shape.at(1))
                            : 0;
  // The model's class IDs index the labels directly.
  const auto labels = sima_examples::load_labels(config.string("model.labels"));

  auto rtsp = groups::RtspEncodedInput(rtsp_options(config, stream));
  const neat::Graph decoder = decoder_graph(config, stream);
  auto video = groups::VideoSender(video_options(config));
  neat::Graph graph;
  graph.connect(rtsp, decoder);
  // A plain link lets Core tee the encoded stream to the decoder and the video sender.
  graph.connect(rtsp, video);
  graph.connect(decoder, backbone_graph(backbone));

  neat::Model::Runner transformer_runner = transformer.build(
      transformer_seed(transformer), neat::Model::RouteOptions{}, transformer_run_options());
  neat::Run source_run = graph.build(run_options());
  std::string metadata_error;
  neat::MetadataSender metadata_sender(metadata_options(config), &metadata_error);
  sima_examples::require(metadata_sender.ok(), metadata_error);
  std::cout << "RF-DETR " << model_id << " " << config.string("input.codec") << ": "
            << config.string("input.source") << " (" << stream.width << "x" << stream.height << "@"
            << stream.fps << ") -> Insight " << config.string("output.insight.host")
            << " video=" << config.integer("output.insight.video_port_base")
            << " metadata=" << config.integer("output.insight.metadata_port_base") << "\n";

  std::atomic<bool>& stop = sima_examples::stop_on_signals();
  std::string feed_error;
  std::thread feeder(feed_transformer, std::ref(source_run), std::ref(transformer_runner),
                     std::cref(proposals_shape), std::ref(stop), std::ref(feed_error));
  int sent = 0;
  std::exception_ptr send_error;
  try {
    sent =
        send_results(transformer_runner, metadata_sender, labels, mask_size,
                     postprocess_options(config, stream), config.integer("inference.frames"), stop);
  } catch (...) {
    send_error = std::current_exception();
  }
  stop = true;
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

  std::cout << "RF-DETR " << model_id << ": sent " << sent << " results to Insight\n";
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
