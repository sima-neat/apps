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

// Detects objects on many RTSP streams with one shared YOLO26 model. Every stream's video and
// its detection boxes go to Insight, one channel per stream.

#include "neat.h"
#include "sima_examples/config.h"
#include "sima_examples/detection_egress.h"
#include "sima_examples/example_utils.h"
#include "sima_examples/labels.h"
#include "sima_examples/pull_status.h"
#include "sima_examples/rtsp_probe.h"
#include "sima_examples/stop_signals.h"

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
namespace neat = simaai::neat;
namespace groups = simaai::neat::nodes::groups;

namespace {

constexpr int kPullTimeoutMs = 100;

struct AppConfig {
  fs::path model_path;
  fs::path labels_path;
  std::vector<std::string> sources;
  std::string codec;
  int width = 0;
  int height = 0;
  int fps = 0; // 0 reads each stream's frame rate with a probe.
  bool rtsp_tcp = true;
  int rtsp_latency_ms = 0;
  double min_score = 0.0;
  double nms_iou = 0.0;
  int max_detections = 0;
  int max_inflight_per_stream = 0;
  int max_inflight_total = 0;
  bool video_enabled = true;
  std::string insight_host;
  int video_port_base = 0;
  int metadata_port_base = 0;
  int decoder_buffers = 0;
  int queue_depth = 0;
  int internal_queue_depth = 0;
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

AppConfig load_app_config(const fs::path& path) {
  const auto config = sima_examples::Config::load(path);
  AppConfig cfg;
  cfg.model_path = fs::absolute(path).parent_path() / config.string("model.path");
  cfg.labels_path = config.string("model.labels");
  cfg.sources = config.string_list("input.source");
  cfg.codec = config.string("input.codec");
  cfg.width = config.integer("input.width");
  cfg.height = config.integer("input.height");
  cfg.fps = config.integer("input.fps");
  cfg.rtsp_tcp = config.boolean("input.rtsp.tcp");
  cfg.rtsp_latency_ms = config.integer("input.rtsp.latency_ms");
  cfg.min_score = config.number("inference.min_score");
  cfg.nms_iou = config.number("inference.nms_iou");
  cfg.max_detections = config.integer("inference.max_detections");
  cfg.max_inflight_per_stream = config.integer("inference.max_inflight_per_stream");
  cfg.max_inflight_total = config.integer("inference.max_inflight_total");
  cfg.video_enabled = config.boolean("output.video_enabled");
  cfg.insight_host = config.string("output.insight.host");
  cfg.video_port_base = config.integer("output.insight.video_port_base");
  cfg.metadata_port_base = config.integer("output.insight.metadata_port_base");
  cfg.decoder_buffers = config.integer("app.decoder_buffers");
  cfg.queue_depth = config.integer("app.queue_depth");
  cfg.internal_queue_depth = config.integer("app.internal_queue_depth");
  config.reject_unread_keys();
  if (cfg.codec != "h264" && cfg.codec != "h265") {
    throw std::runtime_error("input.codec must be h264 or h265");
  }
  return cfg;
}

groups::RtspCodec rtsp_codec(const AppConfig& cfg) {
  return cfg.codec == "h265" ? groups::RtspCodec::H265 : groups::RtspCodec::H264;
}

neat::Model::Options model_options(const AppConfig& cfg) {
  neat::Model::Options options;
  options.verbose = neat::VerboseOptions::quiet();
  options.preprocess.kind = neat::InputKind::Image;
  options.preprocess.enable = neat::AutoFlag::On;
  options.preprocess.color_convert.input_format = neat::PreprocessColorFormat::NV12;
  options.preprocess.preset = neat::NormalizePreset::COCO_YOLO;
  options.decode_type = neat::BoxDecodeType::YoloV26;
  options.score_threshold = cfg.min_score;
  options.nms_iou_threshold = cfg.nms_iou;
  options.top_k = cfg.max_detections;
  return options;
}

// The model every stream shares: "detector_frame" input -> model -> "detections" output.
neat::Graph detector_graph(neat::Model& model, const AppConfig& cfg) {
  auto input_options = model.input_appsrc_options(false);
  input_options.max_bytes = static_cast<std::uint64_t>(cfg.width) * cfg.height * 3 / 2 *
                            static_cast<std::uint64_t>(input_options.pool_max_buffers);
  neat::Graph frames;
  frames.add(neat::nodes::Input("detector_frame", input_options));

  neat::Graph detections;
  detections.add(
      neat::nodes::Output("detections", neat::OutputOptions::EveryFrame(cfg.queue_depth)));

  neat::Graph model_graph = model.graph();
  neat::Graph graph;
  graph.connect(frames, model_graph);
  graph.connect(model_graph, detections);
  return graph;
}

groups::RtspEncodedInputOptions rtsp_options(const AppConfig& cfg, const std::string& url,
                                             int fps) {
  groups::RtspEncodedInputOptions options;
  options.url = url;
  options.codec = rtsp_codec(cfg);
  options.tcp = cfg.rtsp_tcp;
  options.latency_ms = cfg.rtsp_latency_ms;
  options.auto_caps_from_stream = cfg.fps == 0;
  options.source_fps = cfg.fps;
  if (cfg.codec == "h264") {
    options.fallback_h264_width = cfg.width;
    options.fallback_h264_height = cfg.height;
    options.fallback_h264_fps = fps;
  }
  return options;
}

// Hardware decode to NV12 frames at the stream's size, ending at "detector_frame".
neat::Graph decoder_graph(const AppConfig& cfg, int fps) {
  neat::SimaDecodeOptions decode;
  decode.type = cfg.codec == "h265" ? neat::SimaDecodeType::H265 : neat::SimaDecodeType::H264;
  decode.decoder_name = "decoder";
  decode.next_element = "CVU";
  decode.dec_width = cfg.width;
  decode.dec_height = cfg.height;
  decode.dec_fps = fps;
  decode.num_buffers = cfg.decoder_buffers;
  decode.input_buffers = 2;
  decode.decoder_tuning = "throughput-low-latency";
  decode.memory_opt = true;
  neat::Graph graph("decoder");
  graph.add(neat::nodes::SimaDecode(std::move(decode)));
  graph.add(neat::nodes::CapsRaw("NV12", cfg.width, cfg.height, cfg.fps, neat::CapsMemory::Any));
  graph.add(neat::nodes::Output("detector_frame"));
  return graph;
}

std::string stream_id(int index) {
  return "stream" + std::to_string(index);
}

// Feed the shared model the latest frame of each stream, with bounded frames in flight.
neat::GraphLinkOptions detector_link(const AppConfig& cfg, int index) {
  neat::GraphLinkOptions link;
  link.policy = neat::GraphLinkPolicy::RealtimeLatestByStream;
  link.stream_id = stream_id(index);
  link.max_inflight_per_stream = cfg.max_inflight_per_stream;
  link.max_inflight_total = cfg.max_inflight_total;
  return link;
}

neat::GraphLinkOptions video_link() {
  neat::GraphLinkOptions link;
  link.policy = neat::GraphLinkPolicy::RealtimeLatestByStream;
  return link;
}

// Send the encoded stream to Insight unchanged, on video port base + index.
groups::VideoSenderOptions video_options(const AppConfig& cfg, int index) {
  auto options = groups::VideoSenderOptions::Passthrough(rtsp_codec(cfg));
  options.host = cfg.insight_host;
  options.channel = index;
  options.video_port_base = cfg.video_port_base;
  return options;
}

neat::MetadataSenderOptions metadata_options(const AppConfig& cfg, int index) {
  neat::MetadataSenderOptions options;
  options.host = cfg.insight_host;
  options.channel = index;
  options.metadata_port_base = cfg.metadata_port_base;
  return options;
}

neat::GraphOptions graph_options(const AppConfig& cfg) {
  neat::GraphOptions options;
  options.advanced_execution.internal_queue_depth = cfg.internal_queue_depth;
  return options;
}

neat::RunOptions run_options(const AppConfig& cfg) {
  neat::RunOptions options;
  options.preset = neat::RunPreset::Realtime;
  options.queue_depth = cfg.queue_depth;
  options.output_memory = neat::OutputMemory::ZeroCopy;
  return options;
}

// Each source URL's frame rate: the configured one, or a probe when input.fps is 0.
std::map<std::string, int> stream_fps(const AppConfig& cfg) {
  std::map<std::string, int> fps_by_url;
  for (const auto& url : std::set<std::string>(cfg.sources.begin(), cfg.sources.end())) {
    fps_by_url[url] =
        cfg.fps > 0 ? cfg.fps : sima_examples::probe_rtsp_stream_info(url, cfg.rtsp_tcp).fps;
  }
  return fps_by_url;
}

// The boxes in a detections sample, whose first tensor is the BBOX payload.
std::vector<neat::Box> decode_boxes(const neat::Sample& sample, const AppConfig& cfg) {
  const neat::Tensor* bbox = nullptr;
  if (sample.kind == neat::SampleKind::TensorSet && !sample.tensors.empty()) {
    bbox = &sample.tensors.front();
  } else if (sample.kind == neat::SampleKind::Tensor && sample.tensor.has_value()) {
    bbox = &*sample.tensor;
  } else {
    throw std::runtime_error("expected a BBOX tensor in the detections sample");
  }
  return neat::decode_bbox_tensor(*bbox, cfg.width, cfg.height, cfg.max_detections,
                                  /*strict=*/true)
      .boxes;
}

sima_examples::detection_egress::FrameMetadata frame_metadata(const neat::Sample& sample,
                                                              int index) {
  sima_examples::detection_egress::FrameMetadata metadata;
  metadata.stream_index = index;
  metadata.stream_id = sample.stream_id;
  metadata.frame_id = sample.frame_id;
  metadata.pts_ns = sample.pts_ns;
  metadata.dts_ns = sample.dts_ns;
  metadata.duration_ns = sample.duration_ns;
  metadata.input_seq = sample.input_seq;
  metadata.orig_input_seq = sample.orig_input_seq;
  return metadata;
}

// Send every detections sample to its stream's Insight channel until a stop signal.
void send_detections(neat::Run& run, std::vector<neat::MetadataSender>& metadata_senders,
                     const std::vector<std::string>& labels, const AppConfig& cfg,
                     const std::atomic<bool>& stop) {
  while (!stop) {
    neat::Sample sample;
    neat::PullError error;
    const auto status = run.pull("detections", kPullTimeoutMs, sample, &error);
    if (!sima_examples::pull_status_has_sample(status, "detections", error, run.last_error())) {
      continue;
    }

    // With one source, Core names the sample after the decoder instead of the stream.
    int index = 0;
    if (metadata_senders.size() > 1) {
      index = std::stoi(sample.stream_id.substr(std::string("stream").size()));
    }
    const std::string payload = sima_examples::detection_egress::serialize(
        decode_boxes(sample, cfg), labels, cfg.width, cfg.height, frame_metadata(sample, index));
    // A nonblocking send drops the message while the socket buffer is full; the drops are
    // counted in the sender's stats and reported when the run ends. Any other failure throws.
    std::string send_error;
    if (!metadata_senders.at(static_cast<std::size_t>(index)).send_raw_json(payload, &send_error) &&
        !send_error.empty()) {
      throw std::runtime_error("metadata send failed: " + send_error);
    }
  }
}

int run_app(const CliOptions& cli) {
  if (!fs::exists(cli.config_path)) {
    std::cerr << "Error: config file not found: " << cli.config_path.string() << "\n";
    return 2;
  }
  const AppConfig cfg = load_app_config(cli.config_path);
  if (cli.validate_config_only) {
    std::cout << "Config validated: " << cli.config_path.string()
              << " (streams=" << cfg.sources.size() << ")\n";
    return 0;
  }

  const auto labels = sima_examples::load_labels(cfg.labels_path);
  const auto fps_by_url = stream_fps(cfg);
  neat::Model model(cfg.model_path.string(), model_options(cfg));
  const neat::Graph detector = detector_graph(model, cfg);

  neat::Graph graph(graph_options(cfg));
  std::vector<neat::MetadataSender> metadata_senders;
  for (int index = 0; index < static_cast<int>(cfg.sources.size()); ++index) {
    const std::string& url = cfg.sources[static_cast<std::size_t>(index)];
    auto rtsp = groups::RtspEncodedInput(rtsp_options(cfg, url, fps_by_url.at(url)));
    const neat::Graph decoder = decoder_graph(cfg, fps_by_url.at(url));
    graph.connect(rtsp, decoder);
    graph.connect(decoder, detector, detector_link(cfg, index));
    if (cfg.video_enabled) {
      graph.connect(rtsp, groups::VideoSender(video_options(cfg, index)), video_link());
    }
    std::string error;
    metadata_senders.emplace_back(metadata_options(cfg, index), &error);
    sima_examples::require(metadata_senders.back().ok(), error);
  }
  std::cout << "streams=" << cfg.sources.size() << " insight=" << cfg.insight_host
            << " video_ports=" << cfg.video_port_base
            << "+ metadata_ports=" << cfg.metadata_port_base << "+\n";

  const std::atomic<bool>& stop = sima_examples::stop_on_signals();
  // The Run's destructor closes it when send_detections throws.
  neat::Run run = graph.build(run_options(cfg));
  send_detections(run, metadata_senders, labels, cfg, stop);
  run.close();

  for (std::size_t index = 0; index < metadata_senders.size(); ++index) {
    const auto failures = metadata_senders[index].stats().send_failures;
    if (failures > 0) {
      std::cout << "stream " << index << ": " << failures << " metadata messages dropped\n";
    }
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
