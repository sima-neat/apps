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

const std::map<std::string, groups::RtspCodec> kRtspCodecs = {
    {"h264", groups::RtspCodec::H264},
    {"h265", groups::RtspCodec::H265},
};
const std::map<std::string, neat::SimaDecodeType> kDecodeTypes = {
    {"h264", neat::SimaDecodeType::H264},
    {"h265", neat::SimaDecodeType::H265},
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

neat::Model::Options model_options(const sima_examples::Config& config) {
  neat::Model::Options options;
  options.verbose = neat::VerboseOptions::quiet();
  options.preprocess.kind = neat::InputKind::Image;
  options.preprocess.enable = neat::AutoFlag::On;
  options.preprocess.color_convert.input_format = neat::PreprocessColorFormat::NV12;
  options.preprocess.preset = neat::NormalizePreset::COCO_YOLO;
  options.decode_type = neat::BoxDecodeType::YoloV26;
  options.score_threshold = config.number("inference.min_score");
  options.nms_iou_threshold = config.number("inference.nms_iou");
  options.top_k = config.integer("inference.max_detections");
  return options;
}

// The model every stream shares: "detector_frame" input -> model -> "detections" output.
neat::Graph detector_graph(neat::Model& model, const sima_examples::Config& config) {
  neat::Graph frames;
  frames.add(neat::nodes::Input("detector_frame", model.input_appsrc_options(false)));

  neat::Graph detections;
  detections.add(neat::nodes::Output(
      "detections", neat::OutputOptions::EveryFrame(config.integer("app.queue_depth"))));

  neat::Graph model_graph = model.graph();
  neat::Graph graph;
  graph.connect(frames, model_graph);
  graph.connect(model_graph, detections);
  return graph;
}

groups::RtspEncodedInputOptions rtsp_options(const sima_examples::Config& config,
                                             const std::string& url,
                                             const sima_examples::RtspStreamInfo& stream) {
  groups::RtspEncodedInputOptions options;
  options.url = url;
  options.codec = kRtspCodecs.at(config.string("input.codec"));
  options.tcp = config.boolean("input.rtsp.tcp");
  options.latency_ms = config.integer("input.rtsp.latency_ms");
  options.source_fps = config.integer("input.fps");
  options.auto_caps_from_stream = options.source_fps == 0;
  if (options.codec == groups::RtspCodec::H264) {
    options.fallback_h264_width = stream.width;
    options.fallback_h264_height = stream.height;
    options.fallback_h264_fps = stream.fps;
  }
  return options;
}

// Hardware decode to NV12 frames at the stream's size, ending at "detector_frame".
neat::Graph decoder_graph(const sima_examples::Config& config,
                          const sima_examples::RtspStreamInfo& stream) {
  neat::SimaDecodeOptions decode;
  decode.type = kDecodeTypes.at(config.string("input.codec"));
  decode.decoder_name = "decoder";
  decode.next_element = "CVU";
  decode.dec_width = stream.width;
  decode.dec_height = stream.height;
  decode.dec_fps = stream.fps;
  decode.num_buffers = config.integer("app.decoder_buffers");
  decode.input_buffers = 2;
  decode.decoder_tuning = "throughput-low-latency";
  decode.memory_opt = true;
  neat::Graph graph("decoder");
  graph.add(neat::nodes::SimaDecode(std::move(decode)));
  graph.add(neat::nodes::CapsRaw("NV12", stream.width, stream.height, config.integer("input.fps"),
                                 neat::CapsMemory::Any));
  graph.add(neat::nodes::Output("detector_frame"));
  return graph;
}

std::string stream_id(int index) {
  return "stream" + std::to_string(index);
}

// Feed the shared model the latest frame of each stream, with bounded frames in flight.
neat::GraphLinkOptions detector_link(const sima_examples::Config& config, int index) {
  neat::GraphLinkOptions link;
  link.policy = neat::GraphLinkPolicy::RealtimeLatestByStream;
  link.stream_id = stream_id(index);
  link.max_inflight_per_stream = config.integer("inference.max_inflight_per_stream");
  link.max_inflight_total = config.integer("inference.max_inflight_total");
  return link;
}

neat::GraphLinkOptions video_link() {
  neat::GraphLinkOptions link;
  link.policy = neat::GraphLinkPolicy::RealtimeLatestByStream;
  return link;
}

// Send the encoded stream to Insight unchanged, on video port base + index.
groups::VideoSenderOptions video_options(const sima_examples::Config& config, int index) {
  auto options =
      groups::VideoSenderOptions::Passthrough(kRtspCodecs.at(config.string("input.codec")));
  options.host = config.string("output.insight.host");
  options.channel = index;
  options.video_port_base = config.integer("output.insight.video_port_base");
  return options;
}

neat::MetadataSenderOptions metadata_options(const sima_examples::Config& config, int index) {
  neat::MetadataSenderOptions options;
  options.host = config.string("output.insight.host");
  options.channel = index;
  options.metadata_port_base = config.integer("output.insight.metadata_port_base");
  return options;
}

neat::GraphOptions graph_options(const sima_examples::Config& config) {
  neat::GraphOptions options;
  options.advanced_execution.internal_queue_depth = config.integer("app.internal_queue_depth");
  return options;
}

neat::RunOptions run_options(const sima_examples::Config& config) {
  neat::RunOptions options;
  options.preset = neat::RunPreset::Realtime;
  options.queue_depth = config.integer("app.queue_depth");
  options.output_memory = neat::OutputMemory::ZeroCopy;
  return options;
}

// Each stream's frame size and rate: the configured values, and a probe's for each 0. Every
// stream has the same size, because the streams share one model input.
std::vector<sima_examples::RtspStreamInfo> stream_info(const sima_examples::Config& config,
                                                       const std::vector<std::string>& sources) {
  const sima_examples::RtspStreamInfo configured{
      config.integer("input.width"), config.integer("input.height"), config.integer("input.fps")};
  const bool tcp = config.boolean("input.rtsp.tcp");
  std::map<std::string, sima_examples::RtspStreamInfo> by_url;
  for (const auto& url : std::set<std::string>(sources.begin(), sources.end())) {
    by_url[url] = sima_examples::probe_rtsp_stream_info(url, tcp, configured);
  }
  // Core gap: a model shared by several streams accepts one input size, and a stream of another
  // size fails the build with "input spec mismatch".
  const auto& first = by_url.begin()->second;
  for (const auto& [url, stream] : by_url) {
    if (stream.width != first.width || stream.height != first.height) {
      throw std::runtime_error(
          "every input.source stream needs the same frame size, because the streams share one "
          "model: " +
          by_url.begin()->first + " is " + std::to_string(first.width) + "x" +
          std::to_string(first.height) + ", " + url + " is " + std::to_string(stream.width) + "x" +
          std::to_string(stream.height));
    }
  }
  std::vector<sima_examples::RtspStreamInfo> streams;
  for (const auto& url : sources) {
    streams.push_back(by_url.at(url));
  }
  return streams;
}

// The boxes in a detections sample, whose first tensor is the BBOX payload.
std::vector<neat::Box> decode_boxes(const neat::Sample& sample,
                                    const sima_examples::RtspStreamInfo& stream,
                                    int max_detections) {
  const neat::Tensor* bbox = nullptr;
  if (sample.kind == neat::SampleKind::TensorSet && !sample.tensors.empty()) {
    bbox = &sample.tensors.front();
  } else if (sample.kind == neat::SampleKind::Tensor && sample.tensor.has_value()) {
    bbox = &*sample.tensor;
  } else {
    throw std::runtime_error("expected a BBOX tensor in the detections sample");
  }
  return neat::decode_bbox_tensor(*bbox, stream.width, stream.height, max_detections,
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
                     const std::vector<sima_examples::RtspStreamInfo>& streams,
                     const std::vector<std::string>& labels, int max_detections,
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
    const auto& stream = streams.at(static_cast<std::size_t>(index));
    const std::string payload = sima_examples::detection_egress::serialize(
        decode_boxes(sample, stream, max_detections), labels, stream.width, stream.height,
        frame_metadata(sample, index));
    // A nonblocking send drops the message while the socket buffer is full; the drops are
    // counted in the sender's stats and reported when the run ends. Any other failure throws.
    std::string send_error;
    if (!metadata_senders.at(static_cast<std::size_t>(index)).send_raw_json(payload, &send_error) &&
        !send_error.empty()) {
      throw std::runtime_error("metadata send failed: " + send_error);
    }
  }
}

int run_app(const fs::path& config_path) {
  const auto config = sima_examples::Config::load(config_path);
  const auto sources = config.string_list("input.source");
  const auto streams = stream_info(config, sources);
  const auto labels = sima_examples::load_labels(config.string("model.labels"));
  const fs::path model_path = fs::absolute(config_path).parent_path() / config.string("model.path");
  neat::Model model(model_path.string(), model_options(config));
  const neat::Graph detector = detector_graph(model, config);

  neat::Graph graph(graph_options(config));
  std::vector<neat::MetadataSender> metadata_senders;
  for (int index = 0; index < static_cast<int>(sources.size()); ++index) {
    const std::string& url = sources[static_cast<std::size_t>(index)];
    const auto& stream = streams[static_cast<std::size_t>(index)];
    auto rtsp = groups::RtspEncodedInput(rtsp_options(config, url, stream));
    const neat::Graph decoder = decoder_graph(config, stream);
    graph.connect(rtsp, decoder);
    graph.connect(decoder, detector, detector_link(config, index));
    if (config.boolean("output.video_enabled")) {
      graph.connect(rtsp, groups::VideoSender(video_options(config, index)), video_link());
    }
    std::string error;
    metadata_senders.emplace_back(metadata_options(config, index), &error);
    sima_examples::require(metadata_senders.back().ok(), error);
  }
  std::cout << "streams=" << sources.size() << " insight=" << config.string("output.insight.host")
            << " video_ports=" << config.integer("output.insight.video_port_base")
            << "+ metadata_ports=" << config.integer("output.insight.metadata_port_base") << "+\n";

  const std::atomic<bool>& stop = sima_examples::stop_on_signals();
  // The Run's destructor closes it when send_detections throws.
  neat::Run run = graph.build(run_options(config));
  send_detections(run, metadata_senders, streams, labels,
                  config.integer("inference.max_detections"), stop);
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
