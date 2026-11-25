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

// Detects oriented boxes on several aerial RTSP streams with one shared YOLO OBB model. Every
// stream's video and its rotated boxes go to Insight, one channel per stream.

#include "neat.h"
#include "sima_examples/config.h"
#include "sima_examples/example_utils.h"
#include "sima_examples/labels.h"
#include "sima_examples/pull_status.h"
#include "sima_examples/rtsp_probe.h"
#include "sima_examples/stop_signals.h"
#include "utils/oriented_boxes.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
namespace neat = simaai::neat;
namespace groups = simaai::neat::nodes::groups;

namespace {

constexpr int kPullTimeoutMs = 100;
// Ultralytics' IoU for rotated NMS on OBB predictions.
constexpr float kYoloV8ObbNmsIou = 0.7F;

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
// YOLOv8 and YOLO11 share one oriented-box decoder.
const std::map<std::string, neat::BoxDecodeType> kBoxDecodeTypes = {
    {"yolo26", neat::BoxDecodeType::YoloV26Obb},
    {"yolov8", neat::BoxDecodeType::YoloV8Obb},
    {"yolo11", neat::BoxDecodeType::YoloV8Obb},
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
  options.preprocess.kind = neat::InputKind::Image;
  options.preprocess.enable = neat::AutoFlag::On;
  options.preprocess.color_convert.input_format = neat::PreprocessColorFormat::NV12;
  // COCO_YOLO letterboxes, so rotated boxes map back when both axes scale by the same factor.
  options.preprocess.preset = neat::NormalizePreset::COCO_YOLO;
  options.decode_type = config.choice("model.id", kBoxDecodeTypes);
  // YOLO26 OBB runs no NMS and rejects a nonzero threshold. YOLOv8 and YOLO11 OBB need one.
  if (options.decode_type == neat::BoxDecodeType::YoloV8Obb) {
    options.nms_iou_threshold = kYoloV8ObbNmsIou;
  }
  options.score_threshold = static_cast<float>(config.number("inference.min_score"));
  options.top_k = config.integer("inference.max_detections");
  return options;
}

// The model every stream shares: "detector_frame" input -> model -> "detections" output.
neat::Graph detector_graph(neat::Model& model) {
  neat::Graph frames;
  frames.add(neat::nodes::Input("detector_frame", model.input_appsrc_options(false)));

  neat::Graph detections;
  detections.add(neat::nodes::Output("detections"));

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
  options.codec = config.choice("input.codec", kRtspCodecs);
  options.tcp = config.boolean("input.rtsp.tcp");
  options.latency_ms = config.integer("input.rtsp.latency_ms");
  options.source_fps = stream.fps;
  if (options.codec == groups::RtspCodec::H264) {
    options.fallback_h264_width = stream.width;
    options.fallback_h264_height = stream.height;
  }
  return options;
}

// Hardware decode to NV12 frames at the stream's size, ending at "detector_frame".
neat::Graph decoder_graph(const sima_examples::Config& config,
                          const sima_examples::RtspStreamInfo& stream) {
  neat::SimaDecodeOptions decode;
  decode.type = config.choice("input.codec", kDecodeTypes);
  // Names the output buffer "decoder", the default Model::Options::upstream_name.
  decode.decoder_name = "decoder";
  decode.dec_width = stream.width;
  decode.dec_height = stream.height;
  decode.dec_fps = stream.fps;
  neat::Graph graph("decoder");
  graph.add(neat::nodes::SimaDecode(std::move(decode)));
  graph.add(neat::nodes::CapsRaw("NV12", stream.width, stream.height, stream.fps));
  graph.add(neat::nodes::Output("detector_frame"));
  return graph;
}

std::string stream_id(int index) {
  return "stream" + std::to_string(index);
}

// Feed the shared model the latest frames of each stream, tagged with the stream's ID.
neat::GraphLinkOptions detector_link(int index, int streams) {
  neat::GraphLinkOptions link;
  link.policy = neat::GraphLinkPolicy::RealtimeLatestByStream;
  link.stream_id = stream_id(index);
  // Each stream keeps Core's 4 frames in flight. Core's default cap of all streams is
  // min(4 * streams, 8), and skills/neat-application-optimizer/references/tuning.md says to set
  // the cap explicitly.
  link.max_inflight_total = 4 * streams;
  return link;
}

neat::GraphLinkOptions video_link() {
  neat::GraphLinkOptions link;
  link.policy = neat::GraphLinkPolicy::RealtimeLatestByStream;
  return link;
}

// Send the encoded stream to Insight unchanged, on video port base + index.
groups::VideoSenderOptions video_options(const sima_examples::Config& config, int index) {
  // Passthrough(MJPEG) carries RTP/JPEG: each side a multiple of 8, at most 2040 pixels.
  auto options = groups::VideoSenderOptions::Passthrough(config.choice("input.codec", kRtspCodecs));
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

neat::RunOptions run_options() {
  neat::RunOptions options;
  options.preset = neat::RunPreset::Realtime;
  // Auto copies a result that waits in the output queue unless it is in DMA-BUF or framework
  // memory. ZeroCopy never copies.
  options.output_memory = neat::OutputMemory::ZeroCopy;
  return options;
}

// The oriented boxes in a detections sample, already in source-frame pixels.
neat::OrientedDecodeTensors decode_boxes(const neat::Sample& sample) {
  // Core candidate: decode_obb_tensor takes a Tensor, so each example picks the tensor out of the
  // Sample itself.
  if (sample.kind == neat::SampleKind::TensorSet && !sample.tensors.empty()) {
    return neat::decode_obb_tensor(sample.tensors.front());
  }
  if (sample.kind == neat::SampleKind::Tensor && sample.tensor.has_value()) {
    return neat::decode_obb_tensor(*sample.tensor);
  }
  throw std::runtime_error("expected a BBOX_OBB_V1 tensor in the detections sample");
}

// Send every frame's boxes to its stream's Insight channel until each stream sent `frames`
// (0: no limit) or a stop signal; return each stream's count.
std::vector<int> send_detections(neat::Run& run,
                                 std::vector<neat::MetadataSender>& metadata_senders,
                                 const std::vector<std::string>& labels, int frames,
                                 const std::atomic<bool>& stop) {
  std::vector<int> processed(metadata_senders.size(), 0);
  while (!stop && (frames == 0 || *std::min_element(processed.begin(), processed.end()) < frames)) {
    neat::Sample sample;
    neat::PullError error;
    const auto status = run.pull("detections", kPullTimeoutMs, sample, &error);
    if (!sima_examples::pull_status_has_sample(status, "detections", error, run.last_error())) {
      continue;
    }

    // With one source, Core names the sample after the decoder instead of the stream.
    std::size_t index = 0;
    if (metadata_senders.size() > 1) {
      index = std::stoul(sample.stream_id.substr(std::string("stream").size()));
    }
    if (frames > 0 && processed.at(index) >= frames) {
      continue;
    }
    const std::string data = obb::detections_json(decode_boxes(sample), labels);
    const std::int64_t timestamp_ms = sample.pts_ns >= 0 ? sample.pts_ns / 1'000'000 : -1;
    const std::string frame_id = sample.frame_id >= 0 ? std::to_string(sample.frame_id) : "";
    // A nonblocking send drops the message while the socket buffer is full; the drops are
    // counted in the sender's stats and reported when the run ends. Any other failure throws.
    std::string send_error;
    if (!metadata_senders.at(index).send_metadata("object-detection", data, timestamp_ms, frame_id,
                                                  &send_error) &&
        !send_error.empty()) {
      throw std::runtime_error("metadata send failed: " + send_error);
    }
    ++processed[index];
  }
  return processed;
}

int run_app(const fs::path& config_path) {
  const auto config = sima_examples::Config::load(config_path);
  const auto sources = config.string_list("input.source");
  // Each stream's frame size and rate: the configured values, and a probe's for each 0.
  const auto streams = sima_examples::probe_rtsp_streams(
      sources, config.boolean("input.rtsp.tcp"),
      {config.integer("input.width"), config.integer("input.height"), config.integer("input.fps")});
  const auto labels = sima_examples::load_labels(config.string("model.labels"));
  const fs::path model_path = fs::absolute(config_path).parent_path() / config.string("model.path");
  neat::Model model(model_path.string(), model_options(config));
  const neat::Graph detector = detector_graph(model);

  neat::Graph graph;
  std::vector<neat::MetadataSender> metadata_senders;
  for (int index = 0; index < static_cast<int>(sources.size()); ++index) {
    const std::string& url = sources[static_cast<std::size_t>(index)];
    const auto& stream = streams[static_cast<std::size_t>(index)];
    auto rtsp = groups::RtspEncodedInput(rtsp_options(config, url, stream));
    const neat::Graph decoder = decoder_graph(config, stream);
    // A plain link keeps every encoded frame. A dropped one corrupts the decoded frames after it
    // until the next keyframe.
    graph.connect(rtsp, decoder);
    graph.connect(decoder, detector, detector_link(index, static_cast<int>(sources.size())));
    graph.connect(rtsp, groups::VideoSender(video_options(config, index)), video_link());
    std::string error;
    metadata_senders.emplace_back(metadata_options(config, index), &error);
    sima_examples::require(metadata_senders.back().ok(), error);
    std::cout << "[stream " << index << "] rtsp=" << url << " stream=" << stream.width << "x"
              << stream.height << "@" << stream.fps
              << " insight=" << config.string("output.insight.host")
              << " video=" << config.integer("output.insight.video_port_base") + index
              << " metadata=" << metadata_senders.back().metadata_port() << "\n";
  }

  const std::atomic<bool>& stop = sima_examples::stop_on_signals();
  // The Run's destructor closes it when send_detections throws.
  neat::Run run = graph.build(run_options());
  const std::vector<int> processed =
      send_detections(run, metadata_senders, labels, config.integer("inference.frames"), stop);
  run.close();

  // Core candidate: Core forwards fd 1 through a detached thread (start_stdio_filter) that exit()
  // can stop before it writes the last lines. std::endl puts each line into the pipe earlier,
  // which narrows the race but does not close it.
  for (std::size_t index = 0; index < metadata_senders.size(); ++index) {
    std::cout << "[stream " << index << "] processed=" << processed[index] << std::endl;
    const auto failures = metadata_senders[index].stats().send_failures;
    if (failures > 0) {
      std::cout << "[stream " << index << "] " << failures << " metadata messages dropped"
                << std::endl;
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
