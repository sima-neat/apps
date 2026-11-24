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

// Tracks tiny drones in up to four RTSP streams with one shared YOLO26 model, shown in Insight.
// Each stream's decoded frames feed the model. Each stream's tracker gives every drone an ID, and
// Insight draws each frame's tracks over that stream's video, one channel per stream.

#include "neat.h"
#include "sima_examples/config.h"
#include "sima_examples/detection_egress.h"
#include "sima_examples/example_utils.h"
#include "sima_examples/pull_status.h"
#include "sima_examples/rtsp_probe.h"
#include "sima_examples/stop_signals.h"
#include "utils/tracker.h"

#include <algorithm>
#include <atomic>
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
const std::map<std::string, groups::RtspCodec> kRtspCodecs = {
    {"h264", groups::RtspCodec::H264},
    {"h265", groups::RtspCodec::H265},
};
const std::map<std::string, neat::SimaDecodeType> kDecodeTypes = {
    {"h264", neat::SimaDecodeType::H264},
    {"h265", neat::SimaDecodeType::H265},
};
// The model's one class.
const std::vector<std::string> kLabels = {"drone"};
// Frames of one stream the shared model holds at once: Core's default, set so that the cap of all
// streams follows from it (skills/neat-application-optimizer/references/tuning.md).
constexpr int kMaxInflightPerStream = 4;

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

// YOLO26 on the decoded NV12 frames of every stream, which all have the size of `stream`.
neat::Model::Options model_options(const sima_examples::Config& config,
                                   const sima_examples::RtspStreamInfo& stream,
                                   int max_detections) {
  neat::Model::Options options;
  options.preprocess.kind = neat::InputKind::Image;
  options.preprocess.enable = neat::AutoFlag::On;
  options.preprocess.color_convert.input_format = neat::PreprocessColorFormat::NV12;
  options.preprocess.input_max_width = stream.width;
  options.preprocess.input_max_height = stream.height;
  options.preprocess.preset = neat::NormalizePreset::COCO_YOLO;
  options.decode_type = neat::BoxDecodeType::YoloV26;
  // The split heads of a single-class YOLO may not give Core the class count, so
  // Model::Options::num_classes (model/Model.h) sets it.
  options.num_classes = static_cast<int>(kLabels.size());
  options.score_threshold = config.number("inference.min_score");
  options.nms_iou_threshold = config.number("inference.nms_iou");
  options.top_k = max_detections;
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
  // With input.fps at 0, Core also takes the H.264 caps from the stream: above 60 fps, caps set
  // from the probed values alone failed to negotiate in some runs. The H.265 path does not read
  // auto_caps_from_stream (RtspEncodedInput.cpp).
  options.auto_caps_from_stream = config.integer("input.fps") == 0;
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
  graph.add(
      neat::nodes::CapsRaw("NV12", stream.width, stream.height, stream.fps, neat::CapsMemory::Any));
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
  link.max_inflight_per_stream = kMaxInflightPerStream;
  // Core's default cap of all streams is min(4 * streams, 8), which would hold back three or four
  // streams.
  link.max_inflight_total = kMaxInflightPerStream * streams;
  return link;
}

// Send the newest encoded frame, so a slow video sender cannot hold back the decoder.
neat::GraphLinkOptions video_link() {
  neat::GraphLinkOptions link;
  link.policy = neat::GraphLinkPolicy::RealtimeLatestByStream;
  return link;
}

// Send the encoded stream to Insight unchanged, on video port base + index.
groups::VideoSenderOptions video_options(const sima_examples::Config& config, int index) {
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
  options.output_memory = neat::OutputMemory::ZeroCopy;
  return options;
}

drone_tracker::TrackerSettings tracker_settings(const sima_examples::Config& config) {
  drone_tracker::TrackerSettings settings;
  settings.high_score_threshold = config.number("app.tracking.high_score_threshold");
  settings.new_track_threshold = config.number("app.tracking.new_track_threshold");
  settings.max_missing_frames = config.integer("app.tracking.max_missing_frames");
  settings.min_confirmed_hits = config.integer("app.tracking.min_confirmed_hits");
  return settings;
}

// Each stream's frame size and rate: the configured values, and a probe's for each 0. Every
// stream has the same size, because the streams share one model input.
std::vector<sima_examples::RtspStreamInfo> stream_info(const sima_examples::Config& config,
                                                       const std::vector<std::string>& sources) {
  // A copy of the high-density example's stream_info, until #793 moves it into
  // sima_examples/rtsp_probe as probe_rtsp_streams.
  const sima_examples::RtspStreamInfo configured{
      config.integer("input.width"), config.integer("input.height"), config.integer("input.fps")};
  const bool tcp = config.boolean("input.rtsp.tcp");
  std::vector<std::pair<std::string, sima_examples::RtspStreamInfo>> by_url;
  for (const auto& url : sources) {
    if (std::none_of(by_url.begin(), by_url.end(),
                     [&](const auto& probed) { return probed.first == url; })) {
      by_url.emplace_back(url, sima_examples::probe_rtsp_stream_info(url, tcp, configured));
    }
  }
  // Core gap: a model shared by several streams accepts one input size, and a stream of another
  // size fails the build with "input spec mismatch".
  const auto& [first_url, first] = by_url.front();
  for (const auto& [url, stream] : by_url) {
    if (stream.width != first.width || stream.height != first.height) {
      throw std::runtime_error(
          "every input.source stream needs the same frame size, because the streams share one "
          "model: " +
          first_url + " is " + std::to_string(first.width) + "x" + std::to_string(first.height) +
          ", " + url + " is " + std::to_string(stream.width) + "x" + std::to_string(stream.height));
    }
  }
  std::vector<sima_examples::RtspStreamInfo> streams;
  for (const auto& url : sources) {
    streams.push_back(std::find_if(by_url.begin(), by_url.end(), [&](const auto& probed) {
                        return probed.first == url;
                      })->second);
  }
  return streams;
}

// The boxes in a detections sample, whose first tensor is the BBOX payload.
std::vector<neat::Box> decode_boxes(const neat::Sample& sample,
                                    const sima_examples::RtspStreamInfo& stream,
                                    int max_detections) {
  // Core candidate: decode_bbox_tensor takes a Tensor, so each example picks the BBOX tensor out
  // of the Sample itself.
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

// Track every frame's boxes and send the tracks to its stream's Insight channel until each stream
// sent `frames` (0: no limit) or a stop signal; return each stream's count.
std::vector<int> send_tracks(neat::Run& run, std::vector<neat::MetadataSender>& metadata_senders,
                             std::vector<drone_tracker::ObjectTracker>& trackers,
                             const std::vector<sima_examples::RtspStreamInfo>& streams,
                             int max_detections, int frames, const std::atomic<bool>& stop) {
  std::vector<int> processed(metadata_senders.size(), 0);
  while (!stop && (frames == 0 || *std::min_element(processed.begin(), processed.end()) < frames)) {
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
    const auto slot = static_cast<std::size_t>(index);
    if (frames != 0 && processed.at(slot) >= frames) {
      continue;
    }
    const auto& stream = streams.at(slot);
    const auto tracks =
        trackers.at(slot).update(decode_boxes(sample, stream, max_detections), sample.pts_ns);
    const std::string payload = sima_examples::detection_egress::serialize_tracks(
        tracks, kLabels, stream.width, stream.height,
        sima_examples::detection_egress::frame_metadata(sample, index));
    // A nonblocking send drops the message while the socket buffer is full; the drops are
    // counted in the sender's stats and reported when the run ends. Any other failure throws.
    std::string send_error;
    if (!metadata_senders.at(slot).send_raw_json(payload, &send_error) && !send_error.empty()) {
      throw std::runtime_error("metadata send failed: " + send_error);
    }
    ++processed.at(slot);
  }
  return processed;
}

int run_app(const fs::path& config_path) {
  const auto config = sima_examples::Config::load(config_path);
  const auto sources = config.string_list("input.source");
  const auto streams = stream_info(config, sources);
  // Used by both the model and the box decode.
  const int max_detections = config.integer("inference.max_detections");
  const fs::path model_path = fs::absolute(config_path).parent_path() / config.string("model.path");
  neat::Model model(model_path.string(), model_options(config, streams.front(), max_detections));
  const neat::Graph detector = detector_graph(model);

  neat::Graph graph;
  std::vector<neat::MetadataSender> metadata_senders;
  std::vector<drone_tracker::ObjectTracker> trackers;
  const int stream_count = static_cast<int>(sources.size());
  for (int index = 0; index < stream_count; ++index) {
    const std::string& url = sources[static_cast<std::size_t>(index)];
    const auto& stream = streams[static_cast<std::size_t>(index)];
    const neat::Graph rtsp = groups::RtspEncodedInput(rtsp_options(config, url, stream));
    const neat::Graph decoder = decoder_graph(config, stream);
    // A plain link keeps every encoded frame. A dropped one corrupts the decoded frames after it
    // until the next keyframe.
    graph.connect(rtsp, decoder);
    graph.connect(decoder, detector, detector_link(index, stream_count));
    graph.connect(rtsp, groups::VideoSender(video_options(config, index)), video_link());
    std::string error;
    metadata_senders.emplace_back(metadata_options(config, index), &error);
    sima_examples::require(metadata_senders.back().ok(), error);
    trackers.emplace_back(tracker_settings(config), stream.fps);
    std::cout << "[stream " << index << "] rtsp=" << url << " stream=" << stream.width << "x"
              << stream.height << "@" << stream.fps
              << " insight=" << config.string("output.insight.host")
              << " video=" << config.integer("output.insight.video_port_base") + index
              << " metadata=" << metadata_senders.back().metadata_port() << "\n";
  }

  const std::atomic<bool>& stop = sima_examples::stop_on_signals();
  // The Run's destructor closes it when send_tracks throws.
  neat::Run run = graph.build(run_options());
  const auto processed = send_tracks(run, metadata_senders, trackers, streams, max_detections,
                                     config.integer("inference.frames"), stop);
  run.close();

  // Core candidate: after an RTSP run, the process exits with code 0 without flushing stdout, so
  // on a pipe or a file the lines below are lost unless std::endl flushes them (#673).
  for (std::size_t index = 0; index < metadata_senders.size(); ++index) {
    std::cout << "[stream " << index << "] processed=" << processed[index]
              << " tracks_started=" << trackers[index].tracks_started() << std::endl;
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
