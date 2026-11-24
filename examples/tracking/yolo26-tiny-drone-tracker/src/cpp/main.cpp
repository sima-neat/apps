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

// Tracks tiny drones in one RTSP stream with YOLO26, shown in Insight. The decoded frames feed
// the model. The tracker gives each drone an ID, and Insight draws every frame's tracks over the
// video.

#include "neat.h"
#include "sima_examples/config.h"
#include "sima_examples/detection_egress.h"
#include "sima_examples/example_utils.h"
#include "sima_examples/pull_status.h"
#include "sima_examples/rtsp_probe.h"
#include "sima_examples/stop_signals.h"
#include "utils/tracker.h"

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

// YOLO26 on the decoded NV12 frames.
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

groups::RtspEncodedInputOptions rtsp_options(const sima_examples::Config& config,
                                             const sima_examples::RtspStreamInfo& stream) {
  groups::RtspEncodedInputOptions options;
  options.url = config.string("input.source");
  options.codec = config.choice("input.codec", kRtspCodecs);
  options.tcp = config.boolean("input.rtsp.tcp");
  options.latency_ms = config.integer("input.rtsp.latency_ms");
  // With input.fps at 0, Core also takes the caps from the stream: above 60 fps, caps set from
  // the probed values alone failed to negotiate in some runs.
  options.auto_caps_from_stream = config.integer("input.fps") == 0;
  options.source_fps = stream.fps;
  if (options.codec == groups::RtspCodec::H264) {
    options.fallback_h264_width = stream.width;
    options.fallback_h264_height = stream.height;
  }
  return options;
}

// Hardware decode of the RTSP stream to NV12 frames.
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

// Decoded frames -> model -> "detections" output.
neat::Graph detector_graph(neat::Model& model) {
  neat::Graph graph("detector");
  graph.add(model.graph());
  graph.add(neat::nodes::Output("detections"));
  return graph;
}

// Send the encoded stream to Insight unchanged.
groups::VideoSenderOptions video_options(const sima_examples::Config& config) {
  auto options = groups::VideoSenderOptions::Passthrough(config.choice("input.codec", kRtspCodecs));
  options.host = config.string("output.insight.host");
  options.video_port_base = config.integer("output.insight.video_port_base");
  return options;
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

drone_tracker::TrackerSettings tracker_settings(const sima_examples::Config& config) {
  drone_tracker::TrackerSettings settings;
  settings.high_score_threshold = config.number("app.tracking.high_score_threshold");
  settings.new_track_threshold = config.number("app.tracking.new_track_threshold");
  settings.max_missing_frames = config.integer("app.tracking.max_missing_frames");
  settings.min_confirmed_hits = config.integer("app.tracking.min_confirmed_hits");
  return settings;
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

// Track every frame's boxes and send the tracks to Insight until `frames` (0: no limit) or a
// stop signal; return the count.
int send_tracks(neat::Run& run, neat::MetadataSender& metadata_sender,
                drone_tracker::ObjectTracker& object_tracker,
                const sima_examples::RtspStreamInfo& stream, int max_detections, int frames,
                const std::atomic<bool>& stop) {
  int processed = 0;
  while (!stop && (frames == 0 || processed < frames)) {
    neat::Sample sample;
    neat::PullError error;
    const auto status = run.pull("detections", kPullTimeoutMs, sample, &error);
    if (!sima_examples::pull_status_has_sample(status, "detections", error, run.last_error())) {
      continue;
    }
    const auto tracks =
        object_tracker.update(decode_boxes(sample, stream, max_detections), processed);
    const std::string payload = sima_examples::detection_egress::serialize_tracks(
        tracks, kLabels, stream.width, stream.height,
        sima_examples::detection_egress::frame_metadata(sample, 0));
    // A nonblocking send drops the message while the socket buffer is full; the drops are
    // counted in the sender's stats and reported when the run ends. Any other failure throws.
    std::string send_error;
    if (!metadata_sender.send_raw_json(payload, &send_error) && !send_error.empty()) {
      throw std::runtime_error("metadata send failed: " + send_error);
    }
    ++processed;
  }
  return processed;
}

int run_app(const fs::path& config_path) {
  const auto config = sima_examples::Config::load(config_path);
  // The stream's size and rate come from the probe for each value the config leaves at 0.
  const auto stream = sima_examples::probe_rtsp_stream_info(
      config.string("input.source"), config.boolean("input.rtsp.tcp"),
      {config.integer("input.width"), config.integer("input.height"), config.integer("input.fps")});
  // Used by both the model and the box decode.
  const int max_detections = config.integer("inference.max_detections");
  const fs::path model_path = fs::absolute(config_path).parent_path() / config.string("model.path");
  neat::Model model(model_path.string(), model_options(config, stream, max_detections));
  const neat::Graph detector = detector_graph(model);

  neat::Graph graph;
  const neat::Graph source = groups::RtspEncodedInput(rtsp_options(config, stream));
  const neat::Graph decoder = decoder_graph(config, stream);
  graph.connect(source, decoder);
  graph.connect(decoder, detector);
  // A plain link lets Core tee the encoded stream to the decoder and the video sender.
  graph.connect(source, groups::VideoSender(video_options(config)));

  std::string error;
  neat::MetadataSender metadata_sender(metadata_options(config), &error);
  sima_examples::require(metadata_sender.ok(), error);
  drone_tracker::ObjectTracker object_tracker(tracker_settings(config));
  const std::atomic<bool>& stop = sima_examples::stop_on_signals();
  // The Run's destructor closes it when send_tracks throws.
  neat::Run run = graph.build(run_options());
  std::cout << "source=" << config.string("input.source")
            << " codec=" << config.string("input.codec") << " stream=" << stream.width << "x"
            << stream.height << "@" << stream.fps
            << " insight=" << config.string("output.insight.host")
            << " video=" << config.integer("output.insight.video_port_base")
            << " metadata=" << config.integer("output.insight.metadata_port_base") << "\n";
  const int processed = send_tracks(run, metadata_sender, object_tracker, stream, max_detections,
                                    config.integer("inference.frames"), stop);
  run.close();

  // Core candidate: after an RTSP run, the process exits with code 0 without flushing stdout, so
  // on a pipe or a file the lines below are lost unless std::endl flushes them.
  std::cout << "processed=" << processed << " tracks_started=" << object_tracker.tracks_started()
            << std::endl;
  const auto failures = metadata_sender.stats().send_failures;
  if (failures > 0) {
    std::cout << failures << " metadata messages dropped" << std::endl;
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
